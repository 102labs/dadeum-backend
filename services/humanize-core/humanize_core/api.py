from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from humanize_core.config import Settings, get_settings
from humanize_core.debug_log import RewriteDebugLogger
from humanize_core.graph import InputLimitError, RewriteGraphRunner
from humanize_core.jobs import RewriteJobError, RewriteJobManager
from humanize_core.llm import LLMConfigurationError, LLMResponseError, create_llm
from humanize_core.schemas import RewriteJobAccepted, RewriteJobStatus, RewriteRequest, RewriteResponse
from humanize_core.security import verify_core_request

logger = logging.getLogger("humanize_core")


def create_app(
    settings: Settings | None = None,
    graph_runner: RewriteGraphRunner | None = None,
    job_manager: RewriteJobManager | None = None,
) -> FastAPI:
    runtime_settings = settings or get_settings()
    debug_log = RewriteDebugLogger(runtime_settings)

    if graph_runner is None:
        llm = create_llm(
            runtime_settings.model_provider,
            runtime_settings.model_name,
            runtime_settings.openai_api_key,
            runtime_settings.anthropic_api_key,
            openrouter_api_key=runtime_settings.openrouter_api_key,
            openrouter_base_url=runtime_settings.openrouter_base_url,
            openrouter_app_title=runtime_settings.openrouter_app_title,
            openrouter_site_url=runtime_settings.openrouter_site_url,
            rewrite_model_name=runtime_settings.rewrite_model_name,
            rewrite_fallback_model_name=runtime_settings.rewrite_fallback_model_name,
            strict_audit_model_name=runtime_settings.strict_audit_model_name,
            strict_review_model_name=runtime_settings.strict_review_model_name,
        )
        graph_runner = RewriteGraphRunner(runtime_settings, llm, debug_log)

    runtime_job_manager = job_manager or RewriteJobManager(runtime_settings, graph_runner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.job_manager = runtime_job_manager
        await runtime_job_manager.start()
        try:
            yield
        finally:
            await runtime_job_manager.stop()

    app = FastAPI(title="Humanize Core", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/rewrite", response_model=RewriteResponse | RewriteJobAccepted)
    async def rewrite(request: Request):
        raw_body = await request.body()
        await verify_core_request(request, raw_body, runtime_settings)

        try:
            rewrite_request = RewriteRequest.model_validate_json(raw_body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

        if len(rewrite_request.text) > runtime_settings.max_chars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"text length exceeds HUMANIZE_MAX_CHARS ({runtime_settings.max_chars})",
            )

        request_id = request.headers.get("X-Request-Id", "")
        debug_log.event(
            "api.rewrite.accepted",
            request_id=request_id,
            status="accepted",
            details={
                "rewrite_mode": rewrite_request.rewrite_mode,
                "tone": rewrite_request.tone,
                "text_length": len(rewrite_request.text),
                "protected_terms_count": len(rewrite_request.protected_terms),
                "user_intent_length": len(rewrite_request.user_intent),
                "max_rounds": rewrite_request.max_rounds,
                "preserve_formatting": rewrite_request.preserve_formatting,
                "source_text": rewrite_request.text,
                "user_intent": rewrite_request.user_intent,
                "protected_terms": rewrite_request.protected_terms,
            },
        )
        if rewrite_request.rewrite_mode == "strict":
            try:
                accepted = await runtime_job_manager.enqueue(request_id, rewrite_request)
            except RewriteJobError as exc:
                logger.error("rewrite job enqueue failed")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Rewrite job storage is not configured",
                ) from exc
            logger.info(
                "rewrite job queued",
                extra={
                    "request_id": request_id,
                    "job_id": accepted.jobId,
                    "rewrite_mode": rewrite_request.rewrite_mode,
                    "text_length": len(rewrite_request.text),
                },
            )
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=accepted.model_dump(mode="json"),
            )

        try:
            response = await graph_runner.run(rewrite_request, request_id=request_id)
        except InputLimitError as exc:
            debug_log.event(
                "api.rewrite.failed",
                request_id=request_id,
                status="failed",
                error_code="input_limit_exceeded",
                details={"rewrite_mode": rewrite_request.rewrite_mode, "text_length": len(rewrite_request.text)},
            )
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except LLMConfigurationError as exc:
            debug_log.event(
                "api.rewrite.failed",
                request_id=request_id,
                status="failed",
                error_code="model_not_configured",
                details={"rewrite_mode": rewrite_request.rewrite_mode, "text_length": len(rewrite_request.text)},
            )
            logger.error("rewrite failed due to model configuration")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model provider is not configured",
            ) from exc
        except LLMResponseError as exc:
            debug_log.event(
                "api.rewrite.failed",
                request_id=request_id,
                status="failed",
                error_code="invalid_model_response",
                details={"rewrite_mode": rewrite_request.rewrite_mode, "text_length": len(rewrite_request.text)},
            )
            logger.error("rewrite failed due to invalid structured model response")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Model provider returned an invalid structured response",
            ) from exc

        debug_log.event(
            "api.rewrite.succeeded",
            request_id=request_id,
            status="succeeded",
            duration_ms=response.usage.latencyMs,
            details={
                "rewrite_mode": rewrite_request.rewrite_mode,
                "changes_count": len(response.changes),
                "summary_count": len(response.summary),
                "warnings_count": len(response.warnings),
                "input_tokens": response.usage.inputTokens,
                "output_tokens": response.usage.outputTokens,
                "rounds": response.usage.rounds,
                "revised_text_length": len(response.revisedText),
                "revised_text": response.revisedText,
                "changes": [change.model_dump(mode="json") for change in response.changes],
                "summary": response.summary,
                "warnings": response.warnings,
            },
        )
        logger.info(
            "rewrite succeeded",
            extra={
                "request_id": request_id,
                "rewrite_mode": rewrite_request.rewrite_mode,
                "latency_ms": response.usage.latencyMs,
            },
        )
        return response

    @app.get("/v1/rewrite-jobs/{job_id}", response_model=RewriteJobStatus)
    async def get_rewrite_job(job_id: str, request: Request) -> RewriteJobStatus:
        raw_body = await request.body()
        await verify_core_request(request, raw_body, runtime_settings)
        try:
            job_status = runtime_job_manager.get_status(job_id)
        except RewriteJobError as exc:
            logger.error("rewrite job status failed", extra={"job_id": job_id})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rewrite job storage is not configured",
            ) from exc
        if job_status is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rewrite job not found")
        return job_status

    @app.delete("/v1/rewrite-jobs/{job_id}", response_model=RewriteJobStatus)
    async def cancel_rewrite_job(job_id: str, request: Request) -> RewriteJobStatus:
        raw_body = await request.body()
        await verify_core_request(request, raw_body, runtime_settings)
        try:
            job_status = runtime_job_manager.cancel(job_id)
        except RewriteJobError as exc:
            logger.error("rewrite job cancel failed", extra={"job_id": job_id})
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rewrite job storage is not configured",
            ) from exc
        if job_status is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rewrite job not found")
        return job_status

    return app


app = create_app()
