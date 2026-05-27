import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from humanize_core.config import Settings, get_settings
from humanize_core.graph import InputLimitError, RewriteGraphRunner
from humanize_core.llm import LLMConfigurationError, LLMResponseError, create_llm
from humanize_core.schemas import RewriteRequest, RewriteResponse
from humanize_core.security import verify_core_request

logger = logging.getLogger("humanize_core")


def create_app(settings: Settings | None = None, graph_runner: RewriteGraphRunner | None = None) -> FastAPI:
    runtime_settings = settings or get_settings()
    app = FastAPI(title="Humanize Core", version="0.1.0")

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
            fast_model_name=runtime_settings.fast_model_name,
            fast_fallback_model_name=runtime_settings.fast_fallback_model_name,
            strict_detect_model_name=runtime_settings.strict_detect_model_name,
            strict_rewrite_model_name=runtime_settings.strict_rewrite_model_name,
            strict_audit_model_name=runtime_settings.strict_audit_model_name,
            strict_review_model_name=runtime_settings.strict_review_model_name,
            strict_escalation_model_name=runtime_settings.strict_escalation_model_name,
        )
        graph_runner = RewriteGraphRunner(runtime_settings, llm)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/rewrite", response_model=RewriteResponse)
    async def rewrite(request: Request) -> RewriteResponse:
        raw_body = await request.body()
        await verify_core_request(request, raw_body, runtime_settings)

        try:
            rewrite_request = RewriteRequest.model_validate_json(raw_body)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

        try:
            response = await graph_runner.run(rewrite_request)
        except InputLimitError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except LLMConfigurationError as exc:
            logger.error("rewrite failed due to model configuration")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model provider is not configured",
            ) from exc
        except LLMResponseError as exc:
            logger.error("rewrite failed due to invalid structured model response")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Model provider returned an invalid structured response",
            ) from exc

        logger.info(
            "rewrite succeeded",
            extra={
                "request_id": request.headers.get("X-Request-Id"),
                "rewrite_mode": rewrite_request.rewrite_mode,
                "latency_ms": response.usage.latencyMs,
            },
        )
        return response

    return app


app = create_app()
