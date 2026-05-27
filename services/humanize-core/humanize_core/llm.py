import json
from typing import Any, Protocol, TypeVar

from humanize_core.diff import build_fallback_changes, squeeze_spaces
from humanize_core.im_not_ai import prompts
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    DetectionResult,
    FastRewriteResult,
    HumanizeContext,
    NaturalnessReviewResult,
    StrictRewriteResult,
)
from humanize_core.schemas import LLMRewriteResult, RewriteRequest


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


class RewriteLLM(Protocol):
    async def rewrite(self, request: RewriteRequest, document_type: str) -> LLMRewriteResult:
        ...


class StubRewriteLLM:
    """Deterministic local LLM replacement for tests and offline development."""

    async def rewrite(self, request: RewriteRequest, document_type: str) -> LLMRewriteResult:
        revised = squeeze_spaces(request.text).strip()
        if request.concision in {"tighten", "compact"}:
            revised = revised.replace(" 매우 ", " ")
            revised = revised.replace(" 정말 ", " ")
        if request.tone == "formal" and revised and not revised.endswith(("습니다.", "합니다.", ".", "!", "?")):
            revised = f"{revised}."

        return LLMRewriteResult(
            revisedText=revised,
            changes=build_fallback_changes(request.text, revised),
            summary=["문장 표현을 업무 문서에 맞게 다듬었습니다."],
            inputTokens=max(1, len(request.text) // 4),
            outputTokens=max(1, len(revised) // 4),
        )


class OpenAIRewriteLLM:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for OpenAI provider")
        self.api_key = api_key
        self.model_name = model_name

    async def rewrite(self, request: RewriteRequest, document_type: str) -> LLMRewriteResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.responses.create(
            model=self.model_name,
            instructions=_system_prompt(),
            input=_user_prompt(request, document_type),
            text={"format": _openai_rewrite_text_format()},
        )
        content = _extract_openai_output_text(response)
        result = _parse_llm_json(content, request.text)
        usage = response.usage
        if usage:
            result.inputTokens = getattr(usage, "input_tokens", 0) or 0
            result.outputTokens = getattr(usage, "output_tokens", 0) or 0
        return result


class AnthropicRewriteLLM:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is required for Anthropic provider")
        self.api_key = api_key
        self.model_name = model_name

    async def rewrite(self, request: RewriteRequest, document_type: str) -> LLMRewriteResult:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)
        response = await client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            temperature=0.2,
            system=_system_prompt(),
            messages=[{"role": "user", "content": _user_prompt(request, document_type)}],
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        result = _parse_llm_json(content, request.text)
        result.inputTokens = response.usage.input_tokens
        result.outputTokens = response.usage.output_tokens
        return result


TStructuredResult = TypeVar(
    "TStructuredResult",
    FastRewriteResult,
    DetectionResult,
    StrictRewriteResult,
    AuditResult,
    NaturalnessReviewResult,
)


class OpenRouterRewriteLLM:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        app_title: str,
        site_url: str | None,
        model_name: str,
        fast_model_name: str,
        fast_fallback_model_name: str,
        strict_detect_model_name: str,
        strict_rewrite_model_name: str,
        strict_audit_model_name: str,
        strict_review_model_name: str,
        strict_escalation_model_name: str,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_title = app_title
        self.site_url = site_url
        primary_model = model_name if model_name and model_name != "stub" else fast_model_name
        self.fast_models = _dedupe_models([primary_model, fast_fallback_model_name])
        self.detect_models = _dedupe_models([strict_detect_model_name, primary_model])
        self.rewrite_models = _dedupe_models([strict_rewrite_model_name, primary_model, strict_escalation_model_name])
        self.audit_models = _dedupe_models([strict_audit_model_name, primary_model])
        self.review_models = _dedupe_models([strict_review_model_name, primary_model])

    async def rewrite(self, request: RewriteRequest, document_type: str) -> LLMRewriteResult:
        fast_result = await self.rewrite_fast(
            request,
            document_type,
            HumanizeContext(estimatedGenre=document_type).model_dump(),
        )
        return LLMRewriteResult(
            revisedText=fast_result.revisedText,
            changes=fast_result.changes,
            summary=fast_result.summary,
            inputTokens=fast_result.inputTokens,
            outputTokens=fast_result.outputTokens,
        )

    async def rewrite_fast(
        self,
        request: RewriteRequest,
        document_type: str,
        context: dict[str, Any],
    ) -> FastRewriteResult:
        return await self._chat_structured(
            models=self.fast_models,
            schema_name="fast_rewrite_result",
            result_type=FastRewriteResult,
            system=prompts.fast_system_prompt(),
            user=prompts.fast_user_prompt(request, document_type, context),
            max_tokens=5000,
            temperature=0.2,
        )

    async def detect(
        self,
        request: RewriteRequest,
        document_type: str,
        context: dict[str, Any],
    ) -> DetectionResult:
        return await self._chat_structured(
            models=self.detect_models,
            schema_name="detection_result",
            result_type=DetectionResult,
            system=prompts.detect_system_prompt(),
            user=prompts.detect_user_prompt(request, document_type, context),
            max_tokens=5000,
            temperature=0.1,
        )

    async def rewrite_strict(
        self,
        request: RewriteRequest,
        document_type: str,
        context: dict[str, Any],
        detection: DetectionResult,
        previous_revised_text: str | None,
        audit_feedback: list[str],
        review_feedback: list[str],
        *,
        use_escalation: bool = False,
    ) -> StrictRewriteResult:
        models = self.rewrite_models
        if use_escalation:
            models = _dedupe_models([self.rewrite_models[-1], *self.rewrite_models])
        return await self._chat_structured(
            models=models,
            schema_name="strict_rewrite_result",
            result_type=StrictRewriteResult,
            system=prompts.strict_rewrite_system_prompt(),
            user=prompts.strict_rewrite_user_prompt(
                request,
                document_type,
                context,
                detection,
                previous_revised_text=previous_revised_text,
                audit_feedback=audit_feedback,
                review_feedback=review_feedback,
            ),
            max_tokens=6000,
            temperature=0.2,
        )

    async def audit(
        self,
        request: RewriteRequest,
        document_type: str,
        context: dict[str, Any],
        revised_text: str,
        changes: list[dict[str, Any]],
    ) -> AuditResult:
        return await self._chat_structured(
            models=self.audit_models,
            schema_name="audit_result",
            result_type=AuditResult,
            system=prompts.audit_system_prompt(),
            user=prompts.audit_user_prompt(request, document_type, context, revised_text, changes),
            max_tokens=3000,
            temperature=0.0,
        )

    async def review(
        self,
        request: RewriteRequest,
        document_type: str,
        context: dict[str, Any],
        detection: DetectionResult,
        revised_text: str,
        audit_warnings: list[str],
    ) -> NaturalnessReviewResult:
        return await self._chat_structured(
            models=self.review_models,
            schema_name="naturalness_review_result",
            result_type=NaturalnessReviewResult,
            system=prompts.review_system_prompt(),
            user=prompts.review_user_prompt(
                request,
                document_type,
                context,
                detection,
                revised_text,
                audit_warnings,
            ),
            max_tokens=3000,
            temperature=0.1,
        )

    async def _chat_structured(
        self,
        *,
        models: list[str],
        schema_name: str,
        result_type: type[TStructuredResult],
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
    ) -> TStructuredResult:
        from openai import AsyncOpenAI

        headers = {"X-Title": self.app_title}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=headers,
        )

        last_error: Exception | None = None
        for model in models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=_openrouter_response_format(schema_name, result_type.model_json_schema()),
                    extra_body={"provider": {"require_parameters": True}},
                )
                content = _extract_chat_content(response)
                result = result_type.model_validate_json(content)
                input_tokens, output_tokens = _extract_chat_usage(response)
                return result.model_copy(
                    update={"inputTokens": input_tokens, "outputTokens": output_tokens}
                )
            except Exception as exc:  # noqa: BLE001 - sanitize and try configured fallback model.
                last_error = exc
        raise LLMResponseError("OpenRouter structured response failed") from last_error


def create_llm(
    provider: str,
    model_name: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
    *,
    openrouter_api_key: str | None = None,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_app_title: str = "Dadeum Humanize Core",
    openrouter_site_url: str | None = None,
    fast_model_name: str = "openai/gpt-5-mini",
    fast_fallback_model_name: str = "~anthropic/claude-haiku-latest",
    strict_detect_model_name: str = "openai/gpt-5-mini",
    strict_rewrite_model_name: str = "~anthropic/claude-sonnet-latest",
    strict_audit_model_name: str = "openai/gpt-5",
    strict_review_model_name: str = "~anthropic/claude-haiku-latest",
    strict_escalation_model_name: str = "~anthropic/claude-opus-latest",
) -> RewriteLLM:
    normalized = provider.lower().strip()
    if normalized == "stub":
        return StubRewriteLLM()
    if normalized == "openai":
        return OpenAIRewriteLLM(openai_api_key, model_name)
    if normalized == "anthropic":
        return AnthropicRewriteLLM(anthropic_api_key, model_name)
    if normalized == "openrouter":
        return OpenRouterRewriteLLM(
            api_key=openrouter_api_key,
            base_url=openrouter_base_url,
            app_title=openrouter_app_title,
            site_url=openrouter_site_url,
            model_name=model_name,
            fast_model_name=fast_model_name,
            fast_fallback_model_name=fast_fallback_model_name,
            strict_detect_model_name=strict_detect_model_name,
            strict_rewrite_model_name=strict_rewrite_model_name,
            strict_audit_model_name=strict_audit_model_name,
            strict_review_model_name=strict_review_model_name,
            strict_escalation_model_name=strict_escalation_model_name,
        )
    raise LLMConfigurationError(f"Unsupported HUMANIZE_MODEL_PROVIDER: {provider}")


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            deduped.append(model)
    return deduped


def _openrouter_response_format(schema_name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def _extract_chat_content(response: Any) -> str:
    choices = _get_value(response, "choices", []) or []
    first_choice = choices[0] if choices else None
    message = _get_value(first_choice, "message", {}) if first_choice is not None else {}
    content = _get_value(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            text = _get_value(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)
    return ""


def _extract_chat_usage(response: Any) -> tuple[int, int]:
    usage = _get_value(response, "usage", None)
    if not usage:
        return 0, 0
    input_tokens = (
        _get_value(usage, "prompt_tokens", None)
        or _get_value(usage, "input_tokens", None)
        or 0
    )
    output_tokens = (
        _get_value(usage, "completion_tokens", None)
        or _get_value(usage, "output_tokens", None)
        or 0
    )
    return int(input_tokens), int(output_tokens)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _system_prompt() -> str:
    return (
        "You are a Korean business writing rewrite engine. "
        "Do not store or reveal hidden reasoning. "
        "Preserve facts, numbers, dates, names, and protected terms. "
        "Return a structured response with revisedText, changes, and summary. "
        "Each change must contain original, revised, reason, type, and riskLevel."
    )


def _user_prompt(request: RewriteRequest, document_type: str) -> str:
    payload = {
        "text": request.text,
        "document_type": document_type,
        "intensity": request.intensity,
        "concision": request.concision,
        "tone": request.tone,
        "intent": request.intent,
        "protected_terms": request.protected_terms,
        "quality_mode": request.quality_mode,
        "focus_categories": request.focus_categories,
        "max_rounds": request.max_rounds,
        "preserve_formatting": request.preserve_formatting,
    }
    return json.dumps(payload, ensure_ascii=False)


def _openai_rewrite_text_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "rewrite_result",
        "description": "Structured rewrite result without usage metrics.",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "revisedText": {
                    "type": "string",
                    "description": "The polished version of the input text.",
                },
                "changes": {
                    "type": "array",
                    "description": "Important changes made to the text.",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "original": {
                                "type": "string",
                                "description": "A short source excerpt affected by the change.",
                            },
                            "revised": {
                                "type": "string",
                                "description": "The revised expression corresponding to original.",
                            },
                            "reason": {
                                "type": "string",
                                "description": "A concise Korean explanation of why the change was made.",
                            },
                            "type": {
                                "type": "string",
                                "enum": [
                                    "clarity",
                                    "tone",
                                    "concision",
                                    "structure",
                                    "grammar",
                                    "meaning",
                                ],
                            },
                            "riskLevel": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": ["original", "revised", "reason", "type", "riskLevel"],
                    },
                },
                "summary": {
                    "type": "array",
                    "description": "A short Korean summary of the overall rewrite.",
                    "items": {"type": "string"},
                },
            },
            "required": ["revisedText", "changes", "summary"],
        },
    }


def _extract_openai_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _parse_llm_json(content: str, original: str) -> LLMRewriteResult:
    try:
        data = json.loads(content)
        return LLMRewriteResult.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        revised = squeeze_spaces(content).strip() or original
        return LLMRewriteResult(
            revisedText=revised,
            changes=build_fallback_changes(original, revised),
            summary=["모델 응답을 정규화해 반환했습니다."],
        )
