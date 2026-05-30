import copy
import json
from typing import Any, Protocol, TypeVar

from humanize_core.diff import build_fallback_changes, squeeze_spaces
from humanize_core.im_not_ai import prompts
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    RewriteResult,
    HumanizeContext,
    StrictReviewResult,
)
from humanize_core.schemas import LLMRewriteResult, RewriteRequest


class LLMConfigurationError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


MAX_OUTPUT_TOKENS = 20_000


class RewriteLLM(Protocol):
    async def rewrite(self, request: RewriteRequest) -> LLMRewriteResult:
        ...

    async def rewrite_once(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
    ) -> RewriteResult:
        ...


class StubRewriteLLM:
    """Deterministic local LLM replacement for tests and offline development."""

    async def rewrite(self, request: RewriteRequest) -> LLMRewriteResult:
        rewrite_result = await self.rewrite_once(
            request,
            HumanizeContext().model_dump(),
        )
        return _rewrite_result_to_llm_result(rewrite_result)

    async def rewrite_once(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
    ) -> RewriteResult:
        revised = request.text.strip() if request.preserve_formatting else squeeze_spaces(request.text).strip()
        if request.tone == "formal" and revised and not revised.endswith(("습니다.", "합니다.", ".", "!", "?")):
            revised = f"{revised}."
        if request.tone == "friendly":
            revised = _soften_stub_tone(revised)

        return RewriteResult(
            revisedText=revised,
            changes=build_fallback_changes(request.text, revised),
            summary=[_stub_summary(request)],
            inputTokens=max(1, len(request.text) // 4),
            outputTokens=max(1, len(revised) // 4),
        )


class OpenAIRewriteLLM:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for OpenAI provider")
        self.api_key = api_key
        self.model_name = model_name

    async def rewrite(self, request: RewriteRequest) -> LLMRewriteResult:
        rewrite_result = await self.rewrite_once(
            request,
            HumanizeContext().model_dump(),
        )
        return _rewrite_result_to_llm_result(rewrite_result)

    async def rewrite_once(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
    ) -> RewriteResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        response = await client.responses.create(
            model=self.model_name,
            instructions=prompts.rewrite_system_prompt(),
            input=prompts.rewrite_user_prompt(request, context),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            text={"format": _openai_rewrite_text_format()},
        )
        content = _extract_openai_output_text(response)
        result = _parse_llm_json(content, request.text)
        usage = response.usage
        if usage:
            result.inputTokens = getattr(usage, "input_tokens", 0) or 0
            result.outputTokens = getattr(usage, "output_tokens", 0) or 0
        return _llm_result_to_rewrite_result(result)


class AnthropicRewriteLLM:
    def __init__(self, api_key: str | None, model_name: str) -> None:
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is required for Anthropic provider")
        self.api_key = api_key
        self.model_name = model_name

    async def rewrite(self, request: RewriteRequest) -> LLMRewriteResult:
        rewrite_result = await self.rewrite_once(
            request,
            HumanizeContext().model_dump(),
        )
        return _rewrite_result_to_llm_result(rewrite_result)

    async def rewrite_once(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
    ) -> RewriteResult:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=self.api_key)
        response = await client.messages.create(
            model=self.model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.2,
            system=prompts.rewrite_system_prompt(),
            messages=[{"role": "user", "content": prompts.rewrite_user_prompt(request, context)}],
        )
        content = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        result = _parse_llm_json(content, request.text)
        result.inputTokens = response.usage.input_tokens
        result.outputTokens = response.usage.output_tokens
        return _llm_result_to_rewrite_result(result)


TStructuredResult = TypeVar(
    "TStructuredResult",
    RewriteResult,
    AuditResult,
    StrictReviewResult,
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
        rewrite_model_name: str,
        rewrite_fallback_model_name: str,
        strict_audit_model_name: str,
        strict_review_model_name: str,
    ) -> None:
        if not api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required for OpenRouter provider")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_title = app_title
        self.site_url = site_url
        primary_model = model_name if model_name and model_name != "stub" else rewrite_model_name
        self.rewrite_models = _dedupe_models([primary_model, rewrite_fallback_model_name])
        self.audit_models = _dedupe_models([strict_audit_model_name, primary_model])
        self.review_models = _dedupe_models([strict_review_model_name, primary_model])

    async def rewrite(self, request: RewriteRequest) -> LLMRewriteResult:
        rewrite_result = await self.rewrite_once(
            request,
            HumanizeContext().model_dump(),
        )
        return _rewrite_result_to_llm_result(rewrite_result)

    async def rewrite_once(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
    ) -> RewriteResult:
        return await self._chat_structured(
            models=self.rewrite_models,
            schema_name="rewrite_result",
            result_type=RewriteResult,
            system=prompts.rewrite_system_prompt(),
            user=prompts.rewrite_user_prompt(request, context),
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    async def audit(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
        revised_text: str,
        changes: list[dict[str, Any]],
    ) -> AuditResult:
        return await self._chat_structured(
            models=self.audit_models,
            schema_name="audit_result",
            result_type=AuditResult,
            system=prompts.audit_system_prompt(),
            user=prompts.audit_user_prompt(request, context, revised_text, changes),
            max_tokens=MAX_OUTPUT_TOKENS,
        )

    async def review(
        self,
        request: RewriteRequest,
        context: dict[str, Any],
        revised_text: str,
        audit_result: AuditResult,
    ) -> StrictReviewResult:
        return await self._chat_structured(
            models=self.review_models,
            schema_name="preservation_review_result",
            result_type=StrictReviewResult,
            system=prompts.review_system_prompt(),
            user=prompts.review_user_prompt(
                request,
                context,
                revised_text,
                audit_result,
            ),
            max_tokens=MAX_OUTPUT_TOKENS,
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
                    **_openrouter_chat_kwargs(
                        model=model,
                        system=system,
                        user=user,
                        max_tokens=max_tokens,
                        schema_name=schema_name,
                        schema=result_type.model_json_schema(),
                    )
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
    rewrite_model_name: str = "openai/gpt-5-mini",
    rewrite_fallback_model_name: str = "~anthropic/claude-haiku-latest",
    strict_audit_model_name: str = "openai/gpt-5",
    strict_review_model_name: str = "~anthropic/claude-haiku-latest",
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
            rewrite_model_name=rewrite_model_name,
            rewrite_fallback_model_name=rewrite_fallback_model_name,
            strict_audit_model_name=strict_audit_model_name,
            strict_review_model_name=strict_review_model_name,
        )
    raise LLMConfigurationError(f"Unsupported HUMANIZE_MODEL_PROVIDER: {provider}")


def _rewrite_result_to_llm_result(result: RewriteResult) -> LLMRewriteResult:
    return LLMRewriteResult(
        revisedText=result.revisedText,
        changes=result.changes,
        summary=result.summary,
        inputTokens=result.inputTokens,
        outputTokens=result.outputTokens,
    )


def _llm_result_to_rewrite_result(result: LLMRewriteResult) -> RewriteResult:
    return RewriteResult(
        revisedText=result.revisedText,
        changes=result.changes,
        summary=result.summary,
        inputTokens=result.inputTokens,
        outputTokens=result.outputTokens,
    )


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
            "schema": _openrouter_strict_schema(schema),
        },
    }


def _openrouter_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict_schema = copy.deepcopy(schema)
    _normalize_openrouter_schema_node(strict_schema)
    return strict_schema


def _normalize_openrouter_schema_node(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _normalize_openrouter_schema_node(item)
        return

    if not isinstance(node, dict):
        return

    node.pop("default", None)
    node.pop("title", None)

    properties = node.get("properties")
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        node["required"] = list(properties.keys())
        for child in properties.values():
            _normalize_openrouter_schema_node(child)
    elif node.get("type") == "object" and isinstance(node.get("additionalProperties"), dict):
        node["additionalProperties"] = False
        node["properties"] = {}
        node["required"] = []

    for key in ("$defs", "items", "anyOf", "allOf", "oneOf"):
        child = node.get(key)
        if child is not None:
            if key == "$defs" and isinstance(child, dict):
                for definition in child.values():
                    _normalize_openrouter_schema_node(definition)
            else:
                _normalize_openrouter_schema_node(child)


def _openrouter_chat_kwargs(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    # With require_parameters=true, optional sampling controls can remove all
    # eligible providers. Keep the OpenRouter structured request surface minimal.
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "response_format": _openrouter_response_format(schema_name, schema),
        "extra_body": {"provider": {"require_parameters": True}},
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
        "Preserve facts, numbers, dates, names, and quotations. "
        "Return a structured response with revisedText, changes, and summary. "
        "Each change must contain original, revised, reason, type, and riskLevel."
    )


def _user_prompt(request: RewriteRequest) -> str:
    payload = {
        "text": request.text,
        "settings": prompts.request_settings(request),
        "rewrite_guidance": prompts.rewrite_guidance(request),
    }
    return json.dumps(payload, ensure_ascii=False)


def _soften_stub_tone(text: str) -> str:
    if text.endswith("합니다."):
        return f"{text[:-4]}해요."
    return text


def _stub_summary(request: RewriteRequest) -> str:
    tone = {
        "keep": "기존 톤",
        "formal": "격식 있는 톤",
        "friendly": "친근한 톤",
    }[request.tone]
    intent = " 사용자 요청 방향을 반영했습니다." if request.user_intent.strip() else ""
    formatting = "형식을 보존했습니다." if request.preserve_formatting else "필요한 공백을 정리했습니다."
    return f"단일 윤문 기준으로 {tone}을 적용했습니다.{intent} {formatting}"


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
