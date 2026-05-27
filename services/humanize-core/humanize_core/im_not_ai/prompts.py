import json
from typing import Any

from humanize_core.im_not_ai.resources import ai_tell_taxonomy, quick_rules, rewriting_playbook, scholarship
from humanize_core.im_not_ai.schemas import DetectionResult
from humanize_core.schemas import RewriteRequest


def fast_system_prompt() -> str:
    return (
        "You are the backend port of the im-not-ai Korean business rewrite engine. "
        "Perform detection, rewriting, and self-check in one call. "
        "Preserve facts, numbers, dates, names, quotations, protected terms, genre, and register exactly. "
        "Use user_intent, tone, rewrite_mode, and preserve_formatting to choose rewrite strength, tone, and formatting. "
        "Only edit AI-tell style, translationese, rhythm, structure, and business clarity. "
        "Do not expose hidden reasoning. Return only JSON matching the schema."
    )


def fast_user_prompt(request: RewriteRequest, genre_hint: str, context: dict[str, Any]) -> str:
    payload = {
        **_prompt_header("fast", request, genre_hint),
        "im_not_ai_quick_rules": quick_rules(),
        "metrics_before": context.get("metricsBefore", {}),
        "estimated_genre": context.get("estimatedGenre"),
        "preservation_terms": context.get("preservationTerms", []),
        "self_check_required": [
            "고유명사·수치·날짜·인용 100% 보존",
            "변경률 30% 이하, 50% 초과 금지",
            "장르 이탈 없음",
            "register 보존",
            "잔존 S1 패턴 0건",
            "인공 표현 추가 없음",
            "user_intent, tone, preserve_formatting 반영",
        ],
        "must_report": [
            "qualityLevel: A/B/C/D",
            "changeRate: 원문 대비 문자 변경률",
            "rollbackRequired: 변경률 50% 초과 또는 의미 보존 실패 시 true",
            "settingsApplied: user_intent, tone, protected_terms, preserve_formatting 반영 여부",
        ],
        "text": request.text,
    }
    return json.dumps(payload, ensure_ascii=False)


def detect_system_prompt() -> str:
    return (
        "You are ai-tell-detector for Korean text. Identify spans and document-level AI-tell patterns. "
        "Offsets must be based on the original Python string. Exclude numbers, names, quotations, and protected terms. "
        "Return only JSON matching the schema."
    )


def detect_user_prompt(request: RewriteRequest, genre_hint: str, context: dict[str, Any]) -> str:
    payload = {
        **_prompt_header("strict.detect", request, genre_hint),
        "taxonomy": ai_tell_taxonomy(),
        "metrics_before": context.get("metricsBefore", {}),
        "estimated_genre": context.get("estimatedGenre"),
        "preservation_terms": context.get("preservationTerms", []),
        "score_contract": {
            "severityWeightedScore": "S1=5, S2=2, S3=0.5 합계, 0~100 정규화",
            "aiTellDensity": "탐지 span 총 글자 수 / 전체 글자 수",
            "sentenceLengthStats": "mean, stdev, uniformity_warning",
        },
        "text": request.text,
    }
    return json.dumps(payload, ensure_ascii=False)


def strict_rewrite_system_prompt() -> str:
    return (
        "You are korean-style-rewriter for the im-not-ai strict pipeline. "
        "Rewrite only finding-backed AI-tell patterns. Preserve meaning, protected spans, genre, and register. "
        "Use user_intent, tone, rewrite_mode, and preserve_formatting to choose rewrite strength, tone, and formatting. "
        "Do not add new claims, examples, metaphors, facts, or citations. Return only JSON matching the schema."
    )


def strict_rewrite_user_prompt(
    request: RewriteRequest,
    genre_hint: str,
    context: dict[str, Any],
    detection: DetectionResult,
    previous_revised_text: str | None = None,
    audit_feedback: list[str] | None = None,
    review_feedback: list[str] | None = None,
) -> str:
    payload = {
        **_prompt_header("strict.rewrite", request, genre_hint),
        "quick_rules": quick_rules(),
        "rewriting_playbook": rewriting_playbook(),
        "metrics_before": context.get("metricsBefore", {}),
        "estimated_genre": context.get("estimatedGenre"),
        "preservation_terms": context.get("preservationTerms", []),
        "findings": detection.model_dump(exclude={"inputTokens", "outputTokens"}),
        "previous_revised_text": previous_revised_text,
        "audit_feedback": audit_feedback or [],
        "review_feedback": review_feedback or [],
        "diff_contract": {
            "edits": "findingId, before, after, category, reason, action, changeRate를 edit 단위로 기록",
            "findingsResolved": "해결한 finding id",
            "findingsUnresolved": "의도적으로 남긴 finding id와 이유는 summary에 기록",
            "overPolishWarning": "변경률 30% 초과 또는 문체 이탈 위험이면 true",
        },
        "text": request.text,
    }
    return json.dumps(payload, ensure_ascii=False)


def audit_system_prompt() -> str:
    return (
        "You are content-fidelity-auditor. Compare original and rewritten Korean text. "
        "Audit preservation of facts, numbers, dates, names, protected terms, quotations, and claims. "
        "Return only JSON matching the schema."
    )


def audit_user_prompt(
    request: RewriteRequest,
    genre_hint: str,
    context: dict[str, Any],
    revised_text: str,
    changes: list[dict[str, Any]],
) -> str:
    payload = {
        **_prompt_header("strict.audit", request, genre_hint),
        "scholarship_constraints": scholarship(),
        "preservation_terms": context.get("preservationTerms", []),
        "checklist_13": [
            "고유명사",
            "수치·단위",
            "날짜·시간",
            "직접 인용",
            "법률·규정 조문",
            "수식·공식",
            "주장·결론 방향",
            "인과관계",
            "주어 변경 의미",
            "양화·한정",
            "긍정·부정 극성",
            "순서",
            "누락·첨가",
        ],
        "audit_contract": "flaggedEdits에는 findingId/before/after/issue/checklistFailed/action을 기록하고, action은 rollback_required/rewrite_with_hedge_preserved/warning 중 하나를 사용",
        "original_text": request.text,
        "revised_text": revised_text,
        "changes": changes,
    }
    return json.dumps(payload, ensure_ascii=False)


def review_system_prompt() -> str:
    return (
        "You are naturalness-reviewer for Korean business rewriting. "
        "Judge residual AI-tell patterns, over-editing, genre drift, register drift, and whether another rewrite round is needed. "
        "Return only JSON matching the schema."
    )


def review_user_prompt(
    request: RewriteRequest,
    genre_hint: str,
    context: dict[str, Any],
    detection: DetectionResult,
    revised_text: str,
    audit_warnings: list[str],
) -> str:
    payload = {
        **_prompt_header("strict.review", request, genre_hint),
        "quick_rules": quick_rules(),
        "metrics_before": context.get("metricsBefore", {}),
        "preservation_terms": context.get("preservationTerms", []),
        "original_detection": detection.model_dump(exclude={"inputTokens", "outputTokens"}),
        "audit_warnings": audit_warnings,
        "review_contract": {
            "accept": "S1 0, S2 0, 과윤문 없음",
            "accept_with_note": "S1 0, S2 3건 이하, 과윤문 없음",
            "rewrite_round_2": "S1 잔존 또는 S2 4건 이상",
            "rollback_and_rewrite": "과윤문 또는 감사 조건부 통과",
            "hold_and_report": "최대 라운드 뒤에도 S1 3건 이상 또는 심각한 과윤문",
            "quality": "A: S1 0/S2<=2/70%+ 개선, B: S1 0/S2<=4/50%+ 개선, C/D는 원본 기준",
        },
        "original_text": request.text,
        "revised_text": revised_text,
    }
    return json.dumps(payload, ensure_ascii=False)


def _prompt_header(mode: str, request: RewriteRequest, genre_hint: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "genre_hint": genre_hint,
        "settings": request_settings(request),
        "rewrite_guidance": rewrite_guidance(request),
    }


def request_settings(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": request.user_intent,
        "rewrite_mode": request.rewrite_mode,
        "tone": request.tone,
        "protected_terms": request.protected_terms,
        "max_rounds": request.max_rounds,
        "preserve_formatting": request.preserve_formatting,
    }


def rewrite_guidance(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": _user_intent_guidance(request.user_intent),
        "rewrite_mode": _rewrite_mode_guidance(request.rewrite_mode),
        "tone": _tone_guidance(request.tone),
        "protected_terms": _protected_terms_guidance(request.protected_terms),
        "max_rounds": _max_rounds_guidance(request.rewrite_mode, request.max_rounds),
        "formatting": _formatting_guidance(request.preserve_formatting),
        "hard_constraints": [
            "원문에 없는 사실, 예시, 수치, 인용, 근거를 추가하지 않는다.",
            "고유명사, 날짜, 숫자, 단위, 직접 인용, protected_terms는 보존한다.",
            "user_intent가 사실 보존이나 protected_terms와 충돌하면 보존 규칙을 우선한다.",
        ],
    }


def _user_intent_guidance(user_intent: str) -> str:
    intent = user_intent.strip()
    if not intent:
        return "추가 사용자 지시가 없으므로 일반적인 한국어 비즈니스 윤문을 수행한다."
    return "사용자가 원하는 수정 방향이다. 의미 보존 범위 안에서 우선 반영한다: " + intent


def _rewrite_mode_guidance(rewrite_mode: str) -> str:
    if rewrite_mode == "strict":
        return (
            "정밀 윤문 모드다. 문장 흐름, AI 티 패턴, 어색한 번역투, 과윤문 위험을 검토하고 "
            "필요하면 최대 라운드 안에서 재윤문한다. 변경률은 원칙적으로 30% 이하로 유지한다."
        )
    return (
        "빠른 윤문 모드다. 의미와 문서 골격을 거의 그대로 두고 명확성, 어색한 표현, 리듬만 가볍게 다듬는다. "
        "불필요한 재구성은 피하고 변경률은 원칙적으로 20% 이하로 유지한다."
    )


def _tone_guidance(tone: str) -> str:
    if tone == "formal":
        return "격식 있는 비즈니스 문체로 조절한다. 예의 있고 단정한 종결, 과장 없는 전문적 표현을 사용한다."
    if tone == "friendly":
        return "친근하지만 업무 맥락을 해치지 않는 문체로 조절한다. 지나친 구어체, 감탄, 과장 표현은 피한다."
    return "기존 톤과 격식을 유지한다. 톤을 새로 만들기보다 어색한 부분만 자연스럽게 정리한다."


def _protected_terms_guidance(protected_terms: list[str]) -> str:
    if not protected_terms:
        return "사용자가 지정한 보호 용어는 없지만 숫자, 날짜, 고유명사, 직접 인용은 계속 보존한다."
    return "다음 보호 용어는 철자, 띄어쓰기, 대소문자를 그대로 유지한다: " + ", ".join(protected_terms)


def _max_rounds_guidance(rewrite_mode: str, max_rounds: int) -> str:
    if rewrite_mode == "strict":
        return f"정밀 검토는 최대 {max_rounds}라운드까지 수행한다. 각 라운드에서 감사/리뷰 피드백을 반영하되 과윤문은 피한다."
    return "빠른 윤문은 단일 라운드로 끝낸다. max_rounds 값이 있어도 fast 모드에서는 재윤문 루프를 사용하지 않는다."


def _formatting_guidance(preserve_formatting: bool) -> str:
    if preserve_formatting:
        return "원문의 줄바꿈, 문단, 목록, 번호, 표기 구조를 유지하고 문장 내부 표현만 다듬는다."
    return "가독성을 위해 문단, 줄바꿈, 목록 구조를 필요한 범위에서 정리할 수 있다."
