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
        "Only edit AI-tell style, translationese, rhythm, structure, and business clarity. "
        "Do not expose hidden reasoning. Return only JSON matching the schema."
    )


def fast_user_prompt(request: RewriteRequest, document_type: str, context: dict[str, Any]) -> str:
    payload = {
        "mode": "fast",
        "document_type": document_type,
        "settings": _request_settings(request),
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
        ],
        "must_report": [
            "qualityLevel: A/B/C/D",
            "changeRate: 원문 대비 문자 변경률",
            "rollbackRequired: 변경률 50% 초과 또는 의미 보존 실패 시 true",
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


def detect_user_prompt(request: RewriteRequest, document_type: str, context: dict[str, Any]) -> str:
    payload = {
        "mode": "strict.detect",
        "document_type": document_type,
        "settings": _request_settings(request),
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
        "Do not add new claims, examples, metaphors, facts, or citations. Return only JSON matching the schema."
    )


def strict_rewrite_user_prompt(
    request: RewriteRequest,
    document_type: str,
    context: dict[str, Any],
    detection: DetectionResult,
    previous_revised_text: str | None = None,
    audit_feedback: list[str] | None = None,
    review_feedback: list[str] | None = None,
) -> str:
    payload = {
        "mode": "strict.rewrite",
        "document_type": document_type,
        "settings": _request_settings(request),
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
    document_type: str,
    context: dict[str, Any],
    revised_text: str,
    changes: list[dict[str, Any]],
) -> str:
    payload = {
        "mode": "strict.audit",
        "document_type": document_type,
        "settings": _request_settings(request),
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
    document_type: str,
    context: dict[str, Any],
    detection: DetectionResult,
    revised_text: str,
    audit_warnings: list[str],
) -> str:
    payload = {
        "mode": "strict.review",
        "document_type": document_type,
        "settings": _request_settings(request),
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


def _request_settings(request: RewriteRequest) -> dict[str, Any]:
    return {
        "intensity": request.intensity,
        "concision": request.concision,
        "tone": request.tone,
        "intent": request.intent,
        "protected_terms": request.protected_terms,
        "focus_categories": request.focus_categories,
        "preserve_formatting": request.preserve_formatting,
    }
