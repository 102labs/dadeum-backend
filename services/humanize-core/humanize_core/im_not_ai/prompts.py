import json
from typing import Any

from humanize_core.im_not_ai.resources import ai_tell_taxonomy, quick_rules, scholarship
from humanize_core.im_not_ai.schemas import DetectionResult, Finding
from humanize_core.schemas import RewriteRequest


_SEVERITY_ORDER = {"S1": 0, "S2": 1, "S3": 2}

_STRICT_REWRITE_OBJECTIVE = [
    "문장 단위가 아니라 전체 글의 흐름, 리듬, 연결, 명확성을 기준으로 자연스러운 한국어 비즈니스 문장으로 다듬는다.",
    "priority_findings는 우선순위 힌트다. 수정 범위를 finding span으로 제한하지 말고 같은 문장·문단의 어색한 연결, 반복, 번역투도 함께 정리한다.",
    "원문의 주장, 정보량, 순서, 관점, 장르, register, formatting intent는 유지한다.",
    "fast보다 깊게 보되 과장, 새 사실, 장식적 표현, 불필요한 재구성은 하지 않는다.",
]

_STRICT_REWRITE_RECIPES = {
    "A": "AI식 접속어, 번역투, 반복 대명사, 이중 피동, '통해/기반으로'식 추상 연결을 한국어 주어·서술어 중심으로 풀어 쓴다.",
    "B": "괄호 속 영어 병기나 병렬 설명이 흐름을 끊으면 필요한 정보만 남기고 한국어 문장 안에 자연스럽게 통합한다.",
    "C": "기계적인 번호·목록·문단 전개는 유지해야 할 구조만 보존하고, 문장 사이 연결과 설명 순서를 읽기 쉽게 조정한다.",
    "D": "동일한 문장 길이와 결론형 반복을 피하고, 핵심 문장은 또렷하게, 보조 설명은 짧게 배치한다.",
    "E": "첫째/둘째식 나열이나 반복 어미가 단조로우면 항목 간 의미 관계가 드러나도록 연결어와 문장 길이를 조절한다.",
    "F": "전략적/구조화/실행성 같은 명사형 추상어가 겹치면 실제 행위와 대상이 보이는 동사형 표현으로 바꾼다.",
    "G": "균형·신중함 표현이 과하게 중첩되면 판단 기준과 결론을 분명하게 남기고 방어적 완충어를 줄인다.",
    "H": "상투적인 도입·마무리는 원문 의도를 유지하면서 구체적인 업무 맥락과 자연스러운 종결로 정리한다.",
    "I": "피동·진행형·사역형이 겹치면 행위 주체와 결과가 보이는 능동형 문장으로 바꾼다.",
    "J": "목록·불릿·문단 형식은 preserve_formatting 설정을 따르되, 각 항목의 병렬성과 길이 균형을 맞춘다.",
}


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
        "You are an advanced Korean business writing editor for the im-not-ai strict pipeline. "
        "Produce the best natural rewrite for the whole passage: improve flow, rhythm, transitions, clarity, and readability. "
        "Treat detection findings as priority hints, not as the only editable spans. "
        "Preserve meaning, protected spans, facts, claims, order, genre, formatting intent, and register. "
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
        "advanced_rewrite_objective": _STRICT_REWRITE_OBJECTIVE,
        "metrics_before": context.get("metricsBefore", {}),
        "estimated_genre": context.get("estimatedGenre"),
        "preservation_terms": context.get("preservationTerms", []),
        "detection_summary": _compact_detection_summary(detection),
        "priority_findings": _priority_findings(detection),
        "targeted_rewrite_recipes": _targeted_rewrite_recipes(detection),
        "previous_revised_text": previous_revised_text,
        "audit_feedback": audit_feedback or [],
        "review_feedback": review_feedback or [],
        "self_check_required": [
            "원문의 모든 문장·문단이 결과에 반영됐는지 확인한다.",
            "숫자, 날짜, 고유명사, 인용, protected_terms, 핵심 주장과 인과관계를 보존한다.",
            "문장 흐름, 리듬, 연결, 명확성이 fast 초안보다 더 자연스러운지 확인한다.",
            "priority_findings의 S1/S2 신호를 가능한 한 해소하되 보존 규칙과 충돌하면 보존을 우선한다.",
            "새 사실, 예시, 비유, 근거, 과한 마케팅 문구를 추가하지 않는다.",
            "audit_feedback과 review_feedback이 있으면 같은 문제가 반복되지 않도록 반영한다.",
        ],
        "diff_contract": {
            "edits": "의미 있는 변경 단위별로 findingId(optional), before, after, category, reason, action, changeRate를 기록",
            "findingsResolved": "priority_findings 중 개선된 finding id",
            "findingsUnresolved": "보존 또는 문맥상 남긴 finding id와 이유는 summary에 기록",
            "overPolishWarning": "변경률 50% 초과, 핵심어 보존 실패, 문체 이탈, 새 정보 추가 위험이면 true. 30% 초과는 자동 실패가 아니라 리뷰 신호다.",
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
        "Judge whether the draft is better Korean business prose, not merely whether detected spans changed. "
        "Review residual AI-tell patterns, flow, rhythm, readability, over-editing, genre drift, register drift, and whether another rewrite round is needed. "
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
        "review_rubric": [
            "전체 글의 문장 흐름, 리듬, 연결, 명확성이 원문보다 나아졌는지 본다.",
            "탐지 findings만 고친 듯한 국소 수정에 그치지 않았는지 확인한다.",
            "원문 사실, 주장, 순서, 인용, protected_terms가 보존됐는지 본다.",
            "비즈니스 문체를 벗어난 장식적 표현, 새 사실, 과장, 마케팅 문구가 추가됐는지 본다.",
            "결과가 중간에 끊겼거나 일부 문장·문단을 누락했으면 accept하지 않는다.",
        ],
        "metrics_before": context.get("metricsBefore", {}),
        "preservation_terms": context.get("preservationTerms", []),
        "original_detection": _compact_detection_summary(detection),
        "priority_findings": _priority_findings(detection),
        "audit_warnings": audit_warnings,
        "review_contract": {
            "accept": "보존 문제 없음, 결과 완성, 문장 흐름·리듬·명확성 개선, S1 0, 과윤문 없음",
            "accept_with_note": "보존 문제 없음, 결과 완성, S1 0, S2 3건 이하, 경미한 아쉬움만 있음",
            "rewrite_round_2": "보존은 안전하지만 문장 흐름·리듬·명확성 개선 부족, S1 잔존, 또는 S2 4건 이상",
            "rollback_and_rewrite": "과윤문, 문체 이탈, 새 정보 추가, 감사 조건부 통과",
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
            "고도화 윤문 모드다. fast보다 더 깊게 전체 글의 문장 흐름, 리듬, 연결, 명확성, AI 티 패턴, "
            "어색한 번역투를 검토하고 필요하면 최대 라운드 안에서 재윤문한다. 변경률은 품질 목표가 아니라 "
            "보존 위험 신호로만 모니터링한다."
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


def _compact_detection_summary(detection: DetectionResult) -> dict[str, Any]:
    return {
        "estimatedGenre": detection.estimatedGenre,
        "sentenceCount": detection.sentenceCount,
        "detectedCount": detection.detectedCount,
        "aiTellDensity": detection.aiTellDensity,
        "severityWeightedScore": detection.severityWeightedScore,
        "categorySummary": detection.categorySummary,
        "sentenceLengthStats": detection.sentenceLengthStats,
    }


def _priority_findings(detection: DetectionResult, limit: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "id": finding.id,
            "category": finding.category,
            "categoryLabel": finding.categoryLabel,
            "severity": finding.severity,
            "scope": finding.scope,
            "textSpan": _clip(finding.textSpan, 160),
            "reason": _clip(finding.reason, 180),
            "suggestedFix": _clip(finding.suggestedFix, 180),
        }
        for finding in _sorted_findings(detection.findings)[:limit]
    ]


def _targeted_rewrite_recipes(detection: DetectionResult, limit: int = 8) -> list[dict[str, str]]:
    recipes: list[dict[str, str]] = []
    seen: set[str] = set()
    for finding in _sorted_findings(detection.findings):
        key = finding.category if finding.category in _STRICT_REWRITE_RECIPES else finding.category.split("-", 1)[0]
        recipe = _STRICT_REWRITE_RECIPES.get(key)
        if not recipe or key in seen:
            continue
        recipes.append({"category": key, "instruction": recipe})
        seen.add(key)
        if len(recipes) >= limit:
            break
    if recipes:
        return recipes
    return [
        {
            "category": "general",
            "instruction": "원문 전체를 읽고 흐름, 리듬, 연결, 명확성을 개선하되 사실과 문서 골격은 보존한다.",
        }
    ]


def _sorted_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda finding: (
            _SEVERITY_ORDER.get(finding.severity, 9),
            0 if finding.scope == "document" else 1,
            finding.start if finding.start is not None else 10**9,
            finding.category,
            finding.id,
        ),
    )


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"
