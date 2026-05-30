import json
import re
from typing import Any

from humanize_core.im_not_ai.preservation import exact_preserve_targets as build_exact_preserve_targets
from humanize_core.im_not_ai.resources import strict_rules
from humanize_core.im_not_ai.schemas import AuditResult
from humanize_core.schemas import RewriteRequest


_REWRITE_STRUCTURED_OUTPUT_CONTRACT = [
    "revisedText is the single canonical final answer. It must contain the complete rewritten passage from the first original sentence through the final original sentence.",
    "Never put a partial prefix, continuation stub, excerpt, or dangling clause in revisedText.",
    "changes[].original and changes[].revised are local diff snippets only. Do not put the full passage in changes[].revised unless the entire passage truly changed as one unit.",
    "If revisedText and changes[].revised disagree, the response is invalid. Copy the complete final passage into revisedText before returning JSON.",
    "charCountAfter must match revisedText length, not the length of a change snippet.",
    "summary may describe what changed, but it must not be the only place that contains the completed rewrite.",
]
def rewrite_system_prompt() -> str:
    return (
        "You are the backend port of the im-not-ai Korean business rewrite engine. "
        "Perform an active rewrite pass: improve the whole Korean business passage in one call. "
        "Your job is rewriting, not auditing; a later audit will check preservation problems. "
        "Apply the rulebook assertively: remove translationese, reduce repetition, improve word order, rhythm, transitions, and business clarity across the passage. "
        "Keep facts, numbers, dates, names, quotations, URLs, and code intact while applying the requested tone inside the same business context. "
        "Never use preservation as a reason to copy safe surrounding prose unchanged. "
        "revisedText must differ from the original with concrete wording edits whenever any safe expression can be improved. "
        "Use user_intent, tone, and preserve_formatting to choose tone and formatting. "
        "Do not add new claims, examples, metaphors, facts, or citations. "
        "Do not expose hidden reasoning. Return only JSON matching the schema."
    )


def rewrite_user_prompt(request: RewriteRequest, context: dict[str, Any]) -> str:
    payload = {
        **_prompt_header("rewrite", request),
        "rulebook": "active-rewrite-rules",
        "rewrite_pass": "active_rulebook_single_pass",
        "im_not_ai_quick_rules": strict_rules(),
        "rewrite_strategy": "active_rulebook_single_pass: 룰북을 적극 적용해 원문 전체를 바로 윤문하고, 완성본 전체를 revisedText로 반환한다. 보존 감사는 다음 audit 단계가 담당한다.",
        "rewrite_scope": "문장 흐름, 어순, 리듬, 연결, 명확성, 번역투, 반복 구조, AI 티 패턴을 전체 글 기준으로 적극 다듬고 여러 구간에서 실제 표현을 개선하되 원문의 의미와 정보량은 보존한다.",
        "must_edit_policy": [
            "원문을 그대로 반환하는 것은 rewrite 실패다.",
            "보존 대상, 코드, 직접 인용만으로 이루어진 입력이 아니라면 최소 하나 이상의 안전한 표현 개선을 만든다.",
            "보존해야 하는 값은 그대로 두되, 그 주변 문장 흐름·어순·반복·번역투·장황한 연결은 적극적으로 다듬는다.",
            "수치·날짜·직접 인용이 있다는 이유로 전체 문장을 복사하지 않는다.",
            "일반 업무 설명문은 의미가 같아도 표현, 어순, 연결, 종결 중 최소 하나는 더 자연스럽고 간결하게 바뀌어야 한다.",
            "changes가 비어 있거나 revisedText가 원문과 같으면 실패 출력으로 간주한다.",
        ],
        "edit_intensity": {
            "target": "변경률 숫자가 아니라 룰북 신호 해결과 문체 체감성을 목표로 삼는다.",
            "minimum": "S1/S2 신호, 반복 표현, 장황한 설명, 어색한 연결, 번역투가 하나라도 있으면 해당 구간에 실질 수정이 있어야 한다.",
            "avoid": "새 정보 추가, 과한 마케팅 톤, 원문 구조 파괴, 인용·수치·날짜 변경",
        },
        "edit_examples": [
            "켤 수도 있고, 실행할 수도 있습니다 -> 켜거나 실행할 수 있습니다",
            "꺼져 있다면 -> 꺼져 있으면",
            "먼저 목표를 달성하기 위한 작업 계획을 -> 목표 달성을 위한 작업 계획을 먼저",
            "사용할 만한 스킬들을 찾아 정리해줍니다 -> 관련 스킬을 찾아 정리해줍니다",
        ],
        "structured_output_contract": _REWRITE_STRUCTURED_OUTPUT_CONTRACT,
        "self_check_required": [
            "원문의 모든 문장·문단이 결과에 반영됐는지 확인한다.",
            "고유명사·수치·날짜·인용 100% 보존",
            "선택된 tone을 반영하되 업무 문맥과 격식 범위 보존",
            "잔존 S1 패턴 0건",
            "원문에 없는 사실·예시·비유·근거·과한 마케팅 문구 추가 없음",
            "user_intent, tone, preserve_formatting 반영",
        ],
        "must_report": [
            "qualityLevel: A/B/C/D",
            "rollbackRequired: 진단 신호일 뿐 rewrite 단계에서 원문으로 되돌리지 않는다. 의미 보존 실패, 누락, 출력 잘림, 보존 대상 변경이 의심될 때만 true",
            "settingsApplied: user_intent, tone, preserve_formatting 반영 여부",
        ],
        "completion_contract": _completion_contract(request),
        "text": request.text,
    }
    rewrite_priorities = _rewrite_priorities(context)
    if rewrite_priorities:
        payload["rewrite_priorities"] = rewrite_priorities
    return json.dumps(payload, ensure_ascii=False)


def audit_system_prompt() -> str:
    return (
        "You are content-fidelity-auditor. Compare original and rewritten Korean text. "
        "Audit only harmful changes: omissions, additions, changed numbers, dates, units, names, quotations, protected terms, key phrases, claims, causal relations, polarity, order, and meaning drift. "
        "Do not judge style quality or ask for broader polishing. "
        "If there are no harmful changes, return full_pass. If there are harmful changes, list only the exact corrections needed. "
        "Return only JSON matching the schema."
    )


def audit_user_prompt(
    request: RewriteRequest,
    context: dict[str, Any],
    revised_text: str,
    changes: list[dict[str, Any]],
) -> str:
    payload = {
        **_prompt_header("audit", request),
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
        "audit_contract": {
            "purpose": "rewrite 결과가 그대로 반환 가능한지 판단하고, 문제가 있을 때 review 단계가 원복할 수정 지시만 만든다.",
            "flaggedEdits": "before, after, issue, checklistFailed, action, correctionDirection, severity를 기록한다.",
            "actions": {
                "rewrite_required": "의미 보존을 위해 문장 일부를 고쳐야 함",
                "restore_original": "원문 표현을 되살리는 것이 가장 안전함",
                "preserve_exact": "숫자·고유명사·직접 인용 등을 글자 단위로 복원해야 함",
                "warning": "최종 반환은 가능하지만 사람이 알아야 할 경미한 주의점",
            },
            "status": "full_pass는 수정 필요 없음, conditional_pass는 review에서 고칠 항목 있음, fail은 누락/의미변경/잘림처럼 최종 차단 가능성이 큼",
            "do_not_flag": [
                "문체가 더 좋아질 수 있다는 일반 의견",
                "룰북 문제 패턴이 아직 남았다는 스타일 지적",
                "의미 변화가 없는 어순, 조사, 접속어, 문장 길이 조정",
            ],
        },
        "exact_preserve_targets": exact_preserve_targets(request),
        "original_text": request.text,
        "revised_text": revised_text,
        "changes": changes,
    }
    return json.dumps(payload, ensure_ascii=False)


def review_system_prompt() -> str:
    return (
        "You are the final preservation repair step for Korean business rewriting. "
        "Start from the draft rewrite and apply only the audit correction directions. "
        "Restore original numbers, dates, units, names, protected terms, direct quotations, URLs, code, legal clauses, claims, causal relations, order, polarity, and missing key phrases exactly where the audit flagged them. "
        "Do not perform new style polishing or rewrite unflagged sentences. "
        "Return the complete final revised passage and only the local changes you actually repaired. "
        "Return only JSON matching the schema."
    )


def review_user_prompt(
    request: RewriteRequest,
    context: dict[str, Any],
    revised_text: str,
    audit_result: AuditResult,
) -> str:
    payload = {
        **_prompt_header("review", request),
        "fixed_review_routine": [
            "1) preservation_audit.flaggedEdits의 수정 지시를 먼저 반영한다.",
            "2) flaggedEdits에 없는 문장, 문체, 연결, 리듬은 새로 고치지 않는다.",
            "3) 수치·날짜·단위·고유명사·직접 인용·protected_terms는 원문 표기를 글자 단위로 복원한다.",
            "4) 누락, 새 정보 추가, 주장 방향, 인과관계, 순서, 긍정·부정 극성, 양화·한정이 바뀐 부분만 원문에 가깝게 복원한다.",
            "5) 수정이 안전하지 않으면 해당 문장만 원문 표현을 유지한다.",
        ],
        "preservation_audit": audit_result.model_dump(),
        "review_contract": [
            "revisedText에는 최종 완성본 전체를 넣는다. 부분 문장, 이어쓰기, 요약은 실패다.",
            "changes는 audit 지시를 반영해 실제 복원한 로컬 변경만 기록한다.",
            "auditCorrectionsApplied에는 preservation_audit 지시 중 반영한 항목을 요약한다.",
            "finalAuditStatus는 audit 지시를 모두 반영했으면 full_pass로 둔다.",
            "finalBlockingIssues에는 audit 지시를 반영하지 못한 항목만 적는다.",
        ],
        "exact_preserve_targets": exact_preserve_targets(request),
        "original_text": request.text,
        "draft_revised_text": revised_text,
    }
    return json.dumps(payload, ensure_ascii=False)


def _prompt_header(mode: str, request: RewriteRequest) -> dict[str, Any]:
    return {
        "mode": mode,
        "settings": request_settings(request),
        "rewrite_guidance": rewrite_guidance(request),
    }


def request_settings(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": request.user_intent,
        "mode_policy": "single_active_rewrite_with_preservation_audit",
        "tone": request.tone,
        "preserve_formatting": request.preserve_formatting,
    }


def rewrite_guidance(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": _user_intent_guidance(request.user_intent),
        "rewrite_policy": _single_mode_guidance(),
        "tone": _tone_guidance(request.tone),
        "formatting": _formatting_guidance(request.preserve_formatting),
        "hard_constraints": [
            "원문에 없는 사실, 예시, 수치, 인용, 근거를 추가하지 않는다.",
            "고유명사, 날짜, 숫자, 단위, 직접 인용은 보존한다.",
            "user_intent가 사실 보존과 충돌하면 보존 규칙을 우선한다.",
        ],
    }


def exact_preserve_targets(request: RewriteRequest) -> dict[str, list[str]]:
    return build_exact_preserve_targets(request.text, request.protected_terms)


def _user_intent_guidance(user_intent: str) -> str:
    intent = user_intent.strip()
    if not intent:
        return "추가 사용자 지시가 없으므로 일반적인 한국어 비즈니스 윤문을 수행한다."
    return "사용자가 원하는 수정 방향이다. 의미 보존 범위 안에서 우선 반영한다: " + intent


def _single_mode_guidance() -> str:
    return (
        "단일 윤문 루틴이다. rewrite 단계는 룰북을 적극 적용해 전체 글을 먼저 자연스럽게 다듬고, 별도 감사 단계가 "
        "의미 변화와 보존 대상 변경을 검사한다. 보존이 안전한 문장은 어순·연결·반복·번역투를 실제로 개선한다. "
        "변경률은 품질 목표가 아니라 보존 위험 신호로만 본다."
    )


def _tone_guidance(tone: str) -> str:
    if tone == "formal":
        return (
            "격식 있는 비즈니스 문체로 조절한다. 단정한 서술형·하십시오/합니다 계열 종결을 우선하고, "
            "구어적 축약과 느슨한 표현을 줄이며, 전문적이되 과장 없는 어휘를 사용한다. "
            "새 정보나 과한 권위 표현은 추가하지 않는다."
        )
    if tone == "friendly":
        return (
            "자연스럽고 부드러운 업무 문체로 조절한다. 딱딱한 명사화와 직역 표현을 풀고, 연결과 종결을 편하게 다듬되 "
            "업무상 예의와 신뢰감은 유지한다. 지나친 구어체, 감탄, 농담, 과장 표현은 피한다."
        )
    return (
        "기존 톤과 격식을 유지한다. 새 톤을 만들지 않되 어색한 표현, 반복, 장황한 연결, 번역투는 적극적으로 정리한다. "
        "원문의 거리감과 말투를 보존하면서 문장 품질만 높인다."
    )


def _formatting_guidance(preserve_formatting: bool) -> str:
    if preserve_formatting:
        return "원문의 줄바꿈, 문단, 목록, 번호, 표기 구조를 유지하고 문장 내부 표현만 다듬는다."
    return "가독성을 위해 문단, 줄바꿈, 목록 구조를 필요한 범위에서 정리할 수 있다."


def _completion_contract(request: RewriteRequest) -> dict[str, Any]:
    original = request.text.strip()
    sentences = _prompt_sentences(original)
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", original) if paragraph.strip()]
    char_count = len(original)
    sentence_count = len(sentences)
    min_char_ratio = 0.75 if char_count >= 200 else 0.70
    return {
        "scope": "revisedText must contain the complete rewritten passage, never an excerpt, continuation, or summary.",
        "originalCharCount": char_count,
        "minimumSafeCharCount": int(char_count * min_char_ratio),
        "originalSentenceCount": sentence_count,
        "minimumSafeSentenceCount": max(1, int(sentence_count * 0.60)) if sentence_count else 0,
        "originalParagraphCount": len(paragraphs),
        "paragraphPolicy": (
            "preserve paragraph count and order unless preserve_formatting is false"
            if request.preserve_formatting
            else "paragraphs may be adjusted only when all original content remains covered"
        ),
        "failurePolicy": "If a sentence cannot be safely improved, keep it close to the original. Do not shorten the passage to satisfy style rules.",
    }


def _prompt_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in re.finditer(r"[^.!?。！？\n]+[.!?。！？]?", text)
        if match.group(0).strip()
    ]


def _rewrite_priorities(context: dict[str, Any]) -> dict[str, Any]:
    raw_hints = context.get("rulebookHints") or []
    hints: list[dict[str, str]] = []
    for item in raw_hints[:8]:
        if not isinstance(item, dict):
            continue
        hints.append(
            {
                "id": str(item.get("id", "")),
                "category": str(item.get("category", "")),
                "categoryLabel": str(item.get("categoryLabel", "")),
                "severity": str(item.get("severity", "")),
                "scope": str(item.get("scope", "")),
                "suggestedFix": str(item.get("suggestedFix", "")),
            }
        )
    if not hints:
        return {}
    return {
        "purpose": "원문에서 감지된 safe-edit 후보다. 의미·수치·고유명사·인용·protected_terms 보존을 우선하면서 가능한 후보를 rewrite 단계에서 해결한다.",
        "rulebook_hints": hints,
        "priority_policy": [
            "S1/S2 후보를 우선 처리한다.",
            "후보는 감사 결과가 아니라 rewrite 우선순위다.",
            "후보 처리가 의미 보존과 충돌하면 보존을 우선한다.",
            "원문 위치 정보와 protected term 값은 포함하지 않는다.",
        ],
    }
