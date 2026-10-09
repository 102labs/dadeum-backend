import json
import re
from typing import Any

from humanize_core.im_not_ai.preservation import exact_preserve_targets as build_exact_preserve_targets
from humanize_core.im_not_ai.resources import compact_strict_rules, rule_card
from humanize_core.im_not_ai.schemas import AuditResult, ReviewSegment
from humanize_core.schemas import RewriteRequest


_REWRITE_STRUCTURED_OUTPUT_CONTRACT = [
    "revisedText is the single canonical final answer. It must contain the complete rewritten passage from the first original sentence through the final original sentence.",
    "Never put a partial prefix, continuation stub, excerpt, or dangling clause in revisedText.",
    "changes[].original and changes[].revised are local diff snippets only. Do not put the full passage in changes[].revised unless the entire passage truly changed as one unit.",
    "If revisedText and changes[].revised disagree, the response is invalid. Copy the complete final passage into revisedText before returning JSON.",
    "summary may describe what changed, but it must not be the only place that contains the completed rewrite.",
]
CHANGES_CONTRACT = [
    "changes는 사용자에게 '무엇을 왜 바꿨는지' 보여 주는 목록이다. 실제로 바꾼 구간마다 하나씩 기록한다.",
    "original은 원문에 그대로 들어 있는 짧은 조각, revised는 revisedText에 그대로 들어 있는 조각이다. 한 문장을 넘지 않게 자른다.",
    "reason은 한국어 한 문장, 15~40자로 짧게 쓴다. 무엇을 어떻게 바꿨는지만 적는다. "
    "예: \"'에 대해'를 빼고 목적어로 바로 이었습니다.\", \"피동을 능동으로 바꿨습니다.\"",
    "reason에 '사실관계는 그대로다', '의미는 유지했다', '원문 뜻은 같다' 같은 덧말을 붙이지 않는다. 바꾼 내용만 쓴다.",
    "reason은 original/revised 조각에 실제로 보이는 표현만 인용한다. 그 조각에 없는 표현이나 다른 구간의 수정을 끌어오지 않는다.",
    "reason에 룰 번호(A-1, D-2 등), '룰북', 'S1/S2', '탐지' 같은 내부 용어를 쓰지 않는다.",
    "type: clarity=뜻이 더 분명해짐, tone=말투·격식 조정, concision=군더더기 삭제, "
    "structure=어순·문장 분리·연결 변경, grammar=조사·어미·피동·맞춤법, meaning=뜻이 미세하게 달라질 수 있는 수정.",
    "riskLevel: low=의미 동일, medium=뉘앙스가 달라질 수 있음, high=사실·수치·주장에 영향 가능. high는 원칙적으로 만들지 않는다.",
]

SUMMARY_CONTRACT = [
    "summary는 2~4개 항목으로 전체 수정 방향만 적는다. 예: '영어 직역 투 연결어를 한국어 조사로 바꿨습니다.'",
    "summary에도 룰 번호나 내부 용어를 쓰지 않는다.",
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
        "Do not expose hidden reasoning. Return only JSON matching the schema.\n\n"
        "아래는 모든 rewrite 판단의 기준이 되는 한국어 윤문 룰북(im_not_ai_quick_rules)의 압축본이다. "
        "적용 원칙, 보존 규칙, 처리 순서, 전체 룰 목록(제목·심각도)을 담고 있으며, "
        "사용자 메시지의 원문에 이 원칙을 적극 적용한다. "
        "원문에서 자동 탐지된 룰의 상세 카드(윤문 대상·수정 방안·보존 예외)는 사용자 메시지의 "
        "rewrite_priorities.rulebook_hints[].ruleCard로 전달되므로, 탐지된 룰은 카드 기준으로 반드시 처리하고 "
        "목록에만 있는 룰도 원칙과 제목을 근거로 함께 살핀다.\n\n"
        + compact_strict_rules()
    )


def rewrite_user_prompt(request: RewriteRequest, context: dict[str, Any]) -> str:
    payload = {
        **_prompt_header("rewrite", request),
        "rulebook": "active-rewrite-rules",
        "rewrite_pass": "active_rulebook_single_pass",
        "rewrite_strategy": "active_rulebook_single_pass: 시스템 프롬프트의 룰북(im_not_ai_quick_rules)을 적극 적용해 원문 전체를 바로 윤문하고, 완성본 전체를 revisedText로 반환한다. 보존 감사는 다음 audit 단계가 담당한다.",
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
        "changes_contract": CHANGES_CONTRACT,
        "summary_contract": SUMMARY_CONTRACT,
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
        "The rewrite engine intentionally removes AI-tell idioms, converts passives to actives, and tightens endings per its Korean style rulebook; "
        "such edits are the product working as designed, not harm. Do not flag them by themselves. "
        "Flag a style edit only when it changed facts, claims, causality, polarity, quantities, or deontic/epistemic modality "
        "(for example a recommendation or inference turned into a flat assertion). "
        "When you flag lost modality, correctionDirection must propose a minimal fix that restores the modality without reinstating the removed idiom or copying the original sentence. "
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
                "룰북이 지시한 관용구 삭제('시사하는 바가 크다', '지금이야말로 ~할 때다', '것이다' 종결 등), 피동의 능동화, "
                "'~를 통해' 축소, 메타 서술 정리 그 자체. 이는 윤문 엔진의 의도된 동작이다.",
            ],
            "style_edit_policy": (
                "룰북 기반 문체 수정에서 당위(~해야 한다)나 추론(~로 보인다) 같은 양태가 소실된 경우에만 플래그한다. "
                "그때도 correctionDirection은 원문 문장 복원이 아니라, 삭제된 관용구를 되살리지 않으면서 "
                "양태만 되살리는 최소 대안 표현(예: '~해야 한다' 평서형)을 제시한다."
            ),
        },
        "exact_preserve_targets": exact_preserve_targets(request),
        "original_text": request.text,
        "revised_text": revised_text,
        "changes": changes,
    }
    return json.dumps(payload, ensure_ascii=False)


def review_system_prompt() -> str:
    return (
        "You are the preservation repair step for Korean business rewriting. "
        "You receive only the draft sentences that a fidelity audit flagged, each with the matching original "
        "sentence and the corrections to apply. Repair each listed sentence so that the flagged numbers, dates, "
        "names, quotations, protected terms, claims, causal links, polarity, and modality match the original again, "
        "while keeping every other improvement in the draft sentence. "
        "Never restore a removed AI-tell idiom just because it was in the original; restore the meaning, not the wording. "
        "Return one repaired text per segment index. Do not touch sentences that were not listed, do not merge or split "
        "segments, and do not add new information. Return only JSON matching the schema."
    )


def review_user_prompt(
    request: RewriteRequest,
    segments: list[ReviewSegment],
    audit_result: AuditResult,
) -> str:
    payload = {
        **_prompt_header("review", request),
        "repair_routine": [
            "1) 각 segment의 corrections를 draft_sentence에 반영한다. original_sentence는 참고용이다.",
            "2) 수치·날짜·단위·고유명사·직접 인용·protected_terms는 원문 표기를 글자 단위로 복원한다.",
            "3) 추론·권고·가능성 같은 양태가 사라졌다면 양태만 되살린다. 삭제된 상투구를 되살리거나 원문 문장을 통째로 복사하지 않는다.",
            "4) corrections에 없는 표현은 draft_sentence 그대로 둔다.",
            "5) 안전하게 고칠 수 없으면 그 index는 repairedSegments에서 빼고 unresolved에 이유를 적는다.",
        ],
        "output_contract": [
            "repairedSegments[].index는 입력 segment의 index와 같아야 한다.",
            "repairedSegments[].text는 그 문장 하나의 완성본이다. 앞뒤 문장을 붙이지 않는다.",
        ],
        "audit_summary": {"status": audit_result.status, "reason": audit_result.reason},
        "exact_preserve_targets": exact_preserve_targets(request),
        "segments": [segment.model_dump() for segment in segments],
    }
    return json.dumps(payload, ensure_ascii=False)


def explain_changes_system_prompt() -> str:
    return (
        "You label the differences between a Korean source text and its rewritten version for the end user. "
        "For each item you receive the source span, the rewritten span, and sometimes the rewriting model's own "
        "note about that span (draft_reason). Write why the span was changed, as an editor would explain it to "
        "the author: name the concrete expression that was awkward and what it became. Judge from the two spans "
        "and their context; use draft_reason only as a hint and ignore it when it does not match the visible change. "
        "Then write a short summary of the whole rewrite from the items you just explained. "
        "Return only JSON matching the schema."
    )


def explain_changes_user_prompt(
    request: RewriteRequest,
    revised_text: str,
    items: list[dict[str, Any]],
) -> str:
    payload = {
        "mode": "explain_changes",
        "settings": request_settings(request),
        "changes_contract": CHANGES_CONTRACT,
        "summary_contract": SUMMARY_CONTRACT,
        "output_contract": [
            "items[].index는 입력 항목의 index와 같아야 한다. 모든 입력 항목에 하나씩 답한다.",
            "reason은 해당 항목의 original/revised 조각에 보이는 변경만 설명한다. 다른 구간이나 전체 글 이야기는 쓰지 않고, 조각에 없는 표현은 인용하지 않는다. 항목마다 다른 문장으로 쓴다.",
            "한 구간에 변경이 둘 이상이면(예: 쉼표 삭제와 어휘 교체) 둘 다 한 문장 안에서 짧게 짚는다.",
            "reason은 15~40자로 끝낸다. '사실관계는 그대로다' 같은 보존 확인 문구나 평가는 쓰지 않는다.",
            "구간이 사실상 같은 뜻의 표현 교체면 그 이유를 쓰고, 뜻이 달라졌다면 type=meaning과 riskLevel=medium 이상으로 표시한다.",
            "summary는 items에 실제로 있는 변경만 근거로 2~4개 항목을 쓴다. 일어나지 않은 수정을 적지 않는다.",
        ],
        "original_text": request.text,
        "revised_text": revised_text,
        "items": items,
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


_MAX_PROMPT_HINTS = 24
_MAX_HINT_MATCHES = 4


def rulebook_hint_payload(item: dict[str, Any]) -> dict[str, Any]:
    rule_id = str(item.get("category", ""))
    raw_matches = item.get("matches") or []
    matches = [str(match) for match in raw_matches if str(match).strip()][:_MAX_HINT_MATCHES]
    return {
        "id": str(item.get("id", "")),
        "category": rule_id,
        "categoryLabel": str(item.get("categoryLabel", "")),
        "severity": str(item.get("severity", "")),
        "scope": str(item.get("scope", "")),
        "suggestedFix": str(item.get("suggestedFix", "")),
        "occurrences": int(item.get("occurrences", 1) or 1),
        "matches": matches,
        "ruleCard": rule_card(rule_id),
    }


def _rewrite_priorities(context: dict[str, Any]) -> dict[str, Any]:
    raw_hints = context.get("rulebookHints") or []
    hints = [
        rulebook_hint_payload(item)
        for item in raw_hints[:_MAX_PROMPT_HINTS]
        if isinstance(item, dict)
    ]
    if not hints:
        return {}
    return {
        "purpose": (
            "정규식 탐지기가 원문에서 확인한 룰 위반 후보다. 각 후보의 ruleCard(윤문 대상·수정 방안·보존 예외)와 "
            "matches(원문에서 발견된 실제 표현 표본)를 기준으로 해당 지점을 우선 수정한다. "
            "의미·수치·고유명사·인용·protected_terms 보존은 항상 우선한다."
        ),
        "rulebook_hints": hints,
        "priority_policy": [
            "S1 후보를 먼저, S2 후보를 다음으로 처리한다.",
            "matches의 표현은 본문에서 찾아 실제로 고치고, 반복 표현은 occurrences 수만큼 본문 전체에서 처리한다.",
            "후보는 감사 결과가 아니라 rewrite 우선순위다. ruleCard의 보존 예외에 해당하면 유지한다.",
            "후보 처리가 의미 보존과 충돌하면 보존을 우선한다. protected term 값은 후보에 포함하지 않는다.",
        ],
    }


def style_repair_system_prompt() -> str:
    return (
        "You are the residual style repair step of the im-not-ai Korean business rewrite engine. "
        "You receive a draft rewrite plus rulebook violations that a deterministic pattern gate still detects in the draft. "
        "Fix only the listed violations, and smooth paragraph-boundary connectors only when transition_policy asks for it. "
        "Never change facts, numbers, dates, names, quotations, URLs, code, protected terms, claims, or meaning. "
        "Preserve the modality of genuine inference, recommendation, and uncertainty: remove only empty habitual hedging, "
        "and never turn an uncertain claim into an assertion. "
        "Keep every sentence without a listed violation as close to the draft as possible. "
        "Return the complete repaired passage in revisedText; excerpts, summaries, or continuations are failures. "
        "Do not expose hidden reasoning. Return only JSON matching the schema."
    )


def style_repair_user_prompt(
    request: RewriteRequest,
    draft_text: str,
    residual_hints: list[dict[str, Any]],
    smooth_transitions: bool,
) -> str:
    payload: dict[str, Any] = {
        "mode": "style_repair",
        "settings": request_settings(request),
        "repair_contract": [
            "residual_rule_hints에 나열된 위반만 고친다. 나열되지 않은 문장은 초안 그대로 유지한다.",
            "각 후보의 ruleCard 수정 방안을 따르되, 보존 예외에 해당하면 그대로 둔다.",
            "완곡·권고 계열(G-1, G-2, I-4, D-6 등) 수리에서는 실제 추론·권고·불확실성의 양태를 유지하고, "
            "내용 없는 습관성 완곡만 제거한다. 확신 근거가 없는 주장을 단정형으로 바꾸지 않는다. "
            "예: '~기 때문인 것으로 보인다'는 추론이므로 '~영향이다' 같은 단정으로 바꾸지 않는다.",
            "revisedText에는 수리된 완성본 전체를 넣는다. 발췌, 요약, 이어쓰기는 실패다.",
            "의미, 수치, 날짜, 고유명사, 직접 인용, protected term은 바꾸지 않는다.",
            "수리하지 않는 단어와 문장은 draft_text와 글자 단위로 동일하게 유지한다. 새 오타나 용어 변형을 만들지 않는다.",
            "changes에는 실제 수리한 로컬 변경만 기록한다.",
        ],
        "changes_contract": CHANGES_CONTRACT,
        "residual_rule_hints": [
            rulebook_hint_payload(item) for item in residual_hints[:_MAX_PROMPT_HINTS] if isinstance(item, dict)
        ],
        "draft_text": draft_text,
    }
    if smooth_transitions:
        payload["transition_policy"] = (
            "이 초안은 문단 단위로 나뉘어 개별 윤문된 뒤 조립됐다. 문단 첫 문장의 접속·연결 표현이 "
            "앞 문단과 어색하게 이어지면 연결어만 가볍게 다듬는다. 문단의 내용, 순서, 개수는 바꾸지 않는다."
        )
    return json.dumps(payload, ensure_ascii=False)
