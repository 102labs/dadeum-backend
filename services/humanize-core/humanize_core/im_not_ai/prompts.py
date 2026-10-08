import json
import re
from typing import Any

from humanize_core.im_not_ai.audit import detect_register
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
GOOD_KOREAN_STYLE = (
    "좋은 한국어 업무 문장의 기준:\n"
    "- 주어와 서술어가 가깝고, 한 문장에 메시지가 하나다.\n"
    "- 추상 명사와 형식명사 대신 구체 동사로 서술한다. '~의 ~화를 위한 ~의 추진' 같은 명사 사슬을 푼다.\n"
    "- 문장은 접속사가 아니라 내용으로 이어진다. '또한/따라서/즉'은 대부분 지워도 흐름이 남는다.\n"
    "- 영어 직역 투(~에 대해, ~를 통해, ~에 의해, ~되어지다, 그/그녀)는 한국어 조사와 어미로 돌린다.\n"
    "- 상투구('시사하는 바가 크다', '지금이야말로 ~할 때다')는 지우거나 구체 사실로 바꾼다. 다른 상투구로 갈아 끼우지 않는다.\n"
    "- 원문의 격식·종결 등급·거리감을 그대로 둔다. 겸양('드립니다')을 낮추지 않는다.\n"
    "- 쉬운 말을 한자어로 바꾸지 않는다('늦었다'를 '지연되었다'로). 뜻이 같은 다른 단어로 갈아 끼우지 않는다('관리'를 '운영'으로).\n"
    "- 글쓴이의 판단 강도를 바꾸지 않는다. 추측은 추측으로, 예정은 예정으로, 가능은 가능으로 둔다."
)

CHANGES_CONTRACT = [
    "changes는 사용자에게 '무엇을 왜 바꿨는지' 보여 주는 목록이다. 실제로 바꾼 구간마다 하나씩 기록한다.",
    "original은 원문에 그대로 들어 있는 짧은 조각, revised는 revisedText에 그대로 들어 있는 조각이다. 한 문장을 넘지 않게 자른다.",
    "reason은 한국어 한 문장(40~80자)으로, 무엇이 왜 어색했고 어떻게 바꿨는지를 일상어로 쓴다. "
    "예: \"'~에 대해'는 영어 about을 직역한 표현이라 목적어로 바로 이었습니다.\"",
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
        "Apply the rulebook where it applies: remove translationese, AI-signature idioms, and mechanical repetition, and fix awkward word order, so the passage reads as if a careful Korean colleague wrote it. "
        "Edit only what is actually wrong. A sentence that is already natural and breaks no rule is copied unchanged; rewording it is itself an AI tell (over-editing). "
        "Keep facts, numbers, dates, names, quotations, URLs, and code intact. Keep the writer's modality (guess, possibility, plan, recommendation), degree, subject, tense and honorific level; style rules never license changing them. "
        "Use user_intent, tone, and preserve_formatting to choose tone and formatting. "
        "Do not add new claims, examples, metaphors, facts, or citations. "
        "Do not expose hidden reasoning. Return only JSON matching the schema.\n\n"
        + GOOD_KOREAN_STYLE
        + "\n\n"
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
        "edit_policy": [
            "탐지된 룰 위반(rewrite_priorities)과 직접 보이는 AI 티·번역투는 반드시 고친다. 수치·날짜·인용이 있는 문장도 그 주변 표현은 고친다.",
            "이미 자연스럽고 룰 위반이 없는 문장은 글자 그대로 둔다. 바꾸기 위해 바꾸지 않는다. 멀쩡한 문장을 흔드는 것도 AI 티다.",
            "동의어 교체 금지: '관리'→'운영', '결과'→'성과', '늦었다'→'지연되었다'처럼 뜻이 같은 다른 단어로 바꾸지 않는다. 쉬운 말을 한자어로 올리지 않는다.",
            "양태 유지: 추측(~로 보인다, ~것 같다), 가능(~할 수 있었다), 예정(~할 예정이다), 권고(~해야 한다)의 등급을 바꾸지 않는다. "
            "완곡 표현은 같은 글에서 습관처럼 반복될 때만 줄이고, 실제 불확실성·가능성은 그대로 둔다.",
            "정도 표현: 정도부사(매우·정말·대단히)는 지울 수 있지만 다른 강도의 말로 바꾸지 않는다. '대단히 어려운'→'어려운'은 되고 '쉽지 않은'은 안 된다.",
            "주체·시제·진행 유지: 주어를 빼거나 바꿔 행위 주체가 달라지지 않게 한다. '~고 있다'는 같은 글에서 여러 번 반복될 때 일부만 단순 시제로 줄이고, "
            "'지금 진행 중'이라는 뜻이 핵심인 문장은 그대로 둔다.",
            _register_policy(request),
            "지운 상투구를 다른 상투구로 갈아 끼우지 않는다('시사하는 바가 크다'→'여러 측면에서 의미가 있다'는 실패). 지우거나 구체 사실로 바꾼다.",
            "원문이 이미 깨끗하면 revisedText가 원문과 같아도 된다. 그때 changes는 비워 둔다.",
        ],
        "edit_intensity": {
            "target": "변경률 숫자가 아니라 룰북 신호 해결과 문체 체감성을 목표로 삼는다.",
            "minimum": "S1/S2 신호, 반복 표현, 번역투가 있으면 해당 구간에 실질 수정이 있어야 한다.",
            "avoid": "새 정보 추가, 과한 마케팅 톤, 원문 구조 파괴, 인용·수치·날짜 변경, 멀쩡한 문장 손대기, 동의어 교체",
        },
        "edit_examples": [
            "이번 프로젝트에 대해 간략히 공유드립니다 -> 이번 프로젝트를 간략히 공유드립니다 ('에 대해' 제거, '드립니다' 유지)",
            "데이터 분석을 통해 원인을 파악할 수 있었고 -> 데이터를 분석해 원인을 파악할 수 있었고 ('통해' 제거, 가능 양태 유지)",
            "여러 부서에 의해 검토되어졌고 -> 여러 부서가 검토했고 (by-피동과 이중 피동을 능동으로)",
            "결론적으로 하반기 전략의 핵심은 유지율이다 -> 하반기 전략의 핵심은 유지율이다 (결산 라벨 삭제)",
            "우리 팀은 강한 실행력을 가지고 있습니다 -> 우리 팀은 실행력이 강합니다 (have 직역을 형용사 서술로)",
            "하지 말 것: 대응이 40분 늦었습니다 -> 대응이 40분 지연되었습니다 (쉬운 말을 한자어로 바꾼 과윤문)",
            "하지 말 것: 시안은 이번 주까지 나올 거 같고요 -> 시안은 이번 주까지 나올 예정이며 (추측을 확정으로 바꾼 의미 변화)",
            "하지 말 것: 기업들이 앞다투어 도입하고 있다 -> 기업들이 앞다투어 도입한다 (단발 진행형을 지워 '지금 진행 중'이 사라짐)",
        ],
        "structured_output_contract": _REWRITE_STRUCTURED_OUTPUT_CONTRACT,
        "self_check_required": [
            "원문의 모든 문장·문단이 결과에 반영됐는지 확인한다.",
            "고유명사·수치·날짜·인용 100% 보존",
            "선택된 tone을 반영하되 업무 문맥과 격식 범위 보존",
            "잔존 S1 패턴 0건",
            "원문에 없는 사실·예시·비유·근거·과한 마케팅 문구 추가 없음",
            "바뀐 문장마다 '왜 바꿨는지'가 룰 위반 또는 눈에 보이는 어색함으로 설명되는지 확인한다. 설명이 안 되면 원문으로 되돌린다.",
            "추측·예정·가능·권고의 등급, 주어, 높임 등급이 원문과 같은지 확인한다.",
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
            "reason은 해당 구간의 변경만 설명한다. 다른 구간이나 전체 글 이야기는 쓰지 않는다. 항목마다 다른 문장으로 쓴다.",
            "한 구간에 변경이 둘 이상이면(예: 쉼표 삭제와 어휘 교체) 둘 다 한 문장 안에서 짚는다.",
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


_REGISTER_LABELS = {
    "formal": "합쇼체(~습니다/~입니다)",
    "haeyo": "해요체(~해요/~예요)",
    "plain": "해라체(~다/~한다)",
    "mixed": "혼합(해라체와 공손체가 섞임)",
    "unknown": "판별 불가",
}


def request_settings(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": request.user_intent,
        "mode_policy": "single_active_rewrite_with_preservation_audit",
        "tone": request.tone,
        "preserve_formatting": request.preserve_formatting,
        "source_register": _REGISTER_LABELS.get(detect_register(request.text), "판별 불가"),
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


def _register_policy(request: RewriteRequest) -> str:
    if request.tone == "formal":
        return (
            "종결 등급: tone=formal이므로 해요체·반말·구어('~할게요', '~거예요', '좀', '근데')를 합쇼체('~합니다', '~입니다')로 바꾼다. "
            "겸양 표현('드립니다')은 유지한다."
        )
    if request.tone == "friendly":
        return (
            "종결 등급: tone=friendly이므로 딱딱한 관료체 명사화와 피동은 풀되, 공손한 등급(합쇼체 또는 해요체)은 유지한다. "
            "겸양 표현('드립니다')을 낮추지 않는다."
        )
    return (
        "높임 등급 유지: '공유드립니다', '부탁드립니다' 같은 겸양·높임을 낮추지 않는다. "
        "settings.source_register의 종결 등급을 그대로 쓴다(합쇼체는 합쇼체로, 해요체는 해요체로). 혼합이면 가장 많이 쓰인 등급으로 통일한다."
    )


def _tone_guidance(tone: str) -> str:
    if tone == "formal":
        return (
            "격식 있는 비즈니스 문체로 바꾼다. 모든 문장을 하십시오/합니다 계열 종결로 바꾸고, "
            "구어적 축약과 느슨한 표현('좀', '근데', '~거 같고요')을 격식 표현으로 옮긴다. 이때 추측·가능성의 양태는 그대로 둔다"
            "('나올 거 같고요' → '나올 것으로 보입니다', '들어갈 수 있을 거예요' → '시작할 수 있을 것으로 보입니다'). "
            "전문적이되 과장 없는 어휘를 사용하고, 새 정보나 과한 권위 표현은 추가하지 않는다."
        )
    if tone == "friendly":
        return (
            "자연스럽고 부드러운 업무 문체로 조절한다. 딱딱한 명사화와 직역 표현을 풀고, 연결과 종결을 편하게 다듬되 "
            "업무상 예의와 신뢰감은 유지한다. 지나친 구어체, 감탄, 농담, 과장 표현은 피한다."
        )
    return (
        "기존 톤과 격식을 유지한다. settings.source_register의 종결 등급을 그대로 쓰고(합쇼체는 합쇼체로, 해요체는 해요체로), "
        "혼합이면 가장 많이 쓰인 등급으로 통일한다. 새 톤을 만들지 않되 어색한 표현, 반복, 번역투는 정리한다. "
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
