import json
import re
from typing import Any

from humanize_core.im_not_ai.resources import strict_rules
from humanize_core.im_not_ai.schemas import AuditResult
from humanize_core.schemas import RewriteRequest


_STRICT_STRUCTURED_OUTPUT_CONTRACT = [
    "revisedText is the single canonical final answer. It must contain the complete rewritten passage from the first original sentence through the final original sentence.",
    "Never put a partial prefix, continuation stub, excerpt, or dangling clause in revisedText.",
    "changes[].original and changes[].revised are local diff snippets only. Do not put the full passage in changes[].revised unless the entire passage truly changed as one unit.",
    "If revisedText and changes[].revised disagree, the response is invalid. Copy the complete final passage into revisedText before returning JSON.",
    "charCountAfter must match revisedText length, not the length of a change snippet.",
    "summary may describe what changed, but it must not be the only place that contains the completed rewrite.",
]
_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|\d{4}-\d{1,2}-\d{1,2}")
_NUMBER_UNIT_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|원|달러|명|건|개|회|년|월|일|시간|분|초|kg|g|km|m|cm|GB|MB|KB)?"
)
_QUOTE_RE = re.compile(r'"[^"]{1,300}"|“[^”]{1,300}”|‘[^’]{1,300}’|\'[^\']{1,300}\'')
_URL_RE = re.compile(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_CODE_SPAN_RE = re.compile(r"`[^`\n]{1,160}`")


def rewrite_system_prompt() -> str:
    return (
        "You are the backend port of the im-not-ai Korean business rewrite engine. "
        "Perform one full-pass strict rewrite confidently and self-check in one call. "
        "Preserve facts, numbers, dates, names, quotations, and register exactly. "
        "Use strict-rules.md to edit AI-tell style, translationese, rhythm, structure, and business clarity. "
        "Do not become conservative solely because an audit stage exists; improve the whole passage when exact preservation is safe. "
        "Use user_intent, tone, and preserve_formatting to choose tone and formatting. "
        "Do not add new claims, examples, metaphors, facts, or citations. "
        "Do not expose hidden reasoning. Return only JSON matching the schema."
    )


def rewrite_user_prompt(request: RewriteRequest, context: dict[str, Any]) -> str:
    payload = {
        **_prompt_header("rewrite", request),
        "rulebook": "strict-rules.md",
        "strict_rules": strict_rules(),
        "rewrite_strategy": "single_full_pass: strict-rules.md를 기준으로 원문 전체를 한 번 자연스럽게 윤문하고 완성본 전체를 revisedText로 반환한다.",
        "rewrite_scope": "문장 흐름, 리듬, 연결, 명확성, 번역투, 반복 구조, AI 티 패턴을 전체 글 기준으로 다듬되 원문의 의미와 정보량은 보존한다.",
        "no_op_policy": [
            "원문을 그대로 반환하는 것은 실패다. 단, 이미 사람이 쓴 완성도 높은 문장이고 strict-rules.md 수정 후보가 전혀 없을 때만 허용한다.",
            "보존 대상은 글자 단위로 유지하되, 그 주변 문장 흐름·어순·반복·번역투·장황한 연결은 적극적으로 다듬는다.",
            "수치·날짜·직접 인용을 보존해야 한다는 이유로 전체 문장을 복사하지 않는다.",
            "일반 업무 설명문은 의미가 같아도 표현은 더 자연스럽고 간결하게 바뀌어야 한다.",
        ],
        "edit_intensity": {
            "target": "보존이 안전한 일반 문장은 10~25% 수준의 체감 변화가 나도록 다듬는다.",
            "minimum": "strict-rules.md의 S1/S2 신호가 있거나 문장이 장황하면 최소 1개 이상의 실질 수정이 있어야 한다.",
            "avoid": "새 정보 추가, 과한 마케팅 톤, 원문 구조 파괴, 인용·수치·날짜 변경",
        },
        "exact_preserve_targets": exact_preserve_targets(request),
        "structured_output_contract": _STRICT_STRUCTURED_OUTPUT_CONTRACT,
        "self_check_required": [
            "원문의 모든 문장·문단이 결과에 반영됐는지 확인한다.",
            "수치, 날짜, 시간, 단위, 금액, 퍼센트는 글자 단위로 보존한다.",
            "직접 인용, URL, 이메일, 코드 표기, 법률·규정 조문은 원문 그대로 보존한다.",
            "고유명사, 제품명, 모델명, 기관명, 사람 이름, 사용자가 지정한 protected_terms를 보존한다.",
            "핵심 주장, 결론 방향, 인과관계, 순서, 긍정·부정 극성, 양화·한정 표현을 바꾸지 않는다.",
            "register 보존",
            "잔존 S1 패턴 0건",
            "새 사실, 예시, 비유, 근거, 과한 마케팅 문구를 추가하지 않는다.",
            "user_intent, tone, preserve_formatting 반영",
        ],
        "must_report": [
            "qualityLevel: A/B/C/D",
            "changeRate: 원문 대비 문자 변경률",
            "rollbackRequired: 진단 신호일 뿐 rewrite 단계에서 원문으로 되돌리지 않는다. 의미 보존 실패, 누락, 출력 잘림, 보존 대상 변경이 의심될 때만 true",
            "settingsApplied: user_intent, tone, preserve_formatting 반영 여부",
        ],
        "completion_contract": _completion_contract(request),
        "text": request.text,
    }
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
        **_prompt_header("strict.audit", request),
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
                "strict-rules.md 문제 패턴이 아직 남았다는 스타일 지적",
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
        **_prompt_header("strict.review", request),
        "fixed_review_routine": [
            "1) strict_audit.flaggedEdits의 수정 지시를 먼저 반영한다.",
            "2) flaggedEdits에 없는 문장, 문체, 연결, 리듬은 새로 고치지 않는다.",
            "3) 수치·날짜·단위·고유명사·직접 인용·protected_terms는 원문 표기를 글자 단위로 복원한다.",
            "4) 누락, 새 정보 추가, 주장 방향, 인과관계, 순서, 긍정·부정 극성, 양화·한정이 바뀐 부분만 원문에 가깝게 복원한다.",
            "5) 수정이 안전하지 않으면 해당 문장만 원문 표현을 유지한다.",
        ],
        "strict_audit": audit_result.model_dump(),
        "review_contract": [
            "revisedText에는 최종 완성본 전체를 넣는다. 부분 문장, 이어쓰기, 요약은 실패다.",
            "changes는 audit 지시를 반영해 실제 복원한 로컬 변경만 기록한다.",
            "auditCorrectionsApplied에는 strict_audit 지시 중 반영한 항목을 요약한다.",
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
        "mode_policy": "strict_only",
        "tone": request.tone,
        "preserve_formatting": request.preserve_formatting,
    }


def rewrite_guidance(request: RewriteRequest) -> dict[str, Any]:
    return {
        "user_intent": _user_intent_guidance(request.user_intent),
        "rewrite_policy": _strict_only_guidance(),
        "tone": _tone_guidance(request.tone),
        "formatting": _formatting_guidance(request.preserve_formatting),
        "hard_constraints": [
            "원문에 없는 사실, 예시, 수치, 인용, 근거를 추가하지 않는다.",
            "고유명사, 날짜, 숫자, 단위, 직접 인용은 보존한다.",
            "user_intent가 사실 보존과 충돌하면 보존 규칙을 우선한다.",
        ],
    }


def exact_preserve_targets(request: RewriteRequest) -> dict[str, list[str]]:
    text = request.text
    return {
        "protected_terms": _unique([term for term in request.protected_terms if term in text]),
        "numbers_and_units": _regex_values(_NUMBER_UNIT_RE, text),
        "dates": _regex_values(_DATE_RE, text),
        "direct_quotes": _regex_values(_QUOTE_RE, text),
        "urls": _regex_values(_URL_RE, text),
        "emails": _regex_values(_EMAIL_RE, text),
        "code_spans": _regex_values(_CODE_SPAN_RE, text),
    }


def _regex_values(pattern: re.Pattern[str], text: str) -> list[str]:
    return _unique([match.group(0) for match in pattern.finditer(text) if match.group(0).strip()])


def _unique(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _user_intent_guidance(user_intent: str) -> str:
    intent = user_intent.strip()
    if not intent:
        return "추가 사용자 지시가 없으므로 일반적인 한국어 비즈니스 윤문을 수행한다."
    return "사용자가 원하는 수정 방향이다. 의미 보존 범위 안에서 우선 반영한다: " + intent


def _strict_only_guidance() -> str:
    return (
        "단일 strict 윤문 루틴이다. 전체 글을 한 번 자연스럽게 다듬은 뒤, 별도 감사 단계가 의미 변화와 "
        "보존 대상 변경을 검사한다. 감사 단계가 있으므로 rewrite는 과도하게 움츠러들지 말고, 보존이 안전한 "
        "문장은 실제로 읽히는 표현을 개선한다. 변경률은 품질 목표가 아니라 보존 위험 신호로만 본다."
    )


def _tone_guidance(tone: str) -> str:
    if tone == "formal":
        return "격식 있는 비즈니스 문체로 조절한다. 예의 있고 단정한 종결, 과장 없는 전문적 표현을 사용한다."
    if tone == "friendly":
        return "친근하지만 업무 맥락을 해치지 않는 문체로 조절한다. 지나친 구어체, 감탄, 과장 표현은 피한다."
    return "기존 톤과 격식을 유지한다. 톤을 새로 만들기보다 어색한 부분만 자연스럽게 정리한다."


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
