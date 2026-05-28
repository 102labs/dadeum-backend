import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from humanize_core.im_not_ai.schemas import DetectionResult, Finding, SelfCheckItem
from humanize_core.schemas import Change


_SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")
_NUMBER_RE = re.compile(r"\d[\d,./:-]*\d|\d")
_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|\d{4}-\d{2}-\d{2}")
_STANDARD_ABBREV_RE = re.compile(r"\b(?:AI|API|CPU|CSS|GPU|HTML|HTTP|ID|JSON|LLM|MCP|SDK|SQL|UI|URL|UX)\b")
_QUOTE_RE = re.compile(r'"([^"]{1,120})"|“([^”]{1,120})”|‘([^’]{1,120})’')
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")


SEVERITY_WEIGHTS = {"S1": 5.0, "S2": 2.0, "S3": 0.5}
SUPPORTED_QUICK_RULE_IDS = frozenset(
    {
        "A-1",
        "A-2",
        "A-3",
        "A-4",
        "A-5",
        "A-6",
        "A-7",
        "A-8",
        "A-9",
        "A-10",
        "A-11",
        "A-15",
        "A-16",
        "A-18",
        "A-19",
        "B-1",
        "B-2",
        "C-5",
        "C-7",
        "C-8",
        "C-9",
        "C-10",
        "C-11",
        "D-1",
        "D-2",
        "D-3",
        "D-4",
        "D-5",
        "D-6",
        "D-7",
        "E-1",
        "E-2",
        "E-7",
        "F-4",
        "F-5",
        "G-1",
        "G-2",
        "G-3",
        "H-1",
        "H-3",
        "H-4",
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "J-1",
        "J-2",
        "J-3",
    }
)


_QUICK_PATTERNS: tuple[tuple[str, str, str, str, re.Pattern[str], str], ...] = (
    ("A-1", "번역투: ~에 대해", "S1", "span", re.compile(r"에\s*대해(?:서)?"), "목적격 조사나 짧은 절로 바꿉니다."),
    ("A-2", "번역투: ~를 통해", "S1", "span", re.compile(r"(?:를|을)\s*통(?:해|하여)"), "~로, ~해서, ~함으로써 등으로 분산합니다."),
    ("A-3", "번역투: ~에 있어", "S1", "span", re.compile(r"에\s*있(?:어|어서)"), "~에서, ~을 볼 때로 줄입니다."),
    ("A-4", "번역투: ~라는 점에서", "S2", "span", re.compile(r"라는\s*점에서"), "~서, ~라는 이유로 줄입니다."),
    ("A-5", "번역투: 관련하여", "S2", "span", re.compile(r"(?:와|과)\s*관련(?:하여|한|된)|관련(?:하여|한|된)"), "~에, ~의로 줄입니다."),
    ("A-6", "번역투: 기반하여/바탕으로", "S2", "span", re.compile(r"에\s*(?:기반|바탕)(?:하여|한|으로)"), "~로, ~을 보고로 줄입니다."),
    ("A-7", "직역: 가지고 있다", "S1", "span", re.compile(r"가지고\s*있"), "동사나 형용사로 환원합니다."),
    ("A-8", "이중 피동", "S1", "span", re.compile(r"(?:되어진|보여진|쓰여진|잊혀진|여진)"), "능동 또는 단일 피동으로 줄입니다."),
    ("A-9", "피동: ~에 의해", "S2", "span", re.compile(r"에\s*의(?:해|하여)"), "행위자를 주어로 옮깁니다."),
    ("A-10", "가능 표현 남발", "S2", "span", re.compile(r"할\s*수\s*있"), "가능하면 단언형으로 정리합니다."),
    ("A-11", "목적절: ~을 위해", "S2", "span", re.compile(r"(?:을|를)\s*위해"), "~려고, ~위한으로 바꿉니다."),
    (
        "A-15",
        "추상 주어와 만능 동사",
        "S2",
        "span",
        re.compile(r"(?:기술|시대|데이터|변화|혁신|시장|환경|경험|전략|정책|흐름|성과|과제)(?:은|는|이|가)\s*(?:보여|드러내|시사|말해|묻|요구|가능하게|만들|이끌)"),
        "구체 주어 또는 부사절로 환원합니다.",
    ),
    ("A-16", "영어 대명사 직역", "S1", "span", re.compile(r"(?<![가-힣])그(?:녀|것|들)?(?:은|는|이|가|을|를|의|에게|도|만)?(?![가-힣])"), "생략하거나 호칭·명사구로 바꿉니다."),
    (
        "A-18",
        "관계절 좌향 수식",
        "S2",
        "span",
        re.compile(r"(?:(?:[가-힣A-Za-z0-9]+)\s+){3,}[가-힣A-Za-z0-9]+(?:한|하는|된|있는)\s+[가-힣A-Za-z0-9]+"),
        "문장 분리 또는 후치 동격절로 바꿉니다.",
    ),
    ("A-19", "이중 조사", "S2", "span", re.compile(r"(?:에서의|에로의|으로의|에의|으로부터의|로부터의)"), "절이나 구로 풀어 씁니다."),
    ("B-1", "괄호 영어 병기 반복", "S2", "span", re.compile(r"[가-힣][가-힣A-Za-z0-9\s]{0,30}\([A-Za-z][A-Za-z0-9 .&/_-]{2,}\)"), "첫 등장 이후에는 한글만 둡니다."),
    (
        "B-2",
        "영어 용어 비번역",
        "S2",
        "span",
        re.compile(r"\b(?:case|customer|framework|growth|impact|insight|issue|market|performance|risk|solution|strategy|target|trend|value)\b", re.IGNORECASE),
        "업계 표준이 아니면 한국어로 옮깁니다.",
    ),
    ("C-5", "이모지 남발", "S1", "span", _EMOJI_RE, "업무 문서에서는 삭제합니다."),
    ("C-7", "기계적 3단 접속", "S2", "span", re.compile(r"먼저|반면|결국"), "접속사를 줄이거나 본문에 녹입니다."),
    ("C-8", "대칭 대구 공식", "S2", "span", re.compile(r"[가-힣A-Za-z0-9]+인가[,\s·]+[가-힣A-Za-z0-9]+인가"), "한 번만 남기고 평서문으로 바꿉니다."),
    ("C-9", "숫자 괄호 인덱싱", "S2", "span", re.compile(r"(?:\(\d+\)|\d+\)|[①②③④⑤⑥⑦⑧⑨])"), "본문에 녹이거나 단순 줄바꿈으로 바꿉니다."),
    ("C-10", "콜론 부제 헤딩 반복", "S2", "span", re.compile(r"(?m)^.{1,40}:\s*.{1,80}$"), "짧은 헤딩 또는 평서문으로 바꿉니다."),
    ("C-11", "연결어미 뒤 쉼표", "S1", "span", re.compile(r"(?:고|며|지만|면서|아서|어서),"), "불필요한 쉼표를 제거합니다."),
    ("D-2", "AI 관용구", "S1", "span", re.compile(r"시사하는\s*바가\s*크다|주목할\s*만하다"), "삭제하거나 구체 결론으로 바꿉니다."),
    ("D-3", "강조 부사", "S1", "span", re.compile(r"본질적으로|핵심적으로"), "대부분 삭제합니다."),
    ("D-4", "hype 어휘", "S1", "span", re.compile(r"파격적|압도적|막강한|폭발적|대대적|강력한|획기적|치명적"), "구체 표현으로 낮춥니다."),
    ("D-5", "의인화 추상 주어", "S1", "span", re.compile(r"(?:기술|시대|데이터|변화|혁신|시장|환경|흐름)(?:은|는|이|가)\s*(?:묻|부르|말하|요구하)"), "사람·기관 주어로 환원합니다."),
    ("D-6", "결말 공식", "S1", "span", re.compile(r"지금이야말로|할\s*때(?:다|입니다)|해야\s*(?:한다|합니다)|시점(?:이다|입니다)"), "평서형으로 닫습니다."),
    ("D-7", "변환 공식", "S2", "span", re.compile(r"[가-힣A-Za-z0-9]+에서\s+[가-힣A-Za-z0-9]+로|[가-힣A-Za-z0-9]+을\s*넘어\s*[가-힣A-Za-z0-9]+로"), "한 번만 남기고 일반 서술로 바꿉니다."),
    ("E-2", "동일 종결/진행형 매핑", "S2", "span", re.compile(r"고\s*있"), "단순 시제는 현재형·과거형으로 환원합니다."),
    ("F-4", "한자어·영어 명사화 누적", "S2", "span", re.compile(r"[가-힣]{2,}(?:성|적|화)|[A-Za-z]+(?:tion|ment|ness|ity)\b", re.IGNORECASE), "동사·형용사 어근이나 구체 명사로 풉니다."),
    ("F-5", "~적 N 추상 체인", "S2", "span", re.compile(r"[가-힣A-Za-z]+적\s+[가-힣A-Za-z]+"), "명사+명사 또는 풀어쓰기 형태로 바꿉니다."),
    ("G-1", "미래 단정", "S2", "span", re.compile(r"(?:것이다|할\s*것이다)"), "현재형·확정형으로 줄입니다."),
    ("G-2", "추정 남발", "S2", "span", re.compile(r"로\s*보인다|인\s*듯하다|듯하다"), "단언 가능한 곳은 단언합니다."),
    ("G-3", "안전 균형 lexicon", "S2", "span", re.compile(r"양쪽\s*모두|두\s*가지\s*모두|장점도\s*있지만|신중하게|균형"), "구체 비교나 조건부 판단으로 바꿉니다."),
    ("H-1", "문두 접속사", "S2", "span", re.compile(r"(?:^|[.!?\n]\s*)(또한|따라서|즉|나아가|아울러|게다가|더욱이)"), "문장 자체의 흐름으로 연결합니다."),
    ("H-3", "메타 진입", "S1", "span", re.compile(r"이는|이\s*점에서|이\s*관점에서|이\s*말은"), "본문에 녹이거나 삭제합니다."),
    ("H-4", "재정의 접속사 즉 남발", "S2", "span", re.compile(r"즉"), "1회 정도만 남깁니다."),
    ("I-1", "형식명사 결말", "S1", "span", re.compile(r"(?:인|한)\s*것이다"), "평서형으로 줄입니다."),
    ("I-2", "점에 있다 공식", "S2", "span", re.compile(r"(?:은|는)\s*[^.!?\n]{1,60}라는\s+점에\s+있"), "직설형으로 바꿉니다."),
    ("I-3", "의미 설명 결말", "S2", "span", re.compile(r"다는\s*(?:뜻|의미)이다"), "본문에 풀어 씁니다."),
    ("I-4", "권고형 결말 반복", "S2", "span", re.compile(r"해야\s*(?:한다|합니다)"), "평서·단언으로 줄입니다."),
    ("J-1", "마크다운 강조", "S2", "span", re.compile(r"\*\*[^*\n]{1,80}\*\*"), "칼럼·리포트에서는 제거합니다."),
    ("J-3", "불릿 리스트", "S2", "span", re.compile(r"(?m)^\s*[-*]\s+.+$"), "문단 산문으로 통합합니다."),
)


def split_sentences(text: str) -> list[str]:
    return [match.group(0).strip() for match in _SENTENCE_RE.finditer(text) if match.group(0).strip()]


def change_rate(original: str, revised: str) -> float:
    if not original and not revised:
        return 0.0
    ratio = SequenceMatcher(a=original, b=revised).ratio()
    return round((1.0 - ratio) * 100, 2)


def local_detect(
    text: str,
    focus_categories: list[str] | None = None,
    protected_terms: Iterable[str] | None = None,
) -> DetectionResult:
    focus = {category.strip() for category in focus_categories or [] if category.strip()}
    protected_ranges = _protected_ranges(text, protected_terms or [])
    findings: list[Finding] = []
    for rule_id, label, severity, scope, pattern, fix in _QUICK_PATTERNS:
        if not _focus_allows(rule_id, focus):
            continue
        for index, match in enumerate(pattern.finditer(text), start=1):
            if rule_id not in {"B-1", "C-9"} and _overlaps_protected(match.start(), match.end(), protected_ranges):
                continue
            span = match.group(0)
            findings.append(
                Finding(
                    id=f"local-{rule_id}-{index}",
                    category=rule_id,
                    categoryLabel=label,
                    severity=severity,  # type: ignore[arg-type]
                    scope=scope,  # type: ignore[arg-type]
                    textSpan=span,
                    start=match.start(),
                    end=match.end(),
                    reason=f"{rule_id} quick-rule 후보가 감지됐습니다.",
                    suggestedFix=fix,
                )
            )

    findings.extend(_document_level_findings(text, focus, protected_ranges))
    findings = _dedupe_findings(findings)
    category_summary = finding_category_summary(findings)
    sentence_count = len(split_sentences(text))
    return DetectionResult(
        sentenceCount=sentence_count,
        sentenceLengthStats=sentence_length_stats(text),
        detectedCount=len(findings),
        aiTellDensity=finding_density(text, findings),
        severityWeightedScore=finding_score(findings),
        categorySummary=category_summary,
        findings=findings,
    )


def _document_level_findings(
    text: str,
    focus: set[str],
    protected_ranges: list[tuple[int, int]],
) -> list[Finding]:
    findings: list[Finding] = []
    sentences = split_sentences(text)
    if len(sentences) >= 4:
        lengths = [len(sentence) for sentence in sentences]
        mean = sum(lengths) / len(lengths)
        variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
        if variance**0.5 < 8 and _focus_allows("E-1", focus):
            findings.append(
                Finding(
                    id="local-E-1-document",
                    category="E-1",
                    categoryLabel="리듬: 문장 길이 균일",
                    severity="S2",
                    scope="document",
                    reason="문장 길이 표준편차가 낮아 리듬이 균일합니다.",
                    suggestedFix="문단마다 단문과 장문을 섞어 리듬을 조정합니다.",
                )
            )
        if _has_da_ending_streak(sentences) and _focus_allows("E-2", focus):
            findings.append(
                Finding(
                    id="local-E-2-document",
                    category="E-2",
                    categoryLabel="리듬: 동일 종결어미 연속",
                    severity="S2",
                    scope="document",
                    reason="~다 계열 종결이 4문장 이상 연속돼 normalisation 신호가 감지됐습니다.",
                    suggestedFix="종결어미와 문장 구조를 일부 다양화합니다.",
                )
            )

    pivots = ("결론적으로", "따라서", "이를 통해", "그러므로", "요약하면", "정리하면")
    pivot_count = sum(text.count(pivot) for pivot in pivots)
    if pivot_count >= 3 and _focus_allows("D-1", focus):
        findings.append(
            Finding(
                id="local-D-1-document",
                category="D-1",
                categoryLabel="결산 피벗 반복",
                severity="S2",
                scope="document",
                reason=f"결산 피벗 표현이 {pivot_count}회 반복됐습니다.",
                suggestedFix="1~2건만 남기고 나머지는 문맥 속 종결로 바꿉니다.",
            )
        )

    if _count_regex(text, re.compile(r"라는\s*점에서"), protected_ranges) >= 3 and _focus_allows("A-4", focus):
        findings.append(
            Finding(
                id="local-A-4-document",
                category="A-4",
                categoryLabel="번역투: ~라는 점에서 반복",
                severity="S2",
                scope="document",
                reason="~라는 점에서 표현이 3회 이상 반복됐습니다.",
                suggestedFix="~서, ~라는 이유로 등으로 분산합니다.",
            )
        )

    pronoun_count = _count_regex(text, re.compile(r"(?<![가-힣])그(?:녀|것|들)?(?:은|는|이|가|을|를|의|에게|도|만)?(?![가-힣])"), protected_ranges)
    if pronoun_count >= 3 and _focus_allows("A-16", focus):
        findings.append(
            Finding(
                id="local-A-16-document",
                category="A-16",
                categoryLabel="영어 대명사 직역 반복",
                severity="S2",
                scope="document",
                reason=f"그/그녀/그것/그들 계열 대명사가 {pronoun_count}회 반복됐습니다.",
                suggestedFix="절반 이상 생략하거나 호칭·명사구로 바꿉니다.",
            )
        )

    if _count_regex(text, re.compile(r"[가-힣]{2,}(?:성|적|화)|[A-Za-z]+(?:tion|ment|ness|ity)\b", re.IGNORECASE), protected_ranges) > 12 and _focus_allows("F-4", focus):
        findings.append(
            Finding(
                id="local-F-4-document",
                category="F-4",
                categoryLabel="한자어·영어 명사화 누적",
                severity="S2",
                scope="document",
                reason="명사화 접미사(-성/-적/-화/-tion/-ment/-ness/-ity)가 12회를 초과했습니다.",
                suggestedFix="동사·형용사 어근과 구체 명사로 해체합니다.",
            )
        )

    safe_balance_count = sum(text.count(term) for term in ("양쪽 모두", "두 가지 모두", "장점도 있지만", "신중하게", "균형"))
    if safe_balance_count > 4 and _focus_allows("G-3", focus):
        findings.append(
            Finding(
                id="local-G-3-document",
                category="G-3",
                categoryLabel="안전 균형 lexicon 반복",
                severity="S2",
                scope="document",
                reason=f"균형 어휘가 {safe_balance_count}회 반복됐습니다.",
                suggestedFix="구체 기준, 조건부 판단, 한쪽 입장으로 치환합니다.",
            )
        )

    quote_mark_count = len(re.findall(r"[\"“”‘’]", text))
    if quote_mark_count > 10 and _focus_allows("J-2", focus):
        findings.append(
            Finding(
                id="local-J-2-document",
                category="J-2",
                categoryLabel="따옴표 강조 남발",
                severity="S2",
                scope="document",
                reason="따옴표 강조가 5개 어휘를 초과한 것으로 추정됩니다.",
                suggestedFix="핵심 한두 개만 남기고 평어로 바꿉니다.",
            )
        )

    if _mixed_register(text) and _focus_allows("E-7", focus):
        findings.append(
            Finding(
                id="local-E-7-document",
                category="E-7",
                categoryLabel="청자 경어법 일관성 손실",
                severity="S2",
                scope="document",
                reason="한 문서 안에서 해라체·해요체·합쇼체 등 경어 단계가 섞였습니다.",
                suggestedFix="하나의 register로 통일합니다.",
            )
        )
    return findings


def sentence_length_stats(text: str) -> dict[str, float | bool]:
    sentences = split_sentences(text)
    if not sentences:
        return {"mean": 0.0, "stdev": 0.0, "uniformity_warning": False}
    lengths = [len(sentence) for sentence in sentences]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    stdev = variance**0.5
    return {
        "mean": round(mean, 2),
        "stdev": round(stdev, 2),
        "uniformity_warning": len(sentences) >= 4 and stdev < 8,
    }


def finding_score(findings: list[Finding]) -> float:
    return min(100.0, round(sum(SEVERITY_WEIGHTS.get(finding.severity, 1.0) for finding in findings), 2))


def finding_density(text: str, findings: list[Finding]) -> float:
    span_chars = 0
    for finding in findings:
        if finding.start is not None and finding.end is not None:
            span_chars += max(0, finding.end - finding.start)
        else:
            span_chars += len(finding.textSpan or finding.category)
    return round(span_chars / max(len(text), 1), 3)


def finding_category_summary(findings: list[Finding]) -> dict[str, int]:
    return dict(Counter(finding.category.split("-", 1)[0] for finding in findings))


def _protected_ranges(text: str, protected_terms: Iterable[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for pattern in (_NUMBER_RE, _DATE_RE, _QUOTE_RE, _STANDARD_ABBREV_RE):
        ranges.extend((match.start(), match.end()) for match in pattern.finditer(text))
    for term in protected_terms:
        if not term:
            continue
        for match in re.finditer(re.escape(term), text):
            ranges.append((match.start(), match.end()))
    return sorted(ranges)


def _overlaps_protected(start: int, end: int, protected_ranges: list[tuple[int, int]]) -> bool:
    return any(start < protected_end and end > protected_start for protected_start, protected_end in protected_ranges)


def _count_regex(text: str, pattern: re.Pattern[str], protected_ranges: list[tuple[int, int]]) -> int:
    return sum(
        1
        for match in pattern.finditer(text)
        if not _overlaps_protected(match.start(), match.end(), protected_ranges)
    )


def _focus_allows(rule_id: str, focus: set[str]) -> bool:
    if not focus:
        return True
    for category in focus:
        if "-" in category and rule_id == category:
            return True
        if "-" not in category and rule_id.startswith(f"{category}-"):
            return True
    return False


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    deduped: list[Finding] = []
    seen: set[tuple[str, int | None, int | None, str]] = set()
    for finding in findings:
        key = (finding.category, finding.start, finding.end, finding.textSpan)
        if key in seen:
            continue
        deduped.append(finding)
        seen.add(key)
    return deduped


def _has_da_ending_streak(sentences: list[str]) -> bool:
    streak = 0
    for sentence in sentences:
        if re.search(r"(?:다|니다|된다|이다)[.!?。！？]?$", sentence.strip()):
            streak += 1
            if streak >= 4:
                return True
        else:
            streak = 0
    return False


def _mixed_register(text: str) -> bool:
    registers = {
        "plain": bool(re.search(r"(?:^|[.!?\n]\s*)[^.!?\n]{2,}(?:한다|이다|된다)[.!?]?", text)),
        "haeyo": bool(re.search(r"(?:해요|네요|군요|어요)[.!?]?", text)),
        "formal": bool(re.search(r"(?:합니다|습니다|입니다)[.!?]?", text)),
        "hao": bool(re.search(r"(?:하오|하시오|하네)[.!?]?", text)),
    }
    return sum(1 for present in registers.values() if present) >= 2


def self_check_items(
    original: str,
    revised: str,
    protected_terms: list[str],
    residual_findings: list[Finding],
) -> list[SelfCheckItem]:
    rate = change_rate(original, revised)
    s1_count = sum(1 for finding in residual_findings if finding.severity == "S1")
    register_compatible = _formal_register_compatible(original, revised)
    return [
        SelfCheckItem(
            name="고유명사·수치·날짜·인용 보존",
            passed=True,
            note="자동 보존어 추출 없이 통과",
        ),
        SelfCheckItem(
            name="변경률 30% 이하",
            passed=rate <= 30,
            note=f"변경률 {rate:.2f}%",
        ),
        SelfCheckItem(
            name="register 보존",
            passed=register_compatible,
            note="격식체 호환" if register_compatible else "격식체 변화 후보",
        ),
        SelfCheckItem(
            name="잔존 S1 패턴 0건",
            passed=s1_count == 0,
            note=f"S1 잔존 {s1_count}건",
        ),
        SelfCheckItem(
            name="인공 표현 추가 없음",
            passed=rate <= 50,
            note="변경률 기준 통과" if rate <= 50 else "과윤문 후보",
        ),
    ]


def quality_grade(
    original: str,
    revised: str,
    residual_findings: list[Finding],
    checks: list[SelfCheckItem],
) -> tuple[str, str]:
    rate = change_rate(original, revised)
    passed = sum(1 for item in checks if item.passed)
    s1_count = sum(1 for finding in residual_findings if finding.severity == "S1")
    s2_count = sum(1 for finding in residual_findings if finding.severity == "S2")
    if s1_count >= 3 or rate > 50:
        return "D", "S1 잔존 3건 이상이거나 변경률 50% 초과입니다."
    if s1_count >= 1 or passed <= max(len(checks) - 2, 0):
        return "C", "S1 잔존 또는 자체검증 통과 항목 부족으로 strict 모드 권고 대상입니다."
    if s2_count <= 2 and 10 <= rate <= 25 and passed == len(checks):
        return "A", "S1 0건, S2 2건 이하, 변경률 10~25%, 자체검증 전 항목 통과입니다."
    if s2_count <= 4 and passed >= max(len(checks) - 1, 0):
        return "B", "S1 0건, S2 4건 이하, 자체검증 대부분 통과입니다."
    return "C", "S2 잔존 또는 자체검증 결과가 Fast 기준 경계값을 넘었습니다."


def over_polish_signals(original: str, revised: str) -> list[str]:
    signals: list[str] = []
    rate = change_rate(original, revised)
    if rate > 50:
        signals.append("change_rate_over_50")
    if rate > 30:
        signals.append("change_rate_over_30")
    if _formal_register_compatible(original, revised) is False:
        signals.append("register_drift")
    if re.search(r"듯|결|숨결|여운|풍경|서사|빛난다", revised):
        signals.append("literary_tone_added")
    original_terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", original))
    revised_terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", revised))
    if len(original_terms) >= 8:
        retained_ratio = len(original_terms & revised_terms) / max(len(original_terms), 1)
        if retained_ratio < 0.55:
            signals.append("keyword_replacement_excessive")
    return signals


def _formal_register_compatible(original: str, revised: str) -> bool:
    original_formal = bool(re.search(r"습니다|합니다|입니다", original))
    revised_formal = bool(re.search(r"습니다|합니다|입니다", revised))
    return not original_formal or revised_formal


def build_audit_warnings(original: str, revised: str, protected_terms: list[str]) -> list[str]:
    warnings: list[str] = []
    rate = change_rate(original, revised)
    if rate > 50:
        warnings.append(f"변경률이 {rate:.2f}%로 50%를 초과해 과윤문 위험이 큽니다.")
    elif rate > 30:
        warnings.append(f"변경률이 {rate:.2f}%로 30%를 초과했습니다.")
    return warnings


def mark_high_risk_if_needed(changes: list[Change], has_high_risk: bool) -> list[Change]:
    if not has_high_risk:
        return changes
    return [
        change.model_copy(update={"riskLevel": "high"}) if index == 0 else change
        for index, change in enumerate(changes)
    ]
