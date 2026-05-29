import re


_DATE_RE = re.compile(r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|\d{4}-\d{1,2}-\d{1,2}")
_NUMBER_UNIT_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:%|퍼센트|원|달러|명|건|개|회|년|월|일|시간|분|초|kg|g|km|m|cm|GB|MB|KB)?"
)
_QUOTE_RE = re.compile(r'"[^"]{1,300}"|“[^”]{1,300}”|‘[^’]{1,300}’|\'[^\']{1,300}\'')
_URL_RE = re.compile(r"https?://[^\s)>\"]+|www\.[^\s)>\"]+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_CODE_SPAN_RE = re.compile(r"`[^`\n]{1,160}`")


def exact_preserve_targets(text: str, protected_terms: list[str]) -> dict[str, list[str]]:
    return {
        "protected_terms": _unique([term for term in protected_terms if term in text]),
        "numbers_and_units": _regex_values(_NUMBER_UNIT_RE, text),
        "dates": _regex_values(_DATE_RE, text),
        "direct_quotes": _regex_values(_QUOTE_RE, text),
        "urls": _regex_values(_URL_RE, text),
        "emails": _regex_values(_EMAIL_RE, text),
        "code_spans": _regex_values(_CODE_SPAN_RE, text),
    }


def preserved_units_for_text(text: str, protected_terms: list[str]) -> list[tuple[str, list[str]]]:
    targets = exact_preserve_targets(text, protected_terms)
    return [
        ("사용자 보호어", targets["protected_terms"]),
        ("직접 인용", targets["direct_quotes"]),
        ("URL", targets["urls"]),
        ("이메일", targets["emails"]),
        ("코드 표기", targets["code_spans"]),
        ("날짜", targets["dates"]),
        ("수치/단위", targets["numbers_and_units"]),
    ]


def checklist_for_preservation_label(label: str) -> list[int]:
    if label in {"수치/단위"}:
        return [2]
    if label == "날짜":
        return [3]
    if label == "직접 인용":
        return [4]
    if label in {"URL", "이메일", "코드 표기"}:
        return [5, 6]
    return [1, 13]


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
