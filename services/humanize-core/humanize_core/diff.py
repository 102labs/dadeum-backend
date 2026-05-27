import re

from humanize_core.schemas import Change


def squeeze_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text)


def build_fallback_changes(original: str, revised: str) -> list[Change]:
    if original == revised:
        return [
            Change(
                original="",
                revised="",
                reason="원문의 의미와 표현을 유지했습니다.",
                type="clarity",
                riskLevel="low",
            )
        ]

    original_preview = original[:160]
    revised_preview = revised[:160]
    return [
        Change(
            original=original_preview,
            revised=revised_preview,
            reason="문장의 흐름과 전달력을 개선했습니다.",
            type="clarity",
            riskLevel="low",
        )
    ]

