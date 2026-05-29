import re
from difflib import SequenceMatcher

from humanize_core.schemas import Change

_DISPLAY_CONTEXT_CHARS = 14
_MAX_DISPLAY_CHANGES = 12
_MERGE_EQUAL_GAP_CHARS = 10
_MAX_GROUP_CHARS = 180


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


def build_display_safe_changes(original: str, revised: str, changes: list[Change]) -> list[Change]:
    """Return changes whose snippets are exact substrings of original/revised text.

    The SaaS comparison UI highlights by locating changes[].original and
    changes[].revised with exact substring matching. Model-produced review
    changes can be based on an intermediate draft, so this function rebuilds the
    display snippets from the final original/revised pair when needed.
    """

    if _changes_are_display_safe(original, revised, changes):
        return changes

    generated = _build_sequence_changes(original, revised, changes)
    return generated or build_fallback_changes(original, revised)


def _changes_are_display_safe(original: str, revised: str, changes: list[Change]) -> bool:
    if not changes:
        return original == revised

    for change in changes:
        if not _snippet_is_display_safe(original, change.original):
            return False
        if not _snippet_is_display_safe(revised, change.revised):
            return False

    return True


def _snippet_is_display_safe(text: str, snippet: str) -> bool:
    return snippet == "" or snippet in text


def _build_sequence_changes(original: str, revised: str, seed_changes: list[Change]) -> list[Change]:
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

    grouped_opcodes = _changed_opcode_groups(original, revised)
    if not grouped_opcodes:
        return []

    generated: list[Change] = []
    for index, group in enumerate(grouped_opcodes[:_MAX_DISPLAY_CHANGES]):
        original_start, original_end, revised_start, revised_end = group
        seed = seed_changes[min(index, len(seed_changes) - 1)] if seed_changes else None
        generated.append(
            Change(
                original=original[original_start:original_end],
                revised=revised[revised_start:revised_end],
                reason=seed.reason if seed else "원문과 최종 윤문 결과의 차이를 비교 가능한 구간으로 정리했습니다.",
                type=seed.type if seed else "clarity",
                riskLevel=seed.riskLevel if seed else "low",
            )
        )

    if len(grouped_opcodes) > _MAX_DISPLAY_CHANGES:
        generated.append(
            Change(
                original="",
                revised="",
                reason=f"세부 변경 구간이 {len(grouped_opcodes)}건이라 주요 {_MAX_DISPLAY_CHANGES}건만 비교 표시에 사용했습니다.",
                type="clarity",
                riskLevel="low",
            )
        )

    return generated


def _changed_opcode_groups(original: str, revised: str) -> list[tuple[int, int, int, int]]:
    matcher = SequenceMatcher(a=original, b=revised, autojunk=False)
    raw_groups: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None

    for tag, original_start, original_end, revised_start, revised_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        if current is None:
            current = (original_start, original_end, revised_start, revised_end)
            continue

        group_original_start, group_original_end, group_revised_start, group_revised_end = current
        original_gap = original_start - group_original_end
        revised_gap = revised_start - group_revised_end
        merged_original_len = original_end - group_original_start
        merged_revised_len = revised_end - group_revised_start

        if (
            original_gap <= _MERGE_EQUAL_GAP_CHARS
            and revised_gap <= _MERGE_EQUAL_GAP_CHARS
            and max(merged_original_len, merged_revised_len) <= _MAX_GROUP_CHARS
        ):
            current = (group_original_start, original_end, group_revised_start, revised_end)
            continue

        raw_groups.append(current)
        current = (original_start, original_end, revised_start, revised_end)

    if current is not None:
        raw_groups.append(current)

    return [_expand_group(original, revised, group) for group in raw_groups]


def _expand_group(
    original: str,
    revised: str,
    group: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    original_start, original_end, revised_start, revised_end = group
    prefix = min(_DISPLAY_CONTEXT_CHARS, original_start, revised_start)
    suffix = min(
        _DISPLAY_CONTEXT_CHARS,
        len(original) - original_end,
        len(revised) - revised_end,
    )
    return (
        original_start - prefix,
        original_end + suffix,
        revised_start - prefix,
        revised_end + suffix,
    )
