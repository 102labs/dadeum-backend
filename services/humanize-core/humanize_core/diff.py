import re
from difflib import SequenceMatcher

from humanize_core.schemas import Change

_DISPLAY_CONTEXT_CHARS = 14
# Diff groups are shown in document order up to this many; the rest are
# summarised in one trailing note so the comparison UI stays bounded.
_MAX_DISPLAY_CHANGES = 30
# Edits separated by at most this much unchanged text (a space or one short
# token) are shown as one change; anything further apart is a separate edit
# with its own reason.
_MERGE_EQUAL_GAP_CHARS = 4
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
    """Return one change per edited span of the final text, with the model's
    reasons attached where they apply.

    The SaaS comparison UI highlights by locating changes[].original and
    changes[].revised with exact substring matching, so the list is always
    rebuilt from a sequence diff of the final original/revised pair: every
    real edit is listed, snippets are guaranteed substrings, and a model
    change (possibly authored against an intermediate draft) only contributes
    its reason when its snippets still match the span it describes.
    """

    generated = _build_sequence_changes(original, revised, changes)
    return generated or build_fallback_changes(original, revised)


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

    located_seeds = _locate_seeds(original, revised, seed_changes)
    # Each model change explains at most one diff group: the one it overlaps
    # most. Otherwise a sentence-wide snippet would stamp the same reason on
    # every edit inside that sentence.
    assignments = _assign_seeds(grouped_opcodes[:_MAX_DISPLAY_CHANGES], located_seeds)
    generated: list[Change] = []
    shown_groups = grouped_opcodes[:_MAX_DISPLAY_CHANGES]
    for group_index, raw_group in enumerate(shown_groups):
        seed = assignments.get(group_index)
        previous = grouped_opcodes[group_index - 1] if group_index > 0 else None
        following = grouped_opcodes[group_index + 1] if group_index + 1 < len(grouped_opcodes) else None
        original_start, original_end, revised_start, revised_end = _expand_group(
            original, revised, raw_group, previous=previous, following=following
        )
        generated.append(
            Change(
                original=original[original_start:original_end],
                revised=revised[revised_start:revised_end],
                reason=seed.reason if seed else UNEXPLAINED_CHANGE_REASON,
                type=seed.type if seed else "clarity",
                riskLevel=seed.riskLevel if seed else "low",
            )
        )

    if len(grouped_opcodes) > _MAX_DISPLAY_CHANGES:
        generated.append(
            Change(
                original="",
                revised="",
                reason=f"세부 변경 구간이 {len(grouped_opcodes)}건이라 앞에서부터 {_MAX_DISPLAY_CHANGES}건까지만 표시했습니다. 나머지 {len(grouped_opcodes) - _MAX_DISPLAY_CHANGES}건은 윤문 결과에는 반영되어 있습니다.",
                type="clarity",
                riskLevel="low",
            )
        )

    return generated


# Reason used for a diff group that no model-authored change explains. The
# finalize stage hands these groups to the provider's explain step when it has
# one; otherwise the generic text ships as-is.
UNEXPLAINED_CHANGE_REASON = "원문과 최종 윤문 결과의 차이를 비교 가능한 구간으로 정리했습니다."


def is_unexplained_change(change: Change) -> bool:
    return change.reason == UNEXPLAINED_CHANGE_REASON


def _locate_seeds(
    original: str,
    revised: str,
    seeds: list[Change],
) -> list[tuple[Change, tuple[int, int] | None, tuple[int, int] | None]]:
    """Find where each model-authored change sits in the source and result.

    A snippet that is not present (it referred to an intermediate draft) gets
    no range and can never be matched to a diff group, so its reason cannot be
    attached to the wrong edit.
    """

    located = []
    for seed in seeds:
        original_range = _find_range(original, seed.original)
        revised_range = _find_range(revised, seed.revised)
        # A non-empty snippet that is absent means the final edit differs from
        # what the model described (a later stage changed it again), so the
        # reason no longer applies. Empty snippets are insertions/deletions and
        # are matched on the side they do have.
        if seed.original and original_range is None:
            continue
        if seed.revised and revised_range is None:
            continue
        if original_range is None and revised_range is None:
            continue
        located.append((seed, original_range, revised_range))
    return located


def _find_range(text: str, snippet: str) -> tuple[int, int] | None:
    if not snippet:
        return None
    start = text.find(snippet)
    if start < 0:
        return None
    return start, start + len(snippet)


def _assign_seeds(
    groups: list[tuple[int, int, int, int]],
    located_seeds: list[tuple[Change, tuple[int, int] | None, tuple[int, int] | None]],
) -> dict[int, Change]:
    """Map diff-group index -> model change, each seed used for its best group only."""

    candidates: list[tuple[int, int, int]] = []  # (overlap, group_index, seed_index)
    for group_index, (original_start, original_end, revised_start, revised_end) in enumerate(groups):
        for seed_index, (_seed, original_range, revised_range) in enumerate(located_seeds):
            overlap = _overlap((original_start, original_end), original_range) + _overlap(
                (revised_start, revised_end), revised_range
            )
            if overlap > 0:
                candidates.append((overlap, group_index, seed_index))
    assignments: dict[int, Change] = {}
    used_seeds: set[int] = set()
    for _overlap_size, group_index, seed_index in sorted(candidates, reverse=True):
        if group_index in assignments or seed_index in used_seeds:
            continue
        assignments[group_index] = located_seeds[seed_index][0]
        used_seeds.add(seed_index)
    return assignments


def _overlap(a: tuple[int, int], b: tuple[int, int] | None) -> int:
    if b is None:
        return 0
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


_TOKEN_RE = re.compile(r"\s+|[^\s]+")


def _tokens_with_offsets(text: str) -> tuple[list[str], list[int]]:
    """Whitespace-delimited tokens and the char offset where each starts.

    Diffing tokens instead of characters keeps a replaced word as one edit
    ("일이지만" -> "과제이지만") instead of a one-character fragment inside it,
    which is what the comparison UI and the explainer need to see."""
    tokens: list[str] = []
    offsets: list[int] = []
    for match in _TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        offsets.append(match.start())
    offsets.append(len(text))
    return tokens, offsets


def _changed_opcode_groups(original: str, revised: str) -> list[tuple[int, int, int, int]]:
    original_tokens, original_offsets = _tokens_with_offsets(original)
    revised_tokens, revised_offsets = _tokens_with_offsets(revised)
    matcher = SequenceMatcher(a=original_tokens, b=revised_tokens, autojunk=False)
    raw_groups: list[tuple[int, int, int, int]] = []
    current: tuple[int, int, int, int] | None = None

    for tag, token_original_start, token_original_end, token_revised_start, token_revised_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        original_start = original_offsets[token_original_start]
        original_end = original_offsets[token_original_end]
        revised_start = revised_offsets[token_revised_start]
        revised_end = revised_offsets[token_revised_end]

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

    return raw_groups


_BOUNDARY_CHARS = " \t\n.,!?。！？"


def _expand_group(
    original: str,
    revised: str,
    group: tuple[int, int, int, int],
    *,
    previous: tuple[int, int, int, int] | None = None,
    following: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Add a little shared context on both sides, cut at a word boundary so
    the snippets read as phrases instead of mid-word fragments.

    The context is text that is identical in source and result, so it never
    reaches into a neighbouring edit: a snippet must not show another
    change's before/after as if it were part of this one."""
    original_start, original_end, revised_start, revised_end = group
    prefix = min(_DISPLAY_CONTEXT_CHARS, original_start, revised_start)
    if previous is not None:
        prefix = min(prefix, original_start - previous[1], revised_start - previous[3])
    prefix = _shrink_to_boundary_left(original, original_start - prefix, original_start)
    suffix = min(
        _DISPLAY_CONTEXT_CHARS,
        len(original) - original_end,
        len(revised) - revised_end,
    )
    if following is not None:
        suffix = min(suffix, following[0] - original_end, following[2] - revised_end)
    suffix = _shrink_to_boundary_right(original, original_end, original_end + suffix)
    return (
        original_start - prefix,
        original_end + suffix,
        revised_start - prefix,
        revised_end + suffix,
    )


def _shrink_to_boundary_left(text: str, start: int, anchor: int) -> int:
    """Length of context [start, anchor) trimmed so it begins after a boundary char."""
    if start <= 0:
        return anchor - start
    for position in range(start, anchor):
        if text[position - 1] in _BOUNDARY_CHARS:
            return anchor - position
    return 0


def _shrink_to_boundary_right(text: str, anchor: int, end: int) -> int:
    """Length of context [anchor, end) trimmed so it ends before a boundary char."""
    if end >= len(text):
        return end - anchor
    for position in range(end, anchor, -1):
        if text[position] in _BOUNDARY_CHARS:
            return position - anchor
    return 0
