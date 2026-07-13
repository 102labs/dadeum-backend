import re
from functools import lru_cache
from pathlib import Path


_RESOURCE_DIR = Path(__file__).with_name("resources")

_RULE_HEADING_RE = re.compile(r"^## (?P<rule_id>[A-Z]-\d+)\.", re.MULTILINE)
_ANY_HEADING_RE = re.compile(r"^#{1,2} ", re.MULTILINE)


@lru_cache
def load_resource(name: str) -> str:
    if "/" in name or name.startswith("."):
        raise ValueError("resource name must be a plain filename")
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


def strict_rules() -> str:
    return load_resource("strict-rules.md")


@lru_cache
def _rule_sections() -> dict[str, str]:
    text = strict_rules()
    sections: dict[str, str] = {}
    for match in _RULE_HEADING_RE.finditer(text):
        next_heading = _ANY_HEADING_RE.search(text, match.end())
        end = next_heading.start() if next_heading else len(text)
        body = text[match.start():end].strip()
        body = re.sub(r"\n?\*{3}\s*$", "", body).strip()
        sections[match.group("rule_id")] = body
    return sections


def rule_card(rule_id: str) -> str:
    """Full rulebook section (signal, fix, keep, examples) for one rule id."""
    return _rule_sections().get(rule_id, "")


@lru_cache
def compact_strict_rules() -> str:
    """Rulebook with per-rule bodies removed.

    Keeps the global principles (sections 0-3), category preambles, every rule
    heading line as an index, and the trailing global checklists. Per-rule
    detail travels separately as rule cards attached to detected hints.
    """
    kept: list[str] = []
    skipping = False
    for line in strict_rules().splitlines():
        if _RULE_HEADING_RE.match(line):
            kept.append(line)
            skipping = True
            continue
        if line.startswith("#"):
            skipping = False
        if not skipping:
            kept.append(line)
    compact = "\n".join(kept)
    compact = re.sub(r"\n?\*{3}\n?", "\n", compact)
    return re.sub(r"\n{3,}", "\n\n", compact).strip()
