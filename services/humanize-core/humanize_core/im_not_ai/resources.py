from functools import lru_cache
from pathlib import Path


_RESOURCE_DIR = Path(__file__).with_name("resources")


@lru_cache
def load_resource(name: str) -> str:
    if "/" in name or name.startswith("."):
        raise ValueError("resource name must be a plain filename")
    return (_RESOURCE_DIR / name).read_text(encoding="utf-8")


def quick_rules() -> str:
    return load_resource("quick-rules.md")


def strict_rules() -> str:
    return load_resource("stric-rules.md")
