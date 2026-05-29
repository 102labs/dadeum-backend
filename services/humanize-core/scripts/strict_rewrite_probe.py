#!/usr/bin/env python3
"""Run only prepare -> rewrite and print the raw strict-only draft.

This is a local diagnostics tool. It intentionally stops before
audit/review/finalize so the raw rewrite draft can be inspected.
Do not use it from request handlers and do not redirect sensitive input to logs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from humanize_core.config import Settings  # noqa: E402
from humanize_core.graph import RewriteGraphRunner  # noqa: E402
from humanize_core.llm import create_llm  # noqa: E402
from humanize_core.schemas import RewriteRequest  # noqa: E402


def main() -> None:
    args = _parse_args()
    text = _load_text(args)
    output = asyncio.run(_run_probe(args, text))
    if args.raw:
        print(output["rewriteResult"]["revisedText"])
        return
    print(json.dumps(output, ensure_ascii=False, indent=2 if args.pretty else None))


async def _run_probe(args: argparse.Namespace, text: str) -> dict[str, Any]:
    settings = Settings(_env_file=str(CORE_DIR / ".env"))
    request = RewriteRequest(
        text=text,
        user_intent=args.user_intent,
        rewrite_mode="strict",
        tone=args.tone,
        protected_terms=args.protected_term,
        max_rounds=args.max_rounds,
        preserve_formatting=not args.no_preserve_formatting,
    )
    llm = create_llm(
        settings.model_provider,
        settings.model_name,
        settings.openai_api_key,
        settings.anthropic_api_key,
        openrouter_api_key=settings.openrouter_api_key,
        openrouter_base_url=settings.openrouter_base_url,
        openrouter_app_title=settings.openrouter_app_title,
        openrouter_site_url=settings.openrouter_site_url,
        rewrite_model_name=settings.rewrite_model_name,
        rewrite_fallback_model_name=settings.rewrite_fallback_model_name,
        strict_audit_model_name=settings.strict_audit_model_name,
        strict_review_model_name=settings.strict_review_model_name,
    )
    runner = RewriteGraphRunner(settings, llm)
    state: dict[str, Any] = {"request": request, "started_at": time.perf_counter()}
    state.update(await runner._prepare(state))
    state.update(await runner._rewrite(state))

    rewrite_result = state["llm_result"]
    return {
        "stage": "rewrite",
        "stoppedBefore": ["audit", "review", "finalize"],
        "provider": settings.model_provider,
        "modelConfig": {
            "rewriteModelName": settings.rewrite_model_name,
            "rewriteResolvedModels": list(getattr(llm, "rewrite_models", [])),
            "strictAuditModelName": settings.strict_audit_model_name,
            "strictReviewModelName": settings.strict_review_model_name,
        },
        "round": state["round"],
        "input": {
            "charCount": len(request.text),
            "ignoredRewriteMode": request.rewrite_mode,
            "ignoredMaxRounds": request.max_rounds,
            "tone": request.tone,
            "preserveFormatting": request.preserve_formatting,
            "protectedTerms": request.protected_terms,
        },
        "rewriteResult": rewrite_result.model_dump(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the strict-only rewrite stage and print its raw draft.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Text to rewrite. If omitted, stdin is used.")
    source.add_argument("--file", type=Path, help="UTF-8 text file to rewrite.")
    parser.add_argument("--user-intent", default="", help="Optional user intent.")
    parser.add_argument(
        "--tone",
        choices=["keep", "formal", "friendly"],
        default="keep",
        help="Tone control passed to the strict rewrite request.",
    )
    parser.add_argument(
        "--protected-term",
        action="append",
        default=[],
        help="Protected expression to preserve exactly during audit/review.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Compatibility input only; strict-only mode always runs one rewrite routine.",
    )
    parser.add_argument(
        "--no-preserve-formatting",
        action="store_true",
        help="Set preserve_formatting=false.",
    )
    parser.add_argument("--raw", action="store_true", help="Print only revisedText.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _load_text(args: argparse.Namespace) -> str:
    if args.file:
        return args.file.read_text(encoding="utf-8")
    if args.text is not None:
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("Provide --text, --file, or pipe text through stdin.")


if __name__ == "__main__":
    main()
