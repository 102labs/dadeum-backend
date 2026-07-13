#!/usr/bin/env python3
"""Run the golden set through the full rewrite pipeline and score each output.

Scoring reuses the detectors already in the codebase:
  - residual S1/S2: local_detect() applied to the *output* text
  - change rate / over-polish: change_rate(), over_polish_signals()
  - preservation: _local_preservation_flagged_edits() (numbers, dates, quotes, terms)
  - completion: _completion_warnings() (truncation, missing sentences/paragraphs)

Usage:
  python scripts/eval_golden.py                       # run all cases, save report
  python scripts/eval_golden.py --case tr-01 --case ai-02
  python scripts/eval_golden.py --tag 번역투
  python scripts/eval_golden.py --baseline evals/reports/report-XXXX.json
  python scripts/eval_golden.py --stub                # offline smoke test of the harness
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

CORE_DIR = Path(__file__).resolve().parents[1]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from humanize_core.config import Settings  # noqa: E402
from humanize_core.graph import (  # noqa: E402
    RewriteGraphRunner,
    _completion_warnings,
    _local_preservation_flagged_edits,
)
from humanize_core.im_not_ai.audit import change_rate, local_detect, over_polish_signals  # noqa: E402
from humanize_core.llm import create_llm  # noqa: E402
from humanize_core.schemas import RewriteRequest  # noqa: E402

GOLDEN_PATH = CORE_DIR / "evals" / "golden_set.json"
REPORTS_DIR = CORE_DIR / "evals" / "reports"


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args)
    if not cases:
        raise SystemExit("선택된 케이스가 없습니다. --case/--tag 필터를 확인하세요.")

    settings = Settings(_env_file=str(CORE_DIR / ".env"))
    provider = "stub" if args.stub else settings.model_provider
    runner = _build_runner(settings, provider)

    print(f"골든셋 {len(cases)}개 케이스 실행 (provider={provider}, concurrency={args.concurrency})")
    results = asyncio.run(_run_all(runner, cases, args.concurrency))

    report = _build_report(settings, provider, results)
    report_path = _save_report(report, args.report)

    _print_case_table(report)
    _print_summary(report["summary"])
    if args.baseline:
        _print_baseline_diff(report, Path(args.baseline))
    print(f"\n리포트 저장: {report_path}")


# ---------------------------------------------------------------- run cases


async def _run_all(
    runner: RewriteGraphRunner,
    cases: list[dict[str, Any]],
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(case: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await _run_case(runner, case)

    return list(await asyncio.gather(*(run_one(case) for case in cases)))


async def _run_case(runner: RewriteGraphRunner, case: dict[str, Any]) -> dict[str, Any]:
    request = _request_for_case(case)
    started_at = time.perf_counter()
    try:
        response = await runner.run(request, request_id=f"eval-{case['id']}")
    except Exception as exc:  # noqa: BLE001 - keep the batch alive per case
        print(f"  ✗ {case['id']}: {type(exc).__name__}: {exc}")
        return {
            "id": case["id"],
            "tags": case.get("tags", []),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    metrics = _score(request, response.revisedText, response.warnings)
    metrics["latencyMs"] = int((time.perf_counter() - started_at) * 1000)
    metrics["inputTokens"] = response.usage.inputTokens
    metrics["outputTokens"] = response.usage.outputTokens
    print(
        f"  ✓ {case['id']}: 잔존 S1 {metrics['residualS1']}/{metrics['inputS1']}"
        f", S2 {metrics['residualS2']}/{metrics['inputS2']}"
        f", 변경률 {metrics['changeRate']:.1f}%"
        + (f", 보존 이슈 {len(metrics['preservationIssues'])}건" if metrics["preservationIssues"] else "")
    )
    return {
        "id": case["id"],
        "tags": case.get("tags", []),
        "ok": True,
        "metrics": metrics,
        "revisedText": response.revisedText,
    }


def _request_for_case(case: dict[str, Any]) -> RewriteRequest:
    return RewriteRequest(
        text=case["text"],
        user_intent=case.get("user_intent", ""),
        rewrite_mode="strict",
        tone=case.get("tone", "keep"),
        protected_terms=case.get("protected_terms", []),
        preserve_formatting=case.get("preserve_formatting", True),
    )


# ------------------------------------------------------------------ scoring


# over_polish_signals()의 literary_tone_added 패턴과 동일. 원문에 이미 있던
# 표현(예: 결과·해결의 '결')까지 과윤문으로 집계되는 것을 걸러내는 데 쓴다.
_LITERARY_TONE_RE = re.compile(r"듯|결|숨결|여운|풍경|서사|빛난다")


def _score(request: RewriteRequest, revised_text: str, response_warnings: list[str]) -> dict[str, Any]:
    input_detection = local_detect(request.text, protected_terms=request.protected_terms)
    output_detection = local_detect(revised_text, protected_terms=request.protected_terms)
    input_s1, input_s2 = _severity_counts(input_detection.findings)
    residual_s1, residual_s2 = _severity_counts(output_detection.findings)
    polish_signals = over_polish_signals(request.text, revised_text)
    if "literary_tone_added" in polish_signals and _LITERARY_TONE_RE.search(request.text):
        polish_signals = [signal for signal in polish_signals if signal != "literary_tone_added"]
    preservation_issues = [
        edit.issue for edit in _local_preservation_flagged_edits(request, revised_text)
    ]
    return {
        "inputS1": input_s1,
        "inputS2": input_s2,
        "residualS1": residual_s1,
        "residualS2": residual_s2,
        "changeRate": change_rate(request.text, revised_text),
        "overPolishSignals": polish_signals,
        "preservationIssues": preservation_issues,
        "completionWarnings": _completion_warnings(request, revised_text),
        "responseWarnings": response_warnings,
    }


def _severity_counts(findings: list[Any]) -> tuple[int, int]:
    counts = Counter(finding.severity for finding in findings)
    return counts.get("S1", 0), counts.get("S2", 0)


# ------------------------------------------------------------------- report


def _build_report(settings: Settings, provider: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [result for result in results if result.get("ok")]
    failed = [result for result in results if not result.get("ok")]
    metrics = [result["metrics"] for result in succeeded]

    input_s1 = sum(m["inputS1"] for m in metrics)
    residual_s1 = sum(m["residualS1"] for m in metrics)
    input_s2 = sum(m["inputS2"] for m in metrics)
    residual_s2 = sum(m["residualS2"] for m in metrics)
    preservation_fail_cases = [
        result["id"] for result in succeeded if result["metrics"]["preservationIssues"]
    ]
    completion_fail_cases = [
        result["id"] for result in succeeded if result["metrics"]["completionWarnings"]
    ]
    over_polish_cases = [
        result["id"] for result in succeeded if result["metrics"]["overPolishSignals"]
    ]

    summary = {
        "casesRun": len(results),
        "casesFailed": [result["id"] for result in failed],
        "inputS1": input_s1,
        "residualS1": residual_s1,
        "s1ResolutionRate": _rate(input_s1 - residual_s1, input_s1),
        "inputS2": input_s2,
        "residualS2": residual_s2,
        "s2ResolutionRate": _rate(input_s2 - residual_s2, input_s2),
        "meanChangeRate": round(
            sum(m["changeRate"] for m in metrics) / len(metrics), 2
        ) if metrics else 0.0,
        "preservationFailCases": preservation_fail_cases,
        "completionFailCases": completion_fail_cases,
        "overPolishCases": over_polish_cases,
    }
    return {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "models": {
            "rewrite": settings.rewrite_model_name,
            "rewriteFallback": settings.rewrite_fallback_model_name,
            "audit": settings.strict_audit_model_name,
            "review": settings.strict_review_model_name,
        },
        "summary": summary,
        "cases": results,
    }


def _rate(resolved: int, total: int) -> float | None:
    if total <= 0:
        return None
    return round(max(0, resolved) / total, 3)


def _save_report(report: dict[str, Any], override: str | None) -> Path:
    if override:
        path = Path(override)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = REPORTS_DIR / f"report-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------------ display


def _print_case_table(report: dict[str, Any]) -> None:
    header = f"{'케이스':<10} {'잔존S1':>6} {'잔존S2':>6} {'변경률':>7} {'보존':>4} {'완성':>4} {'과윤문':>6}"
    print("\n" + header)
    print("-" * len(header.encode("utf-8")))
    for result in report["cases"]:
        if not result.get("ok"):
            print(f"{result['id']:<12} 실패: {result.get('error', '')}")
            continue
        m = result["metrics"]
        preservation = "OK" if not m["preservationIssues"] else f"✗{len(m['preservationIssues'])}"
        completion = "OK" if not m["completionWarnings"] else f"✗{len(m['completionWarnings'])}"
        over_polish = "-" if not m["overPolishSignals"] else ",".join(m["overPolishSignals"])
        print(
            f"{result['id']:<12} "
            f"{m['residualS1']:>3}/{m['inputS1']:<3} "
            f"{m['residualS2']:>3}/{m['inputS2']:<3} "
            f"{m['changeRate']:>6.1f}% "
            f"{preservation:>4} {completion:>4} {over_polish}"
        )


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n── 요약 " + "─" * 40)
    print(f"실행 케이스        : {summary['casesRun']}개"
          + (f" (실패 {len(summary['casesFailed'])}: {', '.join(summary['casesFailed'])})" if summary["casesFailed"] else ""))
    print(f"S1 해결률          : {_format_rate(summary['s1ResolutionRate'])} (잔존 {summary['residualS1']}/{summary['inputS1']})")
    print(f"S2 해결률          : {_format_rate(summary['s2ResolutionRate'])} (잔존 {summary['residualS2']}/{summary['inputS2']})")
    print(f"평균 변경률        : {summary['meanChangeRate']:.1f}%")
    print(f"보존 실패 케이스   : {_format_cases(summary['preservationFailCases'])}")
    print(f"완성도 경고 케이스 : {_format_cases(summary['completionFailCases'])}")
    print(f"과윤문 신호 케이스 : {_format_cases(summary['overPolishCases'])}")


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _format_cases(case_ids: list[str]) -> str:
    return f"{len(case_ids)}건" + (f" ({', '.join(case_ids)})" if case_ids else "")


def _print_baseline_diff(report: dict[str, Any], baseline_path: Path) -> None:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_cases = {
        case["id"]: case["metrics"]
        for case in baseline.get("cases", [])
        if case.get("ok")
    }
    print(f"\n── 베이스라인 비교: {baseline_path.name} " + "─" * 20)
    regressions: list[str] = []
    for result in report["cases"]:
        if not result.get("ok") or result["id"] not in baseline_cases:
            continue
        current = result["metrics"]
        previous = baseline_cases[result["id"]]
        d_s1 = current["residualS1"] - previous["residualS1"]
        d_s2 = current["residualS2"] - previous["residualS2"]
        d_preservation = len(current["preservationIssues"]) - len(previous["preservationIssues"])
        if d_s1 == 0 and d_s2 == 0 and d_preservation == 0:
            continue
        marks = []
        if d_s1:
            marks.append(f"S1 {d_s1:+d}")
        if d_s2:
            marks.append(f"S2 {d_s2:+d}")
        if d_preservation:
            marks.append(f"보존 이슈 {d_preservation:+d}")
        line = f"{result['id']:<12} {', '.join(marks)}"
        if d_s1 > 0 or d_preservation > 0:
            regressions.append(result["id"])
            line += "  ← 회귀"
        print(line)

    previous_summary = baseline.get("summary", {})
    current_summary = report["summary"]
    print(
        f"\n요약 변화: S1 잔존 {previous_summary.get('residualS1')} → {current_summary['residualS1']}, "
        f"S2 잔존 {previous_summary.get('residualS2')} → {current_summary['residualS2']}, "
        f"평균 변경률 {previous_summary.get('meanChangeRate')}% → {current_summary['meanChangeRate']}%, "
        f"보존 실패 {len(previous_summary.get('preservationFailCases', []))}건 → {len(current_summary['preservationFailCases'])}건"
    )
    if regressions:
        print(f"⚠ 회귀 의심 케이스 {len(regressions)}건: {', '.join(regressions)}")
    else:
        print("회귀 없음")


# -------------------------------------------------------------------- setup


def _build_runner(settings: Settings, provider: str) -> RewriteGraphRunner:
    llm = create_llm(
        provider,
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
    return RewriteGraphRunner(settings, llm)


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    cases = golden["cases"]
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
    if args.tag:
        wanted_tags = set(args.tag)
        cases = [case for case in cases if wanted_tags & set(case.get("tags", []))]
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 자동 채점 평가를 실행합니다.")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="골든셋 JSON 경로")
    parser.add_argument("--case", action="append", default=[], help="특정 케이스 id만 실행 (반복 지정 가능)")
    parser.add_argument("--tag", action="append", default=[], help="특정 태그 케이스만 실행 (반복 지정 가능)")
    parser.add_argument("--concurrency", type=int, default=3, help="동시 실행 케이스 수 (기본 3)")
    parser.add_argument("--report", help="리포트 저장 경로 (기본: evals/reports/report-<timestamp>.json)")
    parser.add_argument("--baseline", help="비교할 이전 리포트 JSON 경로")
    parser.add_argument("--stub", action="store_true", help="LLM 대신 stub 사용 (스크립트 자체 점검용)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
