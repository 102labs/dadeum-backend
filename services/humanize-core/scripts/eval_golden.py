#!/usr/bin/env python3
"""Run the golden set through the full rewrite pipeline and score each output.

Four layers of scoring, from cheapest to most faithful:

  1. Pattern metrics (regex, same detectors the pipeline uses)
       residual S1/S2, change rate, over-polish signals, preservation damage,
       completion warnings. Fast, deterministic, but shares the pipeline's
       blind spots, so treat it as a smoke signal rather than the verdict.
  2. Per-case expectations (deterministic pass/fail)
       must_keep / must_remove / max_count / change-rate bounds / register /
       paragraph count declared on each golden case, plus implicit checks:
       preservation, completion, register preservation, display-safe changes.
  3. Change-list quality
       are changes[] snippets exact substrings, how many reasons fell back to
       the generic diff text, empty or non-Korean reasons, no-op changes.
  4. LLM judge (optional, --judge / --pairwise)
       absolute 1-5 rubric on naturalness, residual AI-tell, meaning, over-edit,
       reason quality, register; and blind A/B against a baseline report.

Stage telemetry (style-gate rounds, audit status, review path, chunking,
per-stage latency) is captured from the graph's debug events so a metric
change can be traced to the stage that caused it.

Usage:
  python scripts/eval_golden.py                       # run all cases, save report
  python scripts/eval_golden.py --case tr-01 --case ai-02
  python scripts/eval_golden.py --tag 번역투
  python scripts/eval_golden.py --runs 3              # repeat each case, report spread
  python scripts/eval_golden.py --judge               # add LLM rubric scores
  python scripts/eval_golden.py --baseline evals/reports/report-XXXX.json [--pairwise]
  python scripts/eval_golden.py --stub                # offline smoke test of the harness
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

CORE_DIR = Path(__file__).resolve().parents[1]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from humanize_core.config import Settings  # noqa: E402
from humanize_core.graph import (  # noqa: E402
    RewriteGraphRunner,
    _completion_warnings,
    _local_preservation_flagged_edits,
)
from humanize_core.im_not_ai.audit import (  # noqa: E402
    change_rate,
    local_detect,
    over_polish_signals,
    split_sentences,
)
from humanize_core.llm import _openrouter_response_format, create_llm  # noqa: E402
from humanize_core.schemas import RewriteRequest, RewriteResponse  # noqa: E402

GOLDEN_PATH = CORE_DIR / "evals" / "golden_set.json"
REPORTS_DIR = CORE_DIR / "evals" / "reports"
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-5.5"
JUDGE_MODEL_ENV = "HUMANIZE_EVAL_JUDGE_MODEL_NAME"

# Reason strings diff.py emits when the model's own change list could not be
# used. Every one of these in a final response means the user saw a generic
# explanation instead of the model's reason for that edit.
FALLBACK_REASONS = (
    "원문의 의미와 표현을 유지했습니다.",
    "문장의 흐름과 전달력을 개선했습니다.",
    "원문과 최종 윤문 결과의 차이를 비교 가능한 구간으로 정리했습니다.",
)
OVERFLOW_REASON_PREFIX = "세부 변경 구간이 "

Register = Literal["formal", "haeyo", "plain", "mixed", "unknown"]


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args)
    if not cases:
        raise SystemExit("선택된 케이스가 없습니다. --case/--tag 필터를 확인하세요.")

    settings = Settings(_env_file=str(CORE_DIR / ".env"))
    provider = "stub" if args.stub else settings.model_provider
    baseline = _load_baseline(args.baseline) if args.baseline else None
    if args.pairwise and baseline is None:
        raise SystemExit("--pairwise 는 --baseline 과 함께 써야 합니다.")

    if args.rejudge:
        _rejudge_main(args, settings, provider, cases, baseline)
        return

    recorder = StageRecorder()
    runner = _build_runner(settings, provider, recorder)
    judge = _build_judge(settings, args, provider) if (args.judge or args.pairwise) else None

    print(
        f"골든셋 {len(cases)}개 케이스 실행 (provider={provider}, runs={args.runs}, "
        f"concurrency={args.concurrency}"
        + (f", judge={judge.model}" if judge else "")
        + ")"
    )
    judge_filter = _judge_filter(args)
    results = asyncio.run(
        _run_all(runner, recorder, cases, args.concurrency, args.runs, judge, baseline, args.pairwise, judge_filter)
    )

    report = _build_report(settings, provider, results, args, judge)
    report_path = _save_report(report, args.report)
    markdown_path = _save_markdown(report, report_path, args.markdown)

    _print_case_table(report)
    _print_summary(report["summary"])
    if baseline is not None:
        _print_baseline_diff(report, baseline, Path(args.baseline))
    print(f"\n리포트 저장: {report_path}")
    print(f"검토용 마크다운: {markdown_path}")


# ------------------------------------------------------------------ rejudge


def _rejudge_main(
    args: argparse.Namespace,
    settings: Settings,
    provider: str,
    cases: list[dict[str, Any]],
    baseline: dict[str, Any] | None,
) -> None:
    """Re-score an existing report's outputs with the judge, without rerunning
    the pipeline. Lets every arm be judged by the same model after the fact."""
    source = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
    args.judge = True
    judge = _build_judge(settings, args, provider)
    wanted = {case["id"]: case for case in cases}
    judge_filter = _judge_filter(args)
    print(f"재심사: {Path(args.rejudge).name} → judge={judge.model}, pairwise={bool(args.pairwise)}")
    results = asyncio.run(_rejudge_all(source, wanted, judge, baseline, args.pairwise, args.concurrency, judge_filter))
    source["cases"] = results
    source["models"]["judge"] = judge.model
    source["config"]["judge"] = True
    source["config"]["pairwise"] = bool(args.pairwise)
    source["config"]["baseline"] = args.baseline
    source["config"]["rejudgedFrom"] = args.rejudge
    source["createdAt"] = datetime.now().isoformat(timespec="seconds")
    metrics = [result["metrics"] for result in results if result.get("ok")]
    source["summary"] = build_summary(results, metrics)
    report_path = _save_report(source, args.report)
    markdown_path = _save_markdown(source, report_path, args.markdown)
    _print_case_table(source)
    _print_summary(source["summary"])
    if baseline is not None:
        _print_baseline_diff(source, baseline, Path(args.baseline))
    print(f"\n리포트 저장: {report_path}")
    print(f"검토용 마크다운: {markdown_path}")


async def _rejudge_all(
    source: dict[str, Any],
    wanted: dict[str, dict[str, Any]],
    judge: "LLMJudge",
    baseline: dict[str, Any] | None,
    pairwise: bool,
    concurrency: int,
    judge_filter: "Callable[[dict[str, Any]], bool] | None",
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def rejudge_one(result: dict[str, Any]) -> dict[str, Any]:
        case = wanted.get(result["id"])
        if not result.get("ok") or case is None or (judge_filter is not None and not judge_filter(case)):
            return result
        request = _request_for_case(case)
        response = response_from_result(result)
        async with semaphore:
            result["metrics"]["judge"] = await judge.score(case, request, response)
            result.pop("pairwise", None)
            if pairwise and baseline is not None:
                baseline_case = baseline.get("casesById", {}).get(case["id"])
                if baseline_case and baseline_case.get("revisedText"):
                    result["pairwise"] = await judge.compare(
                        case,
                        request,
                        current_text=result["revisedText"],
                        baseline_text=baseline_case["revisedText"],
                    )
        print(f"  ✓ {result['id']}: 심사 {result['metrics']['judge']['overall']}")
        return result

    return list(await asyncio.gather(*(rejudge_one(result) for result in source.get("cases", []))))


def response_from_result(result: dict[str, Any]) -> RewriteResponse:
    """Rebuild the RewriteResponse a report row was scored from."""
    from humanize_core.schemas import Change, Usage

    metrics = result.get("metrics", {})
    return RewriteResponse(
        revisedText=result["revisedText"],
        changes=[Change.model_validate(change) for change in result.get("changes", [])],
        summary=list(result.get("summary", [])),
        warnings=list(result.get("warnings", [])),
        usage=Usage(
            inputTokens=int(metrics.get("inputTokens", 0) or 0),
            outputTokens=int(metrics.get("outputTokens", 0) or 0),
            latencyMs=int(metrics.get("latencyMs", 0) or 0),
            rounds=1,
        ),
    )


# ------------------------------------------------------------- stage capture


class StageRecorder:
    """Collects the graph's debug events in memory instead of writing log files."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def event(self, event: str, **kwargs: Any) -> None:
        self.events.append({"event": event, **kwargs})

    def take(self, request_id: str) -> list[dict[str, Any]]:
        taken = [item for item in self.events if item.get("request_id") == request_id]
        self.events = [item for item in self.events if item.get("request_id") != request_id]
        return taken


def stage_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Condense graph.stage.* events into the facts that explain a score."""
    summary: dict[str, Any] = {
        "stages": [],
        "durationsMs": {},
        "chunkCount": 1,
        "styleGateRounds": 0,
        "styleGateImplementation": None,
        "auditStatus": None,
        "auditFlaggedEdits": 0,
        "reviewRan": False,
        "reviewImplementation": None,
        "failedStage": None,
    }
    for item in events:
        step = item.get("step")
        if not step:
            continue
        details = item.get("details") or {}
        if item["event"] == "graph.stage.succeeded":
            summary["stages"].append(step)
            summary["durationsMs"][step] = item.get("duration_ms")
            if step == "rewrite":
                summary["chunkCount"] = int(details.get("chunk_count", 1) or 1)
            elif step == "style_gate":
                summary["styleGateRounds"] = int(details.get("repair_rounds", 0) or 0)
                summary["styleGateImplementation"] = details.get("implementation")
                summary["styleGateInitialScore"] = details.get("initial_severity_score")
                summary["styleGateFinalScore"] = details.get("final_severity_score")
            elif step == "audit":
                summary["auditStatus"] = details.get("result_status")
                summary["auditFlaggedEdits"] = int(details.get("flagged_edits_count", 0) or 0)
            elif step == "review":
                summary["reviewRan"] = True
                summary["reviewImplementation"] = details.get("implementation")
            elif step == "finalize":
                summary["unexplainedChanges"] = int(details.get("unexplained_change_count", 0) or 0)
                summary["explainedChanges"] = int(details.get("explained_change_count", 0) or 0)
                summary["explainFailed"] = bool(details.get("explain_failed", False))
        elif item["event"] == "graph.stage.failed":
            summary["failedStage"] = step
    return summary


# ---------------------------------------------------------------- run cases


async def _run_all(
    runner: RewriteGraphRunner,
    recorder: StageRecorder,
    cases: list[dict[str, Any]],
    concurrency: int,
    runs: int,
    judge: "LLMJudge | None",
    baseline: dict[str, Any] | None,
    pairwise: bool,
    judge_filter: "Callable[[dict[str, Any]], bool] | None" = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(case: dict[str, Any]) -> dict[str, Any]:
        case_judge = judge if judge_filter is None or judge_filter(case) else None
        async with semaphore:
            return await _run_case(runner, recorder, case, runs, case_judge, baseline, pairwise)

    return list(await asyncio.gather(*(run_one(case) for case in cases)))


async def _run_case(
    runner: RewriteGraphRunner,
    recorder: StageRecorder,
    case: dict[str, Any],
    runs: int = 1,
    judge: "LLMJudge | None" = None,
    baseline: dict[str, Any] | None = None,
    pairwise: bool = False,
) -> dict[str, Any]:
    request = _request_for_case(case)
    run_records: list[dict[str, Any]] = []
    for run_index in range(max(1, runs)):
        request_id = f"eval-{case['id']}-r{run_index + 1}"
        started_at = time.perf_counter()
        try:
            response = await runner.run(request, request_id=request_id)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive per case
            print(f"  ✗ {case['id']}: {type(exc).__name__}: {exc}")
            recorder.take(request_id)
            return {
                "id": case["id"],
                "tags": case.get("tags", []),
                "note": case.get("note", ""),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        metrics = score_case(case, request, response)
        metrics["latencyMs"] = latency_ms
        metrics["inputTokens"] = response.usage.inputTokens
        metrics["outputTokens"] = response.usage.outputTokens
        metrics["pipeline"] = stage_summary(recorder.take(request_id))
        if judge is not None and judge.absolute:
            metrics["judge"] = await judge.score(case, request, response)
        run_records.append(
            {
                "metrics": metrics,
                "revisedText": response.revisedText,
                "changes": [change.model_dump(mode="json") for change in response.changes],
                "summary": response.summary,
                "warnings": response.warnings,
            }
        )

    aggregated = aggregate_runs([record["metrics"] for record in run_records])
    representative = run_records[_worst_run_index([record["metrics"] for record in run_records])]
    result = {
        "id": case["id"],
        "tags": case.get("tags", []),
        "note": case.get("note", ""),
        "ok": True,
        "metrics": aggregated,
        "revisedText": representative["revisedText"],
        "changes": representative["changes"],
        "summary": representative["summary"],
        "warnings": representative["warnings"],
    }
    if len(run_records) > 1:
        result["runs"] = run_records

    if pairwise and judge is not None and baseline is not None:
        baseline_case = baseline.get("casesById", {}).get(case["id"])
        if baseline_case and baseline_case.get("revisedText"):
            result["pairwise"] = await judge.compare(
                case,
                request,
                current_text=representative["revisedText"],
                baseline_text=baseline_case["revisedText"],
            )

    m = aggregated
    line = (
        f"  ✓ {case['id']}: 잔존 S1 {_fmt(m['residualS1'])}/{m['inputS1']}"
        f", S2 {_fmt(m['residualS2'])}/{m['inputS2']}"
        f", 변경률 {m['changeRate']:.1f}%"
        f", 기대 {m['expectationsPassed']}/{m['expectationsTotal']}"
    )
    if m["preservationIssues"]:
        line += f", 보존 이슈 {len(m['preservationIssues'])}건"
    if m.get("judge"):
        line += f", 심사 {m['judge']['overall']:.1f}"
    print(line)
    return result


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


def score_case(case: dict[str, Any], request: RewriteRequest, response: RewriteResponse) -> dict[str, Any]:
    revised_text = response.revisedText
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
    completion_warnings = _completion_warnings(request, revised_text)
    register = register_report(case, request.text, revised_text)
    change_quality = change_list_quality(request.text, revised_text, response)
    expectations = evaluate_expectations(
        case,
        request,
        revised_text,
        preservation_issues=preservation_issues,
        completion_warnings=completion_warnings,
        register=register,
        change_quality=change_quality,
    )
    return {
        "inputS1": input_s1,
        "inputS2": input_s2,
        "residualS1": residual_s1,
        "residualS2": residual_s2,
        "changeRate": change_rate(request.text, revised_text),
        "lengthRatio": round(len(revised_text) / max(len(request.text), 1), 3),
        "overPolishSignals": polish_signals,
        "preservationIssues": preservation_issues,
        "completionWarnings": completion_warnings,
        "responseWarnings": list(response.warnings),
        "register": register,
        "changeQuality": change_quality,
        "expectations": expectations,
        "expectationsPassed": sum(1 for item in expectations if item["passed"]),
        "expectationsTotal": len(expectations),
    }


def _severity_counts(findings: list[Any]) -> tuple[int, int]:
    counts = Counter(finding.severity for finding in findings)
    return counts.get("S1", 0), counts.get("S2", 0)


# ---- register ------------------------------------------------------------

_FORMAL_END_RE = re.compile(r"(?:습니다|입니다|합니다|됩니다|십시오|습니까|입니까|니다|시오)$")
_HAEYO_END_RE = re.compile(r"(?:요|죠)$")
_PLAIN_END_RE = re.compile(r"(?:다|까|라|자|냐|지)$")


def detect_register(text: str) -> Register:
    """Dominant sentence-final register: formal(합쇼체), haeyo(해요체), plain(해라체).

    'mixed' only when 해라체 and a polite register share the text without
    either reaching 80%; 합쇼체+해요체 together counts as the dominant one."""
    counts: Counter[str] = Counter()
    for sentence in split_sentences(text):
        body = sentence.rstrip(".!?。！？ \t\"”’'")
        if not body:
            continue
        if _FORMAL_END_RE.search(body):
            counts["formal"] += 1
        elif _HAEYO_END_RE.search(body):
            counts["haeyo"] += 1
        elif _PLAIN_END_RE.search(body):
            counts["plain"] += 1
    if not counts:
        return "unknown"
    polite = counts["formal"] + counts["haeyo"]
    plain = counts["plain"]
    total = polite + plain
    # 합쇼체와 해요체가 섞이는 것은 업무 글에서 정상이다(안내문 끝의 "~주세요").
    # 해라체가 공손체와 섞일 때만 '혼합'으로 본다.
    if plain and polite and max(plain, polite) / total < 0.8:
        return "mixed"
    if plain >= polite:
        return "plain"
    return "formal" if counts["formal"] >= counts["haeyo"] else "haeyo"


def expected_register(case: dict[str, Any], original_register: str) -> str:
    """What the output register must be: a concrete register, 'consistent', 'soft', or 'any'."""
    explicit = case.get("expect_register")
    if explicit:
        return str(explicit)
    tone = case.get("tone", "keep")
    if tone == "formal":
        return "formal"
    if tone == "friendly":
        return "soft"
    if original_register in {"mixed", "unknown"}:
        return "any"
    return original_register


def register_matches(expected: str, actual: str) -> bool:
    if expected == "any":
        return True
    if expected == "consistent":
        return actual != "mixed"
    if expected == "soft":
        return actual in {"haeyo", "formal"}
    return actual == expected


def register_report(case: dict[str, Any], original: str, revised: str) -> dict[str, Any]:
    original_register = detect_register(original)
    revised_register = detect_register(revised)
    expected = expected_register(case, original_register)
    return {
        "original": original_register,
        "revised": revised_register,
        "expected": expected,
        "ok": register_matches(expected, revised_register),
    }


# ---- change list ---------------------------------------------------------

_HANGUL_RE = re.compile(r"[가-힣]")


def change_list_quality(original: str, revised: str, response: RewriteResponse) -> dict[str, Any]:
    changes = response.changes
    fallback = 0
    empty_reason = 0
    non_korean = 0
    noop = 0
    unsafe = 0
    overflow_note = False
    for change in changes:
        if change.reason.startswith(OVERFLOW_REASON_PREFIX):
            overflow_note = True
            continue
        if change.reason in FALLBACK_REASONS:
            fallback += 1
        if not change.reason.strip():
            empty_reason += 1
        elif not _HANGUL_RE.search(change.reason):
            non_korean += 1
        if change.original and change.original == change.revised:
            noop += 1
        if change.original and change.original not in original:
            unsafe += 1
        elif change.revised and change.revised not in revised:
            unsafe += 1
    substantive = sum(1 for change in changes if not change.reason.startswith(OVERFLOW_REASON_PREFIX))
    return {
        "count": substantive,
        "displaySafe": unsafe == 0,
        "unsafeSnippetCount": unsafe,
        "fallbackReasonCount": fallback,
        "emptyReasonCount": empty_reason,
        "nonKoreanReasonCount": non_korean,
        "noopChangeCount": noop,
        "overflowNote": overflow_note,
        "textChangedWithoutChanges": original != revised and substantive == 0,
    }


# ---- expectations --------------------------------------------------------


def evaluate_expectations(
    case: dict[str, Any],
    request: RewriteRequest,
    revised: str,
    *,
    preservation_issues: list[str],
    completion_warnings: list[str],
    register: dict[str, Any],
    change_quality: dict[str, Any],
) -> list[dict[str, Any]]:
    """Deterministic pass/fail checks: implicit ones for every case plus the
    case's own must_keep / must_remove / max_count / rate / paragraph fields."""
    original = request.text
    items: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        items.append({"name": name, "passed": bool(passed), "detail": detail})

    add("보존 대상 유지", not preservation_issues, "; ".join(preservation_issues[:3]))
    add("완성도", not completion_warnings, "; ".join(completion_warnings[:2]))
    add(
        "격식 유지",
        register["ok"],
        f"기대 {register['expected']} / 결과 {register['revised']} (원문 {register['original']})",
    )
    add(
        "변경 목록 표시 가능",
        change_quality["displaySafe"],
        f"비표시 스니펫 {change_quality['unsafeSnippetCount']}건",
    )
    add(
        "변경 사유 존재",
        change_quality["fallbackReasonCount"] == 0
        and change_quality["emptyReasonCount"] == 0
        and not change_quality["textChangedWithoutChanges"],
        f"대체 사유 {change_quality['fallbackReasonCount']}건, 빈 사유 {change_quality['emptyReasonCount']}건",
    )

    for value in case.get("must_keep", []):
        expected_count = original.count(value)
        observed = revised.count(value)
        add(f"유지: {value}", observed >= max(1, expected_count), f"원문 {expected_count}회 / 결과 {observed}회")

    for pattern in case.get("must_remove", []):
        matches = re.findall(pattern, revised, flags=re.MULTILINE)
        add(f"제거: {pattern}", not matches, f"결과에 {len(matches)}회 잔존" if matches else "")

    for pattern, limit in (case.get("max_count") or {}).items():
        count = len(re.findall(pattern, revised, flags=re.MULTILINE))
        add(f"최대 {limit}회: {pattern}", count <= int(limit), f"결과 {count}회")

    rate = change_rate(original, revised)
    if "max_change_rate" in case:
        add(f"변경률 ≤ {case['max_change_rate']}%", rate <= float(case["max_change_rate"]), f"{rate:.1f}%")
    if "min_change_rate" in case:
        add(f"변경률 ≥ {case['min_change_rate']}%", rate >= float(case["min_change_rate"]), f"{rate:.1f}%")

    if "max_length_ratio" in case:
        ratio = len(revised) / max(len(original), 1)
        add(f"길이 비율 ≤ {case['max_length_ratio']}", ratio <= float(case["max_length_ratio"]), f"{ratio:.2f}")

    if "expect_paragraphs" in case:
        paragraphs = [p for p in re.split(r"\n\s*\n", revised.strip()) if p.strip()]
        add(f"문단 {case['expect_paragraphs']}개", len(paragraphs) == int(case["expect_paragraphs"]), f"결과 {len(paragraphs)}개")

    return items


# ---- multi-run aggregation ----------------------------------------------

_MEAN_KEYS = (
    "residualS1",
    "residualS2",
    "changeRate",
    "lengthRatio",
    "latencyMs",
    "inputTokens",
    "outputTokens",
    "expectationsPassed",
)


def aggregate_runs(run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """One metrics dict for N runs: numeric fields are means, list/structured
    fields come from the worst run, and runSpread records min/max."""
    if len(run_metrics) == 1:
        return run_metrics[0]
    worst = dict(run_metrics[_worst_run_index(run_metrics)])
    for key in _MEAN_KEYS:
        values = [float(m[key]) for m in run_metrics if key in m]
        if values:
            worst[key] = round(statistics.fmean(values), 2)
    judge_scores = [m["judge"] for m in run_metrics if m.get("judge")]
    if judge_scores:
        judge_mean: dict[str, Any] = {
            key: round(statistics.fmean(float(score[key]) for score in judge_scores), 2)
            for key in JUDGE_SCORE_KEYS
        }
        worst["judge"] = judge_mean
        judge_mean["issues"] = _dedupe([issue for score in judge_scores for issue in score.get("issues", [])])
    worst["runCount"] = len(run_metrics)
    worst["runSpread"] = {
        key: [min(float(m[key]) for m in run_metrics), max(float(m[key]) for m in run_metrics)]
        for key in ("residualS1", "residualS2", "changeRate", "expectationsPassed")
    }
    return worst


def _worst_run_index(run_metrics: list[dict[str, Any]]) -> int:
    def badness(m: dict[str, Any]) -> tuple[float, float, float]:
        return (
            -float(m.get("expectationsPassed", 0)),
            float(len(m.get("preservationIssues", []))),
            float(m.get("residualS1", 0)),
        )

    return max(range(len(run_metrics)), key=lambda index: badness(run_metrics[index]))


# -------------------------------------------------------------------- judge


JUDGE_SCORE_KEYS = ("naturalness", "aiTell", "meaning", "overEdit", "reasonQuality", "registerFit", "overall")


class JudgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naturalness: int = Field(ge=1, le=5)
    aiTell: int = Field(ge=1, le=5)
    meaning: int = Field(ge=1, le=5)
    overEdit: int = Field(ge=1, le=5)
    reasonQuality: int = Field(ge=1, le=5)
    registerFit: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)


class PairwiseVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preferred: Literal["A", "B", "tie"]
    naturalnessPreferred: Literal["A", "B", "tie"]
    meaningSafer: Literal["A", "B", "tie"]
    reason: str


JUDGE_SYSTEM_PROMPT = (
    "너는 한국어 업무 글 윤문 결과를 심사하는 편집자다. 패턴을 세지 말고 독자로서 읽고 판단한다. "
    "AI 티란 다음을 말한다: 번역투(~에 대해, ~를 통해, ~에 의해 피동, 이중 피동, 그/그녀 대명사), "
    "AI 관용구(결론적으로, 시사하는 바가 크다, 주목할 만하다, 지금이야말로 ~할 때, hype 형용사), "
    "형식명사 결말(~것이다, ~라는 뜻이다), 문두 접속사 남발(또한, 따라서, 즉), 균일한 문장 길이와 종결, "
    "과도한 완곡(~로 보인다, ~일 것이다 반복), 양쪽 모두식 균형 어휘, 불필요한 볼드·이모지·따옴표. "
    "과윤문도 AI 티다: 이미 자연스러운 문장을 바꾸거나, 한 AI 티를 다른 AI 티로 치환하거나, 문학적 수사를 더하는 것. "
    "점수는 1~5 정수다. 5는 숙련된 사람이 쓴 수준, 3은 쓸 만하지만 손볼 곳이 보임, 1은 그대로 쓸 수 없음. "
    "naturalness: 한국어로 자연스럽고 읽기 쉬운가. aiTell: 결과에 AI 티가 남아 있지 않은가(5=없음). "
    "meaning: 사실·수치·날짜·고유명사·인용·주장 방향·인과·양태(추론·권고·가능성)가 모두 보존됐는가. "
    "overEdit: 고칠 곳만 고쳤는가(5=필요한 곳만, 1=멀쩡한 문장을 흔듦). "
    "reasonQuality: changes 목록의 original→revised가 실제 변경과 일치하고 reason이 그 변경을 구체적으로 설명하는가(일반적·틀린 사유는 1~2). "
    "registerFit: 원문의 격식·말투를 유지했거나 요청한 tone을 반영했는가. "
    "overall: 이 결과를 사용자에게 그대로 내보내도 되는가. "
    "issues에는 독자가 바로 알아볼 문제만 짧게 적는다(최대 5개). JSON만 반환한다."
)

PAIRWISE_SYSTEM_PROMPT = (
    "너는 한국어 업무 글 윤문 결과 두 개를 비교하는 편집자다. 같은 원문을 윤문한 A와 B를 받는다. "
    "AI 티 제거, 자연스러움, 의미·수치·양태 보존, 과윤문 여부, 원문 격식 유지를 종합해 더 나은 쪽을 고른다. "
    "차이가 사소하면 tie를 고른다. preferred는 종합 판단, naturalnessPreferred는 읽기 자연스러움만, "
    "meaningSafer는 의미·수치 보존만 본 판단이다. reason은 한국어 두 문장 이내. JSON만 반환한다."
)


class LLMJudge:
    def __init__(self, *, api_key: str, base_url: str, app_title: str, model: str, absolute: bool) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.app_title = app_title
        self.model = model
        self.absolute = absolute

    async def score(self, case: dict[str, Any], request: RewriteRequest, response: RewriteResponse) -> dict[str, Any]:
        payload = judge_payload(case, request, response)
        result = await self._call(JUDGE_SYSTEM_PROMPT, payload, JudgeScore, "rewrite_judge")
        return result.model_dump()

    async def compare(
        self,
        case: dict[str, Any],
        request: RewriteRequest,
        *,
        current_text: str,
        baseline_text: str,
    ) -> dict[str, Any]:
        current_is_a = random.random() < 0.5
        text_a, text_b = (current_text, baseline_text) if current_is_a else (baseline_text, current_text)
        payload = {
            "settings": _judge_settings(case, request),
            "original_text": request.text,
            "candidate_a": text_a,
            "candidate_b": text_b,
        }
        verdict = await self._call(PAIRWISE_SYSTEM_PROMPT, payload, PairwiseVerdict, "rewrite_pairwise")
        mapping = {"A": "current" if current_is_a else "baseline", "B": "baseline" if current_is_a else "current", "tie": "tie"}
        return {
            "preferred": mapping[verdict.preferred],
            "naturalnessPreferred": mapping[verdict.naturalnessPreferred],
            "meaningSafer": mapping[verdict.meaningSafer],
            "reason": verdict.reason,
            "currentShownAs": "A" if current_is_a else "B",
        }

    async def _call(self, system: str, payload: dict[str, Any], result_type: type[BaseModel], schema_name: str) -> Any:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"X-Title": f"{self.app_title} eval judge"},
        )
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=4000,
            response_format=_openrouter_response_format(schema_name, result_type.model_json_schema()),
            extra_body={"provider": {"require_parameters": True}},
        )
        content = response.choices[0].message.content or ""
        return result_type.model_validate_json(content)


def judge_payload(case: dict[str, Any], request: RewriteRequest, response: RewriteResponse) -> dict[str, Any]:
    return {
        "settings": _judge_settings(case, request),
        "original_text": request.text,
        "revised_text": response.revisedText,
        "changes": [
            {
                "original": change.original,
                "revised": change.revised,
                "reason": change.reason,
                "type": change.type,
                "riskLevel": change.riskLevel,
            }
            for change in response.changes
        ],
        "summary": response.summary,
        "warnings": response.warnings,
    }


def _judge_settings(case: dict[str, Any], request: RewriteRequest) -> dict[str, Any]:
    return {
        "case_note": case.get("note", ""),
        "tone": request.tone,
        "user_intent": request.user_intent,
        "protected_terms": request.protected_terms,
        "preserve_formatting": request.preserve_formatting,
    }


def _build_judge(settings: Settings, args: argparse.Namespace, provider: str) -> LLMJudge:
    if not settings.openrouter_api_key:
        raise SystemExit("--judge/--pairwise 에는 OPENROUTER_API_KEY 가 필요합니다 (.env).")
    model = (
        args.judge_model
        or settings.eval_judge_model_name
        or os.environ.get(JUDGE_MODEL_ENV)
        or DEFAULT_JUDGE_MODEL
    )
    if provider == "openrouter" and model in {settings.rewrite_model_name, settings.model_name}:
        print(f"  ⚠ 심사 모델({model})이 윤문 모델과 같습니다. 자기 평가 편향이 생길 수 있습니다.")
    return LLMJudge(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        app_title=settings.openrouter_app_title,
        model=model,
        absolute=bool(args.judge),
    )


# ------------------------------------------------------------------- report


def _build_report(
    settings: Settings,
    provider: str,
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    judge: LLMJudge | None,
) -> dict[str, Any]:
    succeeded = [result for result in results if result.get("ok")]
    metrics = [result["metrics"] for result in succeeded]
    return {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "models": {
            "rewrite": settings.rewrite_model_name,
            "rewriteFallback": settings.rewrite_fallback_model_name,
            "audit": settings.strict_audit_model_name,
            "review": settings.strict_review_model_name,
            "judge": judge.model if judge else None,
        },
        "config": {
            "runs": args.runs,
            "chunkMinChars": settings.chunk_min_chars,
            "styleGateMaxRounds": settings.style_gate_max_rounds,
            "styleGateS2Threshold": settings.style_gate_s2_threshold,
            "reasoningEffort": settings.reasoning_effort,
            "judge": bool(args.judge),
            "pairwise": bool(args.pairwise),
            "judgeCases": list(args.judge_case or []),
            "judgeTags": list(args.judge_tag or []),
            "baseline": args.baseline,
        },
        "summary": build_summary(results, metrics),
        "cases": results,
    }


def build_summary(results: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    succeeded = [result for result in results if result.get("ok")]
    failed = [result for result in results if not result.get("ok")]
    input_s1 = sum(m["inputS1"] for m in metrics)
    residual_s1 = sum(m["residualS1"] for m in metrics)
    input_s2 = sum(m["inputS2"] for m in metrics)
    residual_s2 = sum(m["residualS2"] for m in metrics)
    expectations_passed = sum(m["expectationsPassed"] for m in metrics)
    expectations_total = sum(m["expectationsTotal"] for m in metrics)
    change_count = sum(m["changeQuality"]["count"] for m in metrics)
    fallback_count = sum(m["changeQuality"]["fallbackReasonCount"] for m in metrics)
    pipelines = [m.get("pipeline") or {} for m in metrics]
    judged = [m["judge"] for m in metrics if m.get("judge")]
    pairwise = [result["pairwise"] for result in succeeded if result.get("pairwise")]

    summary: dict[str, Any] = {
        "casesRun": len(results),
        "casesFailed": [result["id"] for result in failed],
        "inputS1": input_s1,
        "residualS1": round(residual_s1, 2),
        "s1ResolutionRate": _rate(input_s1 - residual_s1, input_s1),
        "inputS2": input_s2,
        "residualS2": round(residual_s2, 2),
        "s2ResolutionRate": _rate(input_s2 - residual_s2, input_s2),
        "meanChangeRate": round(statistics.fmean(m["changeRate"] for m in metrics), 2) if metrics else 0.0,
        "expectationsPassed": expectations_passed,
        "expectationsTotal": expectations_total,
        "expectationPassRate": _rate(expectations_passed, expectations_total),
        "casesWithFailedExpectations": [
            result["id"] for result in succeeded if result["metrics"]["expectationsPassed"] < result["metrics"]["expectationsTotal"]
        ],
        "preservationFailCases": [result["id"] for result in succeeded if result["metrics"]["preservationIssues"]],
        "completionFailCases": [result["id"] for result in succeeded if result["metrics"]["completionWarnings"]],
        "overPolishCases": [result["id"] for result in succeeded if result["metrics"]["overPolishSignals"]],
        "registerDriftCases": [result["id"] for result in succeeded if not result["metrics"]["register"]["ok"]],
        "displayUnsafeCases": [result["id"] for result in succeeded if not result["metrics"]["changeQuality"]["displaySafe"]],
        "fallbackReasonRate": _rate(fallback_count, change_count),
        "fallbackReasonCases": [
            result["id"] for result in succeeded if result["metrics"]["changeQuality"]["fallbackReasonCount"]
        ],
        "pipeline": {
            "styleGateFiredCases": sum(1 for p in pipelines if p.get("styleGateRounds", 0) > 0),
            "meanStyleGateRounds": round(statistics.fmean(p.get("styleGateRounds", 0) for p in pipelines), 2) if pipelines else 0.0,
            "reviewRanCases": sum(1 for p in pipelines if p.get("reviewRan")),
            "reviewLocalFallbackCases": sum(
                1 for p in pipelines if p.get("reviewImplementation") == "local_repair_review_fallback"
            ),
            "chunkedCases": sum(1 for p in pipelines if p.get("chunkCount", 1) > 1),
            "auditStatusCounts": dict(Counter(p.get("auditStatus") or "unknown" for p in pipelines)),
        },
        "meanLatencyMs": round(statistics.fmean(m["latencyMs"] for m in metrics), 0) if metrics else 0,
        "totalInputTokens": int(sum(m["inputTokens"] for m in metrics)),
        "totalOutputTokens": int(sum(m["outputTokens"] for m in metrics)),
    }
    if judged:
        judge_summary: dict[str, Any] = {
            key: round(statistics.fmean(float(score[key]) for score in judged), 2) for key in JUDGE_SCORE_KEYS
        }
        summary["judge"] = judge_summary
        judge_summary["casesBelow3"] = [
            result["id"] for result in succeeded if result["metrics"].get("judge", {}).get("overall", 5) < 3
        ]
    if pairwise:
        summary["pairwise"] = {
            "wins": sum(1 for item in pairwise if item["preferred"] == "current"),
            "ties": sum(1 for item in pairwise if item["preferred"] == "tie"),
            "losses": sum(1 for item in pairwise if item["preferred"] == "baseline"),
            "naturalnessWins": sum(1 for item in pairwise if item["naturalnessPreferred"] == "current"),
            "meaningSaferLosses": sum(1 for item in pairwise if item["meaningSafer"] == "baseline"),
        }
    return summary


def _rate(resolved: float, total: float) -> float | None:
    if total <= 0:
        return None
    return round(max(0.0, resolved) / total, 3)


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


def _save_markdown(report: dict[str, Any], report_path: Path, override: str | None) -> Path:
    path = Path(override) if override else report_path.with_suffix(".md")
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def render_markdown(report: dict[str, Any]) -> str:
    """Side-by-side view for a human reviewer: original, result, change list, issues."""
    lines = [f"# 골든셋 평가 {report['createdAt']}", ""]
    lines.append(f"- provider: {report['provider']}, models: {json.dumps(report['models'], ensure_ascii=False)}")
    summary = report["summary"]
    lines.append(
        f"- 기대 통과 {summary['expectationsPassed']}/{summary['expectationsTotal']}"
        f", 잔존 S1 {summary['residualS1']}/{summary['inputS1']}"
        f", 평균 변경률 {summary['meanChangeRate']}%"
        f", 대체 사유 비율 {_format_rate(summary['fallbackReasonRate'])}"
    )
    if summary.get("judge"):
        lines.append(f"- 심사 평균: {json.dumps(summary['judge'], ensure_ascii=False)}")
    lines.append("")
    cases_by_id = {case["id"]: case for case in _load_golden_cases()}
    for result in report["cases"]:
        lines.append(f"## {result['id']}  {result.get('note', '')}")
        if not result.get("ok"):
            lines.append(f"실패: {result.get('error')}")
            lines.append("")
            continue
        m = result["metrics"]
        pipeline = m.get("pipeline") or {}
        lines.append(
            f"- 기대 {m['expectationsPassed']}/{m['expectationsTotal']}, 잔존 S1 {_fmt(m['residualS1'])}/{m['inputS1']}"
            f", S2 {_fmt(m['residualS2'])}/{m['inputS2']}, 변경률 {m['changeRate']:.1f}%"
            f", 격식 {m['register']['original']}→{m['register']['revised']}"
            f", 게이트 {pipeline.get('styleGateRounds', 0)}회, audit {pipeline.get('auditStatus')}"
            f", review {'예' if pipeline.get('reviewRan') else '아니오'}"
        )
        failed = [item for item in m["expectations"] if not item["passed"]]
        if failed:
            lines.append("- 미통과: " + "; ".join(f"{item['name']} ({item['detail']})" if item["detail"] else item["name"] for item in failed))
        if m.get("judge"):
            lines.append(f"- 심사: {json.dumps({k: m['judge'][k] for k in JUDGE_SCORE_KEYS}, ensure_ascii=False)}")
            for issue in m["judge"].get("issues", []):
                lines.append(f"  - {issue}")
        if result.get("pairwise"):
            pw = result["pairwise"]
            lines.append(f"- 기준선 비교: {pw['preferred']} ({pw['reason']})")
        source_text = cases_by_id.get(result["id"], {}).get("text", "")
        lines.append("")
        lines.append("**원문**")
        lines.append("")
        lines.append("> " + source_text.replace("\n", "\n> "))
        lines.append("")
        lines.append("**결과**")
        lines.append("")
        lines.append("> " + result["revisedText"].replace("\n", "\n> "))
        lines.append("")
        if result.get("changes"):
            lines.append("| 원문 | 수정 | 사유 | 유형 |")
            lines.append("|---|---|---|---|")
            for change in result["changes"]:
                lines.append(
                    f"| {_cell(change['original'])} | {_cell(change['revised'])} | {_cell(change['reason'])} | {change['type']}/{change['riskLevel']} |"
                )
            lines.append("")
        if result.get("warnings"):
            lines.append("경고: " + " / ".join(result["warnings"]))
            lines.append("")
    return "\n".join(lines)


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


# ------------------------------------------------------------------ display


def _fmt(value: float | int) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):.1f}"


def _print_case_table(report: dict[str, Any]) -> None:
    has_judge = any(result.get("metrics", {}).get("judge") for result in report["cases"])
    header = f"{'케이스':<10} {'기대':>7} {'잔존S1':>7} {'잔존S2':>7} {'변경률':>7} {'격식':>4} {'게이트':>4} {'리뷰':>4} {'사유':>4}"
    if has_judge:
        header += f" {'심사':>4}"
    header += "  과윤문/미통과"
    print("\n" + header)
    print("-" * len(header.encode("utf-8")))
    for result in report["cases"]:
        if not result.get("ok"):
            print(f"{result['id']:<12} 실패: {result.get('error', '')}")
            continue
        m = result["metrics"]
        pipeline = m.get("pipeline") or {}
        failed_names = [item["name"].split(":")[0] for item in m["expectations"] if not item["passed"]]
        notes = ",".join(m["overPolishSignals"] + failed_names) or "-"
        line = (
            f"{result['id']:<12} "
            f"{_fmt(m['expectationsPassed']):>3}/{m['expectationsTotal']:<3} "
            f"{_fmt(m['residualS1']):>3}/{m['inputS1']:<3} "
            f"{_fmt(m['residualS2']):>3}/{m['inputS2']:<3} "
            f"{m['changeRate']:>6.1f}% "
            f"{'OK' if m['register']['ok'] else '✗':>4} "
            f"{pipeline.get('styleGateRounds', 0):>4} "
            f"{'예' if pipeline.get('reviewRan') else '-':>4} "
            f"{'OK' if m['changeQuality']['fallbackReasonCount'] == 0 else '✗' + str(m['changeQuality']['fallbackReasonCount']):>4}"
        )
        if has_judge:
            line += f" {m.get('judge', {}).get('overall', '-'):>4}"
        line += f"  {notes}"
        print(line)


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n── 요약 " + "─" * 40)
    print(
        f"실행 케이스        : {summary['casesRun']}개"
        + (f" (실패 {len(summary['casesFailed'])}: {', '.join(summary['casesFailed'])})" if summary["casesFailed"] else "")
    )
    print(f"기대 통과율        : {_format_rate(summary['expectationPassRate'])} ({summary['expectationsPassed']}/{summary['expectationsTotal']})")
    print(f"S1 해결률          : {_format_rate(summary['s1ResolutionRate'])} (잔존 {summary['residualS1']}/{summary['inputS1']})")
    print(f"S2 해결률          : {_format_rate(summary['s2ResolutionRate'])} (잔존 {summary['residualS2']}/{summary['inputS2']})")
    print(f"평균 변경률        : {summary['meanChangeRate']:.1f}%")
    print(f"보존 실패 케이스   : {_format_cases(summary['preservationFailCases'])}")
    print(f"완성도 경고 케이스 : {_format_cases(summary['completionFailCases'])}")
    print(f"격식 이탈 케이스   : {_format_cases(summary['registerDriftCases'])}")
    print(f"과윤문 신호 케이스 : {_format_cases(summary['overPolishCases'])}")
    print(f"대체 사유 비율     : {_format_rate(summary['fallbackReasonRate'])} {_format_cases(summary['fallbackReasonCases'])}")
    print(f"표시 불가 변경 목록: {_format_cases(summary['displayUnsafeCases'])}")
    p = summary["pipeline"]
    print(
        f"파이프라인         : 게이트 발동 {p['styleGateFiredCases']}건(평균 {p['meanStyleGateRounds']}회), "
        f"리뷰 {p['reviewRanCases']}건(로컬 대체 {p['reviewLocalFallbackCases']}건), 청크 {p['chunkedCases']}건, "
        f"audit {p['auditStatusCounts']}"
    )
    print(f"평균 지연/토큰     : {summary['meanLatencyMs']:.0f}ms, in {summary['totalInputTokens']} / out {summary['totalOutputTokens']}")
    if summary.get("judge"):
        j = summary["judge"]
        print(
            f"심사 평균(1~5)     : 종합 {j['overall']}, 자연스러움 {j['naturalness']}, AI티없음 {j['aiTell']}, "
            f"의미 {j['meaning']}, 과윤문없음 {j['overEdit']}, 사유 {j['reasonQuality']}, 격식 {j['registerFit']}"
            + (f"  (3점 미만: {', '.join(j['casesBelow3'])})" if j["casesBelow3"] else "")
        )
    if summary.get("pairwise"):
        pw = summary["pairwise"]
        print(
            f"기준선 대비 심사   : 승 {pw['wins']} / 무 {pw['ties']} / 패 {pw['losses']}"
            f" (자연스러움 승 {pw['naturalnessWins']}, 의미 보존 패 {pw['meaningSaferLosses']})"
        )


def _format_rate(rate: float | None) -> str:
    return "n/a" if rate is None else f"{rate * 100:.1f}%"


def _format_cases(case_ids: list[str]) -> str:
    return f"{len(case_ids)}건" + (f" ({', '.join(case_ids)})" if case_ids else "")


def _load_baseline(path: str) -> dict[str, Any]:
    baseline = json.loads(Path(path).read_text(encoding="utf-8"))
    baseline["casesById"] = {case["id"]: case for case in baseline.get("cases", []) if case.get("ok")}
    return baseline


def baseline_diff(report: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-case deltas against a previous report. A case regresses when S1 or
    preservation damage goes up, expectations go down, or the judge drops ≥0.5."""
    rows: list[dict[str, Any]] = []
    for result in report["cases"]:
        previous = baseline.get("casesById", {}).get(result["id"])
        if not result.get("ok") or not previous:
            continue
        current = result["metrics"]
        prev = previous["metrics"]
        row = {
            "id": result["id"],
            "dS1": round(float(current["residualS1"]) - float(prev["residualS1"]), 2),
            "dS2": round(float(current["residualS2"]) - float(prev["residualS2"]), 2),
            "dPreservation": len(current["preservationIssues"]) - len(prev.get("preservationIssues", [])),
            "dExpectations": round(float(current.get("expectationsPassed", 0)) - float(prev.get("expectationsPassed", 0)), 2),
            "dChangeRate": round(float(current["changeRate"]) - float(prev["changeRate"]), 1),
        }
        if current.get("judge") and prev.get("judge"):
            row["dJudge"] = round(float(current["judge"]["overall"]) - float(prev["judge"]["overall"]), 2)
        row["regressed"] = (
            row["dS1"] > 0.5
            or row["dPreservation"] > 0
            or row["dExpectations"] < 0
            or row.get("dJudge", 0) <= -0.5
        )
        row["improved"] = not row["regressed"] and (
            row["dS1"] < -0.5 or row["dExpectations"] > 0 or row.get("dJudge", 0) >= 0.5
        )
        rows.append(row)
    return rows


def _print_baseline_diff(report: dict[str, Any], baseline: dict[str, Any], baseline_path: Path) -> None:
    print(f"\n── 베이스라인 비교: {baseline_path.name} " + "─" * 20)
    rows = baseline_diff(report, baseline)
    regressions = []
    for row in rows:
        marks = []
        if row["dS1"]:
            marks.append(f"S1 {row['dS1']:+g}")
        if row["dS2"]:
            marks.append(f"S2 {row['dS2']:+g}")
        if row["dPreservation"]:
            marks.append(f"보존 이슈 {row['dPreservation']:+d}")
        if row["dExpectations"]:
            marks.append(f"기대 {row['dExpectations']:+g}")
        if row.get("dJudge"):
            marks.append(f"심사 {row['dJudge']:+g}")
        if not marks:
            continue
        line = f"{row['id']:<12} {', '.join(marks)}"
        if row["regressed"]:
            regressions.append(row["id"])
            line += "  ← 회귀"
        elif row["improved"]:
            line += "  ← 개선"
        print(line)

    prev = baseline.get("summary", {})
    cur = report["summary"]
    print(
        f"\n요약 변화: 기대 통과 {prev.get('expectationsPassed', '?')}/{prev.get('expectationsTotal', '?')} → "
        f"{cur['expectationsPassed']}/{cur['expectationsTotal']}, "
        f"S1 잔존 {prev.get('residualS1')} → {cur['residualS1']}, "
        f"S2 잔존 {prev.get('residualS2')} → {cur['residualS2']}, "
        f"평균 변경률 {prev.get('meanChangeRate')}% → {cur['meanChangeRate']}%, "
        f"보존 실패 {len(prev.get('preservationFailCases', []))}건 → {len(cur['preservationFailCases'])}건, "
        f"대체 사유 {_format_rate(prev.get('fallbackReasonRate'))} → {_format_rate(cur['fallbackReasonRate'])}"
    )
    if prev.get("judge") and cur.get("judge"):
        print(f"심사 종합 {prev['judge'].get('overall')} → {cur['judge']['overall']}")
    if regressions:
        print(f"⚠ 회귀 의심 케이스 {len(regressions)}건: {', '.join(regressions)}")
    else:
        print("회귀 없음")


# -------------------------------------------------------------------- setup


def _build_runner(settings: Settings, provider: str, recorder: StageRecorder) -> RewriteGraphRunner:
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
        explain_model_name=settings.explain_model_name,
        reasoning_effort=settings.reasoning_effort,
    )
    return RewriteGraphRunner(settings, llm, recorder)  # type: ignore[arg-type]


def _load_golden_cases(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases = _load_golden_cases(Path(args.golden))
    if args.case:
        wanted = set(args.case)
        cases = [case for case in cases if case["id"] in wanted]
    if args.tag:
        wanted_tags = set(args.tag)
        cases = [case for case in cases if wanted_tags & set(case.get("tags", []))]
    return cases


def _judge_filter(args: argparse.Namespace) -> Callable[[dict[str, Any]], bool] | None:
    """Limit judge/pairwise calls to a subset so a full pipeline run stays cheap."""
    wanted_ids = set(args.judge_case or [])
    wanted_tags = set(args.judge_tag or [])
    if not wanted_ids and not wanted_tags:
        return None
    return lambda case: case["id"] in wanted_ids or bool(wanted_tags & set(case.get("tags", [])))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 자동 채점 평가를 실행합니다.")
    parser.add_argument("--golden", default=str(GOLDEN_PATH), help="골든셋 JSON 경로")
    parser.add_argument("--case", action="append", default=[], help="특정 케이스 id만 실행 (반복 지정 가능)")
    parser.add_argument("--tag", action="append", default=[], help="특정 태그 케이스만 실행 (반복 지정 가능)")
    parser.add_argument("--runs", type=int, default=1, help="케이스당 반복 실행 횟수 (기본 1). 2 이상이면 평균과 편차를 기록")
    parser.add_argument("--concurrency", type=int, default=3, help="동시 실행 케이스 수 (기본 3)")
    parser.add_argument("--report", help="리포트 저장 경로 (기본: evals/reports/report-<timestamp>.json)")
    parser.add_argument("--markdown", help="검토용 마크다운 저장 경로 (기본: 리포트와 같은 이름의 .md)")
    parser.add_argument("--baseline", help="비교할 이전 리포트 JSON 경로")
    parser.add_argument("--judge", action="store_true", help="LLM 심사 점수(1~5)를 케이스마다 추가")
    parser.add_argument("--pairwise", action="store_true", help="--baseline 결과와 블라인드 A/B 심사")
    parser.add_argument("--judge-model", help=f"심사 모델 (기본: ${JUDGE_MODEL_ENV} 또는 {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--judge-case", action="append", default=[], help="심사(--judge/--pairwise)를 이 케이스에만 적용 (반복 지정 가능)")
    parser.add_argument("--judge-tag", action="append", default=[], help="심사를 이 태그 케이스에만 적용 (예: --judge-tag 핵심)")
    parser.add_argument("--stub", action="store_true", help="LLM 대신 stub 사용 (스크립트 자체 점검용)")
    parser.add_argument("--rejudge", help="파이프라인을 다시 돌리지 않고 이 리포트의 결과만 다시 심사 (--judge-model, --pairwise 적용)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
