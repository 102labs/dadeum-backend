"""Provider-free tests for the golden-set eval harness (scripts/eval_golden.py)."""

import importlib.util
import json
from pathlib import Path

import pytest

from humanize_core.config import Settings
from humanize_core.graph import RewriteGraphRunner
from humanize_core.llm import StubRewriteLLM
from humanize_core.schemas import Change, RewriteRequest, RewriteResponse, Usage

CORE_DIR = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("eval_golden", CORE_DIR / "scripts" / "eval_golden.py")
assert _SPEC is not None and _SPEC.loader is not None
eval_golden = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_golden)


def _response(revised: str, changes: list[Change] | None = None, warnings: list[str] | None = None) -> RewriteResponse:
    return RewriteResponse(
        revisedText=revised,
        changes=changes or [],
        summary=[],
        warnings=warnings or [],
        usage=Usage(inputTokens=1, outputTokens=1, latencyMs=1, rounds=1),
    )


# ---------------------------------------------------------------- register


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("보고서를 보냈습니다. 검토 부탁드립니다. 기한은 금요일입니다.", "formal"),
        ("장소를 정했어요. 열 명까지 예약했어요. 못 오시면 알려 주세요.", "haeyo"),
        ("지표를 점검했다. 전환율은 올랐다. 방문자는 줄었다.", "plain"),
        ("일정 공유드립니다. 수요일에 반영된다. 목요일에 배포해요. 당번이 맡는다.", "mixed"),
        ("회의를 엽니다. 자료를 보내 주세요. 기한을 지켜 주시기 바랍니다.", "formal"),
        ("장소를 정했어요. 참석 여부를 알려 주세요. 감사합니다.", "haeyo"),
        ("회의실 예약 현황: 본관 3층 A룸", "unknown"),
    ],
)
def test_detect_register(text, expected):
    assert eval_golden.detect_register(text) == expected


def test_expected_register_follows_tone_and_explicit_override():
    assert eval_golden.expected_register({"tone": "formal"}, "haeyo") == "formal"
    assert eval_golden.expected_register({"tone": "friendly"}, "plain") == "soft"
    assert eval_golden.expected_register({"tone": "keep"}, "haeyo") == "haeyo"
    assert eval_golden.expected_register({"tone": "keep"}, "mixed") == "any"
    assert eval_golden.expected_register({"expect_register": "consistent"}, "mixed") == "consistent"


def test_register_matches_semantics():
    assert eval_golden.register_matches("consistent", "formal")
    assert not eval_golden.register_matches("consistent", "mixed")
    assert eval_golden.register_matches("soft", "haeyo")
    assert not eval_golden.register_matches("soft", "plain")
    assert eval_golden.register_matches("any", "mixed")
    assert not eval_golden.register_matches("formal", "haeyo")


# ------------------------------------------------------------- change list


def test_change_list_quality_flags_fallback_unsafe_and_noop_changes():
    original = "이번 프로젝트에 대해 공유드립니다. 일정은 다음 주입니다."
    revised = "이번 프로젝트를 공유드립니다. 일정은 다음 주입니다."
    response = _response(
        revised,
        changes=[
            Change(original="프로젝트에 대해", revised="프로젝트를", reason="영어 about 직역이라 목적격 조사로 바꿨습니다.", type="clarity"),
            Change(original="일정은", revised="일정은", reason="원문과 최종 윤문 결과의 차이를 비교 가능한 구간으로 정리했습니다.", type="clarity"),
            Change(original="없는 구간", revised="일정", reason="", type="clarity"),
            Change(original="", revised="", reason="세부 변경 구간이 14건이라 주요 12건만 비교 표시에 사용했습니다.", type="clarity"),
        ],
    )

    quality = eval_golden.change_list_quality(original, revised, response)

    assert quality["count"] == 3
    assert quality["fallbackReasonCount"] == 1
    assert quality["emptyReasonCount"] == 1
    assert quality["noopChangeCount"] == 1
    assert quality["unsafeSnippetCount"] == 1
    assert quality["displaySafe"] is False
    assert quality["overflowNote"] is True
    assert quality["textChangedWithoutChanges"] is False


def test_change_list_quality_reports_text_changed_without_changes():
    response = _response("바뀐 문장입니다.", changes=[])
    quality = eval_golden.change_list_quality("원래 문장입니다.", "바뀐 문장입니다.", response)
    assert quality["textChangedWithoutChanges"] is True


# ------------------------------------------------------------ expectations


def _expectations(case: dict, original: str, revised: str) -> dict[str, dict]:
    request = RewriteRequest(text=original, tone=case.get("tone", "keep"))
    register = eval_golden.register_report(case, original, revised)
    quality = eval_golden.change_list_quality(original, revised, _response(revised))
    items = eval_golden.evaluate_expectations(
        case,
        request,
        revised,
        preservation_issues=[],
        completion_warnings=[],
        register=register,
        change_quality=quality,
    )
    return {item["name"]: item for item in items}


def test_expectations_cover_declared_case_fields():
    original = "데이터 분석을 통해 원인에 대해 파악했습니다. 매출은 42억 원입니다. 매출은 42억 원입니다."
    revised = "데이터를 분석해 원인을 파악했습니다. 매출은 42억 원입니다. 매출은 42억 원입니다."
    case = {
        "must_keep": ["42억 원"],
        "must_remove": ["[을를] 통해", "에 대해"],
        "max_count": {"매출": 2},
        "max_change_rate": 40,
        "min_change_rate": 3,
        "max_length_ratio": 1.1,
    }

    items = _expectations(case, original, revised)

    assert items["유지: 42억 원"]["passed"]
    assert items["제거: [을를] 통해"]["passed"]
    assert items["제거: 에 대해"]["passed"]
    assert items["최대 2회: 매출"]["passed"]
    assert items["변경률 ≤ 40%"]["passed"]
    assert items["변경률 ≥ 3%"]["passed"]
    assert items["길이 비율 ≤ 1.1"]["passed"]


def test_expectations_fail_when_value_count_drops_or_pattern_remains():
    original = "매출은 42억 원이고 이익도 42억 원입니다. 이에 대해 논의합니다."
    revised = "매출은 42억 원입니다. 이에 대해 논의합니다."
    case = {"must_keep": ["42억 원"], "must_remove": ["에 대해"], "max_count": {"42억": 1}}

    items = _expectations(case, original, revised)

    assert not items["유지: 42억 원"]["passed"]
    assert "원문 2회 / 결과 1회" in items["유지: 42억 원"]["detail"]
    assert not items["제거: 에 대해"]["passed"]
    assert items["최대 1회: 42억"]["passed"]


def test_expectations_check_register_and_paragraphs():
    original = "일정 공유드립니다.\n\n수요일에 반영됩니다."
    revised = "일정 공유할게요.\n\n수요일에 반영돼요."
    case = {"expect_register": "formal", "expect_paragraphs": 2}

    items = _expectations(case, original, revised)

    assert not items["격식 유지"]["passed"]
    assert "기대 formal / 결과 haeyo" in items["격식 유지"]["detail"]
    assert items["문단 2개"]["passed"]


def test_implicit_expectations_include_preservation_completion_and_reasons():
    original = "원문입니다."
    revised = "결과입니다."
    request = RewriteRequest(text=original)
    register = eval_golden.register_report({}, original, revised)
    quality = eval_golden.change_list_quality(original, revised, _response(revised))
    items = {
        item["name"]: item
        for item in eval_golden.evaluate_expectations(
            {},
            request,
            revised,
            preservation_issues=["수치/단위 보존 대상이 원문보다 적게 남았습니다: 3"],
            completion_warnings=["Rewrite 결과가 비어 있어 완성도 검증을 통과하지 못했습니다."],
            register=register,
            change_quality=quality,
        )
    }
    assert not items["보존 대상 유지"]["passed"]
    assert not items["완성도"]["passed"]
    assert not items["변경 사유 존재"]["passed"]  # text changed but no changes listed
    assert items["변경 목록 표시 가능"]["passed"]


# --------------------------------------------------------- aggregation/diff


def _metrics(**overrides) -> dict:
    base = {
        "inputS1": 2,
        "inputS2": 3,
        "residualS1": 0,
        "residualS2": 1,
        "changeRate": 20.0,
        "lengthRatio": 1.0,
        "latencyMs": 100,
        "inputTokens": 10,
        "outputTokens": 10,
        "preservationIssues": [],
        "completionWarnings": [],
        "overPolishSignals": [],
        "responseWarnings": [],
        "register": {"ok": True},
        "changeQuality": {"count": 2, "fallbackReasonCount": 0, "displaySafe": True},
        "expectations": [],
        "expectationsPassed": 5,
        "expectationsTotal": 5,
    }
    base.update(overrides)
    return base


def test_aggregate_runs_means_numbers_and_keeps_worst_run_lists():
    good = _metrics()
    bad = _metrics(residualS1=2, changeRate=40.0, expectationsPassed=3, preservationIssues=["x"])

    aggregated = eval_golden.aggregate_runs([good, bad])

    assert aggregated["residualS1"] == 1.0
    assert aggregated["changeRate"] == 30.0
    assert aggregated["expectationsPassed"] == 4.0
    assert aggregated["preservationIssues"] == ["x"]
    assert aggregated["runCount"] == 2
    assert aggregated["runSpread"]["residualS1"] == [0.0, 2.0]


def test_aggregate_single_run_is_identity():
    only = _metrics()
    assert eval_golden.aggregate_runs([only]) is only


def test_baseline_diff_marks_regressions_and_improvements():
    report = {
        "cases": [
            {"id": "a", "ok": True, "metrics": _metrics(residualS1=1)},
            {"id": "b", "ok": True, "metrics": _metrics(expectationsPassed=5)},
            {"id": "c", "ok": True, "metrics": _metrics(preservationIssues=["x"])},
        ]
    }
    baseline = {
        "casesById": {
            "a": {"metrics": _metrics(residualS1=0)},
            "b": {"metrics": _metrics(expectationsPassed=3)},
            "c": {"metrics": _metrics()},
        }
    }

    rows = {row["id"]: row for row in eval_golden.baseline_diff(report, baseline)}

    assert rows["a"]["regressed"] and rows["a"]["dS1"] == 1.0
    assert rows["b"]["improved"] and rows["b"]["dExpectations"] == 2.0
    assert rows["c"]["regressed"] and rows["c"]["dPreservation"] == 1


# ----------------------------------------------------------- stage capture


def test_stage_summary_condenses_graph_events():
    events = [
        {"event": "graph.stage.succeeded", "step": "rewrite", "duration_ms": 10, "details": {"chunk_count": 2}},
        {"event": "graph.stage.succeeded", "step": "style_gate", "duration_ms": 20, "details": {"repair_rounds": 1, "implementation": "style_repair"}},
        {"event": "graph.stage.succeeded", "step": "audit", "duration_ms": 5, "details": {"result_status": "conditional_pass", "flagged_edits_count": 1}},
        {"event": "graph.stage.succeeded", "step": "review", "duration_ms": 7, "details": {"implementation": "local_repair_review_fallback"}},
    ]
    summary = eval_golden.stage_summary(events)
    assert summary["chunkCount"] == 2
    assert summary["styleGateRounds"] == 1
    assert summary["auditStatus"] == "conditional_pass"
    assert summary["reviewRan"] is True
    assert summary["reviewImplementation"] == "local_repair_review_fallback"
    assert summary["durationsMs"] == {"rewrite": 10, "style_gate": 20, "audit": 5, "review": 7}


def test_stage_recorder_takes_only_matching_request_events():
    recorder = eval_golden.StageRecorder()
    recorder.event("graph.stage.succeeded", request_id="a", step="rewrite", details={})
    recorder.event("graph.stage.succeeded", request_id="b", step="rewrite", details={})
    assert [item["request_id"] for item in recorder.take("a")] == ["a"]
    assert [item["request_id"] for item in recorder.take("b")] == ["b"]
    assert recorder.events == []


# ------------------------------------------------------------- judge payload


def test_judge_payload_carries_changes_and_settings_but_no_rulebook():
    case = {"note": "n", "tone": "formal"}
    request = RewriteRequest(text="원문", tone="formal", user_intent="간결하게", protected_terms=["X"])
    response = _response(
        "결과",
        changes=[Change(original="원", revised="결", reason="이유", type="tone", riskLevel="medium")],
        warnings=["w"],
    )
    payload = eval_golden.judge_payload(case, request, response)
    assert payload["settings"] == {
        "case_note": "n",
        "tone": "formal",
        "user_intent": "간결하게",
        "protected_terms": ["X"],
        "preserve_formatting": True,
    }
    assert payload["changes"] == [{"original": "원", "revised": "결", "reason": "이유", "type": "tone", "riskLevel": "medium"}]
    assert payload["warnings"] == ["w"]
    assert "rulebook" not in json.dumps(payload)


def test_judge_score_schema_is_openrouter_strict_compatible():
    from humanize_core.llm import _openrouter_response_format

    fmt = _openrouter_response_format("rewrite_judge", eval_golden.JudgeScore.model_json_schema())
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(eval_golden.JUDGE_SCORE_KEYS) | {"issues"}


# ------------------------------------------------------------ golden set


def test_golden_cases_have_unique_ids_and_valid_expectation_fields():
    cases = eval_golden._load_golden_cases()
    ids = [case["id"] for case in cases]
    assert len(ids) == len(set(ids))
    allowed = {
        "id", "tags", "note", "text", "tone", "user_intent", "protected_terms", "preserve_formatting",
        "must_keep", "must_remove", "max_count", "max_change_rate", "min_change_rate",
        "max_length_ratio", "expect_paragraphs", "expect_register",
    }
    for case in cases:
        unknown = set(case) - allowed
        assert not unknown, f"{case['id']}: unknown fields {unknown}"
        for value in case.get("must_keep", []):
            assert value in case["text"], f"{case['id']}: must_keep '{value}' is not in the source text"
        if "expect_register" in case:
            assert case["expect_register"] in {"formal", "haeyo", "plain", "consistent", "soft"}


# ------------------------------------------------------------- end to end


async def test_run_case_with_stub_runner_produces_full_metrics():
    settings = Settings(
        model_provider="stub",
        model_name="stub",
        core_api_key="k",
        signing_secret="s",
        job_store_path=":memory:",
        job_worker_enabled=False,
        debug_log_enabled=False,
    )
    recorder = eval_golden.StageRecorder()
    runner = RewriteGraphRunner(settings, StubRewriteLLM(), recorder)
    case = {
        "id": "t-1",
        "tags": ["t"],
        "note": "stub",
        "text": "이번 프로젝트에 대해 공유드립니다. 일정은 다음 주입니다.",
        "must_keep": ["다음 주"],
        "must_remove": ["에 대해"],
        "max_change_rate": 50,
    }

    result = await eval_golden._run_case(runner, recorder, case, runs=2)

    assert result["ok"]
    metrics = result["metrics"]
    assert metrics["runCount"] == 2
    assert metrics["expectationsTotal"] == 8
    names = {item["name"]: item["passed"] for item in metrics["expectations"]}
    assert names["유지: 다음 주"] is True
    assert names["제거: 에 대해"] is False  # the stub returns the text unchanged
    assert metrics["register"]["original"] == "formal"
    assert metrics["pipeline"]["stages"][0] == "prepare"
    assert metrics["pipeline"]["auditStatus"] == "full_pass"
    assert metrics["pipeline"]["reviewRan"] is False
    assert recorder.events == []
    assert len(result["runs"]) == 2


def test_render_markdown_lists_cases_and_changes():
    report = {
        "createdAt": "now",
        "provider": "stub",
        "models": {},
        "summary": {
            "expectationsPassed": 1,
            "expectationsTotal": 2,
            "residualS1": 0,
            "inputS1": 1,
            "meanChangeRate": 5.0,
            "fallbackReasonRate": 0.0,
        },
        "cases": [
            {
                "id": "tr-01",
                "note": "n",
                "ok": True,
                "metrics": {
                    **_metrics(),
                    "register": {"original": "formal", "revised": "formal", "ok": True},
                    "expectations": [{"name": "제거: 에 대해", "passed": False, "detail": "1회"}],
                    "pipeline": {"styleGateRounds": 1, "auditStatus": "full_pass", "reviewRan": False},
                },
                "revisedText": "결과",
                "changes": [{"original": "a|b", "revised": "c", "reason": "r", "type": "clarity", "riskLevel": "low"}],
                "warnings": ["w"],
            }
        ],
    }
    markdown = eval_golden.render_markdown(report)
    assert "## tr-01" in markdown
    assert "미통과: 제거: 에 대해 (1회)" in markdown
    assert "| a\\|b | c | r | clarity/low |" in markdown
    assert "경고: w" in markdown


def test_judge_filter_limits_by_case_id_or_tag():
    import argparse

    none = eval_golden._judge_filter(argparse.Namespace(judge_case=[], judge_tag=[]))
    assert none is None
    pick = eval_golden._judge_filter(argparse.Namespace(judge_case=["a"], judge_tag=["핵심"]))
    assert pick is not None
    assert pick({"id": "a", "tags": []})
    assert pick({"id": "z", "tags": ["핵심"]})
    assert not pick({"id": "z", "tags": ["기타"]})


def test_golden_core_tag_marks_twelve_representative_cases():
    cases = eval_golden._load_golden_cases()
    assert sum(1 for case in cases if "핵심" in case.get("tags", [])) == 12


def test_response_from_result_rebuilds_scored_response():
    result = {
        "id": "x",
        "revisedText": "결과",
        "changes": [{"original": "a", "revised": "b", "reason": "r", "type": "clarity", "riskLevel": "low"}],
        "summary": ["s"],
        "warnings": ["w"],
        "metrics": {"inputTokens": 3, "outputTokens": 4, "latencyMs": 5},
    }
    response = eval_golden.response_from_result(result)
    assert response.revisedText == "결과"
    assert response.changes[0].original == "a"
    assert response.warnings == ["w"]
    assert response.usage.inputTokens == 3
