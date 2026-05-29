import asyncio
import hashlib
import hmac
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from humanize_core.api import create_app
from humanize_core.config import Settings
from humanize_core.graph import RewriteGraphRunner
from humanize_core.im_not_ai import prompts, resources
from humanize_core.im_not_ai.audit import (
    SUPPORTED_STYLE_RULE_IDS,
    finding_density,
    finding_score,
    local_detect,
)
from humanize_core.im_not_ai.metrics_v2 import (
    compute_all_v2,
    deul_overuse_rate,
    double_passive_count,
    pronoun_density,
)
from humanize_core.im_not_ai.schemas import (
    AuditResult,
    RewriteResult,
    StrictReviewResult,
)
from humanize_core.llm import (
    MAX_OUTPUT_TOKENS,
    OpenAIRewriteLLM,
    OpenRouterRewriteLLM,
    StubRewriteLLM,
    _openai_rewrite_text_format,
    _openrouter_response_format,
)
from humanize_core.schemas import Change
from humanize_core.schemas import RewriteRequest as RewriteRequestForTest


def _settings(**overrides) -> Settings:
    values = {
        "core_api_key": "test-core-key",
        "signing_secret": "test-signing-secret",
        "model_provider": "stub",
        "model_name": "stub",
        "max_chars": 5_000,
        "job_store_path": ":memory:",
        "job_worker_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _payload(**overrides):
    body = {
        "text": "안녕하세요.  2026년 5월 보고서 문장을 더 명확하게 정리해주세요.",
        "user_intent": "",
        "rewrite_mode": "strict",
        "tone": "keep",
        "protected_terms": ["2026년"],
        "max_rounds": 1,
        "preserve_formatting": True,
    }
    body.update(overrides)
    return body


def _signed_headers(raw_body: bytes, *, timestamp: str | None = None, request_id: str = "req_test"):
    timestamp = timestamp or str(int(time.time()))
    body_hash = hashlib.sha256(raw_body).hexdigest()
    signature_payload = f"{timestamp}.{request_id}.{body_hash}"
    signature = hmac.new(
        b"test-signing-secret",
        signature_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return {
        "X-Core-Api-Key": "test-core-key",
        "X-Request-Id": request_id,
        "X-Timestamp": timestamp,
        "X-Body-SHA256": body_hash,
        "X-Signature": signature,
    }


def _strict_review_result(
    request,
    revised_text,
    *,
    warnings=None,
    final_warnings=None,
    blocking=None,
    status="full_pass",
):
    return StrictReviewResult(
        revisedText=revised_text,
        changes=[
            Change(
                original=request.text,
                revised=revised_text,
                reason="strict review 최종 후보입니다.",
                type="clarity",
                riskLevel="low",
            )
        ],
        summary=["strict review가 최종 후보를 구성했습니다."],
        warnings=warnings or [],
        finalAuditStatus=status,
        finalAuditWarnings=final_warnings or [],
        finalBlockingIssues=blocking or [],
    )


def _client() -> TestClient:
    return TestClient(create_app(_settings()))


def test_health_returns_ok():
    client = _client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_api_key_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)
    headers.pop("X-Core-Api-Key")

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_bad_signature_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)
    headers["X-Signature"] = "bad"

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_expired_timestamp_returns_401():
    client = _client()
    raw_body = json.dumps(_payload(), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body, timestamp=str(int(time.time()) - 600))

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 401


def test_body_hash_mismatch_returns_401():
    client = _client()
    signed_body = json.dumps(_payload(text="signed"), ensure_ascii=False).encode("utf-8")
    sent_body = json.dumps(_payload(text="sent"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(signed_body)

    response = client.post("/v1/rewrite", content=sent_body, headers=headers)

    assert response.status_code == 401


def test_invalid_enum_returns_422_after_auth_passes():
    client = _client()
    raw_body = json.dumps(_payload(tone="executive"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_old_contract_fields_return_422_after_auth_passes():
    client = _client()
    body = _payload()
    body["intensity"] = "standard"
    body["concision"] = "tighten"
    body["intent"] = "business_polish"
    body["quality_mode"] = "balanced"
    body["focus_categories"] = []
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_rewrite_request_defaults_match_new_contract():
    request = RewriteRequestForTest.model_validate({"text": "문장을 정리합니다."})

    assert request.user_intent == ""
    assert request.rewrite_mode == "fast"
    assert request.tone == "keep"
    assert request.protected_terms == []
    assert request.max_rounds == 1
    assert request.preserve_formatting is True


def test_rewrite_request_normalizes_intent_and_protected_terms():
    request = RewriteRequestForTest.model_validate(
        {
            "text": "API v1은 2026년에 유지됩니다.",
            "user_intent": "  더 명확하게  ",
            "protected_terms": [" API v1 ", "", " 2026년 "],
        }
    )

    assert request.user_intent == "더 명확하게"
    assert request.protected_terms == ["API v1", "2026년"]


def test_rewrite_success_returns_structured_response():
    client = _client()
    raw_body = json.dumps(_payload(rewrite_mode="fast"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["revisedText"]
    assert data["changes"]
    assert data["summary"]
    assert data["usage"]["rounds"] == 1
    assert isinstance(data["warnings"], list)


def test_stub_rewrite_reflects_user_selected_controls():
    client = _client()
    raw_body = json.dumps(
        _payload(
            text="보고 문장",
            user_intent="더 단정하게 정리",
            rewrite_mode="fast",
            tone="formal",
            preserve_formatting=True,
        ),
        ensure_ascii=False,
    ).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["revisedText"] == "보고 문장."
    assert "격식 있는 톤" in data["summary"][0]
    assert "사용자 요청 방향" in data["summary"][0]
    assert "형식을 보존" in data["summary"][0]


async def test_long_compat_request_uses_single_rewrite_path():
    text = "보고 문장입니다. " * 500
    request = RewriteRequestForTest.model_validate(_payload(text=text, rewrite_mode="fast", max_rounds=3))

    class CapturingRewriteLLM:
        def __init__(self) -> None:
            self.rewrite_calls = 0

        async def rewrite_once(self, request, context):
            self.rewrite_calls += 1
            return RewriteResult(
                revisedText=request.text,
                changes=[
                    Change(
                        original="보고 문장입니다.",
                        revised="보고 문장입니다.",
                        reason="긴 호환 요청도 단일 rewrite 루틴으로 처리합니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["단일 rewrite 루틴을 사용했습니다."],
            )

    llm = CapturingRewriteLLM()
    response = await RewriteGraphRunner(_settings(), llm).run(request)

    assert 4_000 < len(text) <= 5_000
    assert llm.rewrite_calls == 1
    assert response.usage.rounds == 1
    assert not response.warnings


def test_valid_strict_request_returns_accepted_job():
    client = _client()
    raw_body = json.dumps(_payload(rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 202
    data = response.json()
    assert data["jobId"]
    assert data["requestId"] == "req_test"
    assert data["status"] == "queued"
    assert "revisedText" not in data


def test_strict_request_ignores_max_rounds_and_runs_single_routine():
    app = create_app(_settings())
    text = "결론적으로 성과를 냈습니다. 따라서 개선됩니다. 이를 통해 정리합니다. 그러므로 유지합니다."
    raw_body = json.dumps(_payload(text=text, rewrite_mode="strict", max_rounds=3), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)

        assert response.status_code == 202
        job_id = response.json()["jobId"]
        asyncio.run(app.state.job_manager.process_next())

        status_headers = _signed_headers(b"", request_id="req_status")
        status_response = client.get(f"/v1/rewrite-jobs/{job_id}", headers=status_headers)

    assert status_response.status_code == 200
    data = status_response.json()
    assert data["status"] == "succeeded"
    assert data["result"]["usage"]["rounds"] == 1
    assert not any("최대 라운드" in warning for warning in data["result"]["warnings"])


def test_strict_request_defaults_to_async_job_when_max_rounds_omitted():
    client = _client()
    text = "결론적으로 성과를 냈습니다. 따라서 개선됩니다. 이를 통해 정리합니다. 그러므로 유지합니다."
    body = _payload(text=text, rewrite_mode="strict")
    body.pop("max_rounds")
    raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"


def test_job_store_reopens_after_app_lifespan_restart(tmp_path):
    app = create_app(_settings(job_store_path=str(tmp_path / "jobs.sqlite3")))
    raw_body = json.dumps(_payload(rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["jobId"]

    with TestClient(app) as client:
        status_headers = _signed_headers(b"", request_id="req_lifespan_status")
        status_response = client.get(f"/v1/rewrite-jobs/{job_id}", headers=status_headers)

    assert status_response.status_code == 200
    assert status_response.json()["status"] == "queued"


def test_strict_job_worker_processes_queued_jobs(tmp_path):
    app = create_app(
        _settings(
            job_store_path=str(tmp_path / "jobs.sqlite3"),
            job_worker_enabled=True,
            job_poll_interval_seconds=0.01,
        )
    )
    raw_body = json.dumps(_payload(rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["jobId"]

        status_response = None
        for index in range(50):
            status_headers = _signed_headers(b"", request_id=f"req_worker_status_{index}")
            status_response = client.get(f"/v1/rewrite-jobs/{job_id}", headers=status_headers)
            if status_response.json()["status"] == "succeeded":
                break
            time.sleep(0.01)

    assert status_response is not None
    assert status_response.status_code == 200
    data = status_response.json()
    assert data["status"] == "succeeded"
    assert data["result"]["usage"]["rounds"] == 1


def test_prompts_pass_user_selected_rewrite_controls():
    request = RewriteRequestForTest.model_validate(
        _payload(
            user_intent="문장을 더 부드럽게 다듬어 주세요.",
            rewrite_mode="strict",
            tone="friendly",
            max_rounds=3,
            preserve_formatting=True,
        )
    )

    payload = json.loads(
        prompts.rewrite_user_prompt(
            request,
            {},
        )
    )

    assert payload["settings"] == {
        "user_intent": "문장을 더 부드럽게 다듬어 주세요.",
        "mode_policy": "single_active_rewrite_with_preservation_audit",
        "tone": "friendly",
        "preserve_formatting": True,
    }
    assert "우선 반영" in payload["rewrite_guidance"]["user_intent"]
    assert "친근" in payload["rewrite_guidance"]["tone"]
    assert "protected_terms" not in payload["rewrite_guidance"]
    assert "max_rounds" not in payload["rewrite_guidance"]
    assert "rewrite_mode" not in payload["rewrite_guidance"]
    assert "줄바꿈" in payload["rewrite_guidance"]["formatting"]


def test_rewrite_prompt_runs_active_rulebook_single_pass():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="2026년 이 기능은 데이터를 통해 성장을 지원합니다. 전략적 실행성과 구조화가 필요합니다.",
            rewrite_mode="strict",
            max_rounds=2,
        )
    )

    system_prompt = prompts.rewrite_system_prompt()
    payload = json.loads(prompts.rewrite_user_prompt(request, {}))
    rendered_payload = json.dumps(payload, ensure_ascii=False)

    assert "strict detect" not in system_prompt
    assert "strict" not in system_prompt.lower()
    assert "strict" not in rendered_payload.lower()
    assert "active rewrite pass" in system_prompt
    assert "Your job is rewriting, not auditing" in system_prompt
    assert "old fast mode" not in system_prompt
    assert "style_rules" not in payload
    assert "rewriting_playbook" not in payload
    assert "findings" not in payload
    assert "rewrite_strategy" in payload
    assert payload["rewrite_pass"] == "active_rulebook_single_pass"
    assert "must_edit_policy" in payload
    assert any("원문을 그대로 반환하는 것은 rewrite 실패" in item for item in payload["must_edit_policy"])
    assert any("최소 하나 이상의 안전한 표현 개선" in item for item in payload["must_edit_policy"])
    assert "completion_contract" in payload
    assert "structured_output_contract" in payload
    assert "im_not_ai_quick_rules" in payload
    assert "exact_preserve_targets" not in payload
    assert "active_rulebook_single_pass" in payload["rewrite_strategy"]
    assert payload["completion_contract"]["originalCharCount"] == len(request.text)
    assert "complete rewritten passage" in payload["completion_contract"]["scope"]
    assert any("revisedText is the single canonical final answer" in item for item in payload["structured_output_contract"])
    assert any("changes[].original and changes[].revised are local diff snippets only" in item for item in payload["structured_output_contract"])
    assert "Rewrite/Audit Contract" in payload["im_not_ai_quick_rules"]
    assert "detect" not in payload
    assert "문장 흐름" in rendered_payload
    assert "리듬" in rendered_payload
    assert "명확성" in rendered_payload
    assert "룰북을 적극 적용" in payload["rewrite_guidance"]["rewrite_policy"]
    assert "fast mode" not in rendered_payload


def test_review_prompt_applies_audit_corrections_and_reaudits():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="이 기능은 데이터를 통해 성장을 지원합니다.",
            rewrite_mode="strict",
        )
    )
    audit_result = AuditResult(
        status="conditional_pass",
        flaggedEdits=[
            {
                "before": "데이터를 통해",
                "after": "데이터로",
                "issue": "번역투 연결입니다.",
                "checklistFailed": [13],
                "action": "rewrite_required",
                "correctionDirection": "'통해'를 자연스러운 조사로 줄입니다.",
                "severity": "medium",
            }
        ],
        reason="수정 지시가 있습니다.",
    )

    payload = json.loads(
        prompts.review_user_prompt(
            request,
            {},
            "이 기능은 데이터를 통해 성장을 지원합니다.",
            audit_result,
        )
    )

    rendered = json.dumps(payload, ensure_ascii=False)
    assert "strict" not in rendered.lower()
    assert "preservation_audit" in payload
    assert "fixed_review_routine" in payload
    assert "original_detection" not in payload
    assert "residual_detection" not in payload
    assert "새로 고치지 않는다" in rendered
    assert "finalAuditStatus" in rendered


def test_rewrite_prompt_embeds_active_rules_without_detect_stage():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="결론적으로 성과를 통해 결과를 냈습니다.",
            rewrite_mode="strict",
        )
    )
    context = {}

    rewrite_payload = json.loads(prompts.rewrite_user_prompt(request, context))

    assert not hasattr(prompts, "detect_user_prompt")
    assert not hasattr(prompts, "strict_rewrite_user_prompt")
    assert rewrite_payload["rulebook"] == "active-rewrite-rules"
    assert "im_not_ai_quick_rules" in rewrite_payload
    assert "strict_rules" not in rewrite_payload
    assert "A-1" in rewrite_payload["im_not_ai_quick_rules"]
    assert "Do not flag" in rewrite_payload["im_not_ai_quick_rules"]
    assert "A-2 example" in rewrite_payload["im_not_ai_quick_rules"]
    assert "Rewrite/Audit Contract" in rewrite_payload["im_not_ai_quick_rules"]
    assert not hasattr(resources, "ai_tell_taxonomy")
    assert hasattr(resources, "strict_rules")


def test_rewrite_audit_review_prompts_follow_single_routine():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text=(
                "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 "
                "묶어서 폴더를 만들고 자동으로 정리합니다. 두 번째는 소스 검색 기능입니다. 더 이상 "
                "사용자가 직접 피드를 스크롤하지 않아도 필요한 자료를 찾아 작은 묶음으로 반환합니다."
            ),
            rewrite_mode="strict",
            max_rounds=2,
        )
    )
    context = {}

    rewrite_prompt = prompts.rewrite_user_prompt(request, context)
    audit_result = AuditResult(status="full_pass", reason="통과")
    review_prompt = prompts.review_user_prompt(request, context, request.text, audit_result)

    assert "im_not_ai_quick_rules" in json.loads(rewrite_prompt)
    assert "preservation_audit" in json.loads(review_prompt)
    assert "exact_preserve_targets" not in json.loads(rewrite_prompt)
    assert "exact_preserve_targets" in json.loads(review_prompt)
    assert len(review_prompt) < len(rewrite_prompt)


def test_strict_audit_prompt_does_not_embed_scholarship_reference():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="AI 에이전트가 저장된 자료를 정리하고 필요한 자료를 찾아줍니다.",
            rewrite_mode="strict",
            max_rounds=2,
        )
    )
    context = {}

    payload = json.loads(prompts.audit_user_prompt(request, context, request.text, []))
    rendered_payload = json.dumps(payload, ensure_ascii=False)

    assert "strict" not in rendered_payload.lower()
    assert "scholarship_constraints" not in payload
    assert "번역학계" not in rendered_payload
    assert "checklist_13" in payload
    assert "exact_preserve_targets" in payload


def test_removed_reference_resources_have_no_runtime_loaders():
    assert not hasattr(resources, "scholarship")
    assert not hasattr(resources, "rewriting_playbook")
    assert not hasattr(resources, "ai_tell_taxonomy")
    assert hasattr(resources, "strict_rules")


async def test_stub_rewrite_uses_preserve_formatting_switch():
    preserving = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=True)
    )
    normalizing = RewriteRequestForTest.model_validate(
        _payload(text="첫 문장.  둘째 문장.", preserve_formatting=False)
    )
    llm = StubRewriteLLM()

    preserved = await llm.rewrite(preserving)
    normalized = await llm.rewrite(normalizing)

    assert preserved.revisedText == "첫 문장.  둘째 문장."
    assert normalized.revisedText == "첫 문장. 둘째 문장."


async def test_protected_terms_are_restored_by_audit_safety():
    class DropsProtectedTermLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="정책은 유지됩니다.",
                changes=[
                    Change(
                        original="API v1 정책은 유지됩니다.",
                        revised="정책은 유지됩니다.",
                        reason="테스트용 누락입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["표현을 정리했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="API v1 정책은 유지됩니다.", protected_terms=["API v1"])
    )

    response = await RewriteGraphRunner(_settings(), DropsProtectedTermLLM()).run(request)

    assert response.revisedText == request.text
    assert any("감사 지적" in warning for warning in response.warnings)
    assert all(change.revised in response.revisedText for change in response.changes)


async def test_prepare_context_is_empty():
    captured = {}

    class CapturingLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            captured["context"] = context
            return RewriteResult(
                revisedText=request.text,
                changes=[Change(original="", revised="", reason="유지했습니다.", type="clarity", riskLevel="low")],
                summary=["유지했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="분기별 보고 결과와 분석 내용입니다.")
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    assert captured["context"] == {}


def test_local_style_rules_cover_legacy_source_rule_ids():
    expected = {
        "A-1",
        "A-2",
        "A-3",
        "A-4",
        "A-5",
        "A-6",
        "A-7",
        "A-8",
        "A-9",
        "A-10",
        "A-11",
        "A-15",
        "A-16",
        "A-18",
        "A-19",
        "B-1",
        "B-2",
        "C-5",
        "C-7",
        "C-8",
        "C-9",
        "C-10",
        "C-11",
        "D-1",
        "D-2",
        "D-3",
        "D-4",
        "D-5",
        "D-6",
        "D-7",
        "E-1",
        "E-2",
        "E-7",
        "F-4",
        "F-5",
        "G-1",
        "G-2",
        "G-3",
        "H-1",
        "H-3",
        "H-4",
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "J-1",
        "J-2",
        "J-3",
    }

    assert SUPPORTED_STYLE_RULE_IDS == expected


def test_local_detect_covers_reinforced_style_rule_cases():
    cases = [
        ("A-16", "그는 말했다. 그는 다시 말했다. 그것은 반복됐다."),
        ("B-1", "소버린 AI(Sovereign AI)는 중요한 전략이다."),
        ("C-9", "(1) 계획을 세운다. (2) 실행한다."),
        ("E-2", "첫째다. 둘째다. 셋째다. 넷째다."),
        ("E-7", "우리는 실행한다. 그런데 이건 좋아요. 다음 단계입니다."),
        ("F-4", "전략적 실행성과 구조화가 필요하다."),
        ("G-3", "균형 있게 보고 신중하게 판단하며 양쪽 모두와 두 가지 모두의 장점도 있지만 균형이 필요하다."),
        ("J-3", "- 첫 번째 항목\n- 두 번째 항목"),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        assert rule_id in {finding.category for finding in detection.findings}


def test_local_detection_uses_rule_score_and_density():
    detection = local_detect("성과를 통해 결과를 냈다.", focus_categories=["A-2"])

    assert detection.severityWeightedScore == 5.0
    assert detection.aiTellDensity == finding_density("성과를 통해 결과를 냈다.", detection.findings)
    assert finding_score(detection.findings) == 5.0


def test_local_detector_severity_matches_strict_rulebook_for_non_decisive_style_rules():
    cases = [
        ("C-10", "전략: 실행으로 전환"),
        ("H-1", "또한 우리는 실행합니다."),
        ("J-2", '"하나" "둘" "셋" "넷" "다섯" "여섯"'),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        severities = {finding.severity for finding in detection.findings if finding.category == rule_id}
        assert severities == {"S2"}


def test_local_detection_excludes_do_not_spans():
    text = '"데이터를 통해 성장한다"라고 말했다. API는 유지한다.'
    detection = local_detect(text)

    assert "A-2" not in {finding.category for finding in detection.findings}


def test_a16_pronoun_literal_translation_golden_case():
    literal = "메리는 그녀가 그녀를 그리워해서 그녀의 어머니에게 전화했다."
    natural = "메리는 어머니가 그리워서 전화를 걸었다."

    literal_detection = local_detect(literal, focus_categories=["A-16"])
    natural_detection = local_detect(natural, focus_categories=["A-16"])

    assert "A-16" in {finding.category for finding in literal_detection.findings}
    assert "A-16" not in {finding.category for finding in natural_detection.findings}
    assert pronoun_density(literal) > pronoun_density(natural)


def test_a17_deul_overuse_stays_metric_only():
    text = "이러한 데이터들과 정보들과 결과들이 중요한 아이디어들을 보여준다."

    detection = local_detect(text)

    assert "A-17" not in SUPPORTED_STYLE_RULE_IDS
    assert "A-17" not in {finding.category for finding in detection.findings}
    assert deul_overuse_rate(text) > 0


def test_a8_double_passive_golden_case():
    text = "이 문제는 분석되어진다."

    detection = local_detect(text, focus_categories=["A-8"])

    assert "A-8" in {finding.category for finding in detection.findings}
    assert double_passive_count(text) >= 1


async def test_local_review_restores_flagged_sentence_when_preserved_values_are_removed():
    class OverRewriteLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="완전히 다른 결론과 새로운 주장으로 바뀐 문장입니다.",
                changes=[
                    Change(
                        original="원문",
                        revised="완전히 다른 문장",
                        reason="테스트용 과윤문입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["과도하게 바꿨습니다."],
                inputTokens=3,
                outputTokens=4,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="2026년 5월 보고서 문장을 명확하게 정리합니다.", rewrite_mode="fast")
    )
    response = await RewriteGraphRunner(_settings(), OverRewriteLLM()).run(request)

    assert response.revisedText == request.text
    assert response.usage.rounds == 1
    assert not any("원문을 반환" in warning for warning in response.warnings)
    assert any("부분 복원" in warning for warning in response.warnings)


async def test_strict_change_rate_alone_is_review_signal_not_rollback():
    class HighChangeStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="나가 라다 바마 아사 차자 타카 하파 파하 카타 자차 사아 마바 다라 가나.",
                changes=[
                    Change(
                        original=request.text,
                        revised="나가 라다 바마 아사 차자 타카 하파 파하 카타 자차 사아 마바 다라 가나.",
                        reason="테스트용 고변경률 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["변경률은 높지만 보존 누락은 없습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(
                request,
                revised_text,
                final_warnings=audit_result.warnings,
                status=audit_result.status,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="가나 다라 마바 사아 자차 카타 파하 하파 타카 차자 아사 바마 라다 나가.",
            rewrite_mode="strict",
            max_rounds=2,
            protected_terms=[],
        )
    )

    response = await RewriteGraphRunner(_settings(), HighChangeStrictLLM()).run(request)

    assert response.revisedText != request.text
    assert response.usage.rounds == 1
    assert not any("원문을 반환" in warning for warning in response.warnings)


async def test_strict_finalize_rebuilds_review_changes_for_exact_display_matching():
    class UngroundedReviewChangesLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="현대의 업무 환경은 빠르게 변하고 있으며, 조직은 이 변화에 효과적으로 대응해야 합니다.",
                changes=[
                    Change(
                        original="현대의 업무 환경은 빠르게 변화하고 있으며",
                        revised="현대의 업무 환경은 빠르게 변하고 있으며",
                        reason="표현을 간결하게 다듬었습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["초안을 작성했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return StrictReviewResult(
                revisedText=revised_text,
                changes=[
                    Change(
                        original="이러한 변화에 효과적으로 대응하기 위해 더 체계적인 접근이 필요합니다.",
                        revised="이 변화에 효과적으로 대응해야 합니다.",
                        reason="리뷰 단계에서 중간 초안 기준 변경을 보고했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["리뷰를 완료했습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="현대의 업무 환경은 빠르게 변화하고 있으며, 조직은 이러한 변화에 효과적으로 대응해야 합니다.",
            rewrite_mode="strict",
            max_rounds=1,
            protected_terms=[],
        )
    )

    response = await RewriteGraphRunner(_settings(), UngroundedReviewChangesLLM()).run(request)

    assert response.revisedText != request.text
    assert response.changes
    for change in response.changes:
        assert change.original == "" or change.original in request.text
        assert change.revised == "" or change.revised in response.revisedText


async def test_single_strict_graph_returns_after_clean_audit():
    calls = []

    class StrictFakeLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(
                revisedText="2026년 보고서입니다.",
                changes=[
                    Change(
                        original="2026년 보고서입니다.",
                        revised="2026년 보고서입니다.",
                        reason="의미를 유지했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["정밀 파이프라인을 통과했습니다."],
                inputTokens=3,
                outputTokens=4,
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.", inputTokens=5, outputTokens=6)

        async def review(self, request, context, revised_text, audit_result):
            calls.append("review")
            return _strict_review_result(request, revised_text).model_copy(
                update={"inputTokens": 7, "outputTokens": 8}
            )

    runner = RewriteGraphRunner(_settings(), StrictFakeLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict")
        )
    )

    assert calls == ["rewrite", "audit"]
    assert response.usage.rounds == 1
    assert response.usage.inputTokens == 8
    assert response.usage.outputTokens == 10


async def test_strict_rewrite_runs_once_even_when_initial_draft_is_no_op():
    calls = []
    text = (
        "다음은 플랜 모드입니다. 플랜 모드는 바로 구현에 들어가기 전에 먼저 계획을 세우는 기능입니다. "
        "플러스 버튼을 눌러 켤 수도 있고, 슬래시 플래닝 명령어로 실행할 수도 있습니다. "
        "계획이 마음에 들지 않으면 수정하고 싶은 부분을 입력해 다시 다듬을 수 있습니다. "
        "다음으로 MCP 커맨드가 있습니다. MCP 명령어를 입력하면 현재 활성화된 MCP들을 확인할 수 있습니다."
    )

    class NoOpRewriteLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append(("rewrite", {}))
            return RewriteResult(
                revisedText=request.text,
                changes=[],
                summary=["원문을 유지했습니다."],
                inputTokens=3,
                outputTokens=4,
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append(("audit", {"revised_text": revised_text}))
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.", inputTokens=7, outputTokens=8)

    response = await RewriteGraphRunner(_settings(), NoOpRewriteLLM()).run(
        RewriteRequestForTest.model_validate(_payload(text=text, protected_terms=[]))
    )

    assert [call[0] for call in calls] == ["rewrite", "audit"]
    assert calls[1][1]["revised_text"] == text
    assert response.revisedText == text
    assert response.usage.inputTokens == 10
    assert response.usage.outputTokens == 12


async def test_strict_conditional_audit_is_handled_by_review_without_rewrite_loop():
    calls = []
    audit_calls = 0

    class ConditionalAuditLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(
                revisedText="2026년 보고서입니다.",
                changes=[
                    Change(
                        original="2026년 보고서입니다.",
                        revised="2026년 보고서입니다.",
                        reason="의미를 유지했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["정밀 파이프라인을 통과했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            nonlocal audit_calls
            audit_calls += 1
            calls.append("audit")
            return AuditResult(
                status="conditional_pass",
                flaggedEdits=[
                    {
                        "before": "보고서",
                        "after": "자료",
                        "issue": "핵심 표현이 바뀌었습니다.",
                        "checklistFailed": [13],
                        "action": "restore_original",
                        "correctionDirection": "보고서를 원문 표현으로 복원합니다.",
                        "severity": "high",
                    }
                ],
                reason="조건부 감사 결과입니다.",
            )

        async def review(self, request, context, revised_text, audit_result):
            calls.append("review")
            return _strict_review_result(request, revised_text)

    runner = RewriteGraphRunner(_settings(), ConditionalAuditLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict", max_rounds=2)
        )
    )

    assert calls == ["rewrite", "audit", "review"]
    assert response.usage.rounds == 1
    assert response.revisedText == "2026년 보고서입니다."


async def test_strict_conditional_audit_routes_to_review_even_without_flagged_edits():
    calls = []

    class ConditionalStatusOnlyLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(
                revisedText="2026년 보고서입니다.",
                changes=[],
                summary=["초안을 작성했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(
                status="conditional_pass",
                warnings=["감사 모델이 복원 필요 가능성을 표시했습니다."],
                reason="조건부 감사 결과입니다.",
            )

        async def review(self, request, context, revised_text, audit_result):
            calls.append("review")
            return _strict_review_result(
                request,
                revised_text,
                final_warnings=audit_result.warnings,
                status="full_pass",
            )

    response = await RewriteGraphRunner(_settings(), ConditionalStatusOnlyLLM()).run(
        RewriteRequestForTest.model_validate(_payload(text="2026년 보고서입니다."))
    )

    assert calls == ["rewrite", "audit", "review"]
    assert response.revisedText == "2026년 보고서입니다."
    assert any("복원 필요 가능성" in warning for warning in response.warnings)


async def test_strict_returns_truncated_review_candidate_without_terminal_rollback():
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 묶어 "
        "폴더를 만들고 자동으로 정리하는 기능입니다.\n\n"
        "두 번째는 소스 검색 기능입니다. 사용자가 직접 피드를 스크롤하지 않아도 에이전트가 필요한 "
        "자료를 찾아 작은 묶음으로 반환합니다.\n\n"
        "이 기능이 제대로 작동하면 앱은 단순한 캡처 도구를 넘어 작업에 바로 쓰이는 컨텍스트 "
        "시스템이 됩니다."
    )

    class TruncatedStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트가",
                changes=[
                    Change(
                        original="source",
                        revised="truncated",
                        reason="테스트용 잘림 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["중간에서 잘린 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(
                request,
                revised_text,
                final_warnings=audit_result.warnings,
                status=audit_result.status,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=2, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), TruncatedStrictLLM()).run(request)

    assert response.revisedText == "첫 번째는 소스 정리 기능입니다. 에이전트가"
    assert response.usage.rounds == 1
    assert not any("원문을 반환" in warning for warning in response.warnings)
    assert any("출력 잘림" in warning for warning in response.warnings)


async def test_strict_final_warnings_do_not_force_original_when_quote_remains_missing():
    text = (
        '첫 번째는 소스 정리 기능입니다. "소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 '
        '새로운 에이전트 스킬을 만드는 데 사용해줘"라는 요청을 보존해야 합니다. '
        "이 기능은 저장된 자료를 바탕으로 작업 컨텍스트를 만듭니다."
    )

    class MissingQuoteStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다.",
                changes=[
                    Change(
                        original="source",
                        revised="truncated",
                        reason="테스트용 보존 누락 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["보존 문구가 빠진 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            if "소셜미디어 성장" not in revised_text:
                return AuditResult(
                    status="fail",
                    warnings=["직접 인용 또는 핵심 구절이 누락됐습니다."],
                    flaggedEdits=[
                        {
                            "issue": "직접 인용 또는 핵심 구절 누락",
                            "checklistFailed": [4, 13],
                            "action": "restore_original",
                            "correctionDirection": "누락된 직접 인용을 원문 그대로 복원합니다.",
                            "severity": "high",
                        }
                    ],
                    reason="누락 감지",
                )
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(
                request,
                revised_text,
                final_warnings=audit_result.warnings,
                status=audit_result.status,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=1, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), MissingQuoteStrictLLM()).run(request)

    assert response.revisedText == "첫 번째는 소스 정리 기능입니다."
    assert not any("보존되어야 하는 표현이 결과" in warning for warning in response.warnings)
    assert not any("원문을 반환" in warning for warning in response.warnings)
    assert any("직접 인용" in warning for warning in response.warnings)


async def test_clean_audit_skips_review_and_returns_current_draft():
    class HoldReviewLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="문장을 더 세련되고 자연스럽게 정리합니다.",
                changes=[
                    Change(
                        original=request.text,
                        revised="문장을 더 세련되고 자연스럽게 정리합니다.",
                        reason="테스트용 hold 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["hold 대상 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="감사는 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(
                request,
                revised_text,
                warnings=["사람 검토가 필요합니다."],
                blocking=["최종 리뷰에서 의미 보존 위험이 남았습니다."],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="문장을 조금 더 자연스럽게 정리합니다.",
            rewrite_mode="strict",
            max_rounds=1,
            protected_terms=[],
        )
    )
    response = await RewriteGraphRunner(_settings(), HoldReviewLLM()).run(request)

    assert response.revisedText == "문장을 더 세련되고 자연스럽게 정리합니다."
    assert not any("사람 검토" in warning for warning in response.warnings)


async def test_strict_rewrite_does_not_repair_incomplete_revised_text_from_change_candidate():
    audited_texts = []
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 제가 저장해둔 모든 소스를 살펴보고, "
        "관련 있는 자료끼리 묶어서 폴더를 만들고 자동으로 정리하도록 하는 기능입니다. "
        "두 번째는 소스 검색 기능입니다. 더 이상 제가 직접 피드를 스크롤하면서 자료를 찾는 대신, "
        "에이전트에게 “소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 새로운 에이전트 스킬을 만드는 데 사용해줘”라고 "
        "요청할 수 있게 만드는 것입니다. 그러면 에이전트는 현재 작업에 가장 관련성이 높은 자료들을 골라내고, "
        "유용도에 따라 순위를 매긴 뒤 작은 묶음으로 반환해줍니다."
    )
    complete_revised = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 분석해 관련 자료를 묶고 "
        "폴더를 자동으로 구성하는 기능입니다. 두 번째는 소스 검색 기능입니다. 더 이상 피드를 직접 "
        "스크롤하며 자료를 찾는 대신, 에이전트에게 “소셜미디어 성장에 가장 도움이 되는 자료를 찾아서 "
        "새로운 에이전트 스킬을 만드는 데 사용해줘”라고 요청하면 됩니다. 그러면 에이전트는 해당 작업에 "
        "맞는 자료를 추려 유용도 순으로 정렬한 뒤 작은 묶음으로 돌려줍니다."
    )

    class InconsistentStrictLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트에게 ",
                changes=[
                    Change(
                        original=request.text,
                        revised=complete_revised,
                        reason="전체 윤문입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["revisedText 필드만 불완전한 structured output입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            audited_texts.append(revised_text)
            return AuditResult(status="full_pass", reason="보존 검사를 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(
                request,
                revised_text,
                final_warnings=audit_result.warnings,
                status=audit_result.status,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=1, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), InconsistentStrictLLM()).run(request)

    assert audited_texts == ["첫 번째는 소스 정리 기능입니다. 에이전트에게 "]
    assert response.revisedText == "첫 번째는 소스 정리 기능입니다. 에이전트에게 "
    assert not any("revisedText가 불완전" in item for item in response.summary)
    assert not any("원문을 반환" in warning for warning in response.warnings)
    assert any("출력 잘림" in warning for warning in response.warnings)


async def test_strict_review_can_return_safe_final_candidate_without_rewrite_loop():
    rewrite_calls = 0
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 묶어 "
        "폴더를 만들고 자동으로 정리하는 기능입니다.\n\n"
        "두 번째는 소스 검색 기능입니다. 사용자가 직접 피드를 스크롤하지 않아도 에이전트가 필요한 "
        "자료를 찾아 작은 묶음으로 반환합니다.\n\n"
        "이 기능이 제대로 작동하면 앱은 단순한 캡처 도구를 넘어 작업에 바로 쓰이는 컨텍스트 "
        "시스템이 됩니다."
    )
    safe_text = text.replace("자료를 찾아 작은", "자료를 찾아서 작은")

    class ReviewRepairsLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            nonlocal rewrite_calls
            rewrite_calls += 1
            return RewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트가",
                changes=[
                    Change(
                        original="자료",
                        revised="자료",
                        reason="테스트용 strict 초안입니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=[f"{rewrite_calls}라운드 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review(self, request, context, revised_text, audit_result):
            return _strict_review_result(request, safe_text)

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=2, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), ReviewRepairsLLM()).run(request)

    assert response.revisedText == safe_text
    assert response.usage.rounds == 1
    assert rewrite_calls == 1


def test_text_above_core_max_chars_returns_422():
    client = _client()
    raw_body = json.dumps(_payload(text="가" * 5_001), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 422


def test_logs_do_not_include_source_or_rewrite_result(caplog):
    client = _client()
    source = "PRIVACY_SENTINEL_2026 원문입니다."
    raw_body = json.dumps(_payload(text=source, rewrite_mode="fast"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with caplog.at_level(logging.INFO, logger="humanize_core"):
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)

    assert response.status_code == 200
    revised = response.json()["revisedText"]
    assert source not in caplog.text
    assert revised not in caplog.text


def test_strict_job_error_logs_do_not_include_source_text(tmp_path, caplog):
    source = "ASYNC_ERROR_PRIVACY_SENTINEL_2026 원문입니다."

    class FailingGraphRunner:
        async def run(self, request):
            raise RuntimeError(f"provider failed while handling {request.text}")

    app = create_app(
        _settings(job_store_path=str(tmp_path / "jobs.sqlite3")),
        graph_runner=FailingGraphRunner(),
    )
    raw_body = json.dumps(_payload(text=source, rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202

        with caplog.at_level(logging.ERROR, logger="humanize_core"):
            asyncio.run(app.state.job_manager.process_next())

    assert source not in caplog.text
    assert "provider failed" not in caplog.text


def test_strict_job_store_does_not_persist_plaintext_payload_or_result(tmp_path):
    source = "ASYNC_PRIVACY_SENTINEL_2026 원문입니다."
    store_path = tmp_path / "jobs.sqlite3"
    app = create_app(_settings(job_store_path=str(store_path)))
    raw_body = json.dumps(_payload(text=source, rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["jobId"]

        db_bytes = _job_store_bytes(store_path)
        assert source.encode("utf-8") not in db_bytes

        asyncio.run(app.state.job_manager.process_next())
        status_headers = _signed_headers(b"", request_id="req_privacy_status")
        status_response = client.get(f"/v1/rewrite-jobs/{job_id}", headers=status_headers)

    assert status_response.status_code == 200
    data = status_response.json()
    assert data["status"] == "succeeded"
    assert data["result"]["revisedText"] == source
    db_bytes = _job_store_bytes(store_path)
    assert source.encode("utf-8") not in db_bytes


def _job_store_bytes(store_path: Path) -> bytes:
    paths = [store_path, Path(f"{store_path}-wal"), Path(f"{store_path}-shm")]
    chunks = [path.read_bytes() for path in paths if path.exists()]
    return b"".join(chunks)


def test_openai_text_format_uses_strict_json_schema():
    text_format = _openai_rewrite_text_format()

    assert text_format["type"] == "json_schema"
    assert text_format["strict"] is True
    schema = text_format["schema"]
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["revisedText", "changes", "summary"]
    change_schema = schema["properties"]["changes"]["items"]
    assert change_schema["additionalProperties"] is False
    assert change_schema["required"] == ["original", "revised", "reason", "type", "riskLevel"]


async def test_openai_rewrite_uses_responses_structured_output(monkeypatch):
    calls = []

    class FakeResponses:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text=json.dumps(
                    {
                        "revisedText": "개선된 문장입니다.",
                        "changes": [
                            {
                                "original": "원문",
                                "revised": "개선된 문장",
                                "reason": "명확성을 높였습니다.",
                                "type": "clarity",
                                "riskLevel": "low",
                            }
                        ],
                        "summary": ["표현을 정리했습니다."],
                    },
                    ensure_ascii=False,
                ),
                usage=SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenAIRewriteLLM(api_key="test-key", model_name="gpt-5-mini")
    result = await llm.rewrite(
        RewriteRequestForTest.model_validate(_payload()),
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 11
    assert result.outputTokens == 7
    assert calls[0]["model"] == "gpt-5-mini"
    assert calls[0]["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert "response_format" not in calls[0]


async def test_openrouter_rewrite_once_uses_json_schema_and_usage(monkeypatch):
    calls = []
    init_calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "revisedText": "개선된 문장입니다.",
                                    "changes": [
                                        {
                                            "original": "원문",
                                            "revised": "개선된 문장",
                                            "reason": "AI 티를 줄였습니다.",
                                            "type": "clarity",
                                            "riskLevel": "low",
                                        }
                                    ],
                                    "summary": ["표현을 정리했습니다."],
                                    "warnings": [],
                                    "selfCheck": [],
                                    "residualFindings": [],
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=13, completion_tokens=8),
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenRouterRewriteLLM(
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        app_title="Test App",
        site_url="https://example.test",
        model_name="stub",
        rewrite_model_name="openai/gpt-5-mini",
        rewrite_fallback_model_name="~anthropic/claude-haiku-latest",
        strict_audit_model_name="openai/gpt-5",
        strict_review_model_name="~anthropic/claude-haiku-latest",
    )

    result = await llm.rewrite_once(
        RewriteRequestForTest.model_validate(_payload()),
        context={},
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 13
    assert result.outputTokens == 8
    assert init_calls[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert init_calls[0]["default_headers"]["X-Title"] == "Test App"
    assert calls[0]["model"] == "openai/gpt-5-mini"
    assert calls[0]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert "temperature" not in calls[0]


async def test_openrouter_rewrite_once_omits_temperature_for_parameter_routing(monkeypatch):
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "revisedText": "개선된 문장입니다.",
                                    "changes": [],
                                    "summary": ["수정했습니다."],
                                },
                                ensure_ascii=False,
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3),
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )

    llm = OpenRouterRewriteLLM(
        api_key="or-key",
        base_url="https://openrouter.ai/api/v1",
        app_title="Test App",
        site_url=None,
        model_name="openai/gpt-5-mini",
        rewrite_model_name="openai/gpt-5-mini",
        rewrite_fallback_model_name="openai/gpt-5-mini",
        strict_audit_model_name="openai/gpt-5-mini",
        strict_review_model_name="openai/gpt-5-mini",
    )

    result = await llm.rewrite_once(
        RewriteRequestForTest.model_validate(_payload(rewrite_mode="strict")),
        context={},
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 5
    assert result.outputTokens == 3
    assert calls[0]["model"] == "openai/gpt-5-mini"
    assert calls[0]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["extra_body"]["provider"]["require_parameters"] is True
    assert "temperature" not in calls[0]


def test_openrouter_schema_marks_pydantic_default_fields_required():
    response_format = _openrouter_response_format(
        "rewrite_result",
        RewriteResult.model_json_schema(),
    )

    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"].keys())
    assert "default" not in json.dumps(schema)
    assert "title" not in json.dumps(schema)

    finding_schema = schema["$defs"]["Finding"]
    assert finding_schema["required"] == list(finding_schema["properties"].keys())
    assert "textSpan" in finding_schema["required"]
    assert "start" in finding_schema["required"]
    assert "end" in finding_schema["required"]

    assert schema["properties"]["residualFindings"]["items"] == {"$ref": "#/$defs/Finding"}

    review_response_format = _openrouter_response_format(
        "preservation_review_result",
        StrictReviewResult.model_json_schema(),
    )
    rendered_review_format = json.dumps(review_response_format)
    assert review_response_format["json_schema"]["name"] == "preservation_review_result"
    assert "strict_review_result" not in rendered_review_format


def test_metrics_v2_computes_im_not_ai_signal_keys():
    metrics = compute_all_v2("결론적으로, API를 통해 성과를 확인할 수 있다.")

    assert metrics["version"] == "v2.0"
    assert "v2_metrics" in metrics
    assert "normalisation_score" in metrics["v2_metrics"]
    assert "v2_interference_index" in metrics
