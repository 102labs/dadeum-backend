import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from humanize_core.api import create_app
from humanize_core.config import Settings
from humanize_core.debug_log import RewriteDebugLogger
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
    ChangeExplanationResult,
    RepairedSegment,
    RewriteOutput,
    RewriteResult,
    SegmentReviewResult,
    StrictReviewResult,
)
from humanize_core.llm import (
    AnthropicRewriteLLM,
    MAX_OUTPUT_TOKENS,
    LLMResponseError,
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
        "debug_log_enabled": False,
        "debug_log_include_plaintext": False,
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


def test_settings_uses_canonical_rewrite_fallback_env(monkeypatch):
    monkeypatch.setenv("HUMANIZE_REWRITE_FALLBACK_MODEL_NAME", "openai/fallback")

    settings = Settings(_env_file=None)

    assert settings.rewrite_fallback_model_name == "openai/fallback"


def _rulebook_context() -> dict[str, object]:
    return {
        "detectedCount": 2,
        "severityWeightedScore": 8.0,
        "categorySummary": {"A-2": 1, "A-7": 1},
        "rulebookHints": [
            {
                "id": "local-A-2-1",
                "category": "A-2",
                "categoryLabel": "번역투: ~를 통해",
                "severity": "S1",
                "scope": "span",
                "suggestedFix": "~로, ~해서, ~함으로써 등으로 분산합니다.",
            },
            {
                "id": "local-A-7-1",
                "category": "A-7",
                "categoryLabel": "직역: 가지고 있다",
                "severity": "S1",
                "scope": "span",
                "suggestedFix": "동사나 형용사로 환원합니다.",
            },
        ],
    }


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


def _today_log_path(log_dir: Path) -> Path:
    return log_dir / f"{datetime.now().astimezone():%Y-%m-%d}.log"


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


async def test_long_compat_request_is_chunked_at_sentence_boundaries():
    text = "보고 문장입니다. " * 500
    request = RewriteRequestForTest.model_validate(_payload(text=text, rewrite_mode="fast", max_rounds=3))

    class CapturingRewriteLLM:
        def __init__(self) -> None:
            self.chunk_lengths: list[int] = []

        async def rewrite_once(self, request, context):
            self.chunk_lengths.append(len(request.text))
            return RewriteResult(
                revisedText=request.text,
                changes=[
                    Change(
                        original="보고 문장입니다.",
                        revised="보고 문장입니다.",
                        reason="청크 단위로 처리합니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["청크 단위 rewrite를 사용했습니다."],
            )

    llm = CapturingRewriteLLM()
    settings = _settings()
    response = await RewriteGraphRunner(settings, llm).run(request)

    assert 4_000 < len(text) <= 5_000
    # 4,500자, 문단 구분 없는 글도 문장 경계 기준 약 1,000자 청크로 나뉜다.
    assert len(llm.chunk_lengths) == 5
    assert all(length <= settings.chunk_target_chars for length in llm.chunk_lengths)
    assert response.revisedText == text
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


def test_strict_debug_log_writes_stage_events_without_plaintext(tmp_path):
    log_dir = tmp_path / "logs"
    app = create_app(
        _settings(
            job_store_path=str(tmp_path / "jobs.sqlite3"),
            debug_log_enabled=True,
            debug_log_dir=str(log_dir),
        )
    )
    source_text = "민감한 원문 ABC123은 로그에 남으면 안 됩니다."
    raw_body = json.dumps(
        _payload(text=source_text, rewrite_mode="strict", protected_terms=["ABC123"]),
        ensure_ascii=False,
    ).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202
        job_id = response.json()["jobId"]
        asyncio.run(app.state.job_manager.process_next())

        status_headers = _signed_headers(b"", request_id="req_debug_status")
        status_response = client.get(f"/v1/rewrite-jobs/{job_id}", headers=status_headers)

    assert status_response.status_code == 200
    log_path = _today_log_path(log_dir)
    log_content = log_path.read_text(encoding="utf-8")
    lines = log_content.splitlines()

    assert any("INFO | api.py | event=api.rewrite.accepted" in line for line in lines)
    assert any("INFO | jobs.py | event=job.enqueued" in line for line in lines)
    assert any("INFO | jobs.py | event=job.claimed" in line for line in lines)
    assert any("INFO | jobs.py | event=job.succeeded" in line for line in lines)
    assert any("event=graph.stage.succeeded" in line and "step=prepare" in line for line in lines)
    assert any("event=graph.stage.succeeded" in line and "step=rewrite" in line for line in lines)
    assert any("event=graph.stage.succeeded" in line and "step=audit" in line for line in lines)
    assert any("event=graph.stage.succeeded" in line and "step=finalize" in line for line in lines)
    assert all("durationMs=" in line for line in lines if "event=graph.stage.succeeded" in line)
    assert "job.status.read" not in log_content
    assert "민감한 원문" not in log_content
    assert "ABC123" not in log_content
    assert status_response.json()["result"]["revisedText"] not in log_content


def test_strict_debug_log_records_error_code_without_plaintext(tmp_path):
    class InvalidResponseLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            raise LLMResponseError("invalid structured response")

    settings = _settings(
        job_store_path=str(tmp_path / "jobs.sqlite3"),
        job_max_attempts=1,
        debug_log_enabled=True,
        debug_log_dir=str(tmp_path / "logs"),
    )
    app = create_app(settings, graph_runner=RewriteGraphRunner(settings, InvalidResponseLLM()))
    source_text = "실패 로그에도 이 원문은 남으면 안 됩니다."
    raw_body = json.dumps(_payload(text=source_text, rewrite_mode="strict"), ensure_ascii=False).encode("utf-8")
    headers = _signed_headers(raw_body)

    with TestClient(app) as client:
        response = client.post("/v1/rewrite", content=raw_body, headers=headers)
        assert response.status_code == 202
        asyncio.run(app.state.job_manager.process_next())

    log_content = _today_log_path(tmp_path / "logs").read_text(encoding="utf-8")
    failed_event = next(line for line in log_content.splitlines() if "event=job.failed" in line)
    failed_stage = next(line for line in log_content.splitlines() if "event=graph.stage.failed" in line)

    assert "ERROR | jobs.py | event=job.failed" in failed_event
    assert "errorCode=invalid_model_response" in failed_event
    assert "will_retry=false" in failed_event
    assert "ERROR | graph.py | event=graph.stage.failed" in failed_stage
    assert "step=rewrite" in failed_stage
    assert "errorCode=invalid_model_response" in failed_stage
    assert "실패 로그에도" not in log_content


def test_debug_log_sanitizes_accidental_plaintext_detail_keys(tmp_path):
    logger = RewriteDebugLogger(
        _settings(
            debug_log_enabled=True,
            debug_log_dir=str(tmp_path / "logs"),
        )
    )

    logger.event(
        "debug.sanitizer.test",
        details={
            "text": "민감한 본문",
            "result": {"revisedText": "민감한 결과"},
            "text_length": 6,
            "changes_count": 1,
        },
    )

    log_content = _today_log_path(tmp_path / "logs").read_text(encoding="utf-8")

    assert "민감한 본문" not in log_content
    assert "민감한 결과" not in log_content
    assert "text=[REDACTED length=6]" in log_content
    assert "result=[REDACTED fieldCount=1]" in log_content
    assert "text_length=6" in log_content
    assert "changes_count=1" in log_content


def test_debug_log_can_include_plaintext_when_explicitly_enabled(tmp_path):
    logger = RewriteDebugLogger(
        _settings(
            debug_log_enabled=True,
            debug_log_include_plaintext=True,
            debug_log_dir=str(tmp_path / "logs"),
        )
    )

    logger.event(
        "debug.plaintext.test",
        details={
            "text": "디버깅 원문",
            "revised_text": "디버깅 결과",
            "changes": [{"original": "원문", "revised": "결과"}],
        },
    )

    log_content = _today_log_path(tmp_path / "logs").read_text(encoding="utf-8")

    assert "디버깅 원문" in log_content
    assert "디버깅 결과" in log_content
    assert "original=원문" in log_content
    assert "revised=결과" in log_content


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
        "source_register": "해요체(~해요/~예요)",
    }
    assert "우선 반영" in payload["rewrite_guidance"]["user_intent"]
    assert "자연스럽고 부드러운 업무 문체" in payload["rewrite_guidance"]["tone"]
    assert "지나친 구어체" in payload["rewrite_guidance"]["tone"]
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
    assert "must_edit_policy" not in payload
    assert "edit_policy" in payload
    assert any("글자 그대로 둔다" in item for item in payload["edit_policy"])
    assert any("동의어 교체 금지" in item for item in payload["edit_policy"])
    assert any("양태 유지" in item for item in payload["edit_policy"])
    assert any("높임 등급 유지" in item for item in payload["edit_policy"])
    formal_payload = json.loads(
        prompts.rewrite_user_prompt(
            RewriteRequestForTest.model_validate(_payload(text="일정 좀 정리해서 공유할게요.", tone="formal")),
            {},
        )
    )
    assert any("tone=formal이므로" in item and "합쇼체" in item for item in formal_payload["edit_policy"])
    assert not any("높임 등급 유지" in item for item in formal_payload["edit_policy"])
    assert "양태는 그대로 둔다" in formal_payload["rewrite_guidance"]["tone"]
    assert not any("최소 하나는 더 자연스럽고 간결하게 바뀌어야" in item for item in payload["edit_policy"])
    assert any(item.startswith("하지 말 것:") for item in payload["edit_examples"])
    assert "스킬들을" not in rendered_payload
    assert "좋은 한국어 업무 문장의 기준" in system_prompt
    assert "Never use preservation as a reason" not in system_prompt
    assert "over-editing" in system_prompt
    assert "completion_contract" in payload
    assert "structured_output_contract" in payload
    assert "im_not_ai_quick_rules" not in payload
    assert "exact_preserve_targets" not in payload
    assert "active_rulebook_single_pass" in payload["rewrite_strategy"]
    assert payload["completion_contract"]["originalCharCount"] == len(request.text)
    assert "complete rewritten passage" in payload["completion_contract"]["scope"]
    assert any("revisedText is the single canonical final answer" in item for item in payload["structured_output_contract"])
    assert any("changes[].original and changes[].revised are local diff snippets only" in item for item in payload["structured_output_contract"])
    assert "im_not_ai_quick_rules" in system_prompt
    assert "최종 자체 검토 체크리스트" in system_prompt
    assert "detect" not in payload
    assert "문장 흐름" in rendered_payload
    assert "리듬" in rendered_payload
    assert "명확성" in rendered_payload
    assert "룰북을 적극 적용" in payload["rewrite_guidance"]["rewrite_policy"]
    assert "fast mode" not in rendered_payload
    assert "20~40%" not in rendered_payload
    assert "must_report" not in payload
    assert any("룰 번호" in item for item in payload["changes_contract"])
    assert any("40~80자" in item for item in payload["changes_contract"])
    assert "summary_contract" in payload
    assert "charCountAfter" not in rendered_payload


def test_review_prompt_sends_only_flagged_segments():
    from humanize_core.im_not_ai.schemas import FlaggedEdit, ReviewSegment

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="이 기능은 데이터를 통해 성장을 지원합니다. 출시는 2026년 5월입니다.",
            rewrite_mode="strict",
        )
    )
    edit = FlaggedEdit(
        before="2026년 5월",
        after="내년 봄",
        issue="날짜 표기가 바뀌었습니다.",
        checklistFailed=[3],
        action="preserve_exact",
        correctionDirection="날짜를 원문 그대로 복원합니다.",
        severity="high",
    )
    audit_result = AuditResult(status="conditional_pass", flaggedEdits=[edit], reason="수정 지시가 있습니다.")
    segment = ReviewSegment(
        index=1,
        draft_sentence="출시는 내년 봄입니다.",
        original_sentence="출시는 2026년 5월입니다.",
        corrections=[edit],
    )

    payload = json.loads(prompts.review_user_prompt(request, [segment], audit_result))
    rendered = json.dumps(payload, ensure_ascii=False)

    assert "strict" not in rendered.lower()
    assert payload["segments"][0]["index"] == 1
    assert payload["segments"][0]["draft_sentence"] == "출시는 내년 봄입니다."
    assert payload["segments"][0]["corrections"][0]["before"] == "2026년 5월"
    assert "exact_preserve_targets" in payload
    assert "repair_routine" in payload
    assert "repairedSegments" in rendered
    # The review never sees the full draft: only flagged sentences travel.
    assert "데이터를 통해 성장을 지원합니다" not in rendered
    assert "rulebook_hints" not in rendered
    assert "rewrite_priorities" not in rendered
    assert "상투구를 되살리거나" in rendered


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
    assert "im_not_ai_quick_rules" not in rewrite_payload
    assert "strict_rules" not in rewrite_payload

    system_prompt = prompts.rewrite_system_prompt()
    assert "A-1" in system_prompt
    assert "의미 불변" in system_prompt
    assert "최종 자체 검토 체크리스트" in system_prompt
    # The system prompt now carries the compact rulebook: rule headings stay as
    # an index, while per-rule bodies travel as rule cards attached to hints.
    assert "## A-2." in system_prompt
    assert "데이터를 분석해 인사이트를 얻는다" not in system_prompt
    assert "데이터를 분석해 인사이트를 얻는다" in resources.rule_card("A-2")
    assert len(system_prompt) < len(resources.strict_rules())
    assert not hasattr(resources, "ai_tell_taxonomy")
    assert hasattr(resources, "strict_rules")


def test_rewrite_prompt_includes_compact_rulebook_priorities_only_in_rewrite():
    request = RewriteRequestForTest.model_validate(
        _payload(
            text="결과를 통해 성과를 만들고 목적을 가지고 있습니다.",
            rewrite_mode="strict",
        )
    )
    context = _rulebook_context()

    rewrite_payload = json.loads(prompts.rewrite_user_prompt(request, context))
    audit_payload = json.loads(prompts.audit_user_prompt(request, context, request.text, []))
    review_payload = json.loads(
        prompts.review_user_prompt(
            request,
            [],
            AuditResult(status="full_pass", reason="통과"),
        )
    )

    assert "rewrite_priorities" in rewrite_payload
    priority_payload = json.dumps(rewrite_payload["rewrite_priorities"], ensure_ascii=False)
    assert "rulebook_hints" in rewrite_payload["rewrite_priorities"]
    assert rewrite_payload["rewrite_priorities"]["rulebook_hints"][0]["category"] == "A-2"
    assert "번역투: ~를 통해" in priority_payload

    for forbidden in (
        "findings",
        "style_rules",
        "original_detection",
        "residual_detection",
        "textSpan",
        "start",
        "end",
        "결과를 통해",
        "목적을 가지고",
    ):
        assert forbidden not in priority_payload

    rendered_audit = json.dumps(audit_payload, ensure_ascii=False)
    rendered_review = json.dumps(review_payload, ensure_ascii=False)
    assert "rewrite_priorities" not in rendered_audit
    assert "rulebook_hints" not in rendered_audit
    assert "rewrite_priorities" not in rendered_review
    assert "rulebook_hints" not in rendered_review


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
    context = _rulebook_context()

    rewrite_prompt = prompts.rewrite_user_prompt(request, context)
    audit_prompt = prompts.audit_user_prompt(request, context, request.text, [])
    audit_result = AuditResult(status="full_pass", reason="통과")
    review_prompt = prompts.review_user_prompt(request, [], audit_result)

    assert "im_not_ai_quick_rules" not in json.loads(rewrite_prompt)
    assert "rewrite_priorities" in json.loads(rewrite_prompt)
    assert "segments" in json.loads(review_prompt)
    assert "exact_preserve_targets" not in json.loads(rewrite_prompt)
    assert "exact_preserve_targets" in json.loads(audit_prompt)
    assert "exact_preserve_targets" in json.loads(review_prompt)
    assert "rewrite_priorities" not in audit_prompt
    assert "rewrite_priorities" not in review_prompt
    assert "최종 자체 검토 체크리스트" in prompts.rewrite_system_prompt()
    assert "최종 자체 검토 체크리스트" not in audit_prompt
    assert "최종 자체 검토 체크리스트" not in review_prompt


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


def test_tone_guidance_is_stronger_and_distinct_for_requested_tones():
    formal = RewriteRequestForTest.model_validate(_payload(tone="formal"))
    friendly = RewriteRequestForTest.model_validate(_payload(tone="friendly"))
    keep = RewriteRequestForTest.model_validate(_payload(tone="keep"))

    formal_tone = prompts.rewrite_guidance(formal)["tone"]
    friendly_tone = prompts.rewrite_guidance(friendly)["tone"]
    keep_tone = prompts.rewrite_guidance(keep)["tone"]

    assert formal_tone != friendly_tone
    assert formal_tone != keep_tone
    assert friendly_tone != keep_tone
    assert "격식 있는 비즈니스 문체" in formal_tone
    assert "하십시오/합니다" in formal_tone
    assert "구어적 축약" in formal_tone
    assert "자연스럽고 부드러운 업무 문체" in friendly_tone
    assert "직역 표현" in friendly_tone
    assert "지나친 구어체" in friendly_tone
    assert "기존 톤과 격식" in keep_tone


def test_rulebook_and_rewrite_prompt_do_not_target_fixed_change_rate():
    request = RewriteRequestForTest.model_validate(
        _payload(text="결론적으로 성과를 통해 결과를 냈습니다.", rewrite_mode="strict")
    )

    rendered_prompt = prompts.rewrite_user_prompt(request, _rulebook_context())
    strict_rules_text = resources.strict_rules()

    assert "20~40%" not in rendered_prompt
    assert "20~40%" not in strict_rules_text
    assert "changeRate" not in rendered_prompt
    assert "변경률은 품질 목표가 아니라 보존 위험 신호" in rendered_prompt
    assert "장르·문체 유지" in strict_rules_text


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


async def test_prepare_context_contains_compact_rulebook_hints():
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
        _payload(text="성과를 통해 결과를 냈고 목적을 가지고 있습니다.")
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    context = captured["context"]
    assert context["detectedCount"] >= 2
    assert context["severityWeightedScore"] > 0
    assert context["categorySummary"]["A"] >= 2
    assert len(context["rulebookHints"]) <= 24
    assert {hint["category"] for hint in context["rulebookHints"]} >= {"A-2", "A-7"}
    for hint in context["rulebookHints"]:
        assert set(hint) == {
            "id",
            "category",
            "categoryLabel",
            "severity",
            "scope",
            "suggestedFix",
            "occurrences",
            "matches",
        }
        assert hint["occurrences"] >= 1
        assert "textSpan" not in hint
        assert "start" not in hint
        assert "end" not in hint
    hints_by_rule = {hint["category"]: hint for hint in context["rulebookHints"]}
    assert any("통해" in match for match in hints_by_rule["A-2"]["matches"])


async def test_prepare_context_does_not_leak_protected_term_spans():
    captured = {}
    protected = "API v1을 통해"

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
        _payload(text=f"{protected} 결과를 확인했습니다.", protected_terms=[protected])
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    rendered_context = json.dumps(captured["context"], ensure_ascii=False)
    assert protected not in rendered_context
    assert "API v1" not in rendered_context
    assert "textSpan" not in rendered_context
    assert "start" not in rendered_context
    assert "end" not in rendered_context


async def test_prepare_rulebook_hints_do_not_add_extra_llm_calls():
    calls = []

    class CountingLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            assert context["rulebookHints"]
            return RewriteResult(
                revisedText="성과로 결과를 냈고 목적이 있습니다.",
                changes=[
                    Change(
                        original="성과를 통해",
                        revised="성과로",
                        reason="번역투를 줄였습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["번역투를 줄였습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(status="full_pass", reason="통과")

        async def review(self, request, context, revised_text, audit_result):
            calls.append("review")
            return StrictReviewResult(
                revisedText=revised_text,
                changes=[],
                summary=[],
                auditCorrectionsApplied=[],
                finalAuditStatus="full_pass",
                finalBlockingIssues=[],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text="성과를 통해 결과를 냈고 목적을 가지고 있습니다.")
    )

    await RewriteGraphRunner(_settings(), CountingLLM()).run(request)

    assert calls == ["rewrite", "audit"]


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
        "A-12",
        "A-14",
        "A-15",
        "A-16",
        "A-18",
        "A-19",
        "B-1",
        "B-2",
        "C-1",
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
        "F-1",
        "F-4",
        "F-5",
        "G-1",
        "G-2",
        "G-3",
        "H-1",
        "H-2",
        "H-3",
        "H-4",
        "I-1",
        "I-2",
        "I-3",
        "I-4",
        "I-5",
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


def test_local_detector_ignores_common_false_positive_words():
    cases = [
        ("H-3", "성과가 눈에 보이는 수준으로 늘었고 비용을 줄이는 방법도 찾았다."),
        ("H-4", "요청을 즉시 처리했고 즉각 대응했다."),
        ("A-16", "그 사람은 그 결과를 보고 그 자리에서 결정했다."),
        ("C-7", "성과를 반면교사로 삼아 계획을 다듬었다."),
        ("D-6", "우리는 계획을 실행해야 한다."),
        ("G-3", "예산의 균형을 신중하게 검토했다."),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        assert rule_id not in {finding.category for finding in detection.findings}


def test_local_detector_still_flags_true_meta_and_connector_patterns():
    cases = [
        ("H-3", "이는 우리가 준비한 계획의 핵심이다."),
        ("H-4", "핵심 지표, 즉 재방문율을 먼저 본다."),
        ("C-7", "먼저 비용을 줄인다. 반면 품질은 유지한다. 결국 균형이 관건이다."),
        ("A-16", "그는 말했다. 그녀의 의견도 같았다."),
    ]

    for rule_id, text in cases:
        detection = local_detect(text, focus_categories=[rule_id])
        assert rule_id in {finding.category for finding in detection.findings}


async def test_rulebook_hints_include_occurrence_counts():
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
        _payload(text="계획을 통해 정리했고 실행을 통해 검증했으며 회고를 통해 개선했습니다.")
    )

    await RewriteGraphRunner(_settings(), CapturingLLM()).run(request)

    hints = {hint["category"]: hint for hint in captured["context"]["rulebookHints"]}
    assert hints["A-2"]["occurrences"] >= 3


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

        async def review_segments(self, request, segments, audit_result):
            calls.append("review")
            assert [segment.index for segment in segments] == [0]
            assert segments[0].corrections[0].before == "보고서"
            return SegmentReviewResult(
                repairedSegments=[RepairedSegment(index=0, text="2026년 보고서입니다.")],
            )

    runner = RewriteGraphRunner(_settings(), ConditionalAuditLLM())
    response = await runner.run(
        RewriteRequestForTest.model_validate(
            _payload(text="2026년 보고서입니다.", rewrite_mode="strict", max_rounds=2)
        )
    )

    assert calls == ["rewrite", "audit", "review"]
    assert response.usage.rounds == 1
    assert response.revisedText == "2026년 보고서입니다."


async def test_strict_conditional_audit_without_blocking_edits_skips_review():
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

        async def review_segments(self, request, segments, audit_result):
            raise AssertionError("advisory audit output must not trigger a review pass")

    response = await RewriteGraphRunner(_settings(), ConditionalStatusOnlyLLM()).run(
        RewriteRequestForTest.model_validate(_payload(text="2026년 보고서입니다."))
    )

    assert calls == ["rewrite", "audit"]
    assert response.revisedText == "2026년 보고서입니다."
    # Advisory model prose stays in the debug log, not in the user-facing warnings.
    assert not any("복원 필요 가능성" in warning for warning in response.warnings)


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

    # The missing quote is restored from the source sentence that held it;
    # the rest of the draft is kept rather than reverting to the original.
    assert response.revisedText.startswith("첫 번째는 소스 정리 기능입니다.")
    assert "소셜미디어 성장에 가장 도움이 되는 자료" in response.revisedText
    assert response.revisedText != text
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
    assert response.revisedText.startswith("첫 번째는 소스 정리 기능입니다. 에이전트에게 ")
    assert response.revisedText != complete_revised
    assert not any("revisedText가 불완전" in item for item in response.summary)
    assert not any("원문을 반환" in warning for warning in response.warnings)
    assert any("출력 잘림" in warning for warning in response.warnings)


async def test_model_review_that_damages_preserved_values_falls_back_to_local_repair():
    draft = "2026년 5월 출시 일정은 유지합니다. 세부 계획은 다시 정리했습니다."

    class DamagingReviewLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText=draft,
                changes=[
                    Change(
                        original="정리했습니다",
                        revised="다시 정리했습니다",
                        reason="표현을 명확히 했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["일정 표현을 정리했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(
                status="conditional_pass",
                flaggedEdits=[
                    {
                        "before": "정리했습니다",
                        "after": "다시 정리했습니다",
                        "issue": "원문에 없는 '다시'가 추가됐습니다.",
                        "checklistFailed": [13],
                        "action": "restore_original",
                        "correctionDirection": "'다시'를 빼고 원문 표현으로 복원합니다.",
                        "severity": "high",
                    }
                ],
                reason="첨가된 표현이 있습니다.",
            )

        async def review_segments(self, request, segments, audit_result):
            assert [segment.index for segment in segments] == [1]
            # Buggy behavior under test: the repaired sentence introduces a
            # number that is not in the source.
            return SegmentReviewResult(
                repairedSegments=[RepairedSegment(index=1, text="2027 계획은 정리했습니다.")],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="2026년 5월 출시 일정은 유지합니다. 세부 계획은 정리했습니다.",
            rewrite_mode="strict",
        )
    )
    response = await RewriteGraphRunner(_settings(), DamagingReviewLLM()).run(request)

    assert "2027" not in response.revisedText
    assert "2026년 5월" in response.revisedText
    # The local path still applies the audit correction on the draft.
    assert response.revisedText == "2026년 5월 출시 일정은 유지합니다. 세부 계획은 정리했습니다."
    assert any("로컬 복원 결과로 대체했습니다" in warning for warning in response.warnings)
    assert any("보존 대상 훼손이 초안보다 늘었습니다" in warning for warning in response.warnings)


async def test_truncated_draft_ships_with_warning_and_no_segment_review():
    rewrite_calls = 0
    text = (
        "첫 번째는 소스 정리 기능입니다. 에이전트가 저장된 모든 소스를 살펴보고 관련 자료끼리 묶어 "
        "폴더를 만들고 자동으로 정리하는 기능입니다.\n\n"
        "두 번째는 소스 검색 기능입니다. 사용자가 직접 피드를 스크롤하지 않아도 에이전트가 필요한 "
        "자료를 찾아 작은 묶음으로 반환합니다.\n\n"
        "이 기능이 제대로 작동하면 앱은 단순한 캡처 도구를 넘어 작업에 바로 쓰이는 컨텍스트 "
        "시스템이 됩니다."
    )

    class TruncatingLLM:
        async def rewrite(self, request):
            raise AssertionError("strict graph should call node-specific methods")

        async def rewrite_once(self, request, context):
            nonlocal rewrite_calls
            rewrite_calls += 1
            return RewriteResult(
                revisedText="첫 번째는 소스 정리 기능입니다. 에이전트가",
                changes=[],
                summary=[f"{rewrite_calls}라운드 초안입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="모델 감사는 통과했습니다.")

        async def review_segments(self, request, segments, audit_result):
            # A completion failure has no sentence to point at, so the segment
            # review must not be asked to regenerate the passage.
            raise AssertionError("segment review must not run for completion failures")

    request = RewriteRequestForTest.model_validate(
        _payload(text=text, rewrite_mode="strict", max_rounds=2, protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), TruncatingLLM()).run(request)

    assert response.revisedText == "첫 번째는 소스 정리 기능입니다. 에이전트가"
    assert response.usage.rounds == 1
    assert rewrite_calls == 1
    assert any("출력 잘림" in warning for warning in response.warnings)


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
    result = await llm.rewrite_once(
        RewriteRequestForTest.model_validate(_payload()),
        _rulebook_context(),
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 11
    assert result.outputTokens == 7
    assert calls[0]["model"] == "gpt-5-mini"
    assert calls[0]["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0]["text"]["format"]["type"] == "json_schema"
    assert calls[0]["text"]["format"]["strict"] is True
    assert "response_format" not in calls[0]
    user_payload = json.loads(calls[0]["input"])
    assert "rewrite_priorities" in user_payload
    assert user_payload["rewrite_priorities"]["rulebook_hints"][0]["category"] == "A-2"


async def test_anthropic_rewrite_once_uses_prepared_context(monkeypatch):
    calls = []

    class FakeMessages:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="text",
                        text=json.dumps(
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
                    )
                ],
                usage=SimpleNamespace(input_tokens=17, output_tokens=9),
            )

    class FakeAsyncAnthropic:
        def __init__(self, api_key):
            self.api_key = api_key
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )

    llm = AnthropicRewriteLLM(api_key="anthropic-key", model_name="claude-test")
    result = await llm.rewrite_once(
        RewriteRequestForTest.model_validate(_payload()),
        _rulebook_context(),
    )

    assert result.revisedText == "개선된 문장입니다."
    assert result.inputTokens == 17
    assert result.outputTokens == 9
    assert calls[0]["model"] == "claude-test"
    assert calls[0]["max_tokens"] == MAX_OUTPUT_TOKENS
    assert calls[0]["temperature"] == 0.2
    user_payload = json.loads(calls[0]["messages"][0]["content"])
    assert "rewrite_priorities" in user_payload
    assert user_payload["rewrite_priorities"]["rulebook_hints"][0]["category"] == "A-2"


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
        context=_rulebook_context(),
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
    user_payload = json.loads(calls[0]["messages"][1]["content"])
    assert "rewrite_priorities" in user_payload
    assert user_payload["rewrite_priorities"]["rulebook_hints"][0]["category"] == "A-2"


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
        RewriteOutput.model_json_schema(),
    )

    schema = response_format["json_schema"]["schema"]
    assert schema["required"] == list(schema["properties"].keys())
    assert set(schema["required"]) == {"revisedText", "changes", "summary", "warnings"}
    assert "default" not in json.dumps(schema)
    assert "title" not in json.dumps(schema)

    # Internal bookkeeping never reaches the model.
    rendered = json.dumps(schema)
    for internal in ("selfCheck", "residualFindings", "qualityLevel", "changeRate", "inputTokens"):
        assert internal not in rendered

    change_schema = schema["$defs"]["Change"]
    assert change_schema["required"] == ["original", "revised", "reason", "type", "riskLevel"]
    assert "룰 번호" in change_schema["properties"]["reason"]["description"]
    assert "concision" in change_schema["properties"]["type"]["description"]


def test_metrics_v2_computes_im_not_ai_signal_keys():
    metrics = compute_all_v2("결론적으로, API를 통해 성과를 확인할 수 있다.")

    assert metrics["version"] == "v2.0"
    assert "v2_metrics" in metrics
    assert "normalisation_score" in metrics["v2_metrics"]
    assert "v2_interference_index" in metrics


def test_compact_system_prompt_keeps_rule_index_without_bodies():
    compact = resources.compact_strict_rules()
    full = resources.strict_rules()

    assert len(compact) < len(full) * 0.5
    for rule_id in ("A-1", "A-19", "D-4", "I-1", "J-3"):
        assert f"## {rule_id}." in compact
        card = resources.rule_card(rule_id)
        assert card.startswith(f"## {rule_id}.")
    assert "의미 불변" in compact
    assert "최종 자체 검토 체크리스트" in compact
    assert "데이터를 분석해 인사이트를 얻는다" not in compact
    assert resources.rule_card("Z-99") == ""


def test_rewrite_priorities_attach_rule_cards_and_match_samples():
    request = RewriteRequestForTest.model_validate(
        _payload(text="결과를 통해 성과를 만들고 목적을 가지고 있습니다.")
    )
    context = _rulebook_context()
    context["rulebookHints"][0]["matches"] = ["성과를 통해"]

    payload = json.loads(prompts.rewrite_user_prompt(request, context))
    hints = payload["rewrite_priorities"]["rulebook_hints"]

    assert hints[0]["category"] == "A-2"
    assert hints[0]["matches"] == ["성과를 통해"]
    assert hints[0]["ruleCard"].startswith("## A-2.")
    assert "데이터를 분석해 인사이트를 얻는다" in hints[0]["ruleCard"]
    assert hints[1]["matches"] == []
    assert hints[1]["ruleCard"].startswith("## A-7.")


def test_split_text_chunks_reassembles_source_exactly():
    from humanize_core.graph import _split_text_chunks

    text = "첫 문단입니다.\n\n둘째 문단입니다.\n\n\n셋째 문단입니다.\n마지막 줄입니다.\n"
    chunks = _split_text_chunks(text, 12)

    assert len(chunks) >= 2
    assert "".join(chunks) == text


def test_split_text_chunks_cuts_only_at_sentence_ends_without_paragraph_breaks():
    from humanize_core.graph import _split_text_chunks

    sentence = "이번 분기에는 고객 문의 응답 시간을 줄이기 위한 개선 작업을 진행했습니다. "
    text = (sentence * 70).rstrip()
    chunks = _split_text_chunks(text, 1000)

    assert len(chunks) >= 3
    assert "".join(chunks) == text
    for chunk in chunks[:-1]:
        assert len(chunk) >= 1000
        assert chunk.rstrip().endswith("습니다.")


def test_split_sentence_spans_ignores_decimal_points():
    from humanize_core.graph import _split_sentence_spans

    spans = _split_sentence_spans("3분기 매출은 42.7억 원으로 집계됐다. 목표는 50억 원이다.")

    assert spans == ["3분기 매출은 42.7억 원으로 집계됐다. ", "목표는 50억 원이다."]


async def test_chunked_rewrite_splits_long_text_and_reassembles():
    chunk_texts = []

    class ChunkCapturingLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            chunk_texts.append(request.text)
            assert "rulebookHints" in context
            return RewriteResult(
                revisedText=request.text,
                changes=[
                    Change(
                        original="",
                        revised="",
                        reason="유지했습니다.",
                        type="clarity",
                        riskLevel="low",
                    )
                ],
                summary=["유지했습니다."],
            )

    paragraph = "오늘 회의에서 다음 분기 일정을 정리했습니다. 팀별 준비 상황도 함께 점검했습니다."
    text = "\n\n".join(paragraph for _ in range(40))
    request = RewriteRequestForTest.model_validate(
        _payload(text=text, protected_terms=[])
    )

    settings = _settings()
    assert len(text) >= settings.chunk_min_chars
    response = await RewriteGraphRunner(settings, ChunkCapturingLLM()).run(request)

    assert len(chunk_texts) >= 2
    assert all(len(chunk) <= settings.chunk_target_chars + len(paragraph) for chunk in chunk_texts)
    assert response.revisedText == text
    assert response.usage.rounds == 1


async def test_style_gate_repairs_residual_s1_findings():
    calls = []
    captured = {}

    class GatedLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(
                revisedText="이 제품은 강한 경쟁력을 가지고 있다.",
                changes=[],
                summary=["초안입니다."],
            )

        async def style_repair(self, request, draft_text, residual_hints, smooth_transitions):
            calls.append("style_repair")
            captured["hints"] = residual_hints
            captured["smooth"] = smooth_transitions
            return RewriteResult(
                revisedText="이 제품은 경쟁력이 강하다.",
                changes=[],
                summary=["잔존 룰 위반을 수리했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            captured["audited_text"] = revised_text
            return AuditResult(status="full_pass", reason="통과")

    request = RewriteRequestForTest.model_validate(
        _payload(text="이 제품은 강한 경쟁력을 가지고 있다.", protected_terms=[])
    )

    response = await RewriteGraphRunner(_settings(), GatedLLM()).run(request)

    assert calls == ["rewrite", "style_repair", "audit"]
    assert captured["smooth"] is False
    hint_categories = {hint["category"] for hint in captured["hints"]}
    assert "A-7" in hint_categories
    a7_hint = next(hint for hint in captured["hints"] if hint["category"] == "A-7")
    assert any("가지고 있" in match for match in a7_hint["matches"])
    assert captured["audited_text"] == "이 제품은 경쟁력이 강하다."
    assert response.revisedText == "이 제품은 경쟁력이 강하다."
    assert response.usage.rounds == 2


async def test_style_gate_discards_repair_that_regresses_style_score():
    calls = []

    class RegressingLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(
                revisedText="이 팀은 좋은 문화를 가지고 있다.",
                changes=[],
                summary=["초안입니다."],
            )

        async def style_repair(self, request, draft_text, residual_hints, smooth_transitions):
            calls.append("style_repair")
            return RewriteResult(
                revisedText="이 팀은 좋은 문화를 가지고 있다. 그것은 계속 되어진다.",
                changes=[],
                summary=["수리 실패 후보입니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(status="full_pass", reason="통과")

    request = RewriteRequestForTest.model_validate(
        _payload(text="이 팀은 좋은 문화를 가지고 있다.", protected_terms=[])
    )

    response = await RewriteGraphRunner(_settings(), RegressingLLM()).run(request)

    assert calls == ["rewrite", "style_repair", "audit"]
    assert response.revisedText == "이 팀은 좋은 문화를 가지고 있다."
    assert any("S1 잔존" in warning for warning in response.warnings)


async def test_chunked_rewrite_requests_transition_smoothing_once():
    captured_flags = []

    class TransitionLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText=request.text,
                changes=[],
                summary=["유지했습니다."],
            )

        async def style_repair(self, request, draft_text, residual_hints, smooth_transitions):
            captured_flags.append(smooth_transitions)
            return RewriteResult(
                revisedText=draft_text,
                changes=[],
                summary=["문단 연결을 점검했습니다."],
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="통과")

    paragraph = "오늘 회의에서 다음 분기 일정을 정리했습니다. 팀별 준비 상황도 함께 점검했습니다."
    text = "\n\n".join(paragraph for _ in range(40))
    request = RewriteRequestForTest.model_validate(
        _payload(text=text, protected_terms=[])
    )

    response = await RewriteGraphRunner(_settings(), TransitionLLM()).run(request)

    assert captured_flags == [True]
    assert response.revisedText == text


async def test_segment_review_restores_flagged_sentence_and_keeps_other_fixes():
    original = "이 지표는 개선이 필요할 것으로 판단된다. 성과를 통해 결과를 확인했다."
    draft = "이 지표는 개선해야 한다. 성과로 결과를 확인했다."
    calls = []

    class SegmentReviewLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(revisedText=draft, changes=[], summary=["초안입니다."])

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(
                status="conditional_pass",
                reason="완곡 양태가 단정으로 바뀌었습니다.",
                flaggedEdits=[
                    {
                        "before": "개선이 필요할 것으로 판단된다",
                        "after": "개선해야 한다",
                        "issue": "추론 표현이 단정형으로 변경돼 주장 강도가 상승했습니다.",
                        "checklistFailed": [6],
                        "action": "rewrite_required",
                        "correctionDirection": "원문의 추론 양태를 복원합니다.",
                        "severity": "high",
                    }
                ],
            )

        async def review_segments(self, request, segments, audit_result):
            calls.append("review")
            assert len(segments) == 1
            assert segments[0].index == 0
            assert segments[0].draft_sentence == "이 지표는 개선해야 한다."
            assert segments[0].original_sentence == "이 지표는 개선이 필요할 것으로 판단된다."
            # Even an index the graph did not ask for is ignored on splice.
            return SegmentReviewResult(
                repairedSegments=[
                    RepairedSegment(index=0, text="이 지표는 개선이 필요해 보인다."),
                    RepairedSegment(index=1, text="성과를 통해 결과를 확인했다."),
                ],
                inputTokens=11,
                outputTokens=5,
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text=original, protected_terms=[])
    )

    response = await RewriteGraphRunner(_settings(), SegmentReviewLLM()).run(request)

    assert calls == ["rewrite", "audit", "review"]
    assert response.revisedText == "이 지표는 개선이 필요해 보인다. 성과로 결과를 확인했다."
    assert response.usage.inputTokens >= 11
    assert not any("로컬에서 부분 복원" in warning for warning in response.warnings)


async def test_low_severity_rewrite_flag_becomes_warning_without_review():
    calls = []

    class AdvisoryAuditLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            calls.append("rewrite")
            return RewriteResult(revisedText="이 지표는 개선해야 한다.", changes=[], summary=["초안입니다."])

        async def audit(self, request, context, revised_text, changes):
            calls.append("audit")
            return AuditResult(
                status="full_pass",
                reason="경미한 양태 변화가 있습니다.",
                flaggedEdits=[
                    {
                        "before": "개선이 필요할 것으로 판단된다",
                        "after": "개선해야 한다",
                        "issue": "추론 양태가 다소 강해졌습니다.",
                        "checklistFailed": [6],
                        "action": "rewrite_required",
                        "correctionDirection": "필요하면 양태를 완화합니다.",
                        "severity": "low",
                    }
                ],
            )

        async def review_segments(self, request, segments, audit_result):
            raise AssertionError("low-severity flags must not trigger a review pass")

    request = RewriteRequestForTest.model_validate(
        _payload(text="이 지표는 개선이 필요할 것으로 판단된다.", protected_terms=[])
    )
    response = await RewriteGraphRunner(_settings(), AdvisoryAuditLLM()).run(request)

    assert calls == ["rewrite", "audit"]
    assert response.revisedText == "이 지표는 개선해야 한다."
    assert response.warnings == []


async def test_local_repair_skips_style_restore_that_reintroduces_s1():
    original = (
        "이번 조사 결과는 시사하는 바가 크다. "
        "후속 대응은 개선이 필요할 것으로 판단된다. 응답률은 62%였다."
    )
    draft = (
        "이번 조사 결과는 후속 논의가 필요한 지점을 보여준다. "
        "후속 대응은 손봐야 한다. 응답률은 62%였다."
    )

    class NoReviewLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(revisedText=draft, changes=[], summary=["초안입니다."])

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(
                status="conditional_pass",
                reason="문체 수정 검토가 필요합니다.",
                flaggedEdits=[
                    {
                        # Restoring this would reintroduce the D-2 S1 idiom.
                        "before": "이번 조사 결과는 시사하는 바가 크다.",
                        "after": "이번 조사 결과는 후속 논의가 필요한 지점을 보여준다.",
                        "issue": "메타 서술이 제거됐습니다.",
                        "checklistFailed": [7],
                        "action": "rewrite_required",
                        "correctionDirection": "원문 서술을 복원합니다.",
                        "severity": "high",
                    },
                    {
                        # S1-free hedge restore must still apply.
                        "before": "개선이 필요할 것으로 판단된다",
                        "after": "손봐야 한다",
                        "issue": "추론 양태가 단정으로 바뀌었습니다.",
                        "checklistFailed": [6],
                        "action": "rewrite_required",
                        "correctionDirection": "추론 양태를 복원합니다.",
                        "severity": "high",
                    },
                ],
            )

    request = RewriteRequestForTest.model_validate(
        _payload(text=original, protected_terms=[])
    )

    response = await RewriteGraphRunner(_settings(), NoReviewLLM()).run(request)

    # The D-2 idiom stays fixed; the modality hedge is restored.
    assert "시사하는 바가 크다" not in response.revisedText
    assert "후속 논의가 필요한 지점을 보여준다" in response.revisedText
    assert "개선이 필요할 것으로 판단된다" in response.revisedText
    assert any("게이트 수리 결과를 유지" in warning for warning in response.warnings)


def test_display_changes_attach_reasons_by_location_not_index():
    from humanize_core.diff import UNEXPLAINED_CHANGE_REASON, build_display_safe_changes

    original = "이번 프로젝트에 대해 공유드립니다. 데이터 분석을 통해 원인을 파악했습니다. 서비스 개선에 있어 핵심은 속도입니다."
    revised = "이번 프로젝트를 공유드립니다. 데이터를 분석해 원인을 파악했습니다. 서비스 개선에서 핵심은 속도입니다."
    seeds = [
        # Draft-based snippet that no longer exists in the final text: must not
        # be attached to any group.
        Change(original="데이터 분석을 통해", revised="데이터 분석으로", reason="초안 기준 사유", type="clarity"),
        Change(original="개선에 있어", revised="개선에서", reason="'에 있어'를 '에서'로 줄였습니다.", type="grammar"),
        Change(original="프로젝트에 대해", revised="프로젝트를", reason="'에 대해'를 목적어로 이었습니다.", type="clarity"),
    ]

    changes = build_display_safe_changes(original, revised, seeds)

    reasons = [change.reason for change in changes]
    assert reasons == [
        "'에 대해'를 목적어로 이었습니다.",
        UNEXPLAINED_CHANGE_REASON,
        "'에 있어'를 '에서'로 줄였습니다.",
    ]
    assert "초안 기준 사유" not in reasons
    for change in changes:
        assert change.original in original and change.revised in revised


async def test_finalize_explains_every_diff_group_with_model_reasons_as_hints():
    from humanize_core.diff import UNEXPLAINED_CHANGE_REASON
    from humanize_core.im_not_ai.schemas import ChangeExplanation

    original = "이번 프로젝트에 대해 공유드립니다. 데이터 분석을 통해 원인을 파악했습니다."
    revised = "이번 프로젝트를 공유드립니다. 데이터를 분석해 원인을 파악했습니다."
    explain_calls = []

    class ExplainingLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(
                revisedText=revised,
                changes=[Change(original="프로젝트에 대해", revised="프로젝트를", reason="'에 대해'를 목적어로 이었습니다.", type="clarity")],
                summary=["모델이 쓴 요약입니다."],
                inputTokens=10,
                outputTokens=5,
            )

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="통과", inputTokens=3, outputTokens=1)

        async def explain_changes(self, request, revised_text, items):
            explain_calls.append(items)
            return ChangeExplanationResult(
                items=[
                    ChangeExplanation(index=item["index"], reason=f"설명 {item['index']}: 구간을 다듬었습니다.", type="grammar", riskLevel="low")
                    for item in items
                ],
                summary=["설명 단계가 쓴 요약입니다."],
                inputTokens=7,
                outputTokens=2,
            )

    request = RewriteRequestForTest.model_validate(_payload(text=original, protected_terms=[]))
    response = await RewriteGraphRunner(_settings(), ExplainingLLM()).run(request)

    # Both diff groups are sent; the model's reason rides along only for the group it matched.
    assert len(explain_calls) == 1
    items = explain_calls[0]
    assert [item["index"] for item in items] == [0, 1]
    assert items[0]["draft_reason"] == "'에 대해'를 목적어로 이었습니다."
    assert items[1]["draft_reason"] == ""
    assert "통해" in items[1]["original"]
    # Snippets are cut at word boundaries, never mid-word.
    for item in items:
        assert not item["original"].startswith(("트", "터"))
    reasons = [change.reason for change in response.changes]
    assert reasons == ["설명 0: 구간을 다듬었습니다.", "설명 1: 구간을 다듬었습니다."]
    assert UNEXPLAINED_CHANGE_REASON not in reasons
    assert all(change.type == "grammar" for change in response.changes)
    assert response.summary == ["설명 단계가 쓴 요약입니다."]
    assert response.usage.inputTokens == 10 + 3 + 7
    assert response.usage.outputTokens == 5 + 1 + 2


async def test_finalize_keeps_generic_reason_when_explain_call_fails():
    from humanize_core.diff import UNEXPLAINED_CHANGE_REASON

    original = "이번 프로젝트에 대해 공유드립니다."
    revised = "이번 프로젝트를 공유드립니다."

    class FailingExplainLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(revisedText=revised, changes=[], summary=["초안입니다."])

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(status="full_pass", reason="통과")

        async def explain_changes(self, request, revised_text, items):
            raise LLMResponseError("explain failed")

    request = RewriteRequestForTest.model_validate(_payload(text=original, protected_terms=[]))
    response = await RewriteGraphRunner(_settings(), FailingExplainLLM()).run(request)

    assert response.revisedText == revised
    assert [change.reason for change in response.changes] == [UNEXPLAINED_CHANGE_REASON]


def test_review_segments_locate_flagged_sentences_by_after_then_before():
    from humanize_core.graph import _review_segments_for
    from humanize_core.im_not_ai.schemas import FlaggedEdit

    request = RewriteRequestForTest.model_validate(
        _payload(
            text="첫 문장은 그대로입니다. 출시는 2026년 5월입니다. 셋째 문장은 개선이 필요할 것으로 판단된다.",
            protected_terms=[],
        )
    )
    draft = "첫 문장은 그대로입니다. 출시는 내년 봄입니다. 셋째 문장은 손봐야 한다."
    edits = [
        FlaggedEdit(before="2026년 5월", after="내년 봄", issue="날짜", action="preserve_exact", severity="high"),
        # `after` not in the draft: fall back to the original sentence that holds `before`.
        FlaggedEdit(before="개선이 필요할 것으로 판단된다", after="", issue="양태", action="rewrite_required", severity="high"),
        # Nothing to locate: dropped.
        FlaggedEdit(issue="전체 누락", action="restore_original", severity="high"),
    ]

    segments = _review_segments_for(request, draft, edits)

    assert [segment.index for segment in segments] == [1, 2]
    assert segments[0].draft_sentence == "출시는 내년 봄입니다."
    assert segments[0].original_sentence == "출시는 2026년 5월입니다."
    assert segments[1].draft_sentence == "셋째 문장은 손봐야 한다."
    assert segments[1].corrections[0].issue == "양태"


def test_splice_repaired_segments_keeps_separators_and_ignores_unknown_indexes():
    from humanize_core.graph import _review_segments_for, _splice_repaired_segments
    from humanize_core.im_not_ai.schemas import FlaggedEdit

    request = RewriteRequestForTest.model_validate(_payload(text="가 문장. 나 문장.\n\n다 문장.", protected_terms=[]))
    draft = "가 문장. 나 문장!\n\n다 문장."
    segments = _review_segments_for(
        request,
        draft,
        [FlaggedEdit(before="나 문장.", after="나 문장!", issue="x", action="restore_original", severity="high")],
    )
    review = SegmentReviewResult(
        repairedSegments=[
            RepairedSegment(index=1, text="  나 문장.  "),
            RepairedSegment(index=2, text="바꾸면 안 됨"),
        ]
    )

    spliced, repaired = _splice_repaired_segments(draft, segments, review)

    assert spliced == "가 문장. 나 문장.\n\n다 문장."
    assert repaired == [1]


def test_display_changes_use_each_model_reason_once():
    from humanize_core.diff import UNEXPLAINED_CHANGE_REASON, build_display_safe_changes

    original = "모니터링 알림은 정상 동작했지만, 담당자 부재로 대응이 40분 늦었습니다."
    revised = "모니터링 알림은 정상 동작했지만 담당자 부재로 대응이 40분 지연되었습니다."
    # One sentence-wide model change covering two separate edits.
    seeds = [Change(original=original, revised=revised, reason="쉼표를 지우고 어휘를 바꿨습니다.", type="clarity")]

    changes = build_display_safe_changes(original, revised, seeds)

    assert len(changes) == 2
    reasons = [change.reason for change in changes]
    assert reasons.count("쉼표를 지우고 어휘를 바꿨습니다.") == 1
    assert reasons.count(UNEXPLAINED_CHANGE_REASON) == 1


def test_display_change_snippets_end_on_word_boundaries():
    from humanize_core.diff import build_display_safe_changes

    original = "데이터 분석을 통해 고객 이탈의 원인을 파악할 수 있었고 설문 조사를 진행했습니다."
    revised = "데이터를 분석해 고객 이탈의 원인을 파악했고 설문 조사를 진행했습니다."

    for change in build_display_safe_changes(original, revised, []):
        assert change.original in original and change.revised in revised
        assert not change.original[0].isspace() and not change.original[-1].isspace()
        # Context starts right after a boundary and ends right before one.
        start = original.index(change.original)
        end = start + len(change.original)
        assert start == 0 or original[start - 1] in " .,"
        assert end == len(original) or original[end] in " .,"


def test_clean_repaired_segment_rejects_neighbour_copies_and_extra_sentences():
    from humanize_core.graph import _clean_repaired_segment

    bodies = ["첫 문장은 그대로입니다.", "둘째 문장을 고칩니다.", "셋째 문장도 그대로입니다."]

    # Neighbour sentence prepended / appended verbatim is stripped.
    assert _clean_repaired_segment("첫 문장은 그대로입니다. 둘째 문장을 손봤습니다.", 1, bodies) == "둘째 문장을 손봤습니다."
    assert _clean_repaired_segment("둘째 문장을 손봤습니다. 셋째 문장도 그대로입니다.", 1, bodies) == "둘째 문장을 손봤습니다."
    # Another draft sentence embedded elsewhere, or a split into two sentences: rejected.
    assert _clean_repaired_segment("둘째 문장을 손봤습니다. 그리고 첫 문장은 그대로입니다.", 1, bodies) == ""
    assert _clean_repaired_segment("둘째 문장입니다. 손봤습니다.", 1, bodies) == ""
    assert _clean_repaired_segment("   ", 1, bodies) == ""
    # A plain single-sentence repair passes through trimmed.
    assert _clean_repaired_segment("  둘째 문장을 다듬었습니다.  ", 1, bodies) == "둘째 문장을 다듬었습니다."


async def test_review_summary_has_no_internal_notes():
    class BlockingAuditLLM:
        async def rewrite(self, request):
            raise AssertionError("graph should call rewrite_once")

        async def rewrite_once(self, request, context):
            return RewriteResult(revisedText="2026년 보고서를 냈습니다.", changes=[], summary=["표현을 다듬었습니다."])

        async def audit(self, request, context, revised_text, changes):
            return AuditResult(
                status="conditional_pass",
                reason="첨가",
                flaggedEdits=[
                    {
                        "before": "보고서입니다",
                        "after": "보고서를 냈습니다",
                        "issue": "없던 행위가 추가됐습니다.",
                        "action": "restore_original",
                        "severity": "high",
                    }
                ],
            )

    request = RewriteRequestForTest.model_validate(_payload(text="2026년 보고서입니다.", protected_terms=[]))
    response = await RewriteGraphRunner(_settings(), BlockingAuditLLM()).run(request)

    assert response.revisedText == "2026년 보고서입니다."
    assert not any("Audit repair" in item for item in response.summary)
    assert any("로컬에서 부분 복원" in warning for warning in response.warnings)


def test_display_changes_diff_whole_words_not_characters():
    from humanize_core.diff import build_display_safe_changes

    original = "근본적인 혁신이 필요하다. 대단히 어려운 일이지만 3월까지 첫 결과를 내야 한다."
    revised = "근본적인 혁신이 필요하다. 쉽지 않은 과제이지만 3월까지 첫 성과를 내야 한다."

    changes = build_display_safe_changes(original, revised, [])

    originals = [change.original for change in changes]
    # A replaced word shows up whole; no "일이지만" -> "제이지만" fragments, and
    # one change's context never leaks the neighbouring edit's before/after.
    assert any("대단히 어려운 일이지만" in snippet for snippet in originals)
    assert not any(change.revised.startswith("제이지만") for change in changes)
    assert len(changes) == 2
    assert "첫 결과를" in changes[1].original and "첫 성과를" in changes[1].revised
    assert "일이지만" not in changes[1].original
    for change in changes:
        assert change.original in original and change.revised in revised


def test_detector_density_rules_ignore_single_ordinary_uses():
    from humanize_core.im_not_ai.audit import detect_register

    single = "신규 대시보드를 도입하면 업무 효율을 높일 수 있습니다. 자동화 도구로 관리되며, 모니터링은 정상 동작했지만 대응이 늦었습니다."
    categories = {finding.category for finding in local_detect(single).findings}
    # One "할 수 있", one "자동화", one comma after "지만", one "~고 있다"-free text: all ordinary.
    assert "A-10" not in categories
    assert "F-4" not in categories
    assert "C-11" not in categories
    assert "E-2" not in categories
    assert "A-18" not in categories

    dense = "효율을 높일 수 있습니다. 시간을 줄일 수 있고 오류도 발견할 수 있습니다. 지표를 확인할 수 있으며 댓글도 사용할 수 있습니다."
    dense_categories = {finding.category for finding in local_detect(dense).findings}
    assert "A-10" in dense_categories

    commas = "검토했지만, 보류했고, 다시 논의하며, 확정했다."
    assert {f.severity for f in local_detect(commas, focus_categories=["C-11"]).findings} == {"S2"}

    stacked = (
        "사고를 일으킨 화학물질을 생산한 회사에서 일했던 남자를 만났다. "
        "그가 소개한 회사에서 근무했던 동료를 다시 만났다."
    )
    assert "A-18" in {f.category for f in local_detect(stacked, focus_categories=["A-18"]).findings}

    assert detect_register("회의를 엽니다. 자료를 보내 주세요. 기한을 지켜 주시기 바랍니다.") == "formal"
    assert detect_register("일정 공유드립니다. 수요일에 반영된다. 목요일에 배포해요. 당번이 맡는다.") == "mixed"


def test_detector_formal_register_streak_is_not_rhythm_violation():
    formal = "회의를 엽니다. 자료를 보냅니다. 기한을 지킵니다. 결과를 공유합니다. 질문을 받습니다."
    assert "E-2" not in {f.category for f in local_detect(formal, focus_categories=["E-2"]).findings}
    plain = "지표를 분석했다. 사용률은 12%였다. 둘째 주에 올랐다. 셋째 주에도 올랐다."
    assert "E-2" in {f.category for f in local_detect(plain, focus_categories=["E-2"]).findings}


def test_detector_covers_previously_unmatched_rules():
    cases = [
        ("C-1", "첫째, 승인 단계를 줄인다. 둘째, 지표를 남긴다. 셋째, 소통 방식을 바꾼다."),
        ("F-1", "매우 복잡한 승인 단계를 줄이고 정말 중요한 지표만 남긴다."),
        ("I-5", "근본적인 혁신이 필요하다. 조직의 변화가 필요하다."),
        ("A-12", "계약 체결이 이루어졌다."),
        ("H-2", "하지만 어렵다. 그러나 가능하다. 하지만 늦다."),
        ("A-14", "그는 보고했다. 그리고 앉았다. 그리고 떠났다."),
    ]
    for rule_id, text in cases:
        assert rule_id in {f.category for f in local_detect(text, focus_categories=[rule_id]).findings}, rule_id
    # Ordinal enumeration is advisory: never S1, so the style gate does not force it out.
    assert {f.severity for f in local_detect(cases[0][1], focus_categories=["C-1"]).findings} == {"S2"}
    # Below the density threshold nothing fires.
    assert "F-1" not in {f.category for f in local_detect("매우 복잡한 승인 단계를 줄인다.", focus_categories=["F-1"]).findings}
    assert "H-2" not in {f.category for f in local_detect("하지만 어렵다. 그러나 가능하다.", focus_categories=["H-2"]).findings}


def test_compact_rulebook_keeps_fix_lines_for_undetected_rules():
    compact = resources.compact_strict_rules()
    assert "## F-2." in compact
    assert "둘 중 하나만 남긴다" in compact  # F-2 수정 방안
    assert "## A-13." in compact
    assert "필요한 조사를 복원하고 동사로 풀어 쓴다" in compact  # A-13 수정 방안
    assert "중요하고 핵심적인 역할" not in compact  # F-2 윤문 대상/예시 stay in the card
    assert len(compact) < len(resources.strict_rules()) * 0.5


def test_detector_flags_personified_abstract_subjects_with_object_phrases():
    text = "기술의 발전은 우리에게 새로운 질문을 던지고 있다. 변화의 흐름이 대응을 요구한다."
    categories = [f.category for f in local_detect(text, focus_categories=["D-5"]).findings]
    assert categories.count("D-5") == 2
    assert "D-5" not in {f.category for f in local_detect("팀장이 질문을 던졌다.", focus_categories=["D-5"]).findings}
