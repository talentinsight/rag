"""
Deterministic contract tests for the deployed app (src.api_comprehensive_guardrails:app).

External dependencies (RAG pipeline: Weaviate/OpenAI) are replaced by a recording fake;
the real route code, auth, guardrails and response models are exercised. PII is synthetic.
"""
import os
import uuid
from datetime import datetime

os.environ.setdefault("BEARER_TOKEN", "test-token")

import pytest
from fastapi.testclient import TestClient

from src import api_comprehensive_guardrails as api
from src.comprehensive_guardrails import GuardrailResult

TOKEN = "contract-test-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

QUERY_FIELDS = {
    "answer", "question", "pii_masked_input", "chunks_found", "sources", "model",
    "total_tokens", "processing_time_ms", "timestamp", "guardrails_passed",
    "input_guardrails", "output_guardrails", "safety_score",
}
LEGACY_FIELDS = QUERY_FIELDS - {"chunks_found", "sources"}
RECORD_FIELDS = {"category", "passed", "score", "reason", "severity"}

SOURCE = {
    "chunk_id": "chunk_7",
    "section": "3.2 Attention",
    "score": 0.91,
    "content": "An attention function maps a query and a set of key-value pairs to an output.",
}
SYNTHETIC_EMAIL = "jane.doe@acme-mail.net"


class FakePipeline:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def query(self, question, num_chunks=5, min_score=0.0):
        self.calls.append({"question": question, "num_chunks": num_chunks, "min_score": min_score})
        if self.fail:
            raise RuntimeError("vector store unavailable")
        return {
            "answer": "Attention maps queries and key-value pairs to an output.",
            "chunks_found": 1,
            "sources": [dict(SOURCE)],
            "model": "fake-model",
            "total_tokens": 42,
        }


@pytest.fixture
def pipeline(monkeypatch):
    fake = FakePipeline()
    monkeypatch.setattr(api, "BEARER_TOKEN", TOKEN)
    monkeypatch.setattr(api, "rag_pipeline", fake)
    return fake


@pytest.fixture
def client(pipeline):
    # No context manager: lifespan (real pipeline startup) is intentionally not run.
    return TestClient(api.app)


def post(client, path, question, **extra):
    body = {"question": question, "client_id": f"t-{uuid.uuid4().hex}", **extra}
    return client.post(path, json=body, headers=HEADERS)


def result(category, passed, severity, reason="synthetic"):
    return GuardrailResult(passed=passed, score=0.0 if not passed else 1.0, reason=reason,
                           category=category, severity=severity, metadata={})


def force_input(monkeypatch, results):
    passed = all(r.passed for r in results if r.severity in ("high", "critical") or not r.passed)
    monkeypatch.setattr(api.guardrails, "check_input_guardrails_with_pii_filtering",
                        lambda text, client_id="default": (passed, results))


def test_benign_question_returns_full_contract(client, pipeline):
    r = post(client, "/query", "What is the attention mechanism?", num_chunks=3, min_score=0.2)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == QUERY_FIELDS
    assert body["question"] == body["pii_masked_input"] == "What is the attention mechanism?"
    assert body["chunks_found"] == 1
    assert body["sources"] == [SOURCE]
    assert body["model"] == "fake-model" and body["total_tokens"] == 42
    assert isinstance(body["processing_time_ms"], float)
    datetime.fromisoformat(body["timestamp"])
    assert isinstance(body["guardrails_passed"], bool)
    assert 0.0 <= body["safety_score"] <= 1.0
    for rec in body["input_guardrails"] + body["output_guardrails"]:
        assert RECORD_FIELDS <= set(rec)
    assert any(g["category"] == "pii_detection" and g["passed"] for g in body["input_guardrails"])
    assert pipeline.calls == [{"question": "What is the attention mechanism?",
                               "num_chunks": 3, "min_score": 0.2}]


def test_pii_is_masked_before_retrieval_and_generation(client, pipeline, caplog):
    question = f"What is attention? My email is {SYNTHETIC_EMAIL}"
    with caplog.at_level("DEBUG"):
        r = post(client, "/query", question)
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == question  # existing contract: original question echoed
    assert SYNTHETIC_EMAIL not in body["pii_masked_input"]
    assert "[EMAIL_MASKED]" in body["pii_masked_input"]
    pii = [g for g in body["input_guardrails"] if g["category"] == "pii_detection"]
    assert pii and pii[0]["passed"] is False
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["question"] == body["pii_masked_input"]
    assert SYNTHETIC_EMAIL not in pipeline.calls[0]["question"]
    assert SYNTHETIC_EMAIL not in caplog.text  # no raw PII in query-path logs


def test_critical_block_is_structured_and_skips_retrieval(client, pipeline, monkeypatch):
    force_input(monkeypatch, [result("pii_detection", True, "low"),
                              result("self_harm_detection", False, "critical", "blocked-reason")])
    r = post(client, "/query", "synthetic blocked input")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == QUERY_FIELDS
    assert body["sources"] == [] and body["chunks_found"] == 0
    assert body["guardrails_passed"] is False
    assert body["output_guardrails"] == []
    assert body["model"] == "safety-filter"
    assert body["answer"].startswith("BLOCKED")
    blocked = [g for g in body["input_guardrails"] if g["category"] == "self_harm_detection"]
    assert blocked and blocked[0]["passed"] is False and blocked[0]["reason"] == "blocked-reason"
    assert pipeline.calls == []


def test_failed_output_check_keeps_answer_and_reports_check(client, pipeline, monkeypatch):
    monkeypatch.setattr(api.guardrails, "check_output_guardrails",
                        lambda data, start: (False, [result("profanity_filter", False, "medium", "out-problem")]))
    r = post(client, "/query", "What is attention?")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Attention maps queries and key-value pairs to an output."
    assert body["guardrails_passed"] is False
    assert body["output_guardrails"][0]["category"] == "profanity_filter"
    assert body["output_guardrails"][0]["passed"] is False
    assert body["sources"] == [SOURCE]
    assert len(pipeline.calls) == 1


def test_auth_failures_are_not_guardrail_decisions(client, pipeline):
    body = {"question": "What is attention?"}
    assert client.post("/query", json=body).status_code in (401, 403)
    r = client.post("/query", json=body, headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert "guardrails_passed" not in r.json()
    assert pipeline.calls == []


def test_service_failure_is_500_not_guardrail_decision(client, pipeline):
    pipeline.fail = True
    r = post(client, "/query", "What is attention?")
    assert r.status_code == 500
    assert "guardrails_passed" not in r.json()
    assert "Query processing failed" in r.json()["detail"]


def test_legacy_route_uses_shared_path_and_legacy_shape(client, pipeline, monkeypatch):
    calls = []
    original = api._execute_query

    async def spy(request):
        calls.append(request.question)
        return await original(request)

    monkeypatch.setattr(api, "_execute_query", spy)
    r = post(client, "/query-guardrails", f"What is attention? Mail {SYNTHETIC_EMAIL}")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == LEGACY_FIELDS
    assert SYNTHETIC_EMAIL not in body["pii_masked_input"]
    assert len(calls) == 1 and len(pipeline.calls) == 1
    assert SYNTHETIC_EMAIL not in pipeline.calls[0]["question"]


def test_legacy_route_critical_block_is_http_400(client, pipeline, monkeypatch):
    force_input(monkeypatch, [result("self_harm_detection", False, "critical", "blocked-reason")])
    r = post(client, "/query-guardrails", "synthetic blocked input")
    assert r.status_code == 400
    assert r.json()["detail"] == "Request blocked: blocked-reason"
    assert pipeline.calls == []


def test_legacy_route_auth_and_service_errors(client, pipeline):
    assert client.post("/query-guardrails", json={"question": "x"},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    pipeline.fail = True
    assert post(client, "/query-guardrails", "What is attention?").status_code == 500
