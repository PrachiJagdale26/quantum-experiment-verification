"""
Regression: when the live LLM fails (e.g. Vertex 403 billing disabled), the client used to
return canned demo JSON, so every uploaded paper was "analysed" as the same reference paper.
Paper analysis must fall back to the paper's own text, and the client must stop retrying.
"""

import os

from src.llm.client import ConfigurableLLMClient
from src.llm.extractor import ProposalExtractor

PAPER = os.path.join(os.path.dirname(__file__), "..", "data", "sample_papers", "paper_c_vqe_trapped_ion.txt")


def _failing_client(monkeypatch):
    client = ConfigurableLLMClient(demo_mode=False)
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("403 PERMISSION_DENIED. BILLING_DISABLED for project")

    monkeypatch.setattr(client, "is_configured", lambda: (True, "configured"))
    client._provider = "vertex"
    monkeypatch.setattr(client, "_call_vertex", boom)
    return client, calls


def test_failed_live_call_uses_papers_own_text_not_canned_demo(monkeypatch):
    client, calls = _failing_client(monkeypatch)
    text = open(PAPER, encoding="utf-8").read()
    paper = ProposalExtractor(client).extract_research_paper(text, document_title="A Hardware-Efficient VQE Ansatz")
    assert "VQE" in paper.title
    assert any("neutral atom" in s.lower() for s in paper.future_work)
    assert client.live_failure and "billing" in client.live_failure


def test_circuit_breaker_stops_retrying_after_auth_failure(monkeypatch):
    client, calls = _failing_client(monkeypatch)
    extractor = ProposalExtractor(client)
    text = open(PAPER, encoding="utf-8").read()
    extractor.extract_research_paper(text, document_title="t1")
    extractor.extract_research_paper(text, document_title="t2")
    assert calls["n"] == 1


def test_legacy_callers_still_get_demo_fallback(monkeypatch):
    client, _ = _failing_client(monkeypatch)
    out = client.complete("sys", "Analyze this proposal")  # default fallback_to_demo=True
    assert isinstance(out, str) and out
