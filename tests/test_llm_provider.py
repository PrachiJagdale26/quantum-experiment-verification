"""
Gemini provider behaviour (fake client, no network):
- an overloaded model falls back to the next configured model, and the answer records which model replied;
- an overloaded model is skipped for a while instead of delaying every call;
- auth errors are not retried and trip the circuit breaker;
- every call is labelled live/demo and logged in the research database without prompt text.
"""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from src.llm.client import ConfigurableLLMClient
from src.storage import LLMCall, session_scope


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour      # model -> "ok" | exception message
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        b = self.behaviour.get(model, "ok")
        if b != "ok":
            raise RuntimeError(b)
        return SimpleNamespace(text='{"ok": true}')


def _client(monkeypatch, behaviour):
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "fallback-a,fallback-b")
    c = ConfigurableLLMClient(demo_mode=False)
    c._provider, c._model, c._api_key = "gemini", "primary", "test-key"
    fake = SimpleNamespace(models=FakeModels(behaviour))
    monkeypatch.setattr(c, "_get_genai_client", lambda: fake)
    monkeypatch.setattr(c, "is_configured", lambda: (True, "ok"))
    return c, fake.models


def test_overloaded_primary_falls_back_and_records_model(monkeypatch):
    c, models = _client(monkeypatch, {"primary": "503 UNAVAILABLE high demand"})
    out = c.complete("sys json", "user", fallback_to_demo=False, purpose="unit")
    assert out == '{"ok": true}'
    assert c.last_call["model"] == "fallback-a" and c.last_call["live"] is True
    assert models.calls == ["primary", "fallback-a"]          # no pointless retry of an overloaded model


def test_overloaded_model_is_cooled_down(monkeypatch):
    c, models = _client(monkeypatch, {"primary": "503 UNAVAILABLE"})
    c.complete("s", "u", fallback_to_demo=False)
    models.calls.clear()
    c.complete("s", "u", fallback_to_demo=False)
    assert models.calls == ["fallback-a"]


def test_rate_limit_is_retried_once_then_falls_back(monkeypatch):
    monkeypatch.setattr("src.llm.client.time.sleep", lambda s: None)
    c, models = _client(monkeypatch, {"primary": "429 RESOURCE_EXHAUSTED"})
    c.complete("s", "u", fallback_to_demo=False)
    assert models.calls == ["primary", "primary", "fallback-a"]


def test_auth_error_is_not_retried_and_trips_breaker(monkeypatch):
    c, models = _client(monkeypatch, {"primary": "403 PERMISSION_DENIED API key not valid"})
    with pytest.raises(RuntimeError):
        c.complete("s", "u", fallback_to_demo=False)
    assert models.calls == ["primary"]
    assert c.live_failure
    assert c.last_call["live"] is False and c.last_call["success"] is False


def test_demo_output_is_labelled_demo(monkeypatch):
    c = ConfigurableLLMClient(demo_mode=True)
    c.complete("s", "Analyze this proposal")
    assert c.last_call["live"] is False and c.last_call["provider"] == "demo"


def test_calls_are_logged_without_prompt_text(monkeypatch):
    c, _ = _client(monkeypatch, {})
    c.complete("secret system prompt", "secret user text", fallback_to_demo=False, purpose="audit_test")
    with session_scope() as s:
        rows = s.scalars(select(LLMCall).where(LLMCall.purpose == "audit_test")).all()
    assert rows and rows[-1].live and rows[-1].model == "primary"
    assert len(rows[-1].prompt_sha256) == 64
    assert "secret" not in (rows[-1].error or "")
