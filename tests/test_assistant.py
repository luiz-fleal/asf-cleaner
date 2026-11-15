"""Assistant endpoint: no network, the Anthropic call is replaced."""
import pytest
from fastapi.testclient import TestClient

from asf_cleaner import api, assistant


class FakeRes:
    def __init__(self, text="Rain is rising."):
        self.text = text

    def raise_for_status(self):
        pass

    def json(self):
        return {"content": [{"type": "text", "text": self.text}]}


def test_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = TestClient(api.app).post("/assistant", json={"question": "why?"})
    assert r.status_code == 200 and r.json() == {"configured": False, "answer": None}


def test_answer_is_returned_when_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(assistant, "ask", lambda q, lang, ctx: "Chuva em alta.")
    r = TestClient(api.app).post("/assistant", json={"question": "how is it?", "lang": "pt", "context": {"risk": "high"}})
    assert r.json() == {"configured": True, "answer": "Chuva em alta."}


def test_ask_builds_request(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    seen = {}

    def fake_post(url, json, timeout, headers):
        seen.update(url=url, json=json, headers=headers)
        return FakeRes("Chuva em alta.")

    out = assistant.ask("how is it?", "pt", {"risk": "high"}, post=fake_post)
    assert out == "Chuva em alta."
    assert "Portuguese" in seen["json"]["system"]
    assert '"risk": "high"' in seen["json"]["messages"][0]["content"]
    assert seen["headers"]["x-api-key"] == "secret"


def test_inputs_are_bounded():
    q, ctx = assistant.clean("x" * 5000, {"a": "y" * 20000})
    assert len(q) == assistant.MAX_QUESTION and len(ctx) == assistant.MAX_CONTEXT


def test_empty_question_rejected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError):
        assistant.ask("   ", "en", None, post=lambda *a, **k: FakeRes())


def test_upstream_failure_is_a_calm_502(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")

    def boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(assistant, "ask", boom)
    r = TestClient(api.app).post("/assistant", json={"question": "why?"})
    assert r.status_code == 502 and "could not answer" in r.json()["detail"]


def test_oversized_question_is_422():
    r = TestClient(api.app).post("/assistant", json={"question": "x" * 5000})
    assert r.status_code == 422
