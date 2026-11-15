"""Assistant: the Bedrock call is replaced, so no network and no key are needed."""
import pytest
from fastapi.testclient import TestClient

from asf_cleaner import api, assistant


class Res:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"output": {"message": {"role": "assistant", "content": self.content}}}


def tool(actions, text=None):
    c = [{"toolUse": {"toolUseId": "t1", "name": "control_map", "input": {"actions": actions}}}]
    return Res(([{"text": text}] if text else []) + c)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "secret")
    monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
    monkeypatch.delenv("ASF_ASSISTANT_MODEL", raising=False)


def test_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("BEDROCK_API_KEY")
    monkeypatch.setattr(assistant, "env_value", lambda name: None)
    r = TestClient(api.app).post("/assistant", json={"question": "why?"})
    assert r.status_code == 200 and r.json() == {"configured": False, "answer": None, "actions": []}


def test_request_goes_to_bedrock_converse_with_bearer_key():
    seen = {}

    def post(url, json, timeout, headers):
        seen.update(url=url, body=json, headers=headers)
        return Res([{"text": "Hello."}])

    out = assistant.ask("hi", "pt", "No place selected", post=post)
    assert out == {"answer": "Hello.", "actions": []}
    assert seen["url"] == "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-lite-v1%3A0/converse"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert "Brazilian Portuguese" in seen["body"]["system"][0]["text"] and "language the person writes in" in seen["body"]["system"][0]["text"] and "No place selected" in seen["body"]["system"][0]["text"]
    assert seen["body"]["toolConfig"]["tools"][0]["toolSpec"]["name"] == "control_map"


def test_actions_are_checked_and_clamped():
    acts = [
        {"type": "layer", "id": "imerg", "on": True},
        {"type": "layer", "id": "not-a-layer", "on": True},       # unknown id: dropped
        {"type": "region", "id": "amazon"},
        {"type": "region", "id": "../etc"},                       # dropped
        {"type": "fly", "lon": 10, "lat": 80, "zoom": 99},        # outside the map: clamped
        {"type": "disaster", "kind": "earthquake", "magnitude": 99, "lat": -12, "lon": -77},
        {"type": "year", "year": 1999},
        {"type": "rm -rf"},                                       # unknown type: dropped
        {"type": "tab", "tab": "forecast"},
        {"type": "base", "id": "bluemarble"},
    ]
    got = assistant.sanitize(acts)
    assert {"type": "layer", "id": "imerg", "on": True} in got
    assert all(a.get("id") not in ("not-a-layer", "../etc") for a in got)
    fly = next(a for a in got if a["type"] == "fly")
    assert fly["lon"] == -30.0 and fly["lat"] == 24.0 and fly["zoom"] == 15.0
    quake = next(a for a in got if a["type"] == "disaster")
    assert quake["params"]["magnitude"] == 9.0 and quake["lat"] == -12
    assert next(a for a in got if a["type"] == "year")["year"] == 2016
    assert len(got) <= assistant.MAX_ACTIONS


def test_thinking_tags_are_removed():
    out = assistant.ask("q", "en", "", post=lambda *a, **k: Res([{"text": "<thinking>secret plan</thinking>\nThe rain layer is on."}]))
    assert out["answer"] == "The rain layer is on."


def test_tool_only_reply_gets_a_second_round_for_the_sentence():
    calls = []

    def post(url, json, timeout, headers):
        calls.append(json)
        return tool([{"type": "layer", "id": "imerg", "on": True}]) if len(calls) == 1 else Res([{"text": "Rain is on."}])

    out = assistant.ask("show rain", "en", "", post=post)
    assert out["answer"] == "Rain is on." and out["actions"] == [{"type": "layer", "id": "imerg", "on": True}]
    # the second round carries the tool result back
    assert any("toolResult" in b for m in calls[1]["messages"] for b in m["content"])


def test_fixed_sentence_when_the_second_round_fails():
    def post(url, json, timeout, headers):
        if "toolResult" in str(json["messages"]):
            raise RuntimeError("boom")
        return tool([{"type": "layer", "id": "imerg", "on": False}])

    out = assistant.ask("hide rain", "pt", "", post=post)
    assert out["answer"] == "chuva desligada." and out["actions"][0]["on"] is False


def test_history_alternates_and_is_bounded():
    q, st, turns = assistant.clean("x" * 5000, "y" * 9000, [
        {"role": "user", "text": "a"}, {"role": "user", "text": "dup"}, {"role": "assistant", "text": "b"}, {"role": "user", "text": "c"}] * 5)
    assert len(q) == assistant.MAX_QUESTION and len(st) == assistant.MAX_CONTEXT
    roles = [t["role"] for t in turns]
    assert roles[0] == "user" and roles[-1] == "assistant" and all(a != b for a, b in zip(roles, roles[1:]))
    assert len(turns) <= assistant.MAX_HISTORY


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        assistant.ask("   ", "en", "", post=lambda *a, **k: Res([]))


def test_route_returns_answer_and_actions(monkeypatch):
    monkeypatch.setattr(assistant, "ask", lambda q, lang, ctx, hist: {"answer": "ok", "actions": [{"type": "forecast"}]})
    r = TestClient(api.app).post("/assistant", json={"question": "run it", "lang": "en", "context": "x", "history": [{"role": "user", "text": "hi"}]})
    assert r.json() == {"configured": True, "answer": "ok", "actions": [{"type": "forecast"}]}


def test_upstream_failure_is_a_calm_502(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota")

    monkeypatch.setattr(assistant, "ask", boom)
    r = TestClient(api.app).post("/assistant", json={"question": "why?"})
    assert r.status_code == 502 and "could not answer" in r.json()["detail"]


def test_oversized_question_is_422():
    assert TestClient(api.app).post("/assistant", json={"question": "x" * 5000}).status_code == 422


def test_rate_limit_per_client():
    rl = assistant.RateLimit(3)
    assert [rl.allow("a", now=i) for i in range(4)] == [True, True, True, False]
    assert rl.allow("b", now=1) and rl.allow("a", now=100)  # another client, and a new minute
