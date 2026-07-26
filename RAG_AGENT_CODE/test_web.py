"""Tests for the HTTP layer's safety pieces: sessions, rate limit, client IP.

Like test_guardrails.py these need no API key and no network. Importing app.py
does not build the agent (that happens in the lifespan handler), and every unit
below is deliberately shaped to be testable on its own — RateLimiter.check takes
an injectable `now`, so the window can be tested without sleeping.
"""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import app as app_module
from app import (
    GENERIC_FAILURE,
    RateLimiter,
    SessionStore,
    client_key,
    describe_failure,
    new_session_id,
    valid_session_id,
)


# --- Session ids -------------------------------------------------------------
def test_minted_session_id_is_valid():
    assert valid_session_id(new_session_id())


def test_session_ids_are_unique():
    assert new_session_id() != new_session_id()


def test_path_traversal_ids_are_rejected():
    """The id becomes a filename, so traversal must never pass the gate."""
    for bad in ("../../etc/passwd", "../web", "a/b", "web.json", ".."):
        assert not valid_session_id(bad), bad


def test_malformed_ids_are_rejected():
    assert not valid_session_id(None)
    assert not valid_session_id("")
    assert not valid_session_id("deadbeef")                 # too short
    assert not valid_session_id("f" * 33)                   # too long
    assert not valid_session_id("F" * 32)                   # not lowercase
    assert not valid_session_id("g" * 32)                   # not hex


# --- Rate limiter ------------------------------------------------------------
def test_requests_under_the_limit_are_allowed():
    limiter = RateLimiter(limit=3, window_seconds=60)
    assert limiter.check("ip", now=0) == 0
    assert limiter.check("ip", now=1) == 0
    assert limiter.check("ip", now=2) == 0


def test_request_over_the_limit_is_blocked_with_a_wait():
    limiter = RateLimiter(limit=2, window_seconds=60)
    limiter.check("ip", now=0)
    limiter.check("ip", now=1)
    retry_after = limiter.check("ip", now=2)
    assert retry_after > 0
    assert retry_after == 58  # oldest hit at t=0 leaves the window at t=60


def test_window_slides_so_the_client_recovers():
    limiter = RateLimiter(limit=2, window_seconds=60)
    limiter.check("ip", now=0)
    limiter.check("ip", now=1)
    assert limiter.check("ip", now=30) > 0      # still inside the window
    assert limiter.check("ip", now=61) == 0     # first hit has aged out


def test_clients_are_limited_independently():
    limiter = RateLimiter(limit=1, window_seconds=60)
    assert limiter.check("ip-a", now=0) == 0
    assert limiter.check("ip-b", now=0) == 0    # b is not punished for a
    assert limiter.check("ip-a", now=0) > 0


def test_limiter_memory_is_bounded():
    limiter = RateLimiter(limit=5, window_seconds=60, max_keys=10)
    for i in range(50):
        limiter.check(f"ip-{i}", now=0)
    assert len(limiter._hits) <= 10


# --- Client identity ---------------------------------------------------------
def _request(headers: dict, client_host: str = "10.0.0.1") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/api/chat",
            "query_string": b"",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in headers.items()
            ],
            "client": (client_host, 1234),
        }
    )


def test_direct_connection_uses_the_socket_address():
    assert client_key(_request({})) == "10.0.0.1"


def test_forwarded_for_is_preferred_over_the_proxy_address():
    req = _request({"x-forwarded-for": "203.0.113.9"})
    assert client_key(req) == "203.0.113.9"


def test_spoofed_forwarded_for_cannot_bypass_the_limit():
    """A client can forge the leftmost entry; the proxy appends the real one last.

    Taking the rightmost value means every forged prefix maps to the same key,
    so spoofing does not mint a fresh quota.
    """
    forged_a = _request({"x-forwarded-for": "1.1.1.1, 203.0.113.9"})
    forged_b = _request({"x-forwarded-for": "2.2.2.2, 203.0.113.9"})
    assert client_key(forged_a) == client_key(forged_b) == "203.0.113.9"


# --- Session store -----------------------------------------------------------
def test_each_session_gets_its_own_memory(tmp_path):
    store = SessionStore(str(tmp_path), window=10, max_sessions=10, max_stored_messages=40)
    a_mem, a_lock = store.get("a" * 32)
    b_mem, b_lock = store.get("b" * 32)

    assert a_mem is not b_mem
    assert a_lock is not b_lock

    a_mem.add_user_message("only visible to a")
    assert b_mem.all_messages == []   # the bug that started this: no cross-talk


def test_same_session_is_reused(tmp_path):
    store = SessionStore(str(tmp_path), window=10, max_sessions=10, max_stored_messages=40)
    first, _ = store.get("c" * 32)
    second, _ = store.get("c" * 32)
    assert first is second


def test_session_count_is_capped(tmp_path):
    store = SessionStore(str(tmp_path), window=10, max_sessions=3, max_stored_messages=40)
    for i in range(10):
        store.get(f"{i:032x}")
    assert len(store) == 3


def test_evicted_sessions_do_not_leave_files_behind(tmp_path):
    store = SessionStore(str(tmp_path), window=10, max_sessions=2, max_stored_messages=40)
    victim, _ = store.get("d" * 32)
    victim.add_user_message("write a file to disk")
    assert (tmp_path / f"{'d' * 32}.json").exists()

    store.get("e" * 32)
    store.get("f" * 32)   # pushes the victim out of the LRU

    assert not (tmp_path / f"{'d' * 32}.json").exists()


# --- History cap -------------------------------------------------------------
def test_stored_history_is_trimmed(tmp_path):
    store = SessionStore(str(tmp_path), window=10, max_sessions=5, max_stored_messages=4)
    memory, _ = store.get("1" * 32)
    for i in range(10):
        memory.add_user_message(f"message {i}")

    assert len(memory.all_messages) == 4
    assert memory.all_messages[-1].content == "message 9"   # newest kept
    assert memory.all_messages[0].content == "message 6"    # oldest dropped


# --- The endpoint itself -----------------------------------------------------
class _StubService:
    """Stands in for ChatService so these tests need no API key and no LLM call.

    We record the session id every call arrives with — that is what proves the
    endpoint is actually separating visitors.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def ask(self, message: str, session_id: str) -> dict:
        self.calls.append((message, session_id))
        return {"answer": f"echo: {message}", "blocked": False}


@pytest.fixture
def client(monkeypatch):
    """A TestClient with the agent stubbed out.

    Built WITHOUT `with TestClient(...)` on purpose: that form runs the lifespan
    handler, which would construct the real agent (FAISS index + Gemini client).
    """
    stub = _StubService()
    monkeypatch.setattr(app_module, "service", stub)
    monkeypatch.setattr(app_module, "limiter", RateLimiter(limit=100, window_seconds=60))
    return TestClient(app_module.app), stub


def test_chat_returns_an_answer_and_sets_a_session_cookie(client):
    http, _ = client
    res = http.post("/api/chat", json={"message": "what is in the document"})

    assert res.status_code == 200
    assert res.json()["answer"] == "echo: what is in the document"
    assert valid_session_id(res.cookies[app_module.SESSION_COOKIE])


def test_a_visitor_keeps_one_session_across_turns(client):
    http, stub = client
    http.post("/api/chat", json={"message": "first question here"})
    http.post("/api/chat", json={"message": "second question here"})

    assert len({session for _, session in stub.calls}) == 1


def test_two_visitors_get_separate_sessions(client):
    """The original bug: one shared conversation for the whole internet."""
    http, stub = client
    other = TestClient(app_module.app)   # a second browser, its own cookie jar

    http.post("/api/chat", json={"message": "question from visitor one"})
    other.post("/api/chat", json={"message": "question from visitor two"})

    assert stub.calls[0][1] != stub.calls[1][1]


def test_a_forged_session_cookie_is_replaced(client):
    """A traversal attempt must not reach MemoryManager's file path."""
    http, stub = client
    http.cookies.set(app_module.SESSION_COOKIE, "../../etc/passwd")
    http.post("/api/chat", json={"message": "a perfectly normal question"})

    assert valid_session_id(stub.calls[0][1])


def test_oversized_messages_are_rejected(client):
    http, stub = client
    res = http.post("/api/chat", json={"message": "x" * 50_000})

    assert res.status_code == 422
    assert "answer" in res.json()      # the shape the frontend renders
    assert res.json()["blocked"] is True
    assert stub.calls == []            # never reached the agent


def test_rate_limited_clients_get_429_and_retry_after(monkeypatch):
    stub = _StubService()
    monkeypatch.setattr(app_module, "service", stub)
    monkeypatch.setattr(app_module, "limiter", RateLimiter(limit=2, window_seconds=60))
    http = TestClient(app_module.app)

    assert http.post("/api/chat", json={"message": "question number one"}).status_code == 200
    assert http.post("/api/chat", json={"message": "question number two"}).status_code == 200

    res = http.post("/api/chat", json={"message": "question number three"})
    assert res.status_code == 429
    assert int(res.headers["Retry-After"]) > 0
    assert len(stub.calls) == 2        # the third never cost us an LLM call


def test_healthz_reports_ready(client):
    http, _ = client
    assert http.get("/healthz").json() == {"status": "ok", "ready": True}


def test_favicon_does_not_404(client):
    http, _ = client
    assert http.get("/favicon.ico").status_code == 204


# --- Failure classification --------------------------------------------------
class _FakeUpstreamError(Exception):
    pass


def test_quota_errors_tell_the_user_to_wait():
    """The real Gemini failure: grpc RESOURCE_EXHAUSTED with a quota body."""
    exc = _FakeUpstreamError(
        "status = StatusCode.RESOURCE_EXHAUSTED details = 'You exceeded your "
        "current quota, please check your plan and billing details.'"
    )
    message = describe_failure(exc)
    assert "quota" in message.lower()
    assert "try again" in message.lower()


def test_auth_errors_point_at_configuration():
    exc = _FakeUpstreamError("400 API key not valid. Please pass a valid API key.")
    message = describe_failure(exc)
    assert "api key" in message.lower()


def test_unknown_errors_stay_generic():
    assert describe_failure(ValueError("index shard 7 at /opt/render/src")) == GENERIC_FAILURE


def test_no_classification_leaks_the_raw_exception():
    """Whatever branch is taken, upstream text must never reach the client."""
    secrets = [
        "/opt/render/project/src/RAG_AGENT_CODE/faiss_index",
        "AIzaSyExampleKeyMaterial",
        "debug_error_string",
    ]
    for raw in secrets:
        for exc in (ValueError(raw), _FakeUpstreamError(f"quota exceeded {raw}")):
            assert raw not in describe_failure(exc)
