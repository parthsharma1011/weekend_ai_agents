"""FastAPI wrapper around the existing LangGraph RAG agent.

This file adds an HTTP layer. It does not change agent behaviour: it builds the
same objects main.py builds, and runs the same graph. The only difference is
that a turn returns a string instead of printing it.

Everything beyond that — sessions, rate limiting, error handling — exists
because a public URL is not a local CLI. See the comments on each piece.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent import RagAgent
from config import Providers, Settings
from guardrails import GuardrailEngine
from memory import MemoryManager
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic

BASE_DIR = Path(__file__).parent

logger = logging.getLogger("app")

# Read once at import so request-model validation can use the limit below.
# Constructing Settings touches no network — it only reads env vars and .env.
SETTINGS = Settings()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
SESSION_COOKIE = "sid"

# Session ids are minted by US (uuid4().hex) and this pattern is the gate that
# keeps it that way. It matters because MemoryManager turns the id into a file
# path (`memory_dir/<id>.json`): a client-supplied id like "../../etc/foo" would
# otherwise write outside memory_store. 32 lowercase hex chars cannot traverse.
_SESSION_ID_RE = re.compile(r"[0-9a-f]{32}")


def valid_session_id(value: str | None) -> bool:
    """True only for an id in the exact format we issue."""
    return bool(value) and _SESSION_ID_RE.fullmatch(value) is not None


def new_session_id() -> str:
    return uuid.uuid4().hex


class SessionStore:
    """One conversation per visitor, capped in number, each with its own lock.

    Previously the whole app shared a single MemoryManager, so every visitor
    read and wrote one file: user B's turn was primed with user A's history, and
    A's questions could surface in B's answers. Keying by session fixes the
    privacy leak; the LRU cap keeps an unbounded number of visitors from
    becoming an unbounded number of open conversations.
    """

    def __init__(
        self,
        memory_dir: str,
        window: int,
        max_sessions: int,
        max_stored_messages: int,
    ) -> None:
        self._memory_dir = memory_dir
        self._window = window
        self._max_sessions = max_sessions
        self._max_stored_messages = max_stored_messages
        # session_id -> (memory, lock). Ordered so we can evict least-recent.
        self._sessions: OrderedDict[str, tuple[MemoryManager, threading.Lock]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def get(self, session_id: str) -> tuple[MemoryManager, threading.Lock]:
        """Return this session's memory and the lock that guards it."""
        evicted: list[MemoryManager] = []
        with self._lock:
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = (
                    MemoryManager(
                        session_id=session_id,
                        memory_dir=self._memory_dir,
                        window=self._window,
                        max_messages=self._max_stored_messages,
                    ),
                    threading.Lock(),
                )
                self._sessions[session_id] = entry
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                _, (stale, _stale_lock) = self._sessions.popitem(last=False)
                evicted.append(stale)

        # Delete the evicted session's file too, otherwise the disk keeps
        # growing after we have stopped tracking it. An evicted visitor simply
        # starts a fresh conversation. Done outside the registry lock: it is I/O.
        for stale in evicted:
            try:
                stale.clear()
            except OSError as exc:
                logger.warning("could not clear evicted session: %s", exc)

        return entry

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window request cap per client key.

    /api/chat is unauthenticated and each call costs two Gemini requests plus an
    optional Tavily search, all billed to the deploy's own keys. Without a cap,
    one script drains the quota. In-process only: with a single Render instance
    that is the whole picture, but note that it resets on restart and is not
    shared across instances if you ever scale past one.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
    ) -> None:
        self.limit = limit
        self.window = window_seconds
        self.max_keys = max_keys  # bounds the limiter's own memory use
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float:
        """Record a hit. Returns 0.0 if allowed, else seconds to wait."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            self._hits.move_to_end(key)

            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) >= self.limit:
                return max(0.0, hits[0] + self.window - now)

            hits.append(now)
            while len(self._hits) > self.max_keys:
                self._hits.popitem(last=False)
            return 0.0


def client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting.

    Behind Render's proxy, request.client.host is the proxy, so we read
    X-Forwarded-For. We take the RIGHTMOST entry on purpose: the header is
    "client, proxy1, proxy2", and a caller can send a forged X-Forwarded-For
    that the proxy then appends to. The leftmost value is therefore attacker
    controlled (a trivial rate-limit bypass); the rightmost is the one the
    trusted proxy in front of us added.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [part.strip() for part in forwarded.split(",") if part.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# The agent, wrapped
# ---------------------------------------------------------------------------
GENERIC_FAILURE = "The agent hit an internal error. Please try again."

# Matched against the lowercased exception text. Kept to distinctive phrases:
# bare status numbers like "429" would false-positive on ids and timings.
_QUOTA_MARKERS = (
    "resource_exhausted",
    "quota",
    "rate limit",
    "ratelimit",
    "too many requests",
)
_AUTH_MARKERS = (
    "permission_denied",
    "unauthenticated",
    "api key not valid",
    "invalid api key",
    "api_key_invalid",
)


def describe_failure(exc: BaseException) -> str:
    """Turn an upstream exception into a safe, actionable sentence.

    Returning the raw exception leaks filesystem paths and upstream error bodies,
    but a single generic string is useless to whoever has to fix it: "quota
    exhausted" and "your key is rejected" need completely different responses,
    and neither is a secret. So we classify coarsely and say nothing more.
    """
    # The class name is included because some clients (google-api-core) put the
    # useful word only in the type, e.g. ResourceExhausted, while grpc puts it
    # only in the message. Auth is checked first: it is the more specific
    # condition, and a rejected key can surface alongside quota wording.
    text = f"{type(exc).__name__} {exc}".lower()
    if any(marker in text for marker in _AUTH_MARKERS):
        return (
            "The assistant is not configured correctly — its API credentials "
            "were rejected. The site owner needs to check the API key."
        )
    if any(marker in text for marker in _QUOTA_MARKERS):
        return (
            "The assistant has used up its API quota for now. "
            "Please try again in a few minutes."
        )
    return GENERIC_FAILURE


class ChatService:
    """The same wiring as ChatApp in main.py, but returns text instead of printing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.providers = Providers(self.settings)

        self.guardrails = GuardrailEngine.default(self.settings.use_guardrails_ai)
        self.sessions = SessionStore(
            memory_dir=self.settings.memory_dir,
            window=self.settings.memory_window,
            max_sessions=self.settings.max_web_sessions,
            max_stored_messages=self.settings.max_stored_messages,
        )

        retriever = Retriever(
            self.providers.embeddings,
            index_path=self.settings.faiss_index_path,
            top_k=self.settings.top_k,
            min_score=self.settings.retrieval_min_score,
            verbose=False,
        )
        web_tool = WebSearchTool(
            enabled=self.settings.tavily_enabled,
            api_key=self.settings.tavily_api_key,
        )
        # SelfCritic builds an LLM chain in its constructor, so in offline mode
        # it must not be built at all.
        offline = self.settings.offline_mode
        critic = None if offline else SelfCritic(self.providers)
        if offline:
            logger.info("OFFLINE_MODE enabled — serving retrieval results, no LLM calls")
        self.graph = RagAgent(
            self.providers, retriever, web_tool, critic, offline=offline
        ).build()

    def ask(self, user_input: str, session_id: str) -> dict:
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            # checked.reason is our own guardrail copy, safe to show.
            return {"answer": f"Blocked by guardrail: {checked.reason}", "blocked": True}

        memory, lock = self.sessions.get(session_id)

        # Held across the whole turn, not just the writes: two overlapping turns
        # in one session would otherwise read the same history and append out of
        # order, interleaving the transcript. Per-session, so different visitors
        # still run concurrently.
        with lock:
            state = {
                "messages": [HumanMessage(content=checked.text)],
                "chat_history": memory.window_messages(),
                "context": "",
                "answer": "",
                "critique_passed": False,
            }

            try:
                result = self.graph.invoke(state)
            except Exception as exc:
                # Full detail to the log, a classified sentence to the caller:
                # the exception text can carry filesystem paths and upstream API
                # error bodies, but "out of quota" vs "key rejected" is both safe
                # to say and the only part anyone can act on.
                logger.exception("graph invocation failed")
                return {"answer": describe_failure(exc), "error": True}

            answer = result.get("answer", "")
            checked_out = self.guardrails.check_output(answer)
            final = answer if checked_out.allowed else "Output blocked by guardrail."

            memory.add_user_message(checked.text)
            memory.add_ai_message(final)

        return {"answer": final, "blocked": False}


# Built once at startup so the FAISS index and embedding model load a single time,
# not on every request.
service: ChatService | None = None
limiter = RateLimiter(limit=SETTINGS.rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the agent once, before the first request is served."""
    global service
    service = ChatService(SETTINGS)
    yield


app = FastAPI(title="Gemini RAG Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    # An unbounded string here means a caller can post megabytes and have us pay
    # to tokenise it. Rejected by pydantic before any of our code runs.
    message: str = Field(..., min_length=1, max_length=SETTINGS.max_message_chars)


@app.exception_handler(RequestValidationError)
async def on_invalid_request(request: Request, exc: RequestValidationError):
    """Keep the frontend's {"answer": ...} contract on validation failures.

    FastAPI's default 422 body has a different shape, which the UI would render
    as "undefined" rather than telling the user what went wrong.
    """
    return JSONResponse(
        status_code=422,
        content={
            "answer": (
                "That message could not be accepted — it must be between 1 and "
                f"{SETTINGS.max_message_chars} characters."
            ),
            "blocked": True,
        },
    )


@app.get("/healthz")
def healthz() -> dict:
    """Render pings this to confirm the service is alive."""
    return {"status": "ok", "ready": service is not None}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request, response: Response) -> dict:
    if service is None:
        response.status_code = 503
        return {"answer": "Service still starting, try again in a moment.", "error": True}

    retry_after = limiter.check(client_key(request))
    if retry_after > 0:
        response.status_code = 429
        response.headers["Retry-After"] = str(max(1, int(retry_after) + 1))
        return {
            "answer": (
                "Too many requests. Please wait about "
                f"{max(1, int(retry_after) + 1)}s and try again."
            ),
            "error": True,
        }

    message = req.message.strip()
    if not message:
        return {"answer": "Please enter a question.", "blocked": True}

    # Reuse the visitor's session if they already have a valid one, otherwise
    # mint one and pin it to a cookie. HttpOnly because no JS needs to read it;
    # Secure only over HTTPS so local http:// development still works.
    session_id = request.cookies.get(SESSION_COOKIE)
    if not valid_session_id(session_id):
        session_id = new_session_id()
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    response.set_cookie(
        SESSION_COOKIE,
        session_id,
        max_age=60 * 60 * 24 * 7,
        httponly=True,
        samesite="lax",
        secure=forwarded_proto == "https",
    )

    return service.ask(message, session_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Every browser asks for this; without a route each visit logs a 404."""
    return Response(status_code=204)


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
