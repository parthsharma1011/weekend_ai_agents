"""FastAPI wrapper around the existing LangGraph RAG agent.

This file adds an HTTP layer. It does not change agent behaviour: it builds the
same objects main.py builds, and runs the same graph. The only difference is
that a turn returns a string instead of printing it.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import RagAgent
from config import Providers, Settings
from guardrails import GuardrailEngine
from memory import MemoryManager
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic

BASE_DIR = Path(__file__).parent


class ChatService:
    """The same wiring as ChatApp in main.py, but returns text instead of printing."""

    def __init__(self, session_id: str = "web") -> None:
        self.settings = Settings()
        self.providers = Providers(self.settings)

        self.guardrails = GuardrailEngine.default(self.settings.use_guardrails_ai)
        self.memory = MemoryManager(
            session_id=session_id,
            memory_dir=self.settings.memory_dir,
            window=self.settings.memory_window,
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
        critic = SelfCritic(self.providers)
        self.graph = RagAgent(self.providers, retriever, web_tool, critic).build()

    def ask(self, user_input: str) -> dict:
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            return {"answer": f"Blocked by guardrail: {checked.reason}", "blocked": True}

        state = {
            "messages": [HumanMessage(content=checked.text)],
            "chat_history": self.memory.window_messages(),
            "context": "",
            "answer": "",
            "critique_passed": False,
        }

        try:
            result = self.graph.invoke(state)
        except Exception as exc:
            return {"answer": f"Agent error: {exc}", "error": True}

        answer = result.get("answer", "")
        checked_out = self.guardrails.check_output(answer)
        final = answer if checked_out.allowed else "Output blocked by guardrail."

        self.memory.add_user_message(checked.text)
        self.memory.add_ai_message(final)
        return {"answer": final, "blocked": False}


# Built once at startup so the FAISS index and embedding model load a single time,
# not on every request.
service: ChatService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the agent once, before the first request is served."""
    global service
    service = ChatService()
    yield


app = FastAPI(title="Gemini RAG Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.get("/healthz")
def healthz() -> dict:
    """Render pings this to confirm the service is alive."""
    return {"status": "ok", "ready": service is not None}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if service is None:
        return {"answer": "Service still starting, try again in a moment.", "error": True}
    if not req.message.strip():
        return {"answer": "Please enter a question.", "blocked": True}
    return service.ask(req.message)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
