from __future__ import annotations

from functools import cached_property

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# 1) Settings — typed configuration loaded from the environment / .env
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """All knobs for the app, validated and typed by pydantic-settings.

    Each attribute is auto-filled from an environment variable of the SAME
    name in upper-case (e.g. `gemini_api_key` <- GEMINI_API_KEY). Values in
    `.env` are loaded automatically. Unknown variables are ignored.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Google Gemini (the chat LLM — replaces AWS Bedrock) ---------------
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_tokens: int = 1024

    # --- Embeddings (local, free — no API key needed) ----------------------
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- Retrieval / vector store ------------------------------------------
    faiss_index_path: str = "./faiss_index"
    top_k: int = 4
    retrieval_min_score: float = 0.50 #cutoff usually number is 0.25 - 0.33
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Memory layer -------------------------------------------------------
    memory_dir: str = "./memory_store"
    memory_window: int = 10  # how many recent messages to feed back to the LLM

    # --- Optional web-search fallback (Tavily) -----------------------------
    tavily_api_key: str | None = None

    # --- Optional observability (Langfuse) ---------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Guardrails ---------------------------------------------------------
    # Try to use the "famous library" (Guardrails AI) if it is installed.
    use_guardrails_ai: bool = True

    # --- Web layer (app.py only; the CLI in main.py ignores all of these) ---
    # A public URL is a very different threat model from a local CLI: every turn
    # spends two LLM calls (generate + critique) against YOUR key, so the HTTP
    # layer has to bound request size, request rate, and per-visitor state.
    max_message_chars: int = 2000   # reject oversized bodies before tokenising
    rate_limit_per_minute: int = 10  # per client IP
    max_web_sessions: int = 200     # live conversations held in memory (LRU)
    max_stored_messages: int = 40    # per-session history cap, see MemoryManager

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)


# ---------------------------------------------------------------------------
# 2) Providers — build the live objects the rest of the app depends on
# ---------------------------------------------------------------------------
class Providers:
    """Factory for the heavyweight, shared objects (LLM, embeddings, tracer).

    We build each object once and cache it (`cached_property`), so importing
    this module is cheap and the model client is reused across the whole run.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    @cached_property
    def llm(self):
        """The Gemini chat model, via LangChain's Google GenAI integration."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not self.settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. Get a key from "
                "https://aistudio.google.com/apikey (it starts with 'AIza…') "
                "and add it to your .env file."
            )
        return ChatGoogleGenerativeAI(
            model=self.settings.gemini_model,
            google_api_key=self.settings.gemini_api_key,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

    @cached_property
    def embeddings(self):
        """Local sentence embeddings (FastEmbed). No API key, runs offline."""
        from langchain_community.embeddings import FastEmbedEmbeddings

        return FastEmbedEmbeddings(model_name=self.settings.embedding_model)

    @cached_property
    def langfuse_handler(self):
        """A LangChain callback that traces every LLM call, or None if unset."""
        if not self.settings.langfuse_enabled:
            return None
        try:
            from langfuse.callback import CallbackHandler

            return CallbackHandler(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
            )
        except Exception as exc:  # pragma: no cover - observability is optional
            print(f"[config] Langfuse disabled: {exc}")
            return None

    def callbacks(self) -> list:
        """Callback list to pass into every LLM `.invoke(config=...)` call."""
        return [self.langfuse_handler] if self.langfuse_handler else []
