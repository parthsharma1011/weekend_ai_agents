from __future__ import annotations

import json
from pathlib import Path

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict


class MemoryManager:
    """Per-session conversation memory: buffer + windowing + disk persistence."""

    def __init__(
        self,
        session_id: str = "default",
        memory_dir: str = "./memory_store",
        window: int = 10,
    ) -> None:
        self.session_id = session_id
        self.window = window
        self._dir = Path(memory_dir)
        self._buffer = InMemoryChatMessageHistory()
        self.load()  # resume a previous session if one exists

    # -- where this session lives on disk -------------------------------------
    @property
    def _path(self) -> Path:
        return self._dir / f"{self.session_id}.json"

    # -- writing to memory -----------------------------------------------------
    def add_user_message(self, content: str) -> None:
        self._buffer.add_user_message(content)
        self.save()

    def add_ai_message(self, content: str) -> None:
        self._buffer.add_ai_message(content)
        self.save()

    # -- reading from memory ---------------------------------------------------
    @property
    def all_messages(self) -> list[BaseMessage]:
        """The full conversation (every turn ever, this session)."""
        return list(self._buffer.messages)

    def window_messages(self) -> list[BaseMessage]:
        """Only the most recent `window` messages — what the LLM actually sees."""
        return self.all_messages[-self.window :] if self.window else self.all_messages

    # -- persistence -----------------------------------------------------------
    def save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        payload = messages_to_dict(self._buffer.messages)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            for msg in messages_from_dict(payload):
                self._buffer.add_message(msg)
        except Exception as exc:  # corrupt file shouldn't crash the app
            print(f"[memory] Could not load session '{self.session_id}': {exc}")

    def clear(self) -> None:
        self._buffer.clear()
        if self._path.exists():
            self._path.unlink()
