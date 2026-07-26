"""Tests for OFFLINE_MODE — the agent answering with no LLM behind it.

The point of these is not just "does it return something". It is that offline
mode must never touch Gemini: the Providers stub below raises if anything reads
its .llm property, so any accidental LLM construction fails the test loudly
rather than silently costing a quota call in production.
"""

import pytest
from langchain_core.messages import HumanMessage

from agent import NOTHING_FOUND, RagAgent
from config import Settings


# --- Stubs -------------------------------------------------------------------
class _ExplodingProviders:
    """Fails the test if the agent tries to build an LLM."""

    @property
    def llm(self):
        raise AssertionError("offline mode must never construct the Gemini client")

    def callbacks(self) -> list:
        return []


class _FakeRetriever:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def retrieve(self, query: str) -> str:
        return self._text


class _FakeWebTool:
    def __init__(self, available: bool = False, result: str = "") -> None:
        self.available = available
        self._result = result

    def search(self, query: str) -> str:
        return self._result


def _run(retriever, web_tool=None) -> str:
    graph = RagAgent(
        _ExplodingProviders(),
        retriever,
        web_tool or _FakeWebTool(),
        critic=None,
        offline=True,
    ).build()
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="what was the efficacy rate of T001")],
            "chat_history": [],
            "context": "",
            "answer": "",
            "critique_passed": False,
        }
    )
    return result["answer"]


# --- Behaviour ---------------------------------------------------------------
def test_offline_agent_answers_from_the_documents():
    answer = _run(_FakeRetriever("The efficacy rate of trial T001 was 78%."))
    assert "78%" in answer


def test_offline_agent_says_so_rather_than_pretending_to_generate():
    """The passages are quoted, so the UI must not imply they were written."""
    answer = _run(_FakeRetriever("The efficacy rate of trial T001 was 78%."))
    assert "no language model" in answer.lower()


def test_offline_agent_reports_when_nothing_matched():
    assert _run(_FakeRetriever("")) == NOTHING_FOUND


def test_offline_agent_still_uses_web_fallback_when_local_retrieval_is_empty():
    """Only Gemini is removed; Tavily is a separate, optional tool."""
    answer = _run(
        _FakeRetriever(""),
        _FakeWebTool(available=True, result="[Web] T001: efficacy reported at 78%."),
    )
    assert "78%" in answer


def test_building_the_offline_agent_touches_no_llm():
    """Constructing must be lazy too — this is what breaks if the guard is lost."""
    RagAgent(
        _ExplodingProviders(), _FakeRetriever("x"), _FakeWebTool(), critic=None, offline=True
    ).build()


def test_online_agent_still_requires_an_llm():
    """The offline flag must not quietly disable generation for normal runs."""
    with pytest.raises(AssertionError):
        RagAgent(
            _ExplodingProviders(), _FakeRetriever("x"), _FakeWebTool(), critic=None
        )


# --- Configuration -----------------------------------------------------------
def test_offline_mode_defaults_to_off():
    assert Settings().offline_mode is False


def test_offline_mode_reads_the_env_var(monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "true")
    assert Settings().offline_mode is True


def test_providers_refuses_to_build_an_llm_in_offline_mode():
    from config import Providers

    providers = Providers(Settings(offline_mode=True, gemini_api_key="AIzaFakeKey"))
    with pytest.raises(RuntimeError, match="OFFLINE_MODE"):
        _ = providers.llm
