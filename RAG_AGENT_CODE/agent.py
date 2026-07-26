from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import Providers
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic


# --- Shared state passed between nodes ---------------------------------------
# `add_messages` is a reducer: new messages are APPENDED, not overwritten.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    chat_history: list[BaseMessage]
    context: str
    answer: str
    critique_passed: bool


_SYSTEM = (
    "You are a helpful assistant. The context below is content extracted from "
    "documents that have already been uploaded and processed — treat it as THE "
    "document. Answer the question using ONLY the provided context.\n"
    "If the answer is present in the context, state it clearly. If it is "
    "genuinely not in the context, say: 'I could not find that in the provided "
    "documents.' Never say 'no document was provided' — the context below IS "
    "the document.\n\n"
    "Context:\n{context}"
)


NOTHING_FOUND = (
    "I could not find anything about that in the indexed documents. "
    "Try rephrasing, or ask about the Reagan biography or the T001 clinical "
    "trial report."
)

_OFFLINE_PREAMBLE = (
    "Answering directly from the indexed documents (no language model is "
    "running, so these are the matching passages verbatim):"
)


class RagAgent:
    """Builds and compiles the retrieval-augmented-generation graph."""

    def __init__(
        self,
        providers: Providers,
        retriever: Retriever,
        web_tool: WebSearchTool,
        critic: SelfCritic | None = None,
        offline: bool = False,
    ):
        self.providers = providers
        self.retriever = retriever
        self.web_tool = web_tool
        self.critic = critic
        self.offline = offline

        if offline:
            # Deliberately do NOT touch providers.llm: reading that property
            # constructs the Gemini client. In offline mode the key may be
            # absent entirely, and nothing here should require one.
            self._gen_chain = None
            return

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        # LCEL: prompt -> LLM -> plain-string output.
        self._gen_chain = prompt | providers.llm | StrOutputParser()

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def _latest_human(state: AgentState) -> str:
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""

    # -- nodes -----------------------------------------------------------------
    def retrieve_node(self, state: AgentState) -> dict:
        """Fetch top-k chunks from the local FAISS index."""
        question = self._latest_human(state)
        return {"context": self.retriever.retrieve(question) if question else ""}

    def tool_node(self, state: AgentState) -> dict:
        """Fallback: nothing local, so try the web (if Tavily is enabled)."""
        question = self._latest_human(state)
        if not self.web_tool.available:
            print("[agent] No web-search tool configured — skipping fallback.")
            return {"context": ""}
        print("[agent] No local context — falling back to web search.")
        web = self.web_tool.search(question)
        combined = (web + "\n\n" + state.get("context", "")).strip()
        return {"context": combined}

    @staticmethod
    def _extractive_answer(context: str) -> str:
        """Offline answer: hand back the retrieved passages themselves.

        Without an LLM there is nothing to synthesise prose from, so the honest
        thing is to show the source text and say that is what we are doing,
        rather than dress it up as a generated answer.
        """
        context = context.strip()
        if not context:
            return NOTHING_FOUND
        return f"{_OFFLINE_PREAMBLE}\n\n{context}"

    def generate_node(self, state: AgentState) -> dict:
        """Produce the answer — from the LLM, or from the passages if offline."""
        if self.offline:
            return {"answer": self._extractive_answer(state.get("context", ""))}

        answer = self._gen_chain.invoke(
            {
                "context": state.get("context", ""),
                "chat_history": state.get("chat_history", []),
                "question": self._latest_human(state),
            },
            config={"callbacks": self.providers.callbacks()},
        )
        return {"answer": answer}

    def critique_node(self, state: AgentState) -> dict:
        """Second LLM pass: grade and, if needed, rewrite the answer."""
        final = self.critic.critique(
            self._latest_human(state), state["answer"], state.get("context", "")
        )
        return {"answer": final, "critique_passed": True}

    def output_node(self, state: AgentState) -> dict:
        """Publish the final answer as an AIMessage on the message stream."""
        return {"messages": [AIMessage(content=state["answer"])]}

    # -- routing ---------------------------------------------------------------
    def _route_after_retrieve(self, state: AgentState) -> str:
        return "generate_node" if state.get("context", "").strip() else "tool_node"

    # -- assemble --------------------------------------------------------------
    def build(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve_node", self.retrieve_node)
        g.add_node("tool_node", self.tool_node)
        g.add_node("generate_node", self.generate_node)
        g.add_node("output_node", self.output_node)

        g.add_edge(START, "retrieve_node")
        g.add_conditional_edges(
            "retrieve_node",
            self._route_after_retrieve,
            {"tool_node": "tool_node", "generate_node": "generate_node"},
        )
        g.add_edge("tool_node", "generate_node")

        if self.offline:
            # The critique step is a second LLM pass, so it has nothing to do
            # offline — and including it would double the cost of every turn
            # for a model that is not there.
            g.add_edge("generate_node", "output_node")
        else:
            g.add_node("critique_node", self.critique_node)
            g.add_edge("generate_node", "critique_node")
            g.add_edge("critique_node", "output_node")

        g.add_edge("output_node", END)
        return g.compile()
