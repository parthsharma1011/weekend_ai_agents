from __future__ import annotations

import json
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import Providers

_SYSTEM = (
    "You are a strict technical reviewer. You are given a context (extracted "
    "from documents), a question, and an answer. Evaluate the answer ONLY "
    "against the provided context; do not introduce outside information.\n"
    "Respond with ONLY a valid JSON object (no markdown fences), matching:\n"
    '{{"score": <int 1-10>, '
    '"critique": "<one short sentence>", '
    '"improved_answer": "<rewritten answer grounded in context, or null if score >= 7>"}}'
)


class SelfCritic:
    """Grades and, when needed, rewrites an answer using the same LLM."""

    ACCEPT_THRESHOLD = 7 #why 

    def __init__(self, providers: Providers):
        self.providers = providers
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"),
            ]
        )
        self._chain = prompt | providers.llm | StrOutputParser()

    def critique(self, question: str, answer: str, context: str = "") -> str:
        """Return the improved answer if the draft scored low, else the draft."""
        try:
            raw = self._chain.invoke(
                {"context": context, "question": question, "answer": answer},
                config={"callbacks": self.providers.callbacks()},
            )
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(cleaned)

            score = int(result.get("score", 10))
            improved = result.get("improved_answer")

            if score < self.ACCEPT_THRESHOLD and improved and str(improved).lower() != "null":
                print(f"[self-critique] Score {score}/10 — replacing with improved answer.")
                return improved

            print(f"[self-critique] Score {score}/10 — answer accepted.")
            return answer
        except Exception as exc:
            print(f"[self-critique] Skipped (keeping original answer): {exc}")
            return answer
