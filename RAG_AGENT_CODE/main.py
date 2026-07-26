#std lib
from __future__ import annotations #apart from future ignore this for now
import os
import traceback # 1-1.5 exactly 
from langchain_core.messages import HumanMessage

# These lib are internal calls to logic we wrote
from agent import RagAgent
from config import Providers, Settings
from guardrails import GuardrailEngine
from memory import MemoryManager
from retrieval import Retriever, WebSearchTool
from self_critique import SelfCritic


_BANNER ="""
+--------------------------------------------------------------+
| GEMINI RAG AGENT (LANGCHAIN + LANGGRAPH)                     |
| Ask anything about a document in this chatbot                |
| Type 'quit' or 'exit' to end the session                     |
+--------------------------------------------------------------+
"""


class ChatApp:
    def __init__(self, session_id="default"):
        self.settings = Settings()
        self.providers = Providers(self.settings)
        #safety + memory layers
        self.guardrails = GuardrailEngine.default(self.settings.use_guardrails_ai)
        self.memory = MemoryManager(
            session_id=session_id,
            memory_dir=self.settings.memory_dir,
            window=self.settings.memory_window,
        )
        #retrieve
        retriever = Retriever(
            self.providers.embeddings,
            index_path=self.providers.settings.faiss_index_path,
            top_k = self.settings.top_k,
            min_score=self.settings.retrieval_min_score,
            verbose=False #set true once ro see score and calibrate you min_score
        )
        web_tool = WebSearchTool(
            enabled = self.settings.tavily_enabled,
            api_key = self.settings.tavily_api_key
        )
        # Offline mode: no Gemini calls at all, so the critic (which builds an
        # LLM chain up front) must not be constructed either.
        offline = self.settings.offline_mode
        critic = None if offline else SelfCritic(self.providers)
        if offline:
            print("[main] OFFLINE_MODE enabled — answers come straight from the documents.")
        self.graph = RagAgent(
            self.providers, retriever, web_tool, critic, offline=offline
        ).build()
        
    def _handle_turn(self, user_input):
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            print(f"[Guardrail] Input Blocked!!!")
            return
        if checked.reason:
            print(f"[Guardrail] {checked.reason}")
        clean_input = checked.text
        
        state = {
            "messages":[HumanMessage(content=clean_input)],
            "chat_history":self.memory.window_messages(),
            "context":"",
            "answer":"",
            "critique_passed":False,
        }
        try:
            result = self.graph.invoke(state)
        except Exception as e:
            print(f"[Error] Exception during graph invocation: {e}")
            traceback.print_exc()
            return 
    
        answer = result.get("answer","")
        checked_out = self.guardrails.check_output(answer)
        final_answer = (
            answer if checked_out.allowed else "[Guardrail] Output blocked."
        )
        print(f"\nAgent : {final_answer}")
        self.memory.add_user_message(clean_input)
        self.memory.add_ai_message(final_answer)
        
    def run(self):
        print(_BANNER)
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n[main] Session ended.")
                break
            
            if not user_input:
                print("[main] Empty input. Please try again.")
                continue
            if user_input.lower() in {"quit", "exit"}:
                print(f"\n[main] Session ended.")
                break
            
            self._handle_turn(user_input)
            
def main():
    ChatApp().run()
    
    
if __name__ == "__main__":
    main()


#web search -> fired -> when the local context comes back empty but the problem 
# Faiss -> never returns empty chunks top_k= 1234 -> irrevant 
# simi = 0.90 -> less do not ?
# top_k = 4




