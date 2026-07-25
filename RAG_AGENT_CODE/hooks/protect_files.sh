#!/usr/bin/env bash
# PreToolUse hook — the Day 3 out-of-scope FENCE, enforced mechanically.
#
# The worksheet says "the agent must not touch ingest/embeddings, guardrails,
# or self-critique." A worksheet is a promise. This hook is the enforcement.
# That difference is the whole point of blast-radius control: an intention the
# tool cannot violate beats an intention it merely agreed to.
#
# In scope for Day 3 (editable):   RAG/agent.py, RAG/rerank.py, RAG/test/
# Out of scope (blocked here):     the files listed in PROTECTED below.
#
# Honest limitation, worth saying out loud in class: a path-based hook cannot
# protect a *function*. `build_graph()` lives inside agent.py, which learners
# must edit for slice 2. The graph-wiring fence stays a human check —
# `git diff RAG/agent.py` in the L3 proof step.
#
# Contract: exit 0 always. Silence = allow. A JSON body on stdout = deny.
set -euo pipefail

# The heredoc below becomes python's stdin, so the hook payload travels in an
# env var — `json.load(sys.stdin)` would read the script, not the JSON.
CLAUDE_HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, sys

PROTECTED = (
    "RAG/guardrails.py",     # Day 13 exhibit — do not edit
    "RAG/self_critique.py",  # Day 13 exhibit — do not edit
    "RAG/ingest.py",         # fence: no ingest changes
    "RAG/config.py",         # fence: no embedding/LLM changes
    "RAG/offline_stubs.py",  # fence: the stubs are the measurement instrument
)

try:
    payload = json.loads(os.environ.get("CLAUDE_HOOK_INPUT") or "{}")
except Exception:
    sys.exit(0)

tool = payload.get("tool_name", "")
if tool not in ("Edit", "Write", "NotebookEdit"):
    sys.exit(0)

tool_input = payload.get("tool_input", {}) or {}
path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
path = path.replace("\\", "/")

for protected in PROTECTED:
    if path.endswith(protected):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    f"{protected} is OUTSIDE the Day 3 fence. Re-ranking must land in "
                    "RAG/rerank.py and RAG/agent.py only. If you believe this file must "
                    "change, stop and say why — do not work around the fence."
                ),
            }
        }))
        sys.exit(0)
PY
