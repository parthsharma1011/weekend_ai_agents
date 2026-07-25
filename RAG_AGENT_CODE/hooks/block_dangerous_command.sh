#!/usr/bin/env bash
# PreToolUse hook (Bash) — refuse commands that are hard to undo, plus the one
# command that would quietly break the Day 3 experiment: `pip install`.
#
# `pip install` is here on purpose. The worksheet's fence says "NO new
# dependency — no cross-encoder, no sentence-transformers." An agent asked to
# "improve retrieval quality" reaches for a reranker model within two turns.
# Blocking the install turns that fence from a sentence into a wall, and the
# denial message becomes the teaching moment.
#
# `git reset --hard` is deliberately NOT blocked — it is the lab's rewind move.
#
# Contract: exit 0 always. Silence = allow. A JSON body on stdout = deny.
set -euo pipefail

# The heredoc below becomes python's stdin, so the hook payload travels in an
# env var — `json.load(sys.stdin)` would read the script, not the JSON.
CLAUDE_HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, re, sys

try:
    payload = json.loads(os.environ.get("CLAUDE_HOOK_INPUT") or "{}")
except Exception:
    sys.exit(0)

if payload.get("tool_name") != "Bash":
    sys.exit(0)

cmd = (payload.get("tool_input", {}) or {}).get("command", "")

# (regex, why it is blocked)
RULES = [
    (r"\brm\s+(-\w*\s+)*-\w*[rR]\w*f|\brm\s+(-\w*\s+)*-\w*f\w*[rR]",
     "Recursive force-delete. Rewind with `git reset --hard checkpoint-clean` instead."),
    (r"\bsudo\b",
     "No command in this sandbox needs root."),
    (r"\bgit\s+push\b.*(--force|-f)\b",
     "Force-push rewrites shared history. Not part of any Day 3 slice."),
    (r"\bgit\s+clean\b.*-\w*[dxX]",
     "This deletes untracked files, including your FAISS index. Use `git reset --hard checkpoint-clean`."),
    (r"\bchmod\s+(-\w+\s+)*777\b",
     "World-writable permissions. Not needed here."),
    (r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b",
     "Piping a download straight into a shell. Never in a classroom sandbox."),
    (r"\b(pip3?|uv\s+pip)\s+install\b(?!.*-r\s+requirements\.txt)",
     "OUT OF SCOPE FENCE: no new dependency. Re-ranking must be pure Python in "
     "RAG/rerank.py — no cross-encoder, no sentence-transformers. "
     "(`pip install -r requirements.txt` is allowed.)"),
    (r":\(\)\s*\{.*\|.*&.*\}\s*;?\s*:",
     "Fork bomb."),
]

for pattern, reason in RULES:
    if re.search(pattern, cmd):
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Blocked `{cmd.strip()[:80]}` — {reason}",
            }
        }))
        sys.exit(0)
PY
