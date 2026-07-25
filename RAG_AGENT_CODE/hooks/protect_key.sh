#!/usr/bin/env bash
# PreToolUse hook — the agent may USE the API key, never READ it.
#
# Blocks: Read/Edit/Write/Grep on .env, and any Bash command that would print
# its contents (cat/less/grep/strings/base64/...).
# Allows: .env.example, and code that goes through the loader:
#
#     from dotenv import load_dotenv; load_dotenv()
#     API_KEY = os.getenv("GEMINI_API_KEY")
#
# So `python3 -c "import os,dotenv; dotenv.load_dotenv(); print(bool(os.getenv('GEMINI_API_KEY')))"`
# still works and prints True — the agent proves the key loads without ever
# seeing its value.
#
# Contract: exit 0 always. Silence = allow. A JSON body on stdout = deny.
set -euo pipefail

# The heredoc below becomes python's stdin, so the hook payload has to travel
# in an env var — `json.load(sys.stdin)` would read the script, not the JSON.
CLAUDE_HOOK_INPUT="$(cat)" python3 - <<'PY'
import json, os, re, sys

try:
    payload = json.loads(os.environ.get("CLAUDE_HOOK_INPUT") or "{}")
except Exception:
    sys.exit(0)  # unparseable input is not this hook's problem

tool = payload.get("tool_name", "")
tool_input = payload.get("tool_input", {}) or {}


def deny(reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


SECRET = "Secrets are read-only to you. Use os.getenv() after load_dotenv() — never open .env."


def is_secret_path(path):
    if not path:
        return False
    base = os.path.basename(path.rstrip("/"))
    if base.endswith(".example") or base.endswith(".sample"):
        return False
    return base == ".env" or base.startswith(".env.")


# --- File tools ------------------------------------------------------------
if tool in ("Read", "Edit", "Write", "NotebookEdit", "Grep", "Glob"):
    for key in ("file_path", "path", "notebook_path"):
        if is_secret_path(tool_input.get(key, "")):
            deny(f"Blocked {tool} on {tool_input[key]}. {SECRET}")

# --- Bash ------------------------------------------------------------------
if tool == "Bash":
    cmd = tool_input.get("command", "")
    # Strip .env.example / .env.sample so they don't trip the .env match below.
    probe = re.sub(r"\.env\.(example|sample)\b", "", cmd)
    if re.search(r"(^|[\s/=\"'])\.env\b", probe):
        READERS = (
            "cat", "bat", "less", "more", "head", "tail", "grep", "rg", "ag",
            "awk", "sed", "strings", "xxd", "od", "base64", "cp", "mv", "scp",
            "rsync", "curl", "nc", "tar", "zip", "open", "source", "printenv",
        )
        if re.search(r"\b(" + "|".join(READERS) + r")\b", probe):
            deny(f"Blocked: `{cmd.strip()[:80]}` would expose .env contents. {SECRET}")
        if re.search(r"[<>]\s*\.env\b", probe):
            deny(f"Blocked: `{cmd.strip()[:80]}` redirects to/from .env. {SECRET}")
PY
