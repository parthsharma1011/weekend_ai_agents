# Deploying the Gemini RAG Agent to Render

**Everything needed to deploy is already in this repo.** No files to create, no
code to copy. This document explains *what each piece does and why*, so you can
teach it.

Your actual workflow is three steps:

1. `git push`
2. On Render: **New +** → **Blueprint** → pick this repo
3. Paste your `GEMINI_API_KEY` when Render asks

That's it. Render reads `render.yaml`, builds, and serves.

---

## Table of contents

1. [The 3-step deploy](#1-the-3-step-deploy)
2. [Architecture](#2-architecture)
3. [The deployment files, explained](#3-the-deployment-files-explained)
4. [The concepts worth teaching](#4-the-concepts-worth-teaching)
5. [Running it locally first](#5-running-it-locally-first)
6. [Troubleshooting](#6-troubleshooting)
7. [Demo script](#7-demo-script)

---

## 1. The 3-step deploy

### Step 1 — Push

```bash
git push origin main
```

### Step 2 — Create the service on Render

1. Go to <https://dashboard.render.com>
2. **New +** → **Blueprint**
3. Connect GitHub, select `weekend_ai_agents`
4. Render finds `render.yaml` and shows you the service it's about to create
5. Click **Apply**

Because the blueprint declares everything — root directory, build command, start
command, health check — there is nothing to type into the dashboard.

### Step 3 — Provide the secrets

Render will prompt for the two variables marked `sync: false`:

| Variable | Where to get it | Required |
|---|---|---|
| `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> | **Yes** |
| `TAVILY_API_KEY` | <https://tavily.com> | No — leave blank to disable web fallback |

> **Use a long-lived key that starts with `AIza`.** Short-lived `AQ.` tokens
> expire after roughly an hour, and the service will start returning
> `401 UNAUTHENTICATED` with no code change. This is the single most likely way
> for a live demo to break.

First build takes 5–10 minutes (it downloads the embedding model and builds the
FAISS index). After that, your URL is `https://gemini-rag-agent.onrender.com`.

---

## 2. Architecture

```
Browser (static/index.html)
      │  POST /api/chat  {"message": "..."}
      ▼
FastAPI (app.py)
      │
      ▼
ChatService ──► GuardrailEngine.check_input()      ← blocks / redacts
      │
      ▼
LangGraph  retrieve ─► [context empty?] ─► tool (Tavily)
                 │                              │
                 └──────────► generate ◄────────┘
                                 │
                                 ▼
                             critique ─► output
      │
      ▼
GuardrailEngine.check_output() ──► JSON {"answer": "..."}
```

The graph, guardrails, memory and self-critique are **unchanged** from the CLI
version. Deployment changed *how a turn is triggered* and *where the answer
goes* — nothing about how the agent thinks.

---

## 3. The deployment files, explained

Six files make this deployable. Three are new, one was modified, and the agent
itself was not touched at all.

### `RAG_AGENT_CODE/app.py` — the web layer *(new)*

Wraps the existing agent in HTTP. Compare it to `main.py` side by side; that
contrast is the lesson.

| `main.py` (CLI) | `app.py` (web) |
|---|---|
| `_handle_turn()` prints, returns `None` | `ask()` returns `{"answer", "blocked"}` |
| `print("[Guardrail] Input Blocked!!!")` | `{"answer": "Blocked by guardrail: ...", "blocked": True}` |
| Loops on `input()` | One call per HTTP request |
| Session id `"default"` | Session id `"web"` |

`ChatService.__init__` builds exactly what `ChatApp.__init__` builds — same
guardrails, same memory window, same retriever, same Tavily fallback, same
critic, same compiled graph.

**The `lifespan` hook** is the part worth pausing on:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = ChatService()
    yield

app = FastAPI(title="Gemini RAG Agent", lifespan=lifespan)
```

`ChatService()` loads the FAISS index and the embedding model, which takes
seconds. Three options, only one correct:

- **Per request** — every question pays the load cost. Unusably slow.
- **At import time** — CI breaks, because importing would require an API key.
- **At startup (`lifespan`)** — loaded once, after import, before the first
  request. Correct.

> Use `lifespan`, not `@app.on_event("startup")` — the latter is deprecated in
> FastAPI 0.115+.

**Three routes:**

| Route | Purpose |
|---|---|
| `GET /healthz` | Render's health probe; reports whether the agent finished loading |
| `POST /api/chat` | `{"message": "..."}` → `{"answer": "...", "blocked": bool}` |
| `GET /` | Serves the frontend |

**Why not just edit `main.py`?** Because `_handle_turn` prints and returns
`None` — perfect for a terminal, useless over HTTP. `app.py` is a sibling that
shares every underlying component, so `python main.py` still works exactly as
before. **Transport is not logic.**

### `RAG_AGENT_CODE/static/index.html` — the frontend *(new)*

One self-contained page: markup, CSS and JavaScript in a single file. No build
step, no framework, no dependencies — deliberately, because the session is about
deployment, not tooling.

| Part | Role |
|---|---|
| `#log` | Scrolling transcript |
| `#form` submit handler | `POST /api/chat` |
| `add()` | Renders one message using `textContent` |
| `.err` class | Colours guardrail blocks and errors red |
| Disabled button + "Thinking..." | Prevents double-submits |

Two teaching points:

- **No CORS setup anywhere.** FastAPI serves the page *and* the API, so they
  share an origin and the browser never sends a preflight. Split them across
  hosts and CORS immediately becomes your problem.
- **`textContent`, not `innerHTML`.** The agent's reply is untrusted — it can
  contain whatever the model or a retrieved document says. `textContent` makes
  stored-XSS structurally impossible.

### `RAG_AGENT_CODE/requirements.txt` — dependencies *(modified)*

Two lines added:

```
fastapi==0.115.5
uvicorn[standard]==0.32.1
```

Everything else is untouched. Two points:

- **Deployment dependencies are additive.** The CLI never imports FastAPI, so
  `main.py` runs fine without them.
- **Everything is pinned.** The unpinned `langfuse` in the LinkedIn agent is the
  counter-example — that's how a build that works today breaks silently when a
  major version ships.

### `render.yaml` — the blueprint *(new)*

Infrastructure as code. This is why you don't fill in dashboard forms.

```yaml
rootDir: RAG_AGENT_CODE
buildCommand: "pip install -r requirements.txt && python ingest.py ./docs"
startCommand: "uvicorn app:app --host 0.0.0.0 --port $PORT"
healthCheckPath: /healthz
autoDeploy: true
```

Every line earns its place:

| Setting | Why it matters |
|---|---|
| `rootDir` | The app is in a subfolder. Without this, Render looks for `requirements.txt` at the repo root and fails instantly. **The single most common mistake.** |
| `buildCommand` | `ingest.py` rebuilds the FAISS index. It is gitignored, so a fresh clone doesn't have it — see [section 4](#4-the-concepts-worth-teaching). |
| `startCommand` | `$PORT` is assigned by Render at runtime and probed to confirm the deploy worked. Hardcode a port and the health check never passes. |
| `healthCheckPath` | Render polls `/healthz`; a failing probe rolls the deploy back. |
| `autoDeploy: true` | Every push to `main` redeploys. This is your CD. |
| `sync: false` | Declares "this service needs this secret" **without putting the value in git.** |
| `FASTEMBED_CACHE_PATH` | fastembed defaults its model cache to `$TMPDIR`, which isn't guaranteed to survive from build into runtime. Without this, the ~64 MB embedding model downloads at build *and again* on the first request. |

### Why aren't these files inside `RAG_AGENT_CODE/`?

Because two different systems read them, and both look at the **repo root**:

- **`.github/workflows/ci.yml`** — GitHub only discovers workflows in
  `.github/workflows/` at the root. Anywhere else and Actions never runs.
- **`render.yaml`** — Render looks for the blueprint at the root when you
  connect the repo. `rootDir: RAG_AGENT_CODE` inside it is precisely what points
  Render at the subfolder.

Application files (`app.py`, `static/`, `test_guardrails.py`,
`requirements.txt`) *do* live in `RAG_AGENT_CODE/`. Render `cd`s into `rootDir`
before running the build and start commands — which is why `buildCommand` says
`pip install -r requirements.txt`, not `pip install -r RAG_AGENT_CODE/requirements.txt`.

This also scales: to deploy the LinkedIn agent later, add a second service to the
same `render.yaml` with `rootDir: linkedin_generator_agent`. One blueprint,
two services.

That last one is the important idea: you can version-control the *shape* of your
configuration while keeping the *values* out of the repo.

### `.github/workflows/ci.yml` — continuous integration *(new)*

Runs on every push and PR. Two jobs:

**Job `test`** — installs dependencies, then:

| Step | Catches |
|---|---|
| Import smoke test | Both crash bugs we fixed: the missing type annotation on `retrieval_min_score`, and the `self.self._load` typo |
| Settings load | Broken config / bad env mapping |
| Build the FAISS index | Ingestion breakage — the deploy would boot with no index |
| `pytest -q` | Guardrail regressions |

**No API key is needed.** `Providers.llm` is a `cached_property`, so the Gemini
client is only constructed on first *access*. Importing never touches the
network. That's why CI can validate wiring without secrets — a genuinely elegant
consequence of lazy initialisation, and worth pointing out.

**Job `secret-scan`** — fails the build if an API key or a `.env` file was
committed. This is not hypothetical: a live key landed in a tracked file twice
during this project's history. Automating the check is the fix.

### `RAG_AGENT_CODE/test_guardrails.py` — the tests *(new)*

Twelve tests, no API key, no network, runs in 0.04s. The guardrails are pure
functions over strings, which makes them the ideal first test target.

They cover blocking (too short, injection, toxicity), transforming (PII
redaction rewrites rather than rejects), and the engine that threads them —
including that a rewrite by one validator is visible to the next.

---

## 4. The concepts worth teaching

### Artifacts are rebuilt, not committed

`.gitignore` excludes `faiss_index/` and `memory_store/`. So a fresh clone has
no index, and `Retriever._load()` would raise:

```
FileNotFoundError: FAISS index not found at './faiss_index'.
```

The fix is in `buildCommand`: `python ingest.py ./docs`. The `docs/` folder *is*
committed, so the build has its inputs and regenerates the output.

Why not just commit the index? Two reasons, both concrete:

- It's a binary that changes completely on every rebuild — useless diffs.
- It **embeds the source text of your documents.** In this project, an earlier
  index contained a private individual's name, address, phone and email pulled
  from a PDF. Committing it to a public repo would have published all of it.

### Secrets live in the platform, not the repo

`.env` is gitignored. `render.yaml` declares `GEMINI_API_KEY` with `sync: false`
— the *name* is in git, the *value* is entered in Render's dashboard. CI never
needs it at all.

### Ephemeral disk

`memory_store/` is written at runtime, and Render's filesystem resets on every
deploy and every free-tier spin-down. Chat history will not survive. That's fine
for a demo; real persistence needs Postgres or Redis. Worth stating plainly so
nobody is surprised.

### CI gates, CD ships

Right now `autoDeploy: true` means **every** push deploys, even one that fails
CI. Render doesn't know about GitHub Actions.

To gate deploys on CI passing:

1. Render → service → **Settings** → **Auto-Deploy: No**
2. Copy the **Deploy Hook** URL
3. GitHub → **Settings** → **Secrets and variables** → **Actions** → add
   `RENDER_DEPLOY_HOOK`
4. Add `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Render
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    steps:
      - run: curl -fsS -X POST "${{ secrets.RENDER_DEPLOY_HOOK }}"
```

That `if:` line is the entire concept of "CD gated on CI" in one expression.

This file is **not** in the repo on purpose — it fails on every push until the
secret exists, and red X's during a lesson are a distraction. Add it when you
want to teach the gated version.

---

## 5. Running it locally first

Always demo the local version before the cloud one.

```bash
cd RAG_AGENT_CODE

pip install -r requirements.txt
python ingest.py ./docs                          # build the index
python -m uvicorn app:app --reload --port 8000   # run the server
```

Open <http://localhost:8000>.

> **Use `python -m uvicorn`, not bare `uvicorn`.** If you have more than one
> Python (Homebrew *and* Anaconda, say), bare `uvicorn` may resolve to an
> interpreter without LangChain, giving
> `ModuleNotFoundError: No module named 'langchain_core'`. The `-m` form
> guarantees the same interpreter as your `pip install`. Render has a single
> environment, so plain `uvicorn app:app` is correct there.

Check the API directly:

```bash
curl http://localhost:8000/healthz

curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What was the efficacy rate of clinical trial T001?"}'
```

Run the tests:

```bash
cd RAG_AGENT_CODE && pytest -q
```

> On macOS you may need `KMP_DUPLICATE_LIB_OK=TRUE` — an OpenMP conflict between
> FAISS and PyTorch. It does not occur on Render's Linux boxes.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails, `requirements.txt` not found | `rootDir` not applied | Confirm `rootDir: RAG_AGENT_CODE` in `render.yaml` |
| `FileNotFoundError: FAISS index not found` | Index missing at boot | `buildCommand` must include `python ingest.py ./docs` |
| Deploy hangs, then "port scan timeout" | Not bound to `$PORT` | Start command must use `--host 0.0.0.0 --port $PORT` |
| `401 UNAUTHENTICATED` | Key expired or missing | Set `GEMINI_API_KEY`; `AQ.` tokens expire hourly, prefer `AIza` |
| `429 RESOURCE_EXHAUSTED` | Free-tier quota | 20 requests/day per model — and **each turn costs 2 calls** (generate + critique) |
| Always "falling back to web search" | `min_score` too high | `RETRIEVAL_MIN_SCORE=0.35` (already set in `render.yaml`) |
| First request after idle takes ~60s | Free tier spins down | Expected. Warm it up before demoing |
| Very slow first request after deploy | Model cache didn't persist from build | Check `FASTEMBED_CACHE_PATH` resolves to a real path in the service; worst case it re-downloads 64 MB once |
| Chat history vanished | Ephemeral disk | Expected — needs a real database |
| `ModuleNotFoundError` locally | Multiple Pythons | `python -m uvicorn ...` |

Logs: Render dashboard → your service → **Logs**. Nearly every failure above is
one line in there.

---

## 7. Demo script

A ~15-minute arc that builds the ideas in order:

1. **Run the CLI** (`python main.py`). Establish that the agent already works.
   Deployment is about *access*, not intelligence.
2. **Run `python -m uvicorn app:app --reload`.** Same agent, new door. Open
   `app.py`; point out that no agent file changed.
3. **Open the browser.** Ask *"What was the efficacy rate of clinical trial
   T001?"* — a grounded answer from the local index.
4. **Ask *"who is the prime minister of india"*.** Nothing in the docs, so the
   Tavily fallback fires. Show the routing edge in `agent.py`.
5. **Type "hi".** Blocked by `MinWordsValidator`. Guardrails become visible
   rather than theoretical.
6. **Push a deliberate syntax error.** CI goes red. Fix it, CI goes green. That
   contrast *is* the lesson.
7. **Show the Render logs during a cold start** — the build rebuilding the FAISS
   index is the payoff for the artifacts discussion in section 4.

---

## File inventory

```
weekend_ai_agents/
├── render.yaml                        NEW  — Render blueprint
├── deploy.md                          NEW  — this document
├── .github/workflows/ci.yml           NEW  — tests + secret scan
└── RAG_AGENT_CODE/
    ├── app.py                         NEW  — FastAPI layer
    ├── test_guardrails.py             NEW  — 12 tests
    ├── static/index.html              NEW  — frontend
    ├── requirements.txt               MOD  — added fastapi, uvicorn
    ├── agent.py                       unchanged
    ├── config.py                      unchanged
    ├── guardrails.py                  unchanged
    ├── ingest.py                      unchanged
    ├── main.py                        unchanged — CLI still works
    ├── memory.py                      unchanged
    ├── retrieval.py                   unchanged
    └── self_critique.py               unchanged
```

Every file of the agent itself is untouched, and `python main.py` runs exactly
as it did before. That's the takeaway worth leaving students with: **a good
deployment adds a layer, it doesn't rewrite the application.**
