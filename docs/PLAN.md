# LLM Output Arbitration — Build Plan

**Claim under test:** three critics on *different providers*, reconciled by an adjudicator,
catch more planted errors (at an acceptable false-flag rate) than one strong model reviewing alone.

## Architecture

```
question? + output
      │
  parse_input
      │  fan-out (parallel)
 ┌────┼─────────────┐
factual  logic  completeness      (+ optional extra critics, e.g. Ollama)
DeepSeek Gemini  DeepSeek-flash
 └────┼─────────────┘
      │  fan-in
 collect_critiques  ── quote validation (invalid quotes flagged, not trusted)
      │
 detect_disagreements
      │ conditional edge
      ├── unanimous high-confidence pass ──► synthesize_verdict (adjudicator skipped)
      └── otherwise ──► adjudicate ──► synthesize_verdict
```

## Folder structure

```
llm-0utput-arbitration/
├── config.yaml              # providers, models, role assignment, limits (all swappable)
├── .env.example             # DEEPSEEK_API_KEY, GEMINI_API_KEY
├── pyproject.toml           # deps + ruff/mypy/pytest config
├── requirements.lock        # exact versions (used by Docker)
├── Dockerfile, docker-compose.yml          (Phase 7)
├── data/testset.yaml        # YOUR hand-written cases + answer keys   (Phase 4: format + 3 examples ✅)
├── reports/eval_report.md   # generated (+ eval_metrics.json)          (Phase 5 ✅)
├── docs/PLAN.md
├── scripts/
│   ├── case_builder.py      # draft CLEAN answers; `check` coverage     (Phase 4 ✅)
│   ├── evaluate.py          # system vs baseline vs ablations; --resume/--from-run (Phase 5 ✅)
│   ├── run_critics.py       # debug: critics only, no graph             (Phase 1 ✅)
│   ├── arbitrate.py         # live end-to-end run / --mermaid           (Phase 2 ✅)
│   └── seed_demo.py         # demo arbitrations for the UI             (Phase 7)
├── src/arbiter/
│   ├── config.py            # typed config loader                       (Phase 0 ✅)
│   ├── llm/                 # one OpenAI-SDK client for all providers   (Phase 1)
│   │   ├── client.py        #   structured JSON calls, re-asks, semaphore, retries/backoff ✅
│   │   └── telemetry.py     #   tokens, cost, latency, finish_reason → JSONL log ✅
│   ├── schemas/             # Pydantic v2 contracts
│   │   ├── critique.py      #   Issue, Critique                          (Phase 1 ✅)
│   │   ├── disagreement.py  #   IssueGroup, Disagreement, Report         (Phase 2 ✅)
│   │   ├── verdict.py       #   Verdict, ConfirmedIssue, DismissedFlag   (Phase 2 ✅)
│   │   └── testcase.py      #   EvalCase, PlantedSpanIssue/Omission      (✅ early)
│   ├── critics/
│   │   ├── prompts.py       #   one focused system prompt per dimension  (Phase 1 ✅)
│   │   ├── critic.py        #   run a critic → Critique                  (Phase 1 ✅)
│   │   └── quotes.py        #   quote validation + span location         (Phase 1 ✅)
│   ├── spans.py             #   span overlap maths (shared with Phase 5)  (Phase 2 ✅)
│   ├── graph/               #                                            (Phase 2 ✅)
│   │   ├── state.py         #   LangGraph state + reducers
│   │   ├── disagreement.py  #   pure-Python disagreement detector
│   │   ├── nodes.py         #   nodes + routing (fan-out via Send, conditional edge)
│   │   └── build.py         #   graph wiring
│   ├── adjudication/
│   │   ├── base.py          #   Adjudicator protocol + deterministic score (Phase 2 ✅)
│   │   ├── union.py         #   no-LLM union adjudicator = "no adjudicator" ablation (Phase 2 ✅)
│   │   ├── synthesis.py     #   verdict synthesis (coverage-scaled confidence) (Phase 2 ✅)
│   │   ├── prompts.py       #   adjudicator prompt (blind to model names) (Phase 3 ✅)
│   │   └── llm.py           #   LLM adjudicator, one ruling per group     (Phase 3 ✅)
│   ├── baseline.py          #   single-reviewer baseline (same rules)     (Phase 3 ✅)
│   ├── pipeline.py          #   Arbiter: public entry point               (Phase 2 ✅)
│   ├── casebuilder.py       #   drafting, YAML append, coverage report    (Phase 4 ✅)
│   ├── evaluation/          #                                             (Phase 5 ✅)
│   │   ├── findings.py      #   every variant → common Finding list (ablations from 1 run)
│   │   ├── matching.py      #   span rule + blind, cached LLM judge for omissions
│   │   ├── metrics.py       #   recall/precision/FF/accuracy, Wilson CI, exact McNemar, analytics
│   │   ├── runner.py        #   run + persist per case (resume), score
│   │   └── report.py        #   eval_report.md (headline generated from the numbers)
│   ├── storage/db.py        #   SQLite audit trail (WAL, summary columns + full JSON) (Phase 6 ✅)
│   ├── api/                 #   FastAPI: main.py (app factory), schemas.py (OpenAPI) (Phase 6 ✅)
│   └── ui/                  #   Streamlit: app.py + pages/, render.py (pure, tested), data.py (Phase 6 ✅)
└── tests/                   # pytest; every LLM call mocked
```

## Phases (stop for confirmation after each)

| # | Phase | Runnable proof |
|---|-------|----------------|
| 0 | Plan, env, config loader | `pytest` green on Python 3.12 venv |
| 1 | LLM client + 3 critics + quote validation | mocked tests + `scripts/run_critics.py` on one sample (live) |
| 2 | LangGraph orchestration + disagreement detector | graph diagram printed, mocked end-to-end run incl. critic failure + short-circuit |
| 3 | Adjudicator, verdict, single-reviewer baseline | mocked tests + one live arbitration |
| 4 | Test-set format, 3 examples, case builder | **STOP: you write ~40 cases** |
| 5 | Evaluation + ablations + critic analytics + report | `evaluate.py --limit N`, `reports/eval_report.md` |
| 6 | FastAPI + SQLite audit trail + Streamlit explorer | `/docs` OpenAPI page, UI screenshots |
| 7 | Docker + seed + README case study | `docker compose up` |

## Key design decisions (decided up front)

- **One SDK, many providers.** Every provider is reached through the official `openai` SDK
  pointed at an OpenAI-compatible `base_url`. Adding a provider is a config change.
- **Own thin structured-output layer (instructor evaluated, not used).** DeepSeek only supports
  `json_object` (no schema enforcement), so the schema goes in the prompt, the reply is validated
  with Pydantic, and the validation error is sent back as a re-ask (max 2). Gemini additionally
  gets `json_schema` response_format. Doing this ourselves (~60 lines) means every attempt —
  retries AND re-asks — is logged with tokens/cost/finish_reason, which instructor's internal
  retry loop would hide.
- **Quote validation is a post-check, not a re-ask.** A hallucinated quote is *data* about
  critic reliability (we want to count it), so we mark the issue `valid=False` instead of
  silently retrying it away.
- **Async throughout.** `AsyncOpenAI` + a global `asyncio.Semaphore(max_concurrency)` caps
  in-flight calls across critics, cases, and the batch API. The slot is held only while a
  request is in flight, not during backoff sleeps. SDK built-in retries are disabled so our
  loop is the single retry layer (no retry-on-retry blowups).
- **Free-tier 429 handling.** Per-minute limits → exponential backoff honouring
  `Retry-After`; per-day quota exhaustion → fail fast with a clear message (retrying won't help).
- **Fair baseline.** The baseline uses `deepseek-v4-pro` — the same model as the adjudicator —
  so the system has to beat its own strongest component, not a weak strawman.
- **Ablations reuse the same critic calls.** "Each critic alone" and "union without adjudicator"
  are computed from the full run's stored critiques — zero extra API cost, and identical inputs
  across variants.
- **Disagreement detection is deterministic Python**, not an LLM: cheap, testable, explainable.
- **The adjudicator rules, it never adds.** It must confirm or dismiss every issue group (a
  Pydantic validator rejects missing/unknown/duplicate group ids, which triggers a re-ask) and
  cannot report new issues. It runs on the same model as the baseline, so if it could add
  findings the system would just be "baseline + extras" and the comparison would be rigged.
- **Adjudicator failure degrades, not crashes:** union of critic issues at half confidence,
  verdict status `degraded`, reason recorded.
- **Human gate on test cases:** every case has `status: draft|ready`; only `ready` cases are
  evaluated. An unreviewed LLM draft could hide a real error that would be scored as a false
  flag. The case builder only ever drafts CLEAN answers (a test asserts its prompt never
  mentions errors); humans plant and record every error.
- **Evaluation statistics (Phase 5):** every rate gets a 95% Wilson interval; system vs
  baseline is compared PAIRED on the same planted issues with an exact McNemar test (only the
  discordant pairs carry information). Latency = critical-path LLM time (slowest critic +
  adjudicator, timed inside the concurrency slot) so eval concurrency doesn't distort it.
- **UI treats LLM text as hostile (Phase 6):** the evaluated output and every critic/adjudicator
  explanation are html-escaped, and the evidence JSON inlined in the page's <script> is
  additionally escaped so "</script>" can't break out (tested with an injection payload).
  The API is the only writer to SQLite; the UI reads the shared file and submits via the API.
- **Baseline fairness:** same model as the adjudicator, same quote rules, severity definitions
  and per-dimension checklists (shared prompt blocks), same quote validation and pass/fail
  threshold, typed issues so the Phase 5 matching rules apply identically.
- **Two matching rules in evaluation (decided 2026-10-20).** The answer key has two shapes
  (`data/testset.yaml`, validated by `schemas/testcase.py`):
  - *factual / logic* planted issues have an exact `span`; a found **factual/logic** issue
    matches when the spans overlap (threshold settled in Phase 5).
  - *completeness* planted issues have `missing` (what is absent) + optional `anchor` (UI
    highlighting only). A found **completeness** issue matches only if the `match_judge`
    (deepseek-flash) confirms both descriptions name the same missing item. Completeness
    findings never match by span: critics anchor omissions on nearby sentences (observed in the
    Phase 1 live run), which would falsely "catch" unrelated planted errors.
  - The judge is blind to which system raised an issue, runs once per case (all pairs in one
    call) and is cached so system, baseline and ablations get identical decisions.
  - The report splits matches into span vs judge and lists every judge match for spot-checking.
  - Consequence for Phase 3: the baseline's issues must carry a type (factual/logic/completeness)
    so the same rules apply to it.

## Provider facts (official docs, checked 2026-10-19)

- DeepSeek `https://api.deepseek.com`: `deepseek-v4-pro`, `deepseek-flash` (now served by
  V4.1-Flash). Thinking is **on by default** → disabled via `extra_body={"thinking":{"type":"disabled"}}`.
  JSON mode = `json_object` only; prompt must contain the word "json"; can occasionally return
  empty content (handled as a retryable error).
- Gemini `https://generativelanguage.googleapis.com/v1beta/openai/`: `gemini-3.6-flash` (stable,
  free tier). `gemini-3.8-flash` was tried first but measured 43-92 s/call on the free tier.
  Gemini 3 reasoning **cannot be disabled** (`minimal` is rejected with HTTP 400);
  `reasoning_effort: low` still used ~425 hidden thinking tokens per critique. They appear only as
  `total_tokens - prompt - completion`, which telemetry bills as output.
- **Logic critic switched to `gemini-3.5-flash-lite` (2026-10-21).** The free tier allows only
  20 requests/day for `gemini-3.6-flash` (429 body: quotaId
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, quotaValue 20) plus 5/minute; a 40-case
  eval needs 40+. The first full run was stopped at 37 cases after the quota ran out (13 cases
  had a logic critic) and discarded; every case in the reported run uses Flash-Lite. Quotas are
  per model, so Flash-Lite starts fresh. Trade-off: a weaker logic critic.
- Ollama `http://localhost:11434/v1`: optional extra critic only.

## Honest limitations (to carry into the README)

- Critics have no web access: "factual accuracy" = the critic's own knowledge + internal
  consistency, not live verification.
- Critics share training-data blind spots; provider diversity reduces but does not remove this.
- The answer keys are hand-written by one person; ~40 cases gives wide confidence intervals.
- Free-tier rate limits make full eval runs slow.
