# Project context for Claude Code

MSc thesis, 12-week personal execution schedule (formal university registration
is 12 months — that's paperwork, not the working timeline). Day-ahead
electricity price forecasting (hourly + daily baseload), German market
(EPEX-DE benchmark / DE-LU live). Thesis body is written in Farsi; the
separate journal article deliverable is English.

## Non-negotiable constraints
- Data: `BenchmarkLoader` (epftoolbox, thesis results) and
  `EnergyChartsLoader` (Energy-Charts; promoted 2026-08-28 from tool-only to a
  sanctioned thesis feature source) — both keyless, one shared schema
  (`price`, `exog_*`, hourly DatetimeIndex).
  Amended 2026-08-28: no registration-gated source may feed thesis numbers
  without explicit written approval. ENTSO-E is pre-approved pending token
  arrival; that approval does NOT generalise to any other gated source.
  NEVER SCRAPED, and not to be revisited: Investing.com and Trading Economics
  (terms of service, and unreproducible). NEVER PURCHASED: raw commercial fuel
  futures (ICE / Montel / GME). This is why clean spark and dark spreads are
  unbuildable — gas (TTF) and coal (API2) are Montel-licensed, and Ember
  republishes only series DERIVED from them, which makes Ember a citation and
  not a data source.
  Reproducibility rule: external data is fetched ONCE into an immutable
  snapshot with a sha256 provenance record, committed, and read only from
  there. A live API answers differently tomorrow and does so silently. See
  `data/raw/physical/provenance.json` and `data/raw/provenance_benchmark.json`.
  Feature code must never touch the network — enforced by
  `tests/test_no_network_in_features.py`.
- Models: exactly naive, SARIMAX, LEAR-LASSO, LightGBM, LSTM, + weighted
  ensemble. Do not add models (RF/XGBoost/SVR/GRU were deliberately cut).
- Targets: hourly (24-price D+1 vector) and daily (baseload average, both
  direct and hourly-aggregated — the comparison itself answers RQ4).
- Tuning: 50 Optuna trials per model, validation window strictly before
  test window. Walk-forward (rolling-origin) validation only — never
  random splits.
- Metrics: MAE, RMSE, sMAPE, rMAE, Diebold-Mariano. No plain MAPE
  (negative prices exist in the data).
- Leakage rule: no feature may use information after the forecast origin.
  There is an assertion test for this — keep it passing.
- Seed 42 everywhere; every non-trivial decision gets a dated entry in
  logs/decisions.md.
- After the `v1.0-results` tag exists (end of week 7): never rerun or
  modify model results — writing depends on frozen numbers.
- Week-5 checkpoint: compare LightGBM walk-forward results against Lago
  et al.'s published numbers. Outcome decides which gameplan leads —
  Plan A (match/beat the published benchmark) or Plan B (innovation-led
  defense) — see logs/decisions.md 2026-07-11 for the full gameplan.
- Week-7 priority order: static ensemble → regime-aware ensemble →
  France (only if slack; France is now third priority, not a default).

## Gameplan (2026-07-11)
- Plan A = match/beat Lago et al.'s published LEAR/DNN numbers on
  EPEX-DE (the only fair "beat" claim).
- Plan B (built regardless, weeks 5-8) = innovation-led defense, all
  sanctioned scope (not scope creep):
  - Regime-aware ensemble weighting — calm/stressed weight sets switched
    on `regime.stress_threshold_eur_mwh` in configs/evaluation.yaml
    (62.6989 EUR/MWh = train mean + 1.5*std). The week-2 EDA's 3-sigma
    84.04 value left only 3 stressed validation days and was superseded
    on 2026-08-04; k is fixed by a validation-only rule, never by test
    behaviour — see logs/decisions.md 2026-08-04.
  - Calm-vs-stressed + hourly-vs-daily SHAP comparison.
  - OOD stress test: frozen benchmark-era models evaluated on live 2026
    Energy-Charts data (tool-only loader, per the data rule above).
- Stretch goal (week 7, third priority, only if ahead of schedule):
  rerun final models on France (`dataset='FR'`, config change only).
  Nord Pool was considered and rejected (system-price vs. zonal-price
  mismatch with the live API).

## Six formal assumptions (from the approved university proposal)
Must appear in thesis section 3-2. Keep them in mind when writing any
methodology code/comments: (1) stationarity, (2) data availability,
(3) data quality, (4) model generalization, (5) stable market conditions,
(6) model interpretability.

## Scope vs. the approved proposal
The proposal is generic (any of RF/tree/NN, daily-only focus, MAE/MSE/RMSE,
interpretability as an assumption not a deliverable). This project adds,
confirmed by supervisor as approved: named benchmark tied to published
literature (Lago et al. protocol), hourly actually operationalized, fixed
5-model list, live data feed, significance testing, SHAP as a real
deliverable, the PriceCast tool, and a separate journal article. Also
sanctioned (2026-07-11 gameplan decision, not scope creep): regime-aware
ensemble weighting (calm/stressed weight sets) and the OOD stress test of
frozen models on live Energy-Charts data — see Gameplan section below.
Don't scope-creep beyond this list without a logged decision.

## Thesis structure (see thesis/outline.md for full detail)
100-page Farsi body, 5 chapters mapped to Amirkabir's official template:
1. Introduction (7pp) 2. Literature review (17pp) 3. Methodology (37pp)
4. Results & analysis (29pp) 5. Conclusion (10pp). Results/figures produced
by this repo map directly into chapter 3 (methodology sections 3-3
through 3-8) and chapter 4 (results, DM tests, SHAP) — see outline.md for
exact section numbers when generating tables/figures meant for the thesis.

## Session startup (standing instruction, 2026-07-30; extended 2026-08-26)
At the start of every task-oriented session in this project, work through
these four steps before anything else.

**1. Skills.** Invoke the task-observer skill and
superpowers:using-superpowers (also enforced by a SessionStart hook in
.claude/settings.json).

**2. Read the map before reading the code.** `graphify-out/` holds a built
knowledge graph of this repo — nodes, edges and hand-labelled communities over
the whole tree. For any question about structure, call paths, or what touches a
given module, query the graph before grepping cold:

    graphify query "<question>"
    graphify explain "<node>"          # e.g. build_features
    graphify path "<A>" "<B>"

**The graph is NOT in git** — `graphify-out/` is gitignored, because it is
generated output that goes stale on every code change and has no mechanism to
announce that it has. A fresh clone has no graph until it builds one:

    /graphify .                        # first build in a fresh clone
    /graphify . --update               # incremental, after code changes

The rebuild is cheap. Code is extracted structurally by AST with no LLM call,
so a code-only change costs zero tokens; only changed docs need a semantic
pass.

**Current figures live in the generated output, not here.** Read them from
`graphify-out/GRAPH_REPORT.md` (community map, god nodes, known gaps) and
`graphify-out/cost.json` (per-run token cost and file counts). This file
deliberately quotes NO node/edge/community counts: it is read at the start of
every session, so a stale number here is worse than no number — sessions trust
it. The counts were hardcoded until 2026-08-29 and had already drifted by a
full rebuild (they described the graph as of 2026-08-26 while the built graph
was substantially larger). If you ever add a figure back, attach its build date
to it so the staleness is visible on sight.

Standing caution: the graph is a lookup index, never an authority. Confirm
anything it implies against the source file before acting on it — a share of
extracted edges is dropped as dangling and some nodes are isolated, both
reported in GRAPH_REPORT.md for the current build. And it reflects the repo as
of its last build, so refresh after substantial code changes.

**3. Establish where we left off from the files, not from memory.** In
priority order:

    CHECKLIST.md                 what is left, already prioritized P0-P4
    logs/decisions.md            why, plus corrections to earlier claims
    thesis/CONVERSION_QUEUE.md   drafts awaiting conversion
    thesis/page_ledger.csv       pages actually banked
    git log --oneline -15        what actually landed

NEXT_SESSION.md is a handoff snapshot and goes stale — as of 2026-08-26 its
week number was three weeks out of date. Treat its dates and status claims as
things to verify, not facts.

**4. Then plan and execute.** Track multi-step work in a todo list where the
harness offers one, so session state survives compaction; otherwise keep the
running state in CHECKLIST.md, which is the durable version of the same
thing. Honour the mandatory post-task reconciliation rule
in NEXT_SESSION.md: when a technical task finishes, either bank pages or
write a dated deferral line in logs/decisions.md — never neither.

## Conventions
- Config-driven: market/zone/splits come from configs/*.yaml, not code.
- Model wrappers implement fit/predict/save/load on a common interface.
- Figures export once, in final captioned form, to reports/figures/.
- Canonical results table (model x target x metric) auto-exports to LaTeX.
