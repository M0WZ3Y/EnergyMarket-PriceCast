# HANDOFF — new-model / combination design arm

> **STATUS 2026-08-20 — READ BEFORE THE REST OF THIS FILE.**
>
> Two of this document's load-bearing statements have since been resolved,
> one of them against it. Do not act on §4 or §11 without reading these.
>
> 1. **§1's blocking decision is RESOLVED.** `retrain-v2-seed-ensemble` was
>    promoted and merged to `main` on 2026-08-20, kept SUPPLEMENTARY (outside
>    `v1.0-results`). New table `reports/tables/seed_ensemble.{csv,tex}` feeds
>    new thesis section 4-5-2. §11 step 1 is done.
> 2. **§4's novelty premise is RETRACTED.** The novelty gate (§5) was run on
>    2026-08-20 and the primary claim did not survive. Calibration-window
>    averaging was already applied to a DNN in Marcjasz (2020), *Energies*
>    13(18):4605 — before Lago et al. (2021), by one of their own co-authors.
>    Worse, the "parsimonious regression models" sentence §4 relies on is
>    about LONG-TERM SEASONAL COMPONENTS, not calibration windows: Lago et al.
>    never left this open. **The non-linear multi-window arm is abandoned.**
>    Full reasoning and verbatim quotes: `logs/decisions.md` 2026-08-20.
>
> What survives: the cheap post-hoc rungs of §3 as an ABLATION (a null result
> is a valid deliverable under §6.6), and the LEAR multi-window sweep as
> replication rather than novelty. §6, §10.3 and the §10.4 data-source
> conflict remain open.

Written 2026-08-07. Drop this in the repo root (or `docs/`) and read it
before touching anything. It supersedes nothing; `NEXT_SESSION.md`,
`CHECKLIST.md` and `logs/decisions.md` remain authoritative for everything
outside this arm.

---

## 0. Read-first, in this order

1. `NEXT_SESSION.md` — conventions, gotchas, claim discipline
2. `logs/decisions.md` — entries dated 2026-08-04 through 2026-08-07,
   especially the Lago comparison and the seed-ensemble entry
3. `reports/tables/lago_comparison.csv` and `dm_vs_lago.csv`
4. This file

Environment: always `./.venv/Scripts/python.exe`, never bare `python`.
Offline suite: `pytest -m "not network"`.

---

## 1. Verified state as of 2026-08-07

- `main` @ `2670797`, tags `v1.0-results` and `v1.1-ood` intact
- Branch `retrain-v2-seed-ensemble` pushed, **not merged**
- Offline suite 362 passed / 7 deselected
- `thesis/page_ledger.csv` — 0.0 pages banked, dated 2026-08-05
- The ledger gate (`src/ledger_gate.py`) blocks output-producing scripts;
  bypass via `THESIS_SKIP_LEDGER_GATE` writes a trace to `logs/decisions.md`

### Undecided, and blocking this arm

**Whether to promote `retrain-v2-seed-ensemble` to main.** The seed-ensembled
LSTM moves the regime-aware ensemble from MAE 3.5569 (DM p=0.0127 vs Lago's
DNN Ensemble, significantly worse) to 3.4994 (p=0.0803, not significant),
while the static variant remains significantly worse (p=0.0460).

**Resolve this before starting design work.** Any new combination design
must be built on one baseline or the other, and building on the wrong one
means redoing it. If in doubt, promote — the cascade cost is at its minimum
while 0 pages are written, and rises with every page.

Correction to a claim repeated in earlier sessions: the drafts in
`thesis/drafts/` contain **no numeric result values** (verified by grep).
Promotion cascades into `results_canonical`, both DM tables, the Lago
comparison, the OOD ensemble rows and the recalibration experiment — but
not into written prose. §3-7-2 would need an added paragraph describing the
seed ensemble, which is additive, not a rewrite.

---

## 2. What is PROVEN CLOSED — do not re-litigate

**Global convex linear reweighting is exhausted.** L1-optimal weights fitted
directly on the test set (illegitimate, therefore a hard upper bound) score
MAE 3.558; the legitimate validation-fitted regime-aware ensemble already
scores 3.5569. Headroom: −0.001.

**With the seed-ensembled LSTM included**, the same test-fitted oracle scores
3.5019 against Lago's DNN Ensemble at 3.4135. Still 0.088 short with weights
that already cheat. **Ensembling cannot close the gap to the DNN Ensemble.**
Cause: their best single model (DNN 4, 3.592) beats ours (LSTM, 3.873) by
0.28. Averaging reduces variance; it does not manufacture a better learner.

Do not propose: more seeds, more regimes, a different combiner over the
current members, or dropping SARIMAX. Dropping a member is weight-zero and
is inside the bound.

---

## 3. What the bound does NOT close

The oracle was fitted as **one scalar weight per model, applied identically
across all 24 hours, constrained to the probability simplex**
(`src/evaluation/ensemble.py: fit_weights`). The regime-aware variant is two
such vectors. Everything below is outside that family and therefore
unmeasured:

| Design | Outside the bound? | Cost |
|---|---|---|
| Per-hour convex weights (24 vectors) | yes | minutes — post-hoc on saved frames |
| Unconstrained linear stacking (negative weights permitted) | yes | minutes |
| Non-linear meta-learner over member forecasts + context | yes | hours, high overfit risk |
| Calibration-window averaging for **non-linear** models | yes — different mechanism entirely | days of compute |

The first three operate on saved prediction frames. **No retraining, no
touching `v1.0-results`.** That is the cheap end of the design space and
should be exhausted before anything expensive.

---

## 4. The strongest novelty candidate

**Calibration-window averaging applied to non-linear EPF models.**

Two things in Lago et al. (2021), read together:

- They note that combining forecasts across calibration windows
  significantly outperforms the best ex-post selected single window, then
  state it is **unknown whether this extends beyond relatively parsimonious
  regression models**. They then close it for LEAR only (their 4-window LEAR
  Ensemble, windows 56/84/1092/1456).
- Their **DNN** Ensemble averages four DNNs differing in hyperparameters and
  feature selection — all on the same four-year window. Calibration-window
  averaging is **never applied to a deep or tree-based model anywhere in the
  benchmark.**

Their own justification for why it works — long windows fit better, short
windows adapt faster under structural breaks — is the same mechanism your
OOD recalibration result measured independently: short correction windows
(3–7 days) beat long ones, the signature of drift rather than constant bias.

So the design is: run each model family across the window set, combine, and
optionally gate the combination on regime (short-window members up when
stressed, long-window up when calm), using the existing 62.6989 EUR/MWh
classifier.

Supporting facts already in hand:

- `LEARLassoModel` already accepts `calibration_window_days` from config and
  passes it to `LEAR(calibration_window=...)`. **Multi-window LEAR is a
  config sweep, not new code.**
- Your LEAR-LASSO (3.899) already significantly beats their LEAR 1092
  (shipped 3.930, p=0.0013) at the matched window.
- Their window-averaging gained 8.2% on LEAR (3.930 → 3.609 shipped). A
  comparable gain on yours lands near 3.58, which would beat their LEAR
  Ensemble outright.
- Per paper Table 4, LEAR recalibrates in 1–10 s. Three extra windows ×
  728 origins is minutes, not hours.

**Do the LEAR-only multi-window sweep first as a cheap checkpoint** before
committing to the non-linear arm.

---

## 5. PRIOR-ART GATE — do this before building anything

The novelty claim fails if any of this is already published; then it must be
cited as prior work and the contribution shrinks to replication. **Check
first, build second.** Named threats, all from Lago et al.'s own
bibliography, so they are certainly known to the committee:

| Ref | Work | Threatens |
|---|---|---|
| [81] | Marcjasz, Serafin, Weron 2018 — *Selection of calibration windows for day-ahead EPF* | window selection |
| [80] | Hubicka, Marcjasz, Weron 2019 — *Averaging day-ahead EPF across calibration windows* | **window averaging directly** |
| [135] | Serafin, Uniejewski, Weron 2019 — *Averaging predictive distributions across calibration windows* | window averaging, probabilistic |
| [82] | Maciejowska, Uniejewski, Serafin 2020 — *PCA forecast averaging* | combination schemes |
| [134] | Nowotarski, Raviv, Trück, Weron 2014 — *Empirical comparison of alternative schemes for combining EPF forecasts* | **the ablation-of-combination-schemes idea directly** |
| [84] | De Marcos, Bunn, Bello, Reneses 2020 — *Short-term EPF with recurrent regimes and structural breaks* | **regime-conditional forecasting directly** |
| [85] | Nitka, Serafin, Sotiros 2021 — ARHNN, similar-period selection | regime/analogue conditioning |

Specifically establish, and write the answers into `logs/decisions.md`:

1. Has calibration-window averaging been applied to **neural or tree-based**
   EPF models, or only to linear/parsimonious ones? (This is the crux. Lago
   et al. state it is open as of 2021 — verify nothing since has closed it.)
2. Has **regime-gated** window selection or regime-gated forecast
   combination been published? [84] and [85] are the live threats.
3. Has anyone published **per-hour** ensemble weights for EPF?
4. Search beyond the bibliography: Google Scholar for work citing Lago et
   al. 2021 (it is highly cited; sort by relevance for
   "calibration window", "ensemble", "regime"), plus arXiv and the
   International Journal of Forecasting / Applied Energy / Energy Economics
   2021–2026.

If (1) is already done for non-linear models, this arm has no novelty and
should be abandoned in favour of the contributions listed in §8.

---

## 6. IMPROVEMENT GATE — pre-register this before running anything

The novelty gate (§5) asks *is it new*. This asks *did it actually work*,
and it must be fixed in advance. A search over combination designs will
always surface something that looks better on test; without a criterion
declared beforehand, you cannot tell a real gain from the best of twelve
noisy draws. Write the answers into `logs/decisions.md` **before** the
first rung runs, and do not revise them afterwards.

### 6.1 Selection on validation, test used once

Every rung is selected — weights fitted, hyperparameters chosen, rung
adopted or rejected — **on the validation window only**. The test window is
touched once per adopted rung, to report. If a rung is rejected on
validation it does not get a test-set score "just to see". Looking is
selection.

### 6.2 The criterion: MAE / rMAE (one criterion, two units)

Improvement means **lower MAE and lower rMAE**. These are not two hurdles.
rMAE = MAE ÷ (naive2 MAE on the same window), and that denominator is a
constant across all models, so the ranking by rMAE is identical to the
ranking by MAE — Lago et al. state this directly in §5.4.2. On the EPEX-DE
test window the denominator is **≈ 9.13 EUR/MWh** (verify: 3.899/0.427,
3.557/0.390, 3.4135/0.3740 all give ≈ 9.13).

Consequences:
- Nothing can improve one and worsen the other. If a computation appears to
  show that, it is a bug — most likely a mismatched `m=` argument or a
  window misalignment.
- Set the threshold in **MAE**, because it is in interpretable units
  (EUR/MWh). Report **both** in every table; rMAE is the benchmark's
  preferred metric and is what the Lago comparison is stated in.
- The denominator differs between the validation and test windows, so
  **never compare an rMAE across windows** — validation rMAE and test rMAE
  are not on the same scale. The floor below is a validation-window MAE
  floor.

Do **not** use MAPE (excluded by decision — the benchmark's own MAPE runs
~10× its sMAPE on the German market due to negative and near-zero prices).
sMAPE and RMSE may be reported alongside but are not the criterion.

### 6.2.1 Minimum meaningful effect

Declare a floor before running. Suggested: **a rung must improve validation
MAE by ≥ 0.02 EUR/MWh** (≈ 0.002 rMAE) over the previous rung to be
adopted. Below that, the added complexity is not worth it — a gain smaller
than the seed-to-seed spread of your own LSTM (3.873 / 3.875 / 3.898 /
3.925, a spread of 0.052) is inside noise you have already measured. Given
Lago et al. §2.3's critique of unablated hybrid complexity, a 0.005 gain
bought with 24× the parameters is a liability at the defense, not an
achievement.

This number is a judgement call and should be **your** number, set and
logged before the first rung — an examiner will ask you to justify it, and
"we picked it after seeing the results" is not an answer.

### 6.3 Adoption rule

A rung is adopted only if **both**:
- validation MAE improves by ≥ the §6.2.1 floor, **and**
- the DM test on test data, against the previous rung, is p < 0.05 in our
  favour after the §6.4 correction

Anything that improves validation but fails DM is reported as *"improves
point estimate, not significant"* — not as an improvement. Report MAE and
rMAE together for every rung, adopted or not.

### 6.4 Multiple comparisons

Count every DM test run in this arm and apply **Holm–Bonferroni** across
them. Four rungs against three comparators is twelve tests; at uncorrected
α = 0.05 you expect roughly one spurious "win". Report both the raw and
corrected p-values in the table — hiding the raw ones looks evasive,
reporting only the raw ones is wrong.

### 6.5 Stopping rule

Stop climbing the ladder after **two consecutive rungs fail to clear §6.3**.
Do not keep going in the hope that a later rung rescues it; that is the
search this gate exists to prevent.

### 6.6 Report everything

Every rung attempted appears in the final table, adopted or not, with its
validation delta and its p-values. If the ladder produces no adopted rung,
that is the result: *"combination complexity beyond global convex weighting
yields no significant gain on this benchmark"* — a clean answer to an open
question, and unattackable.

### 6.7 Straddle rule

If two variants of the same design land on opposite sides of the threshold —
as the seed ensemble's regime-aware (p = 0.080) and static (p = 0.046)
variants did — **both are reported together, always**. This rule is already
in force elsewhere in the project for the regime-aware-versus-static
comparison. It applies here for the same reason.

---

## 7. Constraints

- Do not modify anything behind `v1.0-results` / `v1.1-ood`. New work goes
  in new namespaces and new branches.
- **Seed 42 rule holds**: 42 is the default, every single-model result uses
  it, other seeds exist only as labelled ensemble members.
- **No new data sources.** Fuel prices, weather and ENTSO-E were considered
  and rejected to preserve protocol equivalence with the benchmark. Adding a
  *modelling technique* is legitimate; adding *data they did not use* is not
  and destroys the comparison.
- Weight fitting on validation only, never on test. There is an existing
  leakage contract; use it.
- Ablate every component. Lago et al. §2.3 is a sustained critique of hybrid
  methods precisely because component contributions are never evaluated. A
  complex combiner without an ablation ladder walks straight into that
  critique. With one — global convex → per-hour convex → per-hour
  unconstrained → regime-gated, each DM-tested against the previous — the
  same complexity becomes a measured finding about where combination
  complexity stops paying.
- A negative result is a valid deliverable. "Window averaging does not
  transfer to non-linear models" answers a question the benchmark authors
  named and left open.

---

## 8. Honest expectations

Beating Lago's DNN Ensemble (3.4135) is **unlikely** and beating it
*significantly* is very unlikely — significance needs a gap near 0.143, and
the oracle bound says ensembling alone cannot get there. The realistic best
outcome is a stronger LEAR arm, a possible clean win over their LEAR
Ensemble, and a defensible answer to an open question.

The thesis does not depend on this arm. The contributions already frozen and
verified are:

1. **The published LEAR table does not reproduce from the shipped
   forecasts** — all five DNN rows reproduce exactly (the control), no LEAR
   variant does. Found, tested, pinned against future releases.
2. **OOD failure is a drifting level shift, not loss of relative skill** —
   diagnosed via the short-window-beats-long-window signature.
3. **Correction asymmetry** — recalibration rescues flexible learners and
   makes LEAR-LASSO worse, implying flexible models absorbed the
   training-era price level into their structure.
4. **Regime-conditional interpretability** (calm vs stressed SHAP).

These are already sufficient for the thesis. This arm is upside, not
foundation. **It must not displace writing** — 24 days to a 60-page review,
0 pages banked. Compute can run unattended overnight; chapters cannot.

---

## 10. OPERATIONAL ARM — live 2026 forecasting (daily + hourly)

**This capability already exists. Do not rebuild it.** `app/pricecast.py` +
`app/forecast_service.py` serve live forecasts from Energy-Charts; both the
hourly and daily-direct routes exist; the OOD arm (`v1.1-ood`) already
evaluated frozen models on 173 live 2026 days. What does not exist is
*acceptable accuracy*.

### 10.1 First, what "real time" means here

Day-ahead EPF is not a streaming problem. The 24 prices for day D are set in
a single once-daily auction around midday on D−1. So "real time" means:
**run once per day, after the day-ahead load and renewables forecasts for D
are published, before gate closure on D−1.** Anything more frequent is
forecasting a number that has already been fixed. Build the scheduler around
that cadence, not around a live tick.

### 10.2 The actual problem

From `v1.1-ood`: on live 2026 data **every trained model is worse than the
naive baseline** (naive 0.808; LEAR-LASSO 1.087, SARIMAX 1.145, LSTM 1.520,
LightGBM 1.828). The recalibration experiment partially fixes this — with a
3-day correction window the regime-aware ensemble reaches rMAE 0.878 and the
static ensemble also crosses below 1.0, while **LEAR-LASSO gets worse under
correction at every window**.

So a deployable 2026 system today is: *ensemble + short-window (3–7 day)
bias correction*. That is the honest current answer, and it is already
measured.

### 10.3 Untested hypothesis — train/serve skew, NOT just regime shift

The OOD failure was diagnosed as a drifting level shift. There is a second,
independent, **untested** cause sitting in the config, and it may be larger:

| | Benchmark training data (2012–2017) | Live serving (2026) |
|---|---|---|
| Price series | EPEX-DE — but pre-Oct-2018 this bidding zone was **DE-AT-LU** | `bzn: "DE-LU"` (Austria split off 01.10.2018) |
| Load | **Amprion zonal** day-ahead forecast | `country: "de"` — national aggregate |
| Renewables | wind+solar from **3 of 4 TSOs** (Amprion, TenneT, 50Hertz) | national aggregate |

The model is being served exogenous features on a **different scale and
different geographic definition** than it was trained on, and the price
target itself is a different bidding zone. German wind+solar capacity also
grew enormously between 2017 and 2026, so live `exog_*` values likely sit
far outside the training support.

**The OOD error ordering is consistent with this.** Covariate shift outside
training support punishes models that cannot extrapolate. Trees cannot;
linear models can. Observed: LightGBM (trees) worst at 1.828, LSTM 1.520,
then SARIMAX 1.145 and LEAR-LASSO (linear) best at 1.087. That is exactly
the ordering covariate shift predicts, and it is *not* what a pure price
level shift alone would predict.

**Test it before building anything on top:** compare the distributions of
each `exog_*` feature between the 2012–2017 training window and the live
2026 window — overlap, range, and fraction of live values outside the
training min/max. Cheap, read-only, and decisive. If the live features are
largely out of support, the fix is feature rescaling or retraining on
current-definition data, **not** more bias correction.

### 10.4 Scope rules are DIFFERENT for this arm — read carefully

The §7 prohibition on new data sources exists **only** to preserve protocol
equivalence with Lago et al. An operational 2026 model is not being compared
to their benchmark, so that constraint does not apply to it.

For the operational arm, adding weather, fuel prices, or full ENTSO-E data
is **legitimate and probably necessary**. The 2012–2017 feature set was
chosen for comparability, not because it is the best available set for
forecasting 2026 prices in a gas-coupled, high-renewables market.

**Keep the two tracks explicitly separate.** Two models, two rule sets, two
sections of the thesis:

- **Benchmark model** — frozen feature set, protocol-equivalent, compared to
  Lago. Chapter 4.
- **Operational model** — free feature set, evaluated on live 2026 data
  against the naive baseline on that same window. Chapter 5.

Never compare the operational model's numbers to Lago's table. Different
data, different period, different market definition — the comparison is
meaningless and an examiner will say so.

### 10.5 Why this is a contribution, not just engineering

The benchmark-optimal model and the 2026-optimal model are **not the same
model**, and you can now demonstrate that with numbers rather than assert
it: LEAR-LASSO is mid-table on the benchmark but the **best** trained model
under OOD, and it is the one that bias correction *harms*. LightGBM is
competitive on the benchmark and the **worst** under OOD.

"Benchmark rank does not predict out-of-distribution rank" is a real finding
about EPF evaluation practice, it speaks directly to the benchmark's own
stated purpose, and it is supported by results you have already frozen.

### 10.6 Deliverable checklist for this arm

- [ ] Feature-support diagnostic (§10.3) — do this first, it may redirect
      everything else
- [ ] Decide and document the bidding-zone mismatch (DE-AT-LU vs DE-LU):
      accept as a limitation, or retrain on current-definition data
- [ ] Daily scheduled run at the D−1 cadence (§10.1), not a live tick
- [ ] Bias-correction layer with the window as a config parameter, default
      3–7 days per the recalibration result
- [ ] Evaluate against the **naive baseline on the 2026 window** — that is
      the only fair comparator here
- [ ] Both routes reported: hourly and daily-direct
- [ ] Attribution string displayed (CC BY 4.0, already in config)

---

## 11. Suggested sequence

1. Resolve the promotion decision (§1)
2. **Novelty gate** (§5) — abandon or proceed on the evidence
3. **Pre-register the improvement gate** (§6) into `logs/decisions.md` —
   criterion, floor, adoption rule, correction, stopping rule — *before* any
   rung runs
4. **Feature-support diagnostic** (§10.3) — cheap, read-only, and it may
   redirect the whole operational arm before effort is spent on it
5. Cheap rungs: per-hour convex → per-hour unconstrained, post-hoc on saved
   frames, DM-tested between rungs under §6.3
6. LEAR-only multi-window sweep (config-only, minutes)
7. Only if 5–6 clear the gate: non-linear multi-window arm
8. Operational arm (§10) — separate track, separate rules, separate chapter
9. Log every rung in `logs/decisions.md`, including null results (§6.6)

Two gates, two different failure modes. §5 catches *"someone already did
this"*. §6 catches *"we fooled ourselves into thinking it worked"*. Passing
one does not excuse the other, and both are cheaper than discovering the
problem at the defense.

And two tracks, two rule sets: the **benchmark** model is bound by protocol
equivalence (§7), the **operational** model is not (§10.4). Confusing them
in either direction — restricting the operational model's features for no
reason, or letting extra data leak into the benchmark comparison — breaks
something important. Keep them in separate chapters.
