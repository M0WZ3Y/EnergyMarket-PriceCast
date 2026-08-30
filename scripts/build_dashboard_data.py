#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dashboard_data.py  —  MVP results dashboard generator (CONSUMER ONLY).

Reads the persisted LEAR-LASSO prediction cache, recomputes the pooled ablation
statistics + the cache-derivable regime slices, attaches the curated block
verdicts, and emits BOTH:

    reports/dashboard/ablation_dashboard.json   (durable, reproducible artifact)
    reports/dashboard/ablation_dashboard.html   (self-contained, double-click)

Design rules this script obeys (enforced in code, not by good intentions):

  * OFFLINE.        No network. A socket guard aborts the process if anything
                    tries to open one. No `import src.*` — the research pipeline,
                    ablation harness, feature pipeline and model code are never
                    touched. Everything here is numpy/scipy over persisted CSVs.

  * FILENAME-PINNED. Variants are resolved by EXACT filename, never by
                    basename.split("__")[0]. The cache holds several generations
                    of some variants (baseline/B1/B5 each appear 2-3x with
                    different MAEs because the common-day pool shifted). Resolving
                    by name would silently mix generations and corrupt every
                    delta and p-value. Each table pins the precise file it uses
                    and verifies its sha256.

  * FROZEN READ-ONLY. data/ablation_cache/ is never written. The exact CSVs the
                    tables consume are copied ONCE into reports/dashboard/inputs/
                    and tracked there, so the dashboard is reproducible from a
                    clean clone without un-gitignoring the frozen cache.

  * SOURCE-INTEGRITY. Curated block verdicts carry decisions.md evidence anchors.
                    Before shipping, the script asserts each cited anchor string
                    is still present in decisions.md and fails loud otherwise, so
                    a future edit can never silently desync the dashboard.

  * REPRODUCTION GATE. The standalone statistics (reimplemented here to avoid
                    importing pipeline code) must reproduce the published
                    p-values and regime MAEs to the digit. A mismatch is a
                    definitional divergence, not a rounding nit — it aborts, and
                    is fixed by aligning to scripts/run_ablation_dm.py, never by
                    tuning the tolerance.

Usage:
    python scripts/build_dashboard_data.py            # build from pinned inputs
    python scripts/build_dashboard_data.py --check    # validate only, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# 0.  Offline guard.  Any socket attempt kills the run.                        #
# --------------------------------------------------------------------------- #
def _no_network(*_a, **_k):
    raise RuntimeError("network access is forbidden in the dashboard generator")

socket.socket = _no_network  # type: ignore[assignment]

# --------------------------------------------------------------------------- #
# 1.  Paths.                                                                   #
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO / "reports" / "dashboard" / "inputs"        # tracked, pinned copies
OUT_DIR = REPO / "reports" / "dashboard"
TEMPLATE = OUT_DIR / "template.html"                          # ships with the repo
DECISIONS = REPO / "logs" / "decisions.md"

WINDOW = {"start": "2017-10-13", "end": "2017-12-31", "n_days": 80, "n_hours": 1920}
VEHICLE = "LEAR-LASSO (LassoLarsIC، criterion=aic)"

# --------------------------------------------------------------------------- #
# 2.  INPUT MANIFEST — the correctness core.  Pin exact files per table.        #
#                                                                              #
#     The cache holds SEVERAL GENERATIONS of baseline / B1_ramp / B5 with       #
#     materially different MAEs, because the cache fingerprint includes the     #
#     common day-pool size and that pool shifted as variants were added.        #
#     Resolving by basename.split("__")[0] therefore picks whichever file sorts #
#     last and silently mixes generations, corrupting every delta and p-value.  #
#                                                                              #
#       gen A  pooled + pairwise tables   baseline MAE 6.2645  spike 13.6412    #
#       gen C  B5 spike-control table     baseline MAE 6.2747  spike 13.6768    #
#                                                                              #
#     Every file is pinned by exact name AND sha256, verified on load.          #
# --------------------------------------------------------------------------- #
BASE_A = "baseline__479cc81ed49b.csv"       # gen A reference
BASE_C_FILE = "baseline__27b3ac2f6ccb.csv"  # gen C reference (B5 control table)

SHA256 = {
    "baseline__479cc81ed49b.csv":            "0e0ede8545e7957e3cf4eaf5b7f01f0fc9067f30768d9ffb300efd1e3864585f",
    "B1_ramp__c5e646a1922e.csv":             "355a264733bfed9840c0be03c76297f7112aae46321c09b7474396e22af317d4",
    "B2_merit_explicit__95baadcab6a5.csv":   "2deaa4e9a078be1966bd2ec33079b1c4bf3f3d44cac955c8f99248227a9b503b",
    "B3_carbon_switch__5049c2c411a8.csv":    "c568182907ce1ee7d78e9737a3cf8cff3833fdbfe71ac5bd38e5edcd43c98bb5",
    "B4_coupling__7a426b3f71ef.csv":         "fab88ffe832d2d300f966a423611451ecf06638d0ed9f4f4dd7a4737eb6fa840",
    "B5_reserve_margin__850e188fccf6.csv":   "57e3201d790cc5f0f20d11edab2ad37f6ca4c19def5aa82a3a6f3b3ed92bc20c",
    "B6_storage__7b960f39f311.csv":          "c5290f1095eff9aefec563c64551911dabf606b14852aa7e01aa893317fb3273",
    "ALL__7d6b7e928c61.csv":                 "a339d8095503e0ca5bc10cbf99318252d81c95f38137ca75ea28d4474775ef8d",
    "B1+B2__40e9d841caeb.csv":               "14b55c5b95f0e13f7f149bdbc66e9378fb1c2a49b3957fa68bf4df29fdb4beb6",
    "B1+B3__107db6153516.csv":               "688643442f410d4bab37f20b3f9ffe5824a2d0afbb1cbe426cdd5d3d2e2079e6",
    "B1+B4__d308c35d5a76.csv":               "e3925ce3b38bf46e601d65fa96261758ce444d587c7d07a2b7fbbe42bb35196a",
    "B1+B5__e1c6aa572ae1.csv":               "c536369ea9e7edb474b1ed7f396c04b2b8eff955320093e23f6449cf335f9da2",
    "B1+B6__8f250b86a4ef.csv":               "129a062caf375f29c4fe61387b6079f349f33b5082b38a625d24284863967631",
    "B1+coupling_split__a2556792e786.csv":   "717834a0185b50119dadabea13320db8b1264d4037ffbcb4593200208ba461cd",
    "B1+coupling_state__7817d8c45cf2.csv":   "f93b8a6983316fa866a8d604a1860f9b0364a4122661835d55b4c8d89ff572fd",
    "B1+headroom__93d48eefcc54.csv":         "9cd0e836e10dcd29e840158a900f070c294dd2ea2d03f1704b05b0bf973afebb",
    "baseline__27b3ac2f6ccb.csv":            "2e91b4a08a5674b82dffc29a171e8f279bcc7abbfd71ece5cb937211688ef3d7",
    "B5_reserve_margin__f75982ed9ed6.csv":   "86f211c1503eaa85beaa12addf90eca59e445ad26a9df3b74306e65afbdc6679",
    "CTRL_noise__dedeba7136e3.csv":          "4d6d0c9a7d94f7723b70f09a6ea83b13379579d2c81dd9aa0414337b493da9b7",
    "CTRL_shifted_load__a48401320e96.csv":   "a8f8a7af3fd0df7a49ff1c7e4000a22c1bcf1920ec61ba6884e7afbcef091b1e",
    "CTRL_scrambled_year__a6ac9b5e1273.csv": "14143f8c7d504fa37734eeebedf476d06257d8880065a3f8479f6df46daf7c08",
}

# gen A — single blocks, compared to BASELINE. List order is display order.
POOLED = [
    dict(key="baseline", file=BASE_A, ref=None,
         name_fa="مبنا", verdict="REFERENCE"),
    dict(key="B1_ramp", file="B1_ramp__c5e646a1922e.csv", ref="baseline",
         name_fa="B1 — بار پسماند + گرادیان", verdict="EFFECTIVE"),
    dict(key="B2_merit", file="B2_merit_explicit__95baadcab6a5.csv", ref="baseline",
         name_fa="B2 — ترتیب شایستگی", verdict="REDUNDANT"),
    dict(key="B3_carbon", file="B3_carbon_switch__5049c2c411a8.csv", ref="baseline",
         name_fa="B3 — کربن + جابه‌جایی سوخت", verdict="TIMESCALE_MISMATCH"),
    dict(key="B4_coupling", file="B4_coupling__7a426b3f71ef.csv", ref="baseline",
         name_fa="B4 — جفت‌شدگی بازار", verdict="MISSPECIFIED"),
    dict(key="B5_reserve", file="B5_reserve_margin__850e188fccf6.csv", ref="baseline",
         name_fa="B5 — کمیابی / حاشیهٔ ذخیره", verdict="DEGENERATE"),
    dict(key="B6_storage", file="B6_storage__7b960f39f311.csv", ref="baseline",
         name_fa="B6 — ذخیره‌سازی / آبی", verdict="MISSPECIFIED"),
    dict(key="ALL", file="ALL__7d6b7e928c61.csv", ref="baseline",
         name_fa="همهٔ بلوک‌ها", verdict="NS"),
]

# gen A — pairwise, compared to B1.  `shown` marks the contract's display rows.
# The Holm family is ALL EIGHT tested pairwise variants, matching the published
# table (decisions.md 2026-08-29, "Pairwise ablation"), so correcting over a
# truncated family can never inflate significance.
PAIRWISE = [
    dict(key="B1_ramp", file="B1_ramp__c5e646a1922e.csv", ref=None, shown=True,
         name_fa="B1 (مرجع)", verdict="REFERENCE"),
    dict(key="B1+coupling_split", file="B1+coupling_split__a2556792e786.csv",
         ref="B1_ramp", shown=True, name_fa="B1 + تفکیک جفت‌شدگی", verdict="NS"),
    dict(key="B1+B2", file="B1+B2__40e9d841caeb.csv", ref="B1_ramp", shown=True,
         name_fa="B1 + ترتیب شایستگی", verdict="NS"),
    dict(key="B1+B3", file="B1+B3__107db6153516.csv", ref="B1_ramp", shown=True,
         name_fa="B1 + کربن", verdict="NS"),
    dict(key="B1+B4", file="B1+B4__d308c35d5a76.csv", ref="B1_ramp", shown=True,
         name_fa="B1 + جفت‌شدگی", verdict="NS"),
    dict(key="B1+B5", file="B1+B5__e1c6aa572ae1.csv", ref="B1_ramp", shown=True,
         name_fa="B1 + کمیابی", verdict="NS"),
    dict(key="B1+B6", file="B1+B6__8f250b86a4ef.csv", ref="B1_ramp", shown=True,
         name_fa="B1 + ذخیره‌سازی", verdict="NS"),
    dict(key="B1+coupling_state", file="B1+coupling_state__7817d8c45cf2.csv",
         ref="B1_ramp", shown=False, name_fa="B1 + وضعیت جفت‌شدگی", verdict="NS"),
    dict(key="B1+headroom", file="B1+headroom__93d48eefcc54.csv", ref="B1_ramp",
         shown=False, name_fa="B1 + هدروم", verdict="NS"),
]

# gen C — B5 spike-control table, its own baseline. Holm family = these 4 tests.
B5_CONTROLS = [
    dict(key="B5", file="B5_reserve_margin__f75982ed9ed6.csv",
         name_fa="کمیابی / حاشیهٔ ذخیره", role_fa="اثرِ مشاهده‌شده",
         ruled_out_fa="—"),
    dict(key="CTRL_noise", file="CTRL_noise__dedeba7136e3.csv",
         name_fa="کنترل: ستون‌های نوفه", role_fa="کنترل",
         ruled_out_fa="رد می‌کند: اثر صرفاً از افزودنِ ستون‌های تصادفی نیست."),
    dict(key="CTRL_shifted_load", file="CTRL_shifted_load__a48401320e96.csv",
         name_fa="کنترل: بارِ جابه‌جاشده", role_fa="کنترل",
         ruled_out_fa="رد می‌کند: اثر از هم‌ترازیِ ساده‌ٔ بار نیست."),
    dict(key="CTRL_scrambled_year", file="CTRL_scrambled_year__a6ac9b5e1273.csv",
         name_fa="کنترل: سالِ درهم‌ریخته", role_fa="کنترل",
         ruled_out_fa="رد می‌کند: اثر از ساختارِ برهم‌کنشِ سال نیست."),
]

HEADLINE_FA = ("هیچ بلوکِ داده‌ٔ بیرونی، B1 را بهبود نداد — نه به‌تنهایی، نه در ترکیب،"
               " نه با AIC و نه با BIC.")

HEADLINE_TEST_LABEL_FA = "آزمون تفکیک‌پذیریِ «B1 در برابر همهٔ بلوک‌ها»"
HEADLINE_TEST_STATEMENT_FA = ("دو مدل از هم تفکیک‌پذیر نیستند؛ افزودن بلوک‌های بیرونی"
                              " سودی نمی‌رساند.")

B5_ANOMALY_FA = ("بهبودِ B5 در رژیمِ جهش واقعی و معنادار است (هولم = ۰٫۰۳۰)، اما"
                 " مکانیزمِ آن ناشناخته است.")
B5_CAVEAT_FA = ("n = ۹۶ ساعتِ جهش، یک پنجرهٔ ۸۰ روزه — هرگز به‌عنوان نتیجهٔ «کمیابی»"
                " بازتفسیر نشود.")
B5_SEGMENT_FA = "رژیمِ جهشِ قیمت"

# --------------------------------------------------------------------------- #
# 3.  Curated block verdicts, transcribed from the persisted record WITH        #
#     evidence anchors re-checked against their source file before shipping.    #
#                                                                              #
#     A source-resolution pass over the working tree and all 129 revisions on   #
#     every branch established what is and is not actually recorded. Three      #
#     figures carried by the design preview did not survive it:                 #
#                                                                              #
#       * B2's "r = 0.956 against exog_2" EXISTS NOWHERE — not in the working   #
#         tree, not in any revision, not in the deleted docs/ tree. No          #
#         collinearity diagnosis for B2 was ever persisted. The evidence line   #
#         states the pairwise ablation result that IS on record instead.        #
#       * B3's "CoV = 0.056" mislabelled the quantity. Both numbers are real    #
#         and mean different things: the EUA carbon CoV is 0.031, and 0.056 is  #
#         its RATIO to the target's own CoV of 0.553 — and the ratio is what    #
#         STATIC_COV_RATIO actually tests. The line now shows both, with the    #
#         noun corrected.                                                       #
#       * B6's evidence cites its recorded +0.902 target-regime delta, which is #
#         harm-on-target — evidence consistent with the verdict, not proof of   #
#         it.                                                                   #
#                                                                              #
#     VERDICT TIERING. The verdict words themselves were produced by            #
#     diagnose_block, whose output was printed and never persisted (the harness #
#     is print-only and no stdout log was captured). For B1/B3/B4/B5 the        #
#     substance is independently recorded — gate-verified statistics, a logged  #
#     reclassification, or a pre-registered and confirmed prediction — so those #
#     read "measured". For B2 and B6 nothing beyond the ablation delta was ever #
#     written down, so their labels are defensible readings of measured         #
#     evidence rather than measurements, and are marked "interpretive". The     #
#     labels are kept; the tier states their standing honestly.                 #
#                                                                              #
#     Each anchor names its own source file, because the record is split: the   #
#     ratio lives in the diagnostic that applies it, not in the decision log.   #
#     Anchors are verified as literal substrings by reading those files as      #
#     TEXT — nothing under src/ is imported or executed.                        #
# --------------------------------------------------------------------------- #
DEC = "logs/decisions.md"
COL = "src/features/collinearity.py"

BLOCKS = [
    dict(id="B1", name_fa="بار پسماند + گرادیان", verdict="EFFECTIVE",
         verdict_status="measured",
         summary_fa="تنها بلوکِ موفق. بهبود ۷٫۷٪ نسبت به مبنا، معنادار.",
         evidence_fa="کاهش MAE از ۶٫۲۶ به ۵٫۷۸ (Δ=−۰٫۴۸)؛ مقدار p خام ۵×۱۰⁻⁵.",
         anchors=[(DEC, "5.7811")]),
    dict(id="B2", name_fa="ترتیب شایستگی (merit order)", verdict="REDUNDANT",
         verdict_status="interpretive",
         summary_fa="افزونه نسبت به exog_2؛ اطلاعات تازه‌ای اضافه نمی‌کند.",
         evidence_fa="آنچه exog_2 پیش‌تر در خود دارد را صریح می‌کند؛ افزودنِ آن به B1"
                     " بهبودی نمی‌دهد (B1+B2 = ۵٫۸۴۹۴، p خام ۰٫۶۳۲، هولم ۱٫۰۰۰).",
         anchors=[(DEC, "0.632")]),
    dict(id="B3", name_fa="کربن + جابه‌جایی سوخت", verdict="TIMESCALE_MISMATCH",
         verdict_status="measured",
         summary_fa="سیگنالِ کربن در مقیاسِ روزانه تقریباً ثابت است؛ قدرت تفکیک ندارد.",
         evidence_fa="سیگنالِ کربن در مقیاسِ روزانه تقریباً ثابت است؛ ضریبِ تغییراتِ"
                     " EUA کربن ۰٫۰۳۱ و نسبتِ آن به ضریبِ تغییراتِ هدف (۰٫۵۵۳) برابرِ"
                     " ۰٫۰۵۶ است — همین نسبت، کمیتی است که آستانهٔ تشخیص روی آن"
                     " می‌سنجد.",
         anchors=[(DEC, "0.031"), (DEC, "0.553"), (COL, "ratio 0.056")]),
    dict(id="B4", name_fa="جفت‌شدگی بازار (market coupling)", verdict="MISSPECIFIED",
         verdict_status="measured",
         summary_fa="بدتصریح‌شده؛ روی رژیمِ هدفِ خودش بدتر عمل می‌کند.",
         evidence_fa="افزایشِ خطا در رژیمِ تنشِ جفت‌شدگی نسبت به مبنا (مسئلهٔ کدگذاری)."
                     " افزونگی از پیش رد شده بود: میانهٔ بیشینهٔ |r| = ۰٫۵۹۹ و هیچ"
                     " ستونی با |r| ≥ ۰٫۹۰.",
         anchors=[(DEC, "0.599")]),
    dict(id="B5", name_fa="کمیابی / حاشیهٔ ذخیره (پروکسیِ ظرفیت)", verdict="DEGENERATE",
         verdict_status="measured",
         summary_fa="تباهیده؛ تابعِ خطیِ دقیقی از بار است و متغیرِ مستقلی نمی‌سازد.",
         evidence_fa="هم‌خطیِ کامل با سریِ بار (باقیماندهٔ درون‌سالی نسبت به exog_1"
                     " برابر ۳×۱۰⁻¹⁵). استثنا: ناهنجاریِ رژیمِ جهش (بخشِ کنترل‌ها).",
         anchors=[(DEC, "3e-15")]),
    dict(id="B6", name_fa="ذخیره‌سازی / آبی (hydro)", verdict="MISSPECIFIED",
         verdict_status="interpretive",
         summary_fa="بدتصریح‌شده؛ نگاشتِ ویژگی به رژیمِ هدف نادرست است.",
         evidence_fa="روی رژیمِ هدفِ خودش بهبودی ندارد؛ Δ=+۰٫۹۰۲ (زیان، نه سود) —"
                     " سنجیده و ثبت‌شده.",
         anchors=[(DEC, "+0.902")]),
]

HYPOTHESES = [
    dict(id="H1", text_fa="مدل‌های آماری/خطی (SARIMAX، LEAR)"),
    dict(id="H2", text_fa="شبکه‌های عصبی بازگشتی (LSTM، GRU)"),
    dict(id="H3", text_fa="ترکیب وزن‌دار مدل‌ها (ensemble)"),
    dict(id="H4", text_fa="خط‌لولهٔ ویژگی‌های فیزیکی"),
]

# --------------------------------------------------------------------------- #
# 4.  Statistics — self-contained reimplementation of scripts/run_ablation_dm.py #
#     (never imported: that module pulls in the research pipeline). The variance #
#     conventions are NOT free parameters — ddof=1 on the daily loss             #
#     differential and the ddof=1 lag autocovariance are what reproduce the      #
#     published p-values to the digit. See GATES.                               #
# --------------------------------------------------------------------------- #
def load_preds(path: Path, expect_sha: str | None = None):
    """Return (err, y_true) as (n_days, 24) arrays, origin-major, hour-minor."""
    raw = path.read_bytes()
    if expect_sha:
        got = hashlib.sha256(raw).hexdigest()
        if got != expect_sha:
            raise ValueError(
                f"sha256 mismatch for {path.name}:\n  got      {got}\n"
                f"  expected {expect_sha}\nthe pinned input changed — refusing to ship."
            )
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    origins = sorted({r["origin"] for r in rows})
    idx = {o: i for i, o in enumerate(origins)}
    err = np.full((len(origins), 24), np.nan)
    ytrue = np.full((len(origins), 24), np.nan)
    for r in rows:
        i, h = idx[r["origin"]], int(r["hour"])
        err[i, h] = float(r["y_true"]) - float(r["y_pred"])
        ytrue[i, h] = float(r["y_true"])
    if np.isnan(err).any():
        raise ValueError(f"{path.name}: incomplete {len(origins)}x24 grid")
    return err, ytrue


def mae(err: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = np.abs(err)
    return float(a[mask].mean()) if mask is not None else float(a.mean())


def dm_multivariate(err_a: np.ndarray, err_b: np.ndarray) -> float:
    """One-sided multivariate DM: p that A has GREATER loss than B (B is better).

    Losses are averaged over the 24 hours of a day BEFORE differencing — one
    auction sets all 24 prices, so treating hours as independent draws would
    inflate the effective sample ~24x and manufacture significance.
    """
    d = np.abs(err_a).mean(axis=1) - np.abs(err_b).mean(axis=1)
    if np.allclose(d, 0):
        return float("nan")
    var = d.var(ddof=1) / d.size
    if var <= 0:
        return float("nan")
    return float(1.0 - stats.norm.cdf(d.mean() / np.sqrt(var)))


def dm_hac(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, int]:
    """One-sided DM with Newey-West HAC variance, for a REGIME subset.

    A regime is a set of hours, not whole days, so the daily aggregation above
    does not apply; the hourly loss differential is autocorrelated instead. The
    bandwidth is derived, not chosen: floor(4*(n/100)**(2/9)); n=96 gives 3.
    """
    d = loss_a - loss_b
    n = d.size
    if n < 30 or np.allclose(d, 0):
        return float("nan"), 0
    bw = int(np.floor(4 * (n / 100) ** (2 / 9)))
    s = float(np.var(d, ddof=1))
    for lag in range(1, bw + 1):
        s += 2 * (1 - lag / (bw + 1)) * float(np.cov(d[lag:], d[:-lag])[0, 1])
    if s <= 0:
        return float("nan"), bw
    return float(1.0 - stats.norm.cdf(d.mean() / np.sqrt(s / n))), bw


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down, order-preserving. NaNs pass through and do NOT
    count toward the family size (matching run_ablation_dm.py)."""
    idx = [i for i, v in enumerate(pvals) if v is not None and not np.isnan(v)]
    m = len(idx)
    adj: list[float] = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda k: pvals[k])):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def jnum(v) -> float | None:
    """JSON-safe: NaN is not valid JSON and the template treats null as 'absent'."""
    if v is None:
        return None
    f = float(v)
    return None if np.isnan(f) else f

# --------------------------------------------------------------------------- #
# 5.  Regime masks — standalone (no pipeline import). MVP-safe subset only:     #
#     spike and negative price come purely from y_true, so they need no exog    #
#     and stay strictly offline. The exog-derived regimes (steep_ramp,          #
#     high_res, low_residual, coupling_stress, high_hydro) need DE.csv and      #
#     ec_*.csv and are the deferred fast-follow.                                #
# --------------------------------------------------------------------------- #
SPIKE_Q = 0.95  # published threshold; validated in the gates below

def regime_masks(ytrue: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "spike": ytrue >= np.quantile(ytrue, SPIKE_Q),
        "negative_price": ytrue < 0.0,
    }

REGIME_META = {
    "spike": dict(name_fa="جهش قیمت", definition_fa="۵٪ بالاییِ قیمتِ واقعی",
                  caveat_fa="جانشینِ دُمِ قیمت است، نه کمیابیِ اندازه‌گیری‌شده."),
    "negative_price": dict(name_fa="قیمت منفی", definition_fa="قیمتِ واقعیِ کمتر از صفر",
                           caveat_fa="مستقیم از y_true؛ بدون نیاز به داده‌ٔ برون‌زا."),
}

DEFERRED_REGIMES = [
    dict(id="steep_ramp", name_fa="شیبِ تندِ تغییر",
         reason_fa="نیازمندِ exog_1/exog_2 — گامِ بعدی"),
    dict(id="high_res", name_fa="نفوذِ بالای تجدیدپذیر",
         reason_fa="نیازمندِ داده‌ٔ برون‌زا — گامِ بعدی"),
    dict(id="low_residual", name_fa="بارِ پسماندِ پایین",
         reason_fa="نیازمندِ داده‌ٔ برون‌زا — گامِ بعدی"),
    dict(id="coupling_stress", name_fa="تنشِ جفت‌شدگی",
         reason_fa="نیازمندِ ec_*.csv — گامِ بعدی"),
    dict(id="high_hydro", name_fa="تولیدِ آبیِ بالا",
         reason_fa="نیازمندِ ec_*.csv — گامِ بعدی"),
]

# --------------------------------------------------------------------------- #
# 6.  Gates.  These must PASS, never be bypassed or retuned to fit.             #
# --------------------------------------------------------------------------- #
def check_source_integrity() -> None:
    """Every figure shown on a block card must still be a literal substring of the
    file it is cited from. Source files are read as TEXT — nothing is imported."""
    cache: dict[str, str] = {}
    missing: list[str] = []
    for b in BLOCKS:
        for relpath, literal in b["anchors"]:
            if relpath not in cache:
                f = REPO / relpath
                if not f.exists():
                    raise FileNotFoundError(
                        f"missing {relpath}; cannot verify curated verdicts")
                cache[relpath] = f.read_text(encoding="utf-8", errors="ignore")
            if literal not in cache[relpath]:
                missing.append(f'{b["id"]}: {literal!r} not in {relpath}')
    if missing:
        raise AssertionError(
            "cited evidence anchors no longer present — a curated verdict has "
            "desynced from the record; refusing to ship:\n    "
            + "\n    ".join(missing)
        )


def check_inputs_present() -> None:
    missing = sorted(f for f in SHA256 if not (INPUT_DIR / f).exists())
    if missing:
        raise FileNotFoundError(
            f"pinned inputs missing from {INPUT_DIR}: {missing}\n"
            f"copy them out of data/ablation_cache/o80_c990_aic/ (read-only)."
        )


def run_gates(E: dict, Y: dict) -> list[dict]:
    """Recomputed vs published. A miss aborts the build; it is fixed by aligning
    to scripts/run_ablation_dm.py, never by loosening the tolerance."""
    ma = regime_masks(Y[BASE_A])["spike"]
    mc = regime_masks(Y[BASE_C_FILE])["spike"]
    lb_c = np.abs(E[BASE_C_FILE])[mc]
    ctrl_p = [dm_hac(lb_c, np.abs(E[c["file"]])[mc])[0] for c in B5_CONTROLS]
    ctrl_h = holm(ctrl_p)

    g = [
        ("baseline pooled MAE (gen A)", mae(E[BASE_A]), 6.2645, 4),
        ("baseline pooled MAE (gen C)", mae(E[BASE_C_FILE]), 6.2747, 4),
        ("baseline spike MAE (gen A)", mae(E[BASE_A], ma), 13.6412, 4),
        ("baseline spike MAE (gen C)", mae(E[BASE_C_FILE], mc), 13.6768, 4),
        ("p(B1 better than baseline)",
         dm_multivariate(E[BASE_A], E["B1_ramp__c5e646a1922e.csv"]), 0.00005, 5),
        ("p(B1 better than ALL)",
         dm_multivariate(E["ALL__7d6b7e928c61.csv"], E["B1_ramp__c5e646a1922e.csv"]),
         0.3854, 4),
        ("p(ALL better than B1)",
         dm_multivariate(E["B1_ramp__c5e646a1922e.csv"], E["ALL__7d6b7e928c61.csv"]),
         0.6146, 4),
        ("p(B1+coupling_split better than B1)",
         dm_multivariate(E["B1_ramp__c5e646a1922e.csv"],
                         E["B1+coupling_split__a2556792e786.csv"]), 0.2289, 4),
        ("B5 spike p raw", ctrl_p[0], 0.0075, 4),
        ("B5 spike p Holm", ctrl_h[0], 0.030, 3),
    ]
    out = [dict(label=l, got=float(got), published=pub,
                ok=round(float(got), dec) == round(pub, dec)) for l, got, pub, dec in g]
    bad = [r for r in out if not r["ok"]]
    if bad:
        lines = "\n".join(
            f"    {r['label']:38s} got {r['got']:.6f}  published {r['published']}"
            for r in bad)
        raise AssertionError(
            "REPRODUCTION GATE FAILED — recomputed statistics do not match the "
            "published analysis:\n" + lines +
            "\nDo not tune to fit. Align the loss aggregation / Holm family to "
            "scripts/run_ablation_dm.py and re-run."
        )
    return out

# --------------------------------------------------------------------------- #
# 7.  Build.                                                                    #
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def _table(specs: list[dict], E: dict, ref_key: str) -> dict:
    """Assemble one reference set. Every row carries its own `reference`, so the
    baseline-referenced and B1-referenced sets can never be mixed by the view."""
    by_key = {s["key"]: s for s in specs}
    ref_file = by_key[ref_key]["file"]
    ref_mae = mae(E[ref_file])

    tested = [s for s in specs if s["ref"] is not None]
    praw = [dm_multivariate(E[by_key[s["ref"]]["file"]], E[s["file"]]) for s in tested]
    pholm = holm(praw)
    stat = {s["key"]: (p, h) for s, p, h in zip(tested, praw, pholm)}

    rows = []
    for s in specs:
        if not s.get("shown", True):
            continue                      # tested for the Holm family, not displayed
        m = mae(E[s["file"]])
        p, h = stat.get(s["key"], (None, None))
        rows.append(dict(
            variant=s["key"], name_fa=s["name_fa"], reference=s["ref"],
            mae=round(m, 4),
            delta=None if s["ref"] is None else round(m - ref_mae, 4),
            pct=None if s["ref"] is None else round(100.0 * (m - ref_mae) / ref_mae, 2),
            p_raw=jnum(p), p_holm=jnum(h), verdict=s["verdict"],
        ))
    return dict(reference=ref_key, reference_mae=round(ref_mae, 4),
                holm_family_size=len(tested), rows=rows)


def build(check_only: bool = False) -> dict:
    check_source_integrity()
    check_inputs_present()

    E: dict[str, np.ndarray] = {}
    Y: dict[str, np.ndarray] = {}
    for fname, sha in SHA256.items():
        e, y = load_preds(INPUT_DIR / fname, sha)
        E[fname], Y[fname] = e, y

    gates = run_gates(E, Y)

    # --- 1. pooled tables ---------------------------------------------------
    vs_baseline = _table(POOLED, E, "baseline")
    vs_baseline["reference_label_fa"] = "نسبت به مبنا"
    vs_b1 = _table(PAIRWISE, E, "B1_ramp")
    vs_b1["reference_label_fa"] = "نسبت به B1"

    e_all, e_b1 = E["ALL__7d6b7e928c61.csv"], E["B1_ramp__c5e646a1922e.csv"]
    headline_test = dict(
        label_fa=HEADLINE_TEST_LABEL_FA,
        p_b1_better=round(dm_multivariate(e_all, e_b1), 4),
        p_all_better=round(dm_multivariate(e_b1, e_all), 4),
        statement_fa=HEADLINE_TEST_STATEMENT_FA,
    )

    # --- 3. regime slices (gen A baseline) ----------------------------------
    masks = regime_masks(Y[BASE_A])
    available = [
        dict(id=rid, name_fa=REGIME_META[rid]["name_fa"], n=int(masks[rid].sum()),
             baseline_mae=round(mae(E[BASE_A], masks[rid]), 4),
             definition_fa=REGIME_META[rid]["definition_fa"],
             caveat_fa=REGIME_META[rid]["caveat_fa"])
        for rid in ("spike", "negative_price")
    ]

    # --- 4. B5 spike controls (gen C, own baseline) -------------------------
    mc = regime_masks(Y[BASE_C_FILE])["spike"]
    lb = np.abs(E[BASE_C_FILE])[mc]
    base_spike = float(lb.mean())
    praw, bws = [], []
    for c in B5_CONTROLS:
        p, bw = dm_hac(lb, np.abs(E[c["file"]])[mc])
        praw.append(p)
        bws.append(bw)
    pholm = holm(praw)
    b5_rows = []
    for c, p, h in zip(B5_CONTROLS, praw, pholm):
        m = mae(E[c["file"]], mc)
        b5_rows.append(dict(
            variant=c["key"], name_fa=c["name_fa"], mae=round(m, 4),
            delta=round(m - base_spike, 4), p_raw=jnum(p), p_holm=jnum(h),
            role_fa=c["role_fa"], ruled_out_fa=c["ruled_out_fa"],
        ))

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": "scripts/build_dashboard_data.py",
            "git_commit": git_commit(),
            "repo_tags": ["v1.0-results", "v1.1-ood", "v1.2-physical-features"],
            "window": WINDOW,
            "vehicle": VEHICLE,
            "is_preview": False,
            "hac_bandwidth": int(bws[0]),
            "generations": {"pooled": "A", "pairwise": "A", "b5_controls": "C"},
            "inputs": {f: SHA256[f] for f in sorted(SHA256)},
            "gates": gates,
        },
        "headline_fa": HEADLINE_FA,
        "hypotheses": HYPOTHESES,
        "pooled": {"vs_baseline": vs_baseline, "vs_B1": vs_b1,
                   "headline_test": headline_test},
        "blocks": [{k: v for k, v in b.items() if k != "anchors"} for b in BLOCKS],
        "regimes": {"available": available, "deferred": DEFERRED_REGIMES},
        "b5_controls": {
            "segment_fa": B5_SEGMENT_FA, "n": int(mc.sum()),
            "baseline_mae": round(base_spike, 4),
            "anomaly_fa": B5_ANOMALY_FA, "caveat_fa": B5_CAVEAT_FA, "rows": b5_rows,
        },
    }


def emit(payload: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "ablation_dashboard.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Inline at build time: file:// fetch() is blocked by browsers, so the data
    # is injected into the template rather than fetched at runtime. This is what
    # lets the dashboard open by double-click with no server and no build step.
    with open(TEMPLATE, encoding="utf-8", newline="") as f:
        tpl = f.read()
    marker = "/*__DASHBOARD_DATA__*/null"
    if tpl.count(marker) != 1:
        raise AssertionError(
            f"template.html must contain exactly one {marker!r}; found {tpl.count(marker)}")
    blob = json.dumps(payload, ensure_ascii=False)
    html = tpl.replace(marker, "/*__DASHBOARD_DATA__*/" + blob)
    with open(OUT_DIR / "ablation_dashboard.html", "w", encoding="utf-8", newline="") as f:
        f.write(html)
    print("wrote reports/dashboard/ablation_dashboard.json"
          " + reports/dashboard/ablation_dashboard.html")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()
    payload = build(check_only=args.check)
    print("\nREPRODUCTION GATE — recomputed vs published")
    for r in payload["meta"]["gates"]:
        print(f"  {'OK ' if r['ok'] else 'FAIL'} {r['label']:38s} "
              f"{r['got']:>12.6f}   published {r['published']}")
    if args.check:
        print("\nvalidation OK (source-integrity + inputs + reproduction gates passed)")
        return
    print()
    emit(payload)


if __name__ == "__main__":
    main()
