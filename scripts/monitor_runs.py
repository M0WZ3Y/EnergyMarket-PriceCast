"""Background-activity monitor — scripts/monitor_runs.py

Discovers the project's long-running background activities from their
on-disk footprints (works no matter which shell/session launched them):

  * walk-forward model runs  -> data/processed/baselines/<model>.csv
    (per-origin checkpoint rows, appended as the run progresses)
  * Optuna tuning            -> data/processed/tuning/*_study.db
    (finished-trial count vs configs/evaluation.yaml optuna.n_trials)

For each activity: state (RUNNING / STALLED / DONE), progress, rate,
time spent, and ETA. Rate comes from the artifact's own ctime->mtime
span, so it is accurate even across pauses/resumes of the same file.

Usage:
    python scripts/monitor_runs.py            # one-shot status table
    python scripts/monitor_runs.py --watch    # refresh every 30s until
                                              # nothing is left running
"""

from __future__ import annotations

import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES_DIR = REPO_ROOT / "data" / "processed" / "baselines"
TUNING_DIR = REPO_ROOT / "data" / "processed" / "tuning"

EXPECTED_ORIGINS = 728  # 2016-01-04 -> 2017-12-31 benchmark test period
STALL_SECONDS = 300  # no file write for 5 min while incomplete = stalled


@dataclass
class Activity:
    name: str
    done: int
    total: int
    started: float  # epoch
    last_write: float  # epoch

    @property
    def state(self) -> str:
        if self.done >= self.total:
            return "DONE"
        if time.time() - self.last_write > STALL_SECONDS:
            return "STALLED"
        return "RUNNING"

    @property
    def rate_s(self) -> float | None:
        if self.done < 2:
            return None
        return (self.last_write - self.started) / self.done

    @property
    def eta_s(self) -> float | None:
        if self.state != "RUNNING" or self.rate_s is None:
            return None
        return (self.total - self.done) * self.rate_s


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _count_csv_origins(path: Path) -> int:
    # 24 rows per completed origin (+1 header); a partially written origin
    # rounds down, matching the resume logic in run_full_baselines.py
    with open(path, "rb") as f:
        lines = sum(1 for _ in f)
    return max(0, (lines - 1) // 24)


def discover() -> list[Activity]:
    activities = []

    for csv in sorted(BASELINES_DIR.glob("*.csv")) if BASELINES_DIR.exists() else []:
        stat = csv.stat()
        activities.append(
            Activity(
                name=f"walk-forward: {csv.stem}",
                done=_count_csv_origins(csv),
                total=EXPECTED_ORIGINS,
                started=stat.st_ctime,
                last_write=stat.st_mtime,
            )
        )

    n_trials_target = 50
    eval_cfg = REPO_ROOT / "configs" / "evaluation.yaml"
    if eval_cfg.exists():
        with open(eval_cfg) as f:
            n_trials_target = yaml.safe_load(f)["optuna"]["n_trials"]

    for db in sorted(TUNING_DIR.glob("*_study.db")) if TUNING_DIR.exists() else []:
        try:
            con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
            finished = con.execute(
                "SELECT COUNT(*) FROM trials WHERE state IN ('COMPLETE', 'PRUNED', 'FAIL')"
            ).fetchone()[0]
            con.close()
        except sqlite3.Error:
            continue
        stat = db.stat()
        activities.append(
            Activity(
                name=f"tuning: {db.stem.replace('_study', '')}",
                done=finished,
                total=n_trials_target,
                started=stat.st_ctime,
                last_write=stat.st_mtime,
            )
        )

    return activities


def _bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + f"] {100 * done / total:3.0f}%"


# What an incomplete activity of each kind blocks, and what stays safe.
PROCEED_RULES = {
    "walk-forward": dict(
        blocked=[
            "the week-5 checkpoint comparison for THIS model",
            "committing this model's results CSV (partial file)",
            "the v1.0-results freeze tag",
        ],
        safe=[
            "developing other models (LSTM wrapper), tests, thesis writing",
            "committing/pushing unrelated code",
            "running the checkpoint for models already DONE",
        ],
        avoid=[
            "launching a second run of the SAME model (would corrupt its CSV)",
            "editing/deleting this model's CSV in data/processed/baselines/",
            "heavy CPU jobs (slows the run; its ETA will stretch)",
            "shutting down the machine (run pauses; resumable, but delays ETA)",
        ],
    ),
    "tuning": dict(
        blocked=[
            "the final walk-forward run for this model (needs tuned params)",
            "committing configs/tuned/ for this model",
        ],
        safe=["all other development and writing"],
        avoid=[
            "a second tuning process on the same study DB",
            "editing configs/evaluation.yaml mid-search",
        ],
    ),
}


def render(activities: list[Activity]) -> str:
    now = datetime.now().strftime("%H:%M:%S")
    header = (
        f"background activities @ {now}\n"
        f"{'activity':<28} {'state':<8} {'progress':<28} "
        f"{'rate':<10} {'spent':<8} {'ETA':<8} finishes"
    )
    lines = [header, "-" * len(header.splitlines()[-1])]
    for a in activities:
        rate = f"{a.rate_s:.1f}s/unit" if a.rate_s else "-"
        spent = _fmt_dur(a.last_write - a.started)
        eta = _fmt_dur(a.eta_s)
        finishes = (
            datetime.fromtimestamp(time.time() + a.eta_s).strftime("%H:%M")
            if a.eta_s
            else "-"
        )
        lines.append(
            f"{a.name:<28} {a.state:<8} {_bar(a.done, a.total):<28} "
            f"{rate:<10} {spent:<8} {eta:<8} {finishes}"
        )
    if not activities:
        lines.append("(no tracked activities found)")

    incomplete = [a for a in activities if a.state != "DONE"]
    if incomplete:
        lines.append("")
        lines.append("PROCEED? YES — parallel work is allowed; the run only owns its own output files.")
        for a in incomplete:
            kind = a.name.split(":")[0].strip()
            rules = PROCEED_RULES.get(kind)
            if not rules:
                continue
            lines.append(f"\nwhile '{a.name}' is {a.state}:")
            lines.append("  blocked until done:")
            lines.extend(f"    - {item}" for item in rules["blocked"])
            lines.append("  safe to proceed with:")
            lines.extend(f"    - {item}" for item in rules["safe"])
            lines.append("  avoid:")
            lines.extend(f"    - {item}" for item in rules["avoid"])
            if a.state == "STALLED":
                lines.append(
                    "  recommended: run has stopped writing -- resume it with:"
                )
                lines.append(
                    f"    .venv\\Scripts\\python.exe scripts\\run_full_baselines.py"
                    f" {a.name.split(':')[1].strip()}"
                )
    else:
        lines.append("")
        lines.append("PROCEED? YES — no incomplete activities; all results are final.")
    return "\n".join(lines)


def main() -> None:
    watch = "--watch" in sys.argv
    while True:
        acts = discover()
        print(render(acts), flush=True)
        running = [a for a in acts if a.state == "RUNNING"]
        if not watch or not running:
            if watch:
                print("\nnothing left running -- monitor exiting", flush=True)
            break
        print(flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
