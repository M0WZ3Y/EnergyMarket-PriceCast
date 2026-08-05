"""Task monitor — scripts/task_monitor.py

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
    python scripts/task_monitor.py            # one-shot status table
    python scripts/task_monitor.py --watch    # refresh every 30 min,
                                              # notify on stop/finish
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


def expected_origins_from_config() -> int:
    """Number of walk-forward origins the benchmark test period contains.

    Derived, not hardcoded: every progress bar and every DONE/STALLED verdict
    in this monitor is a comparison against this number, so a literal would
    go silently wrong the moment configs/data.yaml changes market or
    years_test (the France stretch goal changes exactly that file).

    epftoolbox's read_data defines a "year" as 52 weeks = 364 days and carves
    the trailing `24 * 364 * years_test` hours off as the test split
    (epftoolbox/data/_datasets.py). One origin per test day, step_days=1, so
    the origin count is years_test * 364 — 728 for the current DE config
    (2016-01-04 -> 2017-12-31).
    """
    days_per_year = 364  # epftoolbox's 52-week year, not 365
    with open(REPO_ROOT / "configs" / "data.yaml") as f:
        years_test = int(yaml.safe_load(f)["benchmark"]["years_test"])
    return years_test * days_per_year


EXPECTED_ORIGINS = expected_origins_from_config()
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
            "the checkpoint comparison for THIS model",
            "committing this model's results CSV (partial file)",
            "the v1.0-results freeze tag",
            "the next model's tuning/walk-forward EXECUTION (heavy CPU -- "
            "queue it behind this run)",
        ],
        safe=[
            "analysis of models already DONE (checkpoint, DM tests, daily aggregation)",
            "CPU-light code prep for later milestones (tuning/ensemble scripts, tests)",
            "committing/pushing code that does not touch this model's outputs",
            "waiting -- all current milestones are sequential behind this run",
        ],
        avoid=[
            "launching a second run of the SAME model (would corrupt its CSV)",
            "editing/deleting this model's CSV in data/processed/baselines/",
            "heavy CPU jobs incl. network training (slows the run; ETA stretches)",
            "shutting down the machine (run pauses; resumable, but delays ETA)",
            "thesis drafting (parked by user decision 2026-07-30 until the "
            "project side is fully done)",
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
        f"task monitor @ {now}\n"
        f"{'activity':<28} {'state':<8} {'progress':<28} "
        f"{'rate':<10} {'spent':<8} {'ETA':<8} finishes"
    )
    lines = [header, "-" * len(header.splitlines()[-1])]
    for a in activities:
        # `is not None`, not truthiness: a rate of exactly 0.0s/unit is a
        # legitimate (very fast) measurement, and "-" means "not measurable".
        rate = f"{a.rate_s:.1f}s/unit" if a.rate_s is not None else "-"
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
                cmd = resume_command_for(a.name)
                lines.append(
                    "  recommended: run has stopped writing -- resume it with:"
                )
                if cmd:
                    printable = " ".join(arg.replace("/", "\\") for arg in cmd)
                    lines.append(f"    .venv\\Scripts\\python.exe {printable}")
                else:
                    lines.append(
                        f"    (no resume command known for '{a.name}' -- "
                        "add one to RESUME_COMMANDS)"
                    )
    else:
        lines.append("")
        lines.append("PROCEED? YES — no incomplete activities; all results are final.")
    return "\n".join(lines)


WATCH_INTERVAL_S = 1800  # 30 minutes

# How to resume each artifact if its producing process is gone
# (csv/db stem -> command args after the venv python)
RESUME_COMMANDS = {
    "naive": ["scripts/run_full_baselines.py", "naive"],
    "sarimax": ["scripts/run_full_baselines.py", "SARIMAX"],
    "lear_lasso": ["scripts/run_full_baselines.py", "LEAR-LASSO"],
    "lightgbm": ["scripts/run_full_baselines.py", "LightGBM"],
    # The LSTM walk-forward is the longest-running and most-interrupted job
    # in the project, so it is the one entry that must not be missing.
    "lstm": ["scripts/run_full_baselines.py", "LSTM"],
    # run_ensemble.py rewrites BOTH ensemble frames in a single pass -- there
    # is no per-frame argument and no append-resume, so either stem maps to
    # the same whole-script re-run. It is seconds of work, not hours.
    "ensemble_static": ["scripts/run_ensemble.py"],
    "ensemble_regime": ["scripts/run_ensemble.py"],
    "tuning: lightgbm": ["scripts/tune_lightgbm.py"],
    "tuning: lstm": ["scripts/tune_lstm.py"],
}
_PROC_PATTERN = "run_full_baselines|tune_lightgbm|tune_lstm|run_ensemble"


def _resume_key(name: str) -> str:
    """Normalize an activity name or CSV/DB stem to a RESUME_COMMANDS key.

    Accepts "lstm", "walk-forward: lstm" and "tuning: lstm". A tuning
    activity keeps its "tuning: " prefix because it resumes with tune_*.py,
    never with the walk-forward runner -- dropping the prefix would map it
    onto the wrong script entirely.
    """
    key = name.strip()
    if key.startswith("tuning"):
        return f"tuning: {key.split(':', 1)[1].strip()}" if ":" in key else key
    return key.split(":", 1)[1].strip() if ":" in key else key


def resume_command_for(name: str) -> list[str] | None:
    """Resume command args (everything after the venv python), or None.

    The single source of resume advice: both the STALLED hint in render()
    and _resume_incomplete() route through here. They used to derive the
    model name independently, and disagreed -- render() built the name by
    lowercasing the CSV stem, producing 'sarimax'/'lear_lasso'/'lightgbm',
    all of which run_full_baselines.py rejects with "no matching models",
    and it pointed tuning activities at the walk-forward runner. One lookup
    means the two paths cannot drift apart again.
    """
    return RESUME_COMMANDS.get(_resume_key(name))


def _find_run_pids() -> list[int]:
    """PIDs of python processes running this project's long jobs.

    Raises RuntimeError if the probe itself fails. Returning [] on failure
    would fail OPEN: a PowerShell timeout happens precisely when the
    machine is loaded by a live run, and an empty list reads as "nothing is
    running", which lets _resume_incomplete launch a DUPLICATE process
    appending to the same CSV (the corruption PROCEED_RULES warns about;
    it hit validation_preds/lightgbm.csv on 2026-08-02). Unknown must never
    mean "not running".
    """
    import subprocess

    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                f"Where-Object {{ $_.CommandLine -match '{_PROC_PATTERN}' }} | "
                "Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        return [int(line) for line in out.split() if line.strip().isdigit()]
    except Exception as exc:
        raise RuntimeError(
            f"could not determine whether project jobs are running: {exc!r}"
        ) from exc


def _pause_all() -> None:
    """Stop all running project jobs. Safe by design: every job
    checkpoints per origin/trial, so at most the in-flight unit is lost
    and a later resume continues from the last complete unit. Use before
    shutting the machine down."""
    import subprocess

    try:
        pids = _find_run_pids()
    except RuntimeError as exc:
        # "no processes found" and "could not look" are different answers;
        # reporting the second as the first would tell the user it is safe
        # to shut the machine down while a run may still be writing.
        print(f"cannot pause -- process probe failed: {exc}", flush=True)
        print("re-run the probe, or check Task Manager before shutting down.", flush=True)
        return
    if not pids:
        print("nothing to pause -- no project job processes found", flush=True)
        return
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    print(
        f"paused {len(pids)} job process(es); safe to shut down. "
        "Press [r] after restart to resume.",
        flush=True,
    )


def _resume_incomplete(activities) -> None:
    """Relaunch every incomplete activity that has no live process,
    detached so it survives this monitor window closing. Output goes to
    logs/runs/<name>.log."""
    import subprocess

    try:
        running_pids = _find_run_pids()
    except RuntimeError as exc:
        # Fail closed: without a trustworthy answer we cannot rule out a
        # live process, and a duplicate appending to the same CSV is far
        # worse than a resume that has to be re-triggered by hand.
        print(f"refusing to resume -- {exc}", flush=True)
        print(
            "nothing was launched. Re-try once the machine is responsive, "
            "or start the run manually after checking Task Manager.",
            flush=True,
        )
        return
    if running_pids:
        print("job process(es) already running -- not spawning duplicates", flush=True)
        return
    launched = 0
    log_dir = REPO_ROOT / "logs" / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    for a in activities:
        if a.state == "DONE":
            continue
        key = _resume_key(a.name)
        cmd = resume_command_for(a.name)
        if not cmd:
            print(f"no resume command known for '{a.name}' -- skipped", flush=True)
            continue
        log_path = log_dir / f"{key.replace(': ', '_')}.log"
        with open(log_path, "a") as log:
            subprocess.Popen(
                [str(python)] + cmd,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=0x00000008 | 0x00000200,  # DETACHED | NEW_PROCESS_GROUP
            )
        print(f"resumed '{a.name}' (log: {log_path})", flush=True)
        launched += 1
    if not launched:
        print("nothing incomplete to resume", flush=True)


def _set_keep_awake(active: bool) -> None:
    """While a job is RUNNING and the watch window is open, stop Windows
    from sleeping (display may still turn off). Released automatically
    when nothing runs or the monitor exits."""
    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if active else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception:
        pass


def _wait_for_key_or_timeout(seconds: float) -> str:
    """Block up to `seconds`, returning early on user input.

    Returns "refresh" (Enter/Space pressed), "quit" ('q' pressed), or
    "timeout". Uses msvcrt on Windows consoles; falls back to a plain
    sleep (timeout only) where no console keyboard is available.
    """
    try:
        import msvcrt
    except ImportError:
        time.sleep(seconds)
        return "timeout"

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            key = msvcrt.getwch().lower()
            if key == "q":
                return "quit"
            if key == "p":
                return "pause"
            if key == "r":
                return "resume"
            return "refresh"  # Enter, Space, or any other key
        time.sleep(0.25)
    return "timeout"


def _notify(title: str, text: str) -> None:
    """Non-blocking desktop notification (Windows message box on top,
    in its own thread) + console bell. Used for stop/stall/done events."""
    print(f"\a*** {title}: {text}", flush=True)
    try:
        import ctypes
        import threading

        MB_SYSTEMMODAL, MB_ICONWARNING = 0x1000, 0x30
        threading.Thread(
            target=ctypes.windll.user32.MessageBoxW,
            args=(None, text, title, MB_SYSTEMMODAL | MB_ICONWARNING),
            daemon=True,
        ).start()
    except Exception:
        pass  # console bell already fired


def main() -> None:
    watch = "--watch" in sys.argv
    prev_states: dict[str, str] = {}
    while True:
        acts = discover()
        print(render(acts), flush=True)

        # Notify on state transitions: a run that stopped writing while
        # incomplete (crash/kill/error) or one that just finished.
        for a in acts:
            prev = prev_states.get(a.name)
            if prev is not None and prev != a.state:
                if a.state == "STALLED":
                    cmd = resume_command_for(a.name)
                    how = (
                        f"Resume: python {' '.join(cmd)}"
                        if cmd
                        else "No resume command known -- see RESUME_COMMANDS."
                    )
                    _notify(
                        "Background run STOPPED",
                        f"{a.name} stopped at {a.done}/{a.total} "
                        f"({100 * a.done / a.total:.0f}%). It is no longer "
                        f"writing output -- likely crashed or was killed. {how}",
                    )
                elif a.state == "DONE" and prev == "RUNNING":
                    _notify("Background run finished", f"{a.name} completed {a.total}/{a.total}.")
            prev_states[a.name] = a.state

        # Keep watching while anything is incomplete -- a STALLED run stays
        # on watch so a resume is picked up and re-notified on completion.
        incomplete = [a for a in acts if a.state != "DONE"]
        running = [a for a in acts if a.state == "RUNNING"]
        _set_keep_awake(bool(running))
        if not watch or not incomplete:
            _set_keep_awake(False)
            if watch:
                print("\nnothing left running -- monitor exiting", flush=True)
            break
        awake_note = "   [machine kept awake]" if running else ""
        print(
            "\n[Enter/Space] refresh   [p] pause jobs (pre-shutdown)   "
            f"[r] resume   [q] quit   (auto-refresh {WATCH_INTERVAL_S // 60} min)"
            f"{awake_note}",
            flush=True,
        )
        action = _wait_for_key_or_timeout(WATCH_INTERVAL_S)
        if action == "quit":
            _set_keep_awake(False)
            print("monitor quit by user", flush=True)
            break
        if action == "pause":
            _pause_all()
        elif action == "resume":
            _resume_incomplete(acts)
        print(flush=True)


if __name__ == "__main__":
    main()
