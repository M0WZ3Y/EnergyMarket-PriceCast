"""The dashboard generator must stay pinned to the numbers it published.

`scripts/build_dashboard_data.py` carries the physical-feature ablation
results out of the repo and into a shareable artifact. It already gates
itself hard -- ten reproduction checks against the published values, a
sha256 per pinned input, and an evidence-anchor check against the source
files -- but until this module existed *nothing ran those gates*. They
fired only when a human typed the command. `pytest` stayed green while the
one artifact an examiner is most likely to open could have drifted from the
frozen tables without a sound.

That is the gap this file closes: the generator's own gates become part of
the suite, and the committed JSON is diffed against a fresh rebuild so any
change to a statistic fails here rather than in a reviewer's browser.

EVERYTHING RUNS IN A SUBPROCESS, DELIBERATELY. Importing the generator has a
process-wide side effect: at import time it rebinds `socket.socket` to a
function that raises, as its offline guard. That is correct for a standalone
script and poisonous inside a shared pytest process -- it would break every
later test that touches a socket, in test-order-dependent ways. So this
module never imports the generator; it shells out. Do not "simplify" this
into a direct import.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
GENERATOR = REPO_ROOT / "scripts" / "build_dashboard_data.py"
INPUT_DIR = REPO_ROOT / "reports" / "dashboard" / "inputs"
COMMITTED_JSON = REPO_ROOT / "reports" / "dashboard" / "ablation_dashboard.json"
TEMPLATE = REPO_ROOT / "reports" / "dashboard" / "template.html"

# meta fields that legitimately change on every rebuild: a wall-clock stamp and
# the HEAD commit. Everything else in the payload is a claim about the results
# and must not move.
VOLATILE_META = {"generated_at", "git_commit"}


def _python() -> str:
    return str(PY) if PY.exists() else sys.executable


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_python(), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


@pytest.fixture(scope="module")
def fresh_payload() -> dict:
    """Rebuild the payload from the pinned inputs, writing nothing.

    Uses --check so the build runs its full gate sequence but never touches
    reports/dashboard/, keeping the working tree clean during a test run.
    """
    proc = _run([
        "-c",
        "import json,runpy,sys;"
        "m=runpy.run_path(r'%s');"
        "sys.stdout.write('@@JSON@@'+json.dumps(m['build'](check_only=True)))"
        % GENERATOR,
    ])
    assert proc.returncode == 0, f"generator build failed:\n{proc.stdout}\n{proc.stderr}"
    assert "@@JSON@@" in proc.stdout, f"no payload emitted:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.split("@@JSON@@", 1)[1])


def test_check_mode_exits_clean():
    """`--check` is the cheap gate a human runs; it must stay runnable."""
    proc = _run([str(GENERATOR), "--check"])
    assert proc.returncode == 0, f"--check failed:\n{proc.stdout}\n{proc.stderr}"
    assert "validation OK" in proc.stdout


def test_every_reproduction_gate_passes(fresh_payload):
    """All ten published values must still reproduce from the pinned inputs."""
    gates = fresh_payload["meta"]["gates"]
    assert len(gates) == 10, f"gate count changed: {len(gates)} (expected 10)"
    failed = [g["label"] for g in gates if not g["ok"]]
    assert not failed, f"reproduction gates failed: {failed}"


def test_committed_artifact_equals_a_fresh_build(fresh_payload):
    """The shipped JSON must be what the generator produces today.

    This is the regression guard proper. If a pinned input, a statistic, a
    verdict or a Persian string changes, the committed artifact and the fresh
    build diverge and this fails -- naming the offending key path.
    """
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))

    for payload in (committed, fresh_payload):
        for key in VOLATILE_META:
            payload["meta"].pop(key, None)

    assert fresh_payload == committed, (
        "reports/dashboard/ablation_dashboard.json is stale or the generator "
        "changed a published number. Re-run "
        "`./.venv/Scripts/python.exe scripts/build_dashboard_data.py` and "
        "review the diff before committing it."
    )


def test_pinned_inputs_match_their_recorded_hashes(fresh_payload):
    """Each pinned CSV must still hash to the value the payload records.

    Recomputed here independently of the generator, so a bug in its own
    verification cannot mask a changed input.
    """
    recorded = fresh_payload["meta"]["inputs"]
    assert recorded, "payload recorded no pinned inputs"

    for fname, expected in sorted(recorded.items()):
        path = INPUT_DIR / fname
        assert path.exists(), f"pinned input missing: {fname}"
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == expected, (
            f"{fname} changed on disk: recorded {expected}, found {got}. "
            "These bytes are pinned -- check .gitattributes line-ending "
            "normalisation before assuming the data moved."
        )


def test_generator_imports_no_pipeline_code():
    """Consumer-only is a hard constraint, so assert it rather than trust it.

    The dashboard must never be able to re-run, re-fit or re-freeze anything.
    Importing `src.*` would give it that reach.
    """
    source = GENERATOR.read_text(encoding="utf-8")
    offenders = re.findall(r"^\s*(?:from|import)\s+src\b.*$", source, re.MULTILINE)
    assert not offenders, f"generator imports pipeline code: {offenders}"


def test_template_has_exactly_one_data_marker():
    """Injection is a single string replace; two markers would corrupt it."""
    assert TEMPLATE.exists(), "reports/dashboard/template.html is missing"
    count = TEMPLATE.read_text(encoding="utf-8").count("/*__DASHBOARD_DATA__*/null")
    assert count == 1, f"expected exactly one data marker, found {count}"


def test_shipped_html_carries_no_preview_banner():
    """`is_preview` must be false in what ships; the banner is a draft marker."""
    committed = json.loads(COMMITTED_JSON.read_text(encoding="utf-8"))
    assert committed["meta"]["is_preview"] is False
