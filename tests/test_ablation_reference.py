"""The ablation must measure each variant against what it ADDS TO.

This pins a bug that produced a real misreading rather than a hypothetical
one. Every pairwise variant CONTAINS B1, so comparing it to the baseline
credits B1's large aggregate gain to the block under test. The physics check
then flagged B1+headroom and B1+B3 as "improved aggregate MAE but missed its
target regime" -- the signature of a feature tracking a correlate rather than
its mechanism -- when the aggregate gain was B1's all along and the added
block was simply worse than B1 on its own target.

The conclusions drawn at the time were unaffected, because the vs-B1 numbers
were computed separately by hand. The harness would have misled the next run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_harness():
    path = REPO_ROOT / "scripts" / "run_full_physical_ablation.py"
    spec = importlib.util.spec_from_file_location("_abl", path)
    mod = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = ["run_full_physical_ablation.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = saved
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_pairwise_variants_are_measured_against_b1(harness):
    """A B1+X variant must compare to B1, never to the baseline."""
    available = set(harness.VARIANTS) | {"baseline"}
    pairwise = [n for n in harness.VARIANTS if n.startswith("B1+")]
    assert pairwise, "no pairwise variants defined -- the test would be vacuous"
    for name in pairwise:
        ref = harness.reference_for(name, available)
        assert ref == "B1_ramp", (
            f"{name} contains B1 but is measured against '{ref}'. Its aggregate "
            "gain would be credited to the added block instead of to B1."
        )


def test_single_block_variants_are_measured_against_baseline(harness):
    available = set(harness.VARIANTS) | {"baseline"}
    for name in ("B1_ramp", "B2_merit_explicit", "B4_coupling", "ALL"):
        if name in harness.VARIANTS:
            assert harness.reference_for(name, available) == "baseline"


def test_missing_reference_falls_back_to_baseline_not_self(harness):
    """If the declared reference was not run, fall back to the baseline.

    Silently self-comparing would report a delta of exactly zero for every
    regime, which reads as 'this block changes nothing' -- a wrong conclusion
    that looks like a real measurement.
    """
    available = {"baseline", "B1+B2"}  # B1_ramp deliberately absent
    ref = harness.reference_for("B1+B2", available)
    assert ref == "baseline"
    assert ref != "B1+B2"


def test_every_declared_reference_is_a_real_variant(harness):
    """A reference naming a variant that does not exist would silently
    degrade to the baseline for every run, reinstating the original bug."""
    known = set(harness.VARIANTS) | {"baseline"}
    for variant, ref in harness.VARIANT_REFERENCE.items():
        assert variant in harness.VARIANTS, f"unknown variant '{variant}'"
        assert ref in known, f"'{variant}' references unknown variant '{ref}'"


def test_reference_is_never_the_variant_itself(harness):
    for variant, ref in harness.VARIANT_REFERENCE.items():
        assert ref != variant, f"'{variant}' is its own reference"
