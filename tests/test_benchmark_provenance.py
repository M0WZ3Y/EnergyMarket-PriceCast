"""The benchmark input is pinned, and is the one the frozen results came from.

The tags v1.0-results and v1.1-ood ship NO benchmark data -- only
data/raw/.gitkeep -- so reproducing the reported thesis numbers depended on
re-downloading DE.csv from the epftoolbox server. If that source ever moves or
revises its data, the frozen results become unreproducible with no error and no
warning. That is precisely the failure the snapshot rule was written to
prevent, and it had been applied to the physical data but not to the primary
benchmark input.

These tests make the pin verifiable rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = REPO_ROOT / "data" / "raw" / "DE.csv"
PROV = REPO_ROOT / "data" / "raw" / "provenance_benchmark.json"


@pytest.fixture(scope="module")
def record() -> dict:
    if not PROV.exists():
        pytest.skip("benchmark provenance not present in this checkout")
    return json.loads(PROV.read_text(encoding="utf-8"))["de_benchmark"]


def test_benchmark_csv_matches_its_recorded_hash(record):
    """The load-bearing check: these bytes are the recorded bytes.

    A mismatch means the file was re-downloaded or edited since it was pinned,
    and results computed against it are not the results on record.
    """
    if not CSV.exists():
        pytest.skip("DE.csv not present in this checkout")
    actual = hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert actual == record["sha256"], (
        f"DE.csv does not match its provenance record.\n"
        f"  expected {record['sha256']}\n"
        f"  actual   {actual}\n"
        "The benchmark input has changed since it was pinned. The frozen "
        "v1.0-results / v1.1-ood numbers were produced from the recorded "
        "version, not this one."
    )


def test_recorded_shape_matches_the_file(record):
    if not CSV.exists():
        pytest.skip("DE.csv not present in this checkout")
    d = pd.read_csv(CSV, index_col=0, parse_dates=True)
    assert len(d) == record["n_rows"]
    assert list(d.columns) == record["columns"]
    assert str(d.index.min()) == record["date_start"]
    assert str(d.index.max()) == record["date_end"]


def test_provenance_states_which_tags_it_binds_to(record):
    """Someone reproducing from a tag must be able to discover that the tag
    ships no data, WITHOUT reading git history."""
    assert set(record["binds_to_tags"]) == {"v1.0-results", "v1.1-ood"}
    notes = record["notes"]
    assert "ship NO benchmark data" in notes or "ship no benchmark data" in notes.lower()
    assert "v1.1-ood" in notes and "live_ood_de.csv" in notes


def test_provenance_carries_the_feature_matrix_check(record):
    """The hash alone proves the bytes; this proves the bytes still produce the
    matrix the frozen results were built on."""
    chk = record["feature_matrix_check"]
    assert chk["shape"] == [2177, 247]
    assert chk["md5_of_hash_pandas_object"] == "6c8f0c5d66e5895b69d0c15d3c061c5c"


@pytest.mark.skipif(not CSV.exists(), reason="DE.csv not present")
def test_default_feature_matrix_still_reproduces(record):
    """End-to-end: the pinned file must still build the exact matrix the
    frozen results were produced from."""
    from src.data.loader import BenchmarkLoader, load_config
    from src.features.pipeline import build_features

    tr, te = BenchmarkLoader(load_config()).load()
    X, _ = build_features(pd.concat([tr, te]))
    chk = record["feature_matrix_check"]
    assert list(X.shape) == chk["shape"]
    h = hashlib.md5(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
    assert h == chk["md5_of_hash_pandas_object"], (
        "the pinned DE.csv no longer reproduces the frozen feature matrix"
    )
