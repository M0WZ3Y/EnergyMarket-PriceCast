"""Provenance records for pinned data snapshots — src/data/sources/provenance.py

Every external series this project uses is fetched ONCE, written to an
immutable on-disk snapshot, and committed. Feature code then reads only the
snapshot, never the network. This module is the record that makes that
claim checkable rather than merely asserted.

Why a live API cannot back a thesis feature: an API answers differently
tomorrow. A result computed against a moving source is not reproducible, and
a reviewer re-running the pipeline a year later would silently get different
numbers with no error and no indication anything changed. Pinning converts
that silent drift into a loud hash mismatch.

Each record carries what a citation and a reproduction both need:
  source, url, licence, fetch timestamp, date range, row count, sha256 of
  the exact bytes on disk.

`verify_snapshot()` recomputes the hash and fails on any mismatch, so a
snapshot that has been edited, truncated or regenerated cannot be used
without someone noticing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DIR = REPO_ROOT / "data" / "raw" / "physical"
MANIFEST_PATH = SNAPSHOT_DIR / "provenance.json"


@dataclass(frozen=True)
class ProvenanceRecord:
    """One pinned series: what it is, where it came from, and its exact bytes."""

    key: str
    source: str
    url: str
    licence: str
    fetched_at_utc: str
    date_start: str
    date_end: str
    n_rows: int
    n_cols: int
    sha256: str
    filename: str
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str | Path) -> str:
    """sha256 of a file's bytes, streamed so a large snapshot stays cheap."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, ProvenanceRecord]:
    """Read the manifest. Returns {} when no snapshot has been taken yet.

    A missing manifest is a legitimate state (nobody has run the fetch
    script), so it is not an error here -- but `snapshot.load_series` treats
    a missing SERIES as a hard failure, because a feature silently getting
    no data is the failure mode this whole module exists to prevent.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: ProvenanceRecord(**v) for k, v in raw.items()}


def write_manifest(
    records: dict[str, ProvenanceRecord], path: str | Path = MANIFEST_PATH
) -> None:
    """Write the manifest, merging into whatever is already recorded.

    Merging rather than replacing means fetching one series does not erase
    the provenance of the others -- a partial re-fetch is a normal thing to
    do (one source rate-limits, another does not) and must not destroy the
    record of series it did not touch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(path)
    existing.update(records)
    payload = {k: v.as_dict() for k, v in sorted(existing.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def verify_snapshot(
    key: str, manifest: dict[str, ProvenanceRecord] | None = None
) -> ProvenanceRecord:
    """Confirm a pinned series is present and byte-identical to its record.

    Raises rather than returning a flag: a caller that has to remember to
    check a boolean will eventually forget, and the consequence here is
    training on data that is not the data the results were produced from.
    """
    manifest = load_manifest() if manifest is None else manifest
    if key not in manifest:
        raise FileNotFoundError(
            f"no pinned snapshot for '{key}'. Run "
            f"scripts/fetch_physical_snapshot.py to create one. Feature code "
            f"must never fetch this series live."
        )
    rec = manifest[key]
    path = SNAPSHOT_DIR / rec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"manifest records '{key}' as {rec.filename}, but that file is "
            f"missing from {SNAPSHOT_DIR}"
        )
    actual = sha256_file(path)
    if actual != rec.sha256:
        raise ValueError(
            f"snapshot '{key}' does not match its provenance record.\n"
            f"  expected sha256 {rec.sha256}\n"
            f"  actual   sha256 {actual}\n"
            f"The file has been edited or re-fetched since it was pinned. "
            f"Results computed against it are not the results on record."
        )
    return rec


def format_provenance_table(manifest: dict[str, ProvenanceRecord] | None = None) -> str:
    """The provenance table printed at the end of a run (and citable)."""
    manifest = load_manifest() if manifest is None else manifest
    if not manifest:
        return "(no pinned snapshots)"
    rows = [
        f"{'key':<22} {'source':<16} {'licence':<12} {'range':<25} "
        f"{'rows':>7} {'sha256':<12} fetched"
    ]
    rows.append("-" * 120)
    for k, r in sorted(manifest.items()):
        rows.append(
            f"{k:<22} {r.source:<16} {r.licence:<12} "
            f"{r.date_start[:10] + '..' + r.date_end[:10]:<25} "
            f"{r.n_rows:>7} {r.sha256[:12]:<12} {r.fetched_at_utc}"
        )
    return "\n".join(rows)
