"""Read-only access to the pinned physical snapshot — src/data/sources/snapshot.py

Feature code reads external data through this module and ONLY through this
module. It performs no network call and imports nothing that can: `clients`
is deliberately not imported here, so an accidental live fetch from a feature
path would have to be added visibly rather than inherited by accident.
`tests/test_no_network_in_features.py` enforces that at runtime by failing on
any socket the feature path opens.

Every load verifies the file against its provenance record first, so a
snapshot that has been edited or re-fetched since it was pinned fails loudly
instead of quietly changing results.

ALIGNMENT is the other job here. External series arrive on their own hourly
grid; the benchmark price index is tz-naive local German time with exactly 24
hours per day. `align_to_index` reindexes onto the benchmark calendar and
NEVER fills: a missing hour stays NaN and is dropped downstream by
build_features, because an interpolated value in a physical driver is
indistinguishable from a measured one once it reaches a model.
"""

from __future__ import annotations

import pandas as pd

from src.data.sources.provenance import SNAPSHOT_DIR, load_manifest, verify_snapshot


def available() -> list[str]:
    """Keys present in the pinned snapshot. Empty when none has been taken."""
    return sorted(load_manifest())


def load_series(key: str, parse_dates: bool = True) -> pd.DataFrame:
    """Load one pinned series, after verifying its hash.

    Raises FileNotFoundError when the series was never pinned -- callers must
    treat that as "this feature block cannot run", never as "no data this
    period".
    """
    rec = verify_snapshot(key)
    path = SNAPSHOT_DIR / rec.filename
    df = pd.read_csv(path, index_col=0)
    if parse_dates:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[df.index.notna()].sort_index()
    return df


def has(key: str) -> bool:
    """Whether a series is pinned AND verifies. Never raises.

    Feature blocks use this to skip cleanly when a Tier-B series is absent,
    which is the normal state until an ENTSO-E token arrives.
    """
    try:
        verify_snapshot(key)
        return True
    except (FileNotFoundError, ValueError):
        return False


def align_to_index(
    df: pd.DataFrame, index: pd.DatetimeIndex, prefix: str = ""
) -> pd.DataFrame:
    """Reindex an external series onto the benchmark hourly calendar.

    Two failure modes this handles explicitly rather than by accident:

    DUPLICATE TIMESTAMPS. Converting UTC to local German time makes the
    October DST hour occur twice. Keeping 'first' is a real choice, not a
    default -- averaging the two would invent a value that never cleared.

    MISSING HOURS. Never filled. A gap stays NaN and build_features drops the
    day. Filling would put an interpolated number into a physical driver,
    where it becomes indistinguishable from a measured one.
    """
    out = df[~df.index.duplicated(keep="first")].sort_index()
    out = out.reindex(index)
    if prefix:
        out.columns = [f"{prefix}{c}" for c in out.columns]
    return out


def coverage(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Fraction of `index` each column actually covers.

    Printed by the fetch/ablation scripts. A block whose input covers a small
    slice of the benchmark window would silently shrink the usable training
    set once NaN rows are dropped; this makes that visible before it happens
    rather than as an unexplained row-count drop.
    """
    aligned = align_to_index(df, index)
    return aligned.notna().mean().sort_values()
