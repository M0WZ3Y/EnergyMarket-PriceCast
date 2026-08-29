"""Feature code must never touch the network.

The reproducibility rule for this project is that external data is fetched
once, pinned to an immutable on-disk snapshot with a provenance hash, and
read from there forever after. That rule is worth nothing unless something
enforces it: a live fetch added to a feature path would work perfectly on the
day it was written, produce slightly different numbers every month
afterwards, and never raise.

So this file blocks the socket. Any attempt to open one from inside a feature
build fails the test with the call site attached.

It also pins the two documented traps as executable facts rather than
comments, since both are the kind of thing that gets "simplified" later by
someone who does not know why the constant is odd:
  * Energy-Charts serves nothing before 2015 (contradicting its "since 2011"
    reputation), and reports that as a misleading HTTP 400.
  * The German bidding zone is DE-AT-LU before 2018-10-01, DE-LU after.
"""

from __future__ import annotations

import datetime as dt
import socket

import pandas as pd
import pytest


class NetworkAccessDenied(RuntimeError):
    """Raised in place of opening a socket during a feature build."""


@pytest.fixture
def no_network(monkeypatch):
    """Make any socket connection raise, with the target attached."""

    def deny(*args, **kwargs):  # noqa: ANN002, ANN003
        raise NetworkAccessDenied(
            f"feature code attempted a network connection: {args!r}. "
            "External data must come from the pinned snapshot "
            "(src/data/sources/snapshot.py), never a live fetch -- a live "
            "source makes the result irreproducible, and silently so."
        )

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    return deny


def _hourly(n_days: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2015-01-01", periods=24 * n_days, freq="h")
    n = len(idx)
    hour = pd.Index(range(n)) % 24
    return pd.DataFrame(
        {
            "price": 40.0 + (hour * 0.5),
            "exog_1": 20000.0 + 300 * hour,
            "exog_2": 5000.0 + 100 * hour,
        },
        index=idx,
    )


def test_default_feature_build_makes_no_network_call(no_network):
    from src.features.pipeline import build_features

    X, Y = build_features(_hourly())
    assert len(X) > 0


def test_physical_blocks_make_no_network_call(no_network):
    from src.features.pipeline import build_features, load_feature_config

    cfg = dict(load_feature_config())
    cfg["physical_blocks"] = {
        "residual_load_block": True,
        "residual_load_gradient_block": True,
        "merit_order_block": {"window": 20},
        "scarcity_block": {"window": 20},
    }
    X, _ = build_features(_hourly(), cfg)
    assert any(c.startswith("resload_") for c in X.columns)


def test_snapshot_module_imports_no_network_client():
    """snapshot.py must not import clients.py.

    Import-level separation is what keeps the socket ban from depending on
    anyone's discipline: a live fetch from a feature path would have to be
    added visibly, not inherited by accident through an import.
    """
    import src.data.sources.snapshot as snap

    src = open(snap.__file__, encoding="utf-8").read()
    assert "from src.data.sources.clients" not in src
    assert "import requests" not in src


def test_snapshot_reader_makes_no_network_call(no_network):
    from src.data.sources import snapshot

    # Works whether or not a snapshot has been taken; the point is that
    # neither path reaches for the network.
    assert isinstance(snapshot.available(), list)


def test_energy_charts_coverage_floor_is_pinned():
    """2015, not 2011. Verified by probing /price, /public_power and /cbpf."""
    from src.data.sources.clients import EC_DATA_START

    assert EC_DATA_START == dt.date(2015, 1, 1)


def test_german_bidding_zone_switches_at_the_right_date():
    """DE-LU returns 404 for the benchmark window; the zone was DE-AT-LU."""
    from src.data.sources.clients import de_bidding_zone

    assert de_bidding_zone(dt.date(2015, 6, 1)) == "DE-AT-LU"
    assert de_bidding_zone(dt.date(2018, 9, 30)) == "DE-AT-LU"
    assert de_bidding_zone(dt.date(2018, 10, 1)) == "DE-LU"
    assert de_bidding_zone(dt.date(2020, 1, 1)) == "DE-LU"


def test_entsoe_client_is_inert_without_a_token(monkeypatch):
    """No token must mean a loud failure, never an empty frame."""
    from src.data.sources.clients import ENTSOE_TOKEN_ENV, EntsoeClient, TokenNotConfigured

    monkeypatch.delenv(ENTSOE_TOKEN_ENV, raising=False)
    c = EntsoeClient()
    assert not c.configured
    with pytest.raises(TokenNotConfigured, match=ENTSOE_TOKEN_ENV):
        c.generation_outages("10Y1001A1001A83F", "201501010000", "201501020000")


def test_entsoe_token_is_never_hardcoded():
    """The token comes from the environment and is never committed."""
    from src.data.sources import clients

    src = open(clients.__file__, encoding="utf-8").read()
    # A real ENTSO-E token is a 36-char UUID. Any UUID-shaped literal in this
    # file would be a committed secret.
    import re

    uuids = re.findall(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", src, re.I
    )
    assert uuids == [], f"UUID-shaped literal in clients.py: {uuids}"
    assert 'os.environ.get(ENTSOE_TOKEN_ENV)' in src
