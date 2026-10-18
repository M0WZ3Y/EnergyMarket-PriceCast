import json

import pandas as pd

from scripts import run_health as H

B = "Europe/Berlin"


def at(s):
    return pd.Timestamp(s, tz=B)


def test_clean_run_raises_nothing():
    assert H.alerts(at("2026-10-16 07:45"), 0, set(), []) == []  # before the ex-ante window: no alarm yet


def test_failed_run_and_failed_refresh_alert():
    a = H.alerts(at("2026-10-18 02:30"), 1, set(), ["shadow"])
    assert any("failed (exit 1)" in x for x in a) and any("shadow model failed" in x for x in a)


def test_missing_exante_before_auction_alerts_only_inside_window():
    issued = set()
    assert any("ex-ante" in x for x in H.alerts(at("2026-10-16 11:15"), 0, issued, []))
    assert not H.alerts(at("2026-10-16 10:45"), 0, issued, [])
    assert not any("ex-ante" in x for x in H.alerts(at("2026-10-16 12:15"), 0, issued, []))  # auction over
    assert not H.alerts(at("2026-10-16 11:15"), 0, {"2026-10-17.exante"}, [])


def test_missing_shadow_in_the_evening():
    assert any("shadow" in x for x in H.alerts(at("2026-10-16 19:30"), 0, {"2026-10-17.exante"}, []))
    assert not H.alerts(at("2026-10-16 19:30"), 0, {"2026-10-17.exante", "2026-10-17.shadow"}, [])
    assert not H.alerts(at("2026-10-16 18:45"), 0, set(), [])  # A69 may still be on its way


def test_status_keeps_last_success_across_a_failure(tmp_path):
    p = tmp_path / "status.json"
    H.write_status(at("2026-10-16 07:45"), 0, [], "2026-10-16T08:15:00Z", path=p)
    s = H.write_status(at("2026-10-16 10:15"), 1, ["x"], None, path=p)
    assert s["last_success"] == "2026-10-16 07:45" and s["last_exit"] == 1
    assert json.loads(p.read_text())["alerts"] == ["x"]
    assert H.write_status(at("2026-10-16 07:45"), 0, [], "2026-10-16T08:15:00Z", path=p)["next_run"] == "2026-10-16 10:15"


def test_issued_forecasts_reads_write_once_files(tmp_path):
    (tmp_path / "2026-10-17.exante.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("")
    assert H.issued_forecasts(tmp_path) == {"2026-10-17.exante"}


def test_fetch_failure_alerts_even_when_the_run_exits_cleanly():
    # a revoked token: daily_run logs the 401 in latest.json and exits 0 (2026-10-17)
    errs = ["fetch failed: ENTSO-E request failed (A44): HTTPError: 401 Client Error: Unauthorized"]
    a = H.alerts(at("2026-10-17 18:45"), 0, {"2026-10-18.exante"}, [], errs)
    assert any("401" in x and "fetch" in x.lower() for x in a)
    assert not H.alerts(at("2026-10-17 18:45"), 0, {"2026-10-18.exante"}, [], [])


def test_run_errors_reads_latest_json(tmp_path):
    p = tmp_path / "latest.json"
    assert H.run_errors(p) == []
    p.write_text(json.dumps({"errors": ["fetch failed: x"]}))
    assert H.run_errors(p) == ["fetch failed: x"]
