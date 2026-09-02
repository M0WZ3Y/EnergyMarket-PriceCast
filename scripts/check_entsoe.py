"""Is the ENTSO-E Transparency Platform up, and is our token valid?

Run this when the Platform's scheduled maintenance ends. It answers exactly
one question -- can the token be used -- and it distinguishes the three
outcomes that look alike from the outside.

WHY THE DISCRIMINATOR EXISTS. During the 2026-09-02 maintenance the API
returned HTTP 503 to *every* request: a valid token, a deliberately invalid
token, and no token at all. The outage sat in front of authentication, so a
503 says nothing whatsoever about the token. Absence of a 401 is NOT evidence
of validity, and this script refuses to report success on that basis. It
sends a known-bad token alongside the real one and compares.

READS THE TOKEN FROM THE ENVIRONMENT ONLY. Never pass a token on the command
line: the terminal tooling records approved shell commands into a local
approval cache, which is how the token leaked on 2026-09-02.
The variable is `ENTSOE_API_TOKEN`, set at Windows *User* scope.

    Bash    ./.venv/Scripts/python.exe scripts/check_entsoe.py
    Note    a User-scope variable is NOT inherited by shells that were already
            open. If it reports "not set", either open a new shell or, in
            PowerShell, load it for this process only:
              $env:ENTSOE_API_TOKEN =
                  [Environment]::GetEnvironmentVariable('ENTSOE_API_TOKEN','User')

This script touches the network deliberately and is NOT part of the feature
pipeline, which `tests/test_no_network_in_features.py` keeps offline.
"""
from __future__ import annotations

import os
import re
import sys
import time

import requests

API = "https://web-api.tp.entsoe.eu/api"
DE_LU = "10Y1001A1001A82H"
FR = "10YFR-RTE------C"
KNOWN_BAD = "00000000-0000-0000-0000-000000000000"

RE_TOKEN = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

# The canary is A44 day-ahead prices: the series we already hold from another
# source, so a success can be sanity-checked against known data. The other
# three are the documents a token would actually unblock -- they are probed
# only after the canary passes, because probing them during an outage just
# reprints the same maintenance page four times.
CANARY = ("A44 day-ahead prices (canary)", {
    "documentType": "A44", "in_Domain": DE_LU, "out_Domain": DE_LU,
    "periodStart": "202508010000", "periodEnd": "202508020000"})

UNBLOCKS = [
    ("A80 generation outages  (the true scarcity signal)", {
        "documentType": "A80", "biddingZone_Domain": DE_LU,
        "periodStart": "202508010000", "periodEnd": "202508020000"}),
    ("A61 forecast NTC        (market coupling)", {
        "documentType": "A61", "contract_MarketAgreement.Type": "A01",
        "in_Domain": DE_LU, "out_Domain": FR,
        "periodStart": "202508010000", "periodEnd": "202508020000"}),
    ("A72 hydro reservoir     (storage)", {
        "documentType": "A72", "processType": "A16", "in_Domain": DE_LU,
        "periodStart": "202508010000", "periodEnd": "202508080000"}),
]


def redact(text: object) -> str:
    return RE_TOKEN.sub("<REDACTED>", str(text))


def visible_text(html: str) -> str:
    """Strip an HTML error page down to its message. The 503 body is a full
    styled page; the useful part is one sentence buried in it."""
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body)).strip()


def get(params: dict, token: str, timeout: int = 90) -> requests.Response:
    return requests.get(API, params={**params, "securityToken": token}, timeout=timeout)


def probe(label: str, params: dict, token: str, retries: int = 3) -> str | None:
    """Return 'ok' | 'auth' | 'down' | 'error'. Prints one line per probe."""
    for attempt in range(1, retries + 1):
        try:
            r = get(params, token)
            if r.status_code == 503 and attempt < retries:
                time.sleep(5)
                continue
            if r.status_code == 200:
                xml = r.text
                root = re.search(r"<(\w+)[ >]", xml)
                n = xml.count("<TimeSeries>")
                print(f"  OK    {label}: {len(xml):,} bytes, "
                      f"root={root.group(1) if root else '?'}, TimeSeries={n}")
                return "ok" if n else "empty"
            msg = visible_text(r.text)[:180]
            print(f"  FAIL  {label}: HTTP {r.status_code} :: {redact(msg)}")
            return {401: "auth", 403: "auth", 503: "down"}.get(r.status_code, "error")
        except requests.RequestException as exc:
            if attempt < retries:
                time.sleep(5)
                continue
            print(f"  FAIL  {label}: {type(exc).__name__} {redact(exc)[:160]}")
            return "error"
    return "down"


def main() -> int:
    token = os.environ.get("ENTSOE_API_TOKEN", "").strip()
    if not token:
        print("ENTSOE_API_TOKEN is not set in this process.\n"
              "  It lives at Windows User scope and is not inherited by shells\n"
              "  opened before it was set. See this file's docstring.")
        return 2
    print(f"token: {len(token)} chars, ends ...{token[-4:]}\n")

    print("Canary + auth discriminator")
    real = probe(*CANARY, token=token)
    bad = probe("A44 with a KNOWN-BAD token", CANARY[1], token=KNOWN_BAD, retries=1)

    print()
    if real == "down":
        print("VERDICT: Platform still DOWN. "
              + ("The known-bad token gets the same 503, so the outage is in "
                 "front of authentication\n         and the token remains "
                 "UNVERIFIED -- this is not a token problem."
                 if bad == "down" else
                 "The known-bad token got a different answer, so auth IS "
                 "reachable;\n         investigate before assuming maintenance."))
        return 1
    if real == "auth":
        print("VERDICT: Token REJECTED. The Platform is reachable and it "
              "refused this credential.\n         Re-request access, or try "
              "ENTSOE_API_TOKEN_PREV (the earlier token, kept at User scope).")
        return 1
    if real not in ("ok", "empty"):
        print("VERDICT: Inconclusive -- neither a clean success nor a clean "
              "rejection. Read the line above.")
        return 1

    if bad != "auth":
        print("WARNING: the known-bad token was NOT rejected "
              f"(got '{bad}'). The canary's success may not prove the token "
              "is\n         doing any work. Treat validity as unconfirmed.")

    print("VERDICT: Token WORKS. Probing the three documents it unblocks:\n")
    for label, params in UNBLOCKS:
        probe(label, params, token=token)

    print("\nNEXT STEP IS A DECISION, NOT A BUILD.")
    print("  A working token does not authorise using it. `v1.0-results` and")
    print("  `v1.1-ood` are frozen and PROJECT_SPEC.md forbids rerunning or modifying")
    print("  model results after the tag. See docs/HANDOFF_entsoe.md before")
    print("  writing any feature code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
