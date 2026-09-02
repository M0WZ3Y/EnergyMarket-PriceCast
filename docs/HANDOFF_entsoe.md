# ENTSO-E handoff — run this when the maintenance ends

**Status as of 2026-09-02:** token configured, **UNVERIFIED**, nothing built
with it. The Transparency Platform is in scheduled maintenance and returns
HTTP 503 to everything.

---

## 1. The one command

```bash
./.venv/Scripts/python.exe scripts/check_entsoe.py
```

If it prints `ENTSOE_API_TOKEN is not set`, the variable exists but this shell
predates it — a Windows *User* variable is not inherited by already-open
shells. Either open a new terminal, or load it for one process:

```powershell
$env:ENTSOE_API_TOKEN = [Environment]::GetEnvironmentVariable('ENTSOE_API_TOKEN','User')
```

**Never pass the token as a command-line argument.** The terminal tooling
records approved shell commands into a local approval cache; that is exactly
how it leaked on 2026-09-02. The script reads the environment only.

## 2. Reading the verdict

The script prints one of three verdicts and exits 0 on success, 1 otherwise.

| Verdict | Meaning | Do next |
|---|---|---|
| **Platform still DOWN** | Maintenance continues. | Nothing. Re-run later. |
| **Token REJECTED** | Platform is up and refused the credential. | Re-request access, or try `ENTSOE_API_TOKEN_PREV` (§4). |
| **Token WORKS** | Canary returned 200 with data. | Go to §3 — and stop, because the next step is a decision, not a build. |

### Why there is a known-bad-token probe

During the 2026-09-02 outage the API returned 503 to a valid token, a garbage
token, **and no token at all**. The outage sits in front of authentication, so
a 503 carries no information about the credential. **Absence of a 401 is not
evidence of validity.** The script sends a deliberately invalid token
alongside the real one and compares; it refuses to claim success unless the
bad token is actually rejected. Do not simplify this away.

## 3. If the token works — the decision gate

**A working token does not authorise using it.** Read this before writing any
feature code.

`v1.0-results` and `v1.1-ood` are frozen. PROJECT_SPEC.md: *never rerun or modify
model results after the tag — writing depends on frozen numbers.* The
technical phase closed at `b53dcfb`, and CHECKLIST.md lists no ENTSO-E work at
any priority.

What the token unblocks — the three series `src/features/price_formation.py`
records as `blocked_by` a token, and the reason
`regimes.UNAVAILABLE_SEGMENTS` has no `outage_scarcity` or `reservoir_hydro`
segment:

| Doc | Series | Why it was wanted |
|---|---|---|
| `A80` | generation outages | The **true scarcity signal**. The current feature set can only build a capacity-based *tightness proxy*, and installed capacity is a yearly step while availability varies daily — see `logs/decisions.md` 2026-08-29 ("zero residual is not zero information"). This is the one with real scientific value. |
| `A61` | forecast NTC | Cross-border market coupling. |
| `A72` | hydro reservoir | Storage. |

Three options, none of them default:

1. **Verify and reword only.** Keep the freeze. The 2026-09-02 decision entry
   concluded the token's value is *"prose, not numbers"*: the "blocked by a
   token" framing in sections 3-x and the limitations chapter is now factually
   stale, and the honest statement is that the data became reachable **after**
   the results were frozen. Costs nothing, breaks nothing.
2. **Snapshot now, decide later.** Fetch A80/A61/A72 into an immutable
   sha256-stamped snapshot under `data/raw/` and commit it, per the
   reproducibility rule (`data/raw/physical/provenance.json` is the pattern).
   Wire nothing into features. Keeps the option open without touching the
   freeze; the API answers differently tomorrow, so the snapshot is what makes
   any later decision executable.
3. **Build and reopen the freeze.** Requires a new dated entry in
   `logs/decisions.md` **before** any code, stating what is reopened and why,
   plus a scope amendment. Do not start this in the last week before a
   supervisor deadline.

Whichever is chosen, log it. The 2026-08-28 amendment pre-approved ENTSO-E and
**only** ENTSO-E as a registration-gated source; that approval does not
generalise.

## 4. Where things live

| Thing | Location |
|---|---|
| Current token | `ENTSOE_API_TOKEN`, Windows **User** scope, 36-char UUID |
| Previous token | `ENTSOE_API_TOKEN_PREV`, same scope — kept because neither is verified; delete once one is confirmed |
| Client | `src/data/sources/clients.py` → `EntsoeClient`; reads the env var, raises `TokenNotConfigured` when absent, never writes the token to disk or logs |
| Probe | `scripts/check_entsoe.py` (this handoff's command) |
| History | `logs/decisions.md` — 2026-08-28 (the amendment), 2026-09-02 (token arrival, configuration, non-use) |

Neither token appears in the local approval cache as of 2026-09-02;
verified by direct search. The file is gitignored but remains a leak vector,
so re-check after any session where a token is handled.

## 5. Related, and more urgent

`scripts/run_feature_support.py` (§10.3, closed 2026-09-02) found that the
live 2026 exogenous features sit almost entirely outside their training
support — `exog_1` overlap **0.0002**, 99.86% above the training maximum — and
that a single scalar restores overlap to 0.87. That is an
Amprion-zonal-vs-national **definition mismatch in the serving path**, not
market drift, and it reinterprets the whole OOD result.

It needs no token, and it is prose-ready for 3-2, 4-7 and the limitations
chapter. If time is short, write that up before touching any of this.
