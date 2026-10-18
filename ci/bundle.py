"""Encrypted private bundle: `pack` runs on the author's PC, `fetch` on the runner.

Models and unpublished results travel as ONE AES-256-GCM file attached to a GitHub
Release (assets on a public repo are public; the encryption keeps them private).
The key is 32 random bytes, base64, in env BUNDLE_KEY. It is never printed, never
put on a command line, never committed.

    python ci/bundle.py pack --out pricecast_bundle.bin
    python ci/bundle.py fetch [--src <url-or-path>]
    python ci/bundle.py seal-cache | unseal-cache   # data/live_cache/entsoe_recent.csv <-> state/entsoe_recent.csv.enc
"""
from __future__ import annotations

import argparse
import base64
import binascii
import fnmatch
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ["models/live", "data/processed/live_v2", "data/raw/live_v2",
           "reports/tables/live_ladder.csv"]
MANIFEST = "manifest.json"
MAX_PLAINTEXT = 2**31 - 1
DEFAULT_REPO = "M0WZ3Y/EnergyMarket-PriceCast"
RELEASE_ASSET = "pricecast_bundle.bin"
CACHE_PLAIN = "data/live_cache/entsoe_recent.csv"
CACHE_SEALED = "state/entsoe_recent.csv.enc"


def key_from_env() -> bytes:
    raw = os.environ.get("BUNDLE_KEY", "").strip()
    if not raw:
        raise SystemExit("BUNDLE_KEY is not set (base64 of 32 random bytes expected)")
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise SystemExit("BUNDLE_KEY is not valid base64") from None
    if len(key) != 32:
        raise SystemExit("BUNDLE_KEY must decode to exactly 32 bytes")
    return key


def _files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.is_file())
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"bundle path missing: {p}")
    return out


def _encrypt(plain: bytes, key: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plain, None)


def _decrypt(blob: bytes, key: bytes, what: str = "bundle") -> bytes:
    if len(blob) < 28:
        raise ValueError(f"{what} authentication failed")
    try:
        return AESGCM(key).decrypt(blob[:12], blob[12:], None)
    except InvalidTag:
        raise ValueError(f"{what} authentication failed") from None


def seal(src: Path, out: Path, key: bytes) -> None:
    """One file, nonce || ciphertext (same AES-256-GCM as the bundle)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(_encrypt(Path(src).read_bytes(), key))


def unseal(src: Path, out: Path, key: bytes) -> None:
    plain = _decrypt(Path(src).read_bytes(), key, "sealed file")  # nothing written unless it verifies
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(plain)


def pack(paths: list[Path], out: Path, key: bytes, root: Path = REPO_ROOT,
         exclude: list[str] | None = None) -> dict[str, str]:
    manifest: dict[str, str] = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in _files([Path(p) for p in paths]):
            rel = f.resolve().relative_to(Path(root).resolve()).as_posix()
            if any(fnmatch.fnmatch(rel, g) for g in exclude or ()):
                continue
            data = f.read_bytes()
            manifest[rel] = hashlib.sha256(data).hexdigest()
            z.writestr(rel, data)
        z.writestr(MANIFEST, json.dumps(manifest, sort_keys=True))
    plain = buf.getvalue()
    if len(plain) > MAX_PLAINTEXT:
        raise ValueError(f"bundle is {len(plain)} bytes; in-memory AES-GCM limit is {MAX_PLAINTEXT}")
    Path(out).write_bytes(_encrypt(plain, key))
    return manifest


def _read_src(src: str) -> bytes:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=300) as r:
            return r.read()
    return Path(src).read_bytes()


def _safe(name: str) -> None:
    win = PureWindowsPath(name)
    if (name.startswith(("/", "\\")) or PurePosixPath(name).is_absolute() or win.is_absolute()
            or win.drive or ".." in PurePosixPath(name.replace("\\", "/")).parts):
        raise ValueError(f"unsafe member path: {name}")


def fetch(src: str, key: bytes, dest_root: Path) -> None:
    blob = _read_src(src)
    plain = _decrypt(blob, key)
    with zipfile.ZipFile(io.BytesIO(plain)) as z:
        manifest = json.loads(z.read(MANIFEST))
        names = [n for n in z.namelist() if n != MANIFEST]
        for n in list(names) + list(manifest):
            _safe(n)
        if set(names) != set(manifest):
            raise ValueError("manifest does not match bundle members")
        staged = {}
        for n in names:
            data = z.read(n)
            if hashlib.sha256(data).hexdigest() != manifest[n]:
                raise ValueError(f"sha256 mismatch: {n}")
            staged[n] = data
    dest_root = Path(dest_root)
    for n, data in staged.items():  # nothing is written until every member verified
        target = dest_root / n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def release_url(tag: str) -> str:
    # Actions sets GITHUB_REPOSITORY, so the runner follows its own repo.
    repo = os.environ.get("GITHUB_REPOSITORY") or DEFAULT_REPO
    return f"https://github.com/{repo}/releases/download/{tag}/{RELEASE_ASSET}"


def default_src() -> str:
    tag_file = REPO_ROOT / "ci" / "BUNDLE_RELEASE"
    if not tag_file.exists():
        raise SystemExit("ci/BUNDLE_RELEASE is missing and --src was not given")
    return release_url(tag_file.read_text().strip())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Encrypted private bundle")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True)
    p.add_argument("--root", type=Path, default=None, help="checkout holding the private files (default: this repo)")
    p.add_argument("--exclude", action="append", default=[], help="repo-relative glob to leave out (repeatable)")
    f = sub.add_parser("fetch")
    f.add_argument("--src")
    sub.add_parser("seal-cache")
    sub.add_parser("unseal-cache")
    a = ap.parse_args(argv)
    key = key_from_env()
    if a.cmd == "pack":
        root = a.root or REPO_ROOT
        man = pack([root / p for p in PRIVATE], Path(a.out), key, root=root, exclude=a.exclude)
        print(f"packed {len(man)} files -> {a.out}")
    elif a.cmd == "seal-cache":
        seal(REPO_ROOT / CACHE_PLAIN, REPO_ROOT / CACHE_SEALED, key)
        print("cache sealed")
    elif a.cmd == "unseal-cache":
        if not (REPO_ROOT / CACHE_SEALED).exists():
            print("no sealed cache yet")
            return
        unseal(REPO_ROOT / CACHE_SEALED, REPO_ROOT / CACHE_PLAIN, key)
        print("cache restored")
    else:
        fetch(a.src or default_src(), key, REPO_ROOT)
        print("bundle restored")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        sys.exit(f"error: {e}")
