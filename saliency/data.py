"""Load measured files from the GitHub repo (or a local checkout)."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

GITHUB_OWNER = "testing-crsscl"
GITHUB_REPO = "saliency-kyiv"
GITHUB_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/main/data"

HERE = Path(__file__).resolve().parent.parent / "data"


def _read_bytes(name: str) -> bytes:
    local = HERE / name
    if local.exists():
        return local.read_bytes()
    url = f"{RAW_BASE}/{name}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def load_json(name: str):
    raw = _read_bytes(name)
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8"))


def load_grid():
    for name in ("kyiv-grid.json.gz", "kyiv-grid.json"):
        try:
            return load_json(name)
        except Exception:
            continue
    raise FileNotFoundError("kyiv-grid.json not found locally or on GitHub")


def load_bundle():
    """Grid, WVS, sirens, reconstructed Van Dijcke. Same files the instrument ships."""
    grid = load_grid()
    wvs = load_json("wvs-kyiv.json")
    sirens = load_json("sirens-kyiv.json")
    vd = load_json("vandijcke-reconstructed.json")
    return {"grid": grid, "wvs": wvs, "sirens": sirens, "vd": vd, "source": GITHUB_URL}


def source_label() -> str:
    if (HERE / "wvs-kyiv.json").exists():
        return f"local checkout ({HERE})"
    return RAW_BASE
