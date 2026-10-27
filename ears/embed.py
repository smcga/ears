"""Per-bar learned embeddings (CLAP, two models concatenated), cached on disk.

Each bar is embedded from that bar's audio only (no look-ahead), so the vectors can be used for
causal, bar-by-bar prediction as well as for similarity maps and "sounds like" comparisons.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import soundfile as sf
import soxr

from .analysis import Analysis
from .paths import output_dir

SR = 48000
CACHE_VERSION = "v1"


def _cache_path(a: Analysis) -> str:
    st = os.stat(a.path)
    key = f"{CACHE_VERSION}|{os.path.normcase(a.path)}|{st.st_mtime_ns}|{st.st_size}|{a.grid.bpm:g}"
    h = hashlib.sha1(key.encode()).hexdigest()[:12]
    d = output_dir() / "cache"
    d.mkdir(exist_ok=True)
    return str(d / f"{os.path.splitext(a.name)[0]}_{h}.npy")


def bar_embeddings(a: Analysis) -> np.ndarray:
    """(n_bars, D) unit vectors. Bar k is embedded from bar k's audio only."""
    p = _cache_path(a)
    if os.path.exists(p):
        return np.load(p)
    import contextlib
    import sys

    from . import describe as d

    x, sr = sf.read(a.path, dtype="float32", always_2d=True)
    y = soxr.resample(x.mean(axis=1), sr, SR, quality="HQ").astype(np.float32)
    g = a.grid
    wins = []
    for bar in range(1, g.n_bars + 1):
        s = int(g.t_of_bar(bar) * SR)
        e = int(min(a.duration, g.t_of_bar(bar + 1)) * SR)
        seg = y[s:e]
        if len(seg) < SR // 2:
            seg = np.pad(seg, (0, SR // 2 - len(seg)))
        wins.append(seg)
    with d._lock, contextlib.redirect_stdout(sys.stderr):
        per_model = d._audio_emb(wins)
    E = np.concatenate(per_model, axis=1)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    np.save(p, E.astype(np.float32))
    return E


def section_embeddings(a: Analysis, E: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    out = []
    for s in a.sections or [{"start": 1, "end": a.grid.n_bars}]:
        v = E[s["start"] - 1: s["end"]].mean(axis=0)
        out.append((s["start"], s["end"], v / max(np.linalg.norm(v), 1e-9)))
    return out
