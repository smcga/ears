"""Optional stem separation with Demucs (htdemucs) + a short report per stem."""
from __future__ import annotations

import contextlib
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf
import soxr

from .analysis import (BAND_NAMES, NOTE_NAMES, Grid, camelot, harmonic_chroma, key_candidates, key_name,
                       spectral_pass, to_db)

STEM_ORDER = ("drums", "bass", "other", "vocals")
_model = None
_lock = threading.Lock()

def available() -> tuple[bool, str]:
    try:
        import demucs  # noqa: F401
        import torch
    except Exception as exc:  # pragma: no cover - depends on install
        return False, f"Demucs/PyTorch not installed ({exc})."
    return True, "cuda" if torch.cuda.is_available() else "cpu"

def _get_model():
    global _model
    if _model is None:
        from demucs.pretrained import get_model

        with contextlib.redirect_stdout(sys.stderr):
            _model = get_model("htdemucs")
        _model.eval()
    return _model

def stem_paths(path: str) -> tuple[str, dict]:
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(path), f"{base}_stems")
    return out_dir, {n: os.path.join(out_dir, f"{base}_{n}.wav") for n in STEM_ORDER}

def separate(path: str, force: bool = False) -> tuple[dict, dict]:
    out_dir, paths = stem_paths(path)
    src_mtime = os.path.getmtime(path)
    if not force and all(os.path.isfile(p) and os.path.getmtime(p) >= src_mtime for p in paths.values()):
        return paths, {"reused_existing": True}

    import torch
    from demucs.apply import apply_model

    with _lock:
        model = _get_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        x = x[:, :2]
        if sr != model.samplerate:
            x = soxr.resample(x, sr, model.samplerate, quality="HQ")
            sr = model.samplerate
        wav = torch.from_numpy(np.ascontiguousarray(x.T))
        ref = wav.mean(0)
        mu, sd = ref.mean(), ref.std() + 1e-8
        t0 = time.time()
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            out = apply_model(model, ((wav - mu) / sd)[None], device=device, shifts=1, split=True,
                              overlap=0.25, progress=False)[0]
        out = (out * sd + mu).cpu().numpy()
        if device == "cuda":
            torch.cuda.empty_cache()
        os.makedirs(out_dir, exist_ok=True)
        for name, src in zip(model.sources, out):
            if name in paths:
                sf.write(paths[name], src.T, sr, subtype="FLOAT")
        return paths, {"reused_existing": False, "separation_sec": round(time.time() - t0, 1), "device": device}

def _ranges(mask: np.ndarray) -> str:
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append(f"{s + 1}-{i}" if i - s > 1 else f"{s + 1}")
            s = None
    return ", ".join(out) if out else "none"

def stem_report(name: str, path: str, grid: Grid) -> tuple[dict, np.ndarray]:
    import pyloudnorm as pyln

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    n = len(x)
    nb = grid.n_bars
    edges = grid.sample_edges(n, sr)
    pw = np.mean(x * x, axis=1)
    bar_db = to_db(np.add.reduceat(pw, edges[:-1], dtype=np.float64) / np.maximum(np.diff(edges), 1))
    top = float(bar_db.max())
    active = bar_db > max(top - 20.0, -55.0)

    rep: dict = {"file": os.path.basename(path)}
    try:
        rep["lufs"] = round(float(pyln.Meter(sr).integrated_loudness(x)), 1)
    except Exception:
        pass
    rep["peak_dbfs"] = round(float(20 * np.log10(max(np.abs(x).max(), 1e-9))), 1)
    rep["active_bars"] = _ranges(active)
    rep["rms_per_8_bars"] = [int(round(float(v))) for v in
                             (to_db(np.mean(10 ** (bar_db[i:i + 8] / 10))) for i in range(0, nb, 8))]
    mid = x.mean(axis=1)
    sp = spectral_pass(mid, None, sr, n_rows=8)
    bands = to_db(sp["band_m"].mean(axis=0))
    rep["bands_db"] = {b: int(round(float(v))) for b, v in zip(BAND_NAMES, bands)}

    if name in ("bass", "other", "vocals") and active.any():
        # Stems are already drum-free, so skip HPSS; bass reads from C1 (33 Hz) up.
        vec = harmonic_chroma(mid, sr, None, fmin_note="C1" if name == "bass" else "C2",
                              n_octaves=5 if name == "bass" else 6, hpss=False)[0]
        if vec.sum() > 0:
            share = vec / vec.sum()
            top3 = np.argsort(share)[::-1][:4]
            rep["main_pitch_classes"] = ", ".join(f"{NOTE_NAMES[i]} {share[i] * 100:.0f}%" for i in top3)
            if name != "bass":
                c = key_candidates(vec)
                if c:
                    rep["key_guess"] = (f"{key_name(c[0][1], c[0][2])} ({camelot(c[0][1], c[0][2])}), "
                                        f"r={c[0][0]:.2f}; runner-up {key_name(c[1][1], c[1][2])}")
    return rep, bar_db
