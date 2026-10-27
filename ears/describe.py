"""Descriptive tags with CLAP (audio-text embeddings), scored per section and for the whole track.

Two LAION CLAP models are averaged for robustness. Weights come from Hugging Face's own safetensors
conversions (SFconvertbot PRs), pinned to exact commits: safetensors files hold numbers only, no code.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

import numpy as np
import soundfile as sf
import soxr

from .analysis import Analysis, mmss

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MODELS = (
    ("laion/larger_clap_music_and_speech", "8a2acdbf328883b3cc38b575179e5950cf2deeb4"),
    ("laion/clap-htsat-unfused", "79b58ed25fc00386262a2bea4b19fd21dc4310a0"),
)
SR = 48000
WIN = 10.0  # CLAP listens to 10 s at a time

GENRE = ["techno", "minimal techno", "acid techno", "house", "deep house", "tech house", "UK garage", "2-step garage",
         "drum and bass", "jungle", "dubstep", "trance", "progressive house", "breakbeat", "electro", "synthwave",
         "ambient", "downtempo", "lo-fi hip hop", "hip hop", "trap", "pop", "disco", "afro house", "hardstyle"]
VOCAB = {
    "genre": [f"{g} music" for g in GENRE],
    "groove": ["four-on-the-floor kick drum", "swung shuffling groove", "straight rigid quantized groove",
               "broken syncopated beat", "half-time groove", "fast breakbeats", "no drums"],
    "texture": ["gritty distorted sound", "clean polished production", "warm analog sound", "cold digital sound",
                "lo-fi dusty sound", "bright crisp highs", "dark muffled sound", "airy spacious reverb",
                "dry upfront mix", "glitchy chopped sounds"],
    "elements": ["plucked synth", "lush synth pads", "supersaw lead", "acid 303 bassline", "deep sub bass",
                 "reese bass", "wobble bass", "piano", "sung vocals", "vocal chops", "guitar", "strings",
                 "arpeggiated synth", "risers and sweeps", "claps and hi-hats"],
    "mood": ["euphoric", "dark", "melancholic", "uplifting", "aggressive", "hypnotic", "chill and relaxed",
             "energetic", "dreamy", "tense"],
}
NOTE = ("CLAP scores are relative similarities between the audio and each phrase (typically 0.1-0.5): compare "
        "tags within a category, not absolute values. 'margin' = how far the top tag stands above that category's "
        "median. It's a learned impression from big web datasets, not hearing: genre names especially are fuzzy, "
        "and it can't judge whether something sounds good.")

_models: list = []
_text_cache: dict = {}
_lock = threading.Lock()


def _load():
    if _models:
        return _models
    import torch
    from transformers import ClapModel, ClapProcessor
    from transformers.utils import logging as tlog

    tlog.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(sys.stderr):
        for rid, rev in MODELS:
            try:
                proc = ClapProcessor.from_pretrained(rid, revision=rev)
                model = ClapModel.from_pretrained(rid, revision=rev, use_safetensors=True)
            except OSError:  # offline: use the cached copy
                proc = ClapProcessor.from_pretrained(rid, revision=rev, local_files_only=True)
                model = ClapModel.from_pretrained(rid, revision=rev, use_safetensors=True, local_files_only=True)
            _models.append((proc, model.eval().to(device), device))
    return _models


def available() -> tuple[bool, str]:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"transformers/PyTorch not installed ({exc})."
    return True, "ok"


def _norm(t):
    import torch

    return torch.nn.functional.normalize(t, dim=-1)


def _text_emb(phrases: list[str]) -> list[np.ndarray]:
    import torch

    out = []
    for i, (proc, model, dev) in enumerate(_load()):
        missing = [p for p in phrases if (i, p) not in _text_cache]
        if missing:
            with torch.no_grad():
                e = _norm(model.get_text_features(**proc(text=missing, return_tensors="pt", padding=True).to(dev)))
            for p, v in zip(missing, e.cpu().numpy()):
                _text_cache[(i, p)] = v
        out.append(np.stack([_text_cache[(i, p)] for p in phrases]))
    return out


def _audio_emb(windows: list[np.ndarray]) -> list[np.ndarray]:
    import torch

    out = []
    for proc, model, dev in _load():
        embs = []
        for a in range(0, len(windows), 16):
            with torch.no_grad():
                inp = proc(audio=windows[a:a + 16], sampling_rate=SR, return_tensors="pt").to(dev)
                embs.append(_norm(model.get_audio_features(**inp)).cpu().numpy())
        out.append(np.concatenate(embs))
    return out


def _windows_for(y: np.ndarray, t0: float, t1: float, n_max: int = 3) -> list[np.ndarray]:
    w = int(WIN * SR)
    span = t1 - t0
    if span <= WIN:
        starts = [max(0.0, (t0 + t1) / 2 - WIN / 2)]
    else:
        n = min(n_max, max(1, int(span // WIN)))
        starts = [t0 + (span - WIN) * (k + 0.5) / n for k in range(n)]
    wins = []
    for s in starts:
        seg = y[int(s * SR): int(s * SR) + w]
        if len(seg) < w:
            seg = np.pad(seg, (0, w - len(seg)))
        wins.append(seg.astype(np.float32))
    return wins


def describe(a: Analysis, extra: list[str] | None = None) -> dict:
    x, sr = sf.read(a.path, dtype="float32", always_2d=True)
    y = soxr.resample(x.mean(axis=1), sr, SR, quality="HQ")
    g = a.grid
    secs = a.sections or [{"start": 1, "end": g.n_bars, "rms": 0.0}]

    windows, owner = [], []
    for k, s in enumerate(secs):
        t0, t1 = g.t_of_bar(s["start"]), min(a.duration, g.t_of_bar(s["end"] + 1))
        if t1 - t0 < 2.0:
            continue
        for w in _windows_for(y, t0, t1):
            windows.append(w)
            owner.append(k)

    phrases = [p for cat in VOCAB.values() for p in cat]
    extra = [e.strip() for e in (extra or []) if e.strip()][:12]
    all_phrases = phrases + extra
    with _lock, contextlib.redirect_stdout(sys.stderr):
        T = _text_emb(all_phrases)
        A = _audio_emb(windows)
    # (n_windows, n_phrases) similarity, averaged over the two models
    S = np.mean([ae @ te.T for ae, te in zip(A, T)], axis=0)
    owner = np.array(owner)

    # Section scores = mean over its windows; track = loudness-weighted mean over sections
    sec_ids = sorted(set(owner.tolist()))
    sec_S = {k: S[owner == k].mean(axis=0) for k in sec_ids}
    w = np.array([10 ** (secs[k].get("rms", 0.0) / 20) * (secs[k]["end"] - secs[k]["start"] + 1) for k in sec_ids])
    track_S = np.tensordot(w / w.sum(), np.stack([sec_S[k] for k in sec_ids]), axes=1)

    idx = {p: i for i, p in enumerate(all_phrases)}

    def top(scores, cat, n=3):
        vals = np.array([scores[idx[p]] for p in VOCAB[cat]])
        order = np.argsort(-vals)[:n]
        return [(VOCAB[cat][i].removesuffix(" music"), float(vals[i])) for i in order], float(vals.max() - np.median(vals))

    whole = {}
    for cat in VOCAB:
        best, margin = top(track_S, cat)
        whole[cat] = {"top": [f"{t} {s:.2f}" for t, s in best], "margin": round(margin, 2)}

    lines = []
    for k in sec_ids:
        s = secs[k]
        sc = sec_S[k]
        parts = [top(sc, "genre", 1)[0][0][0], top(sc, "groove", 1)[0][0][0]]
        parts += [t for t, _ in top(sc, "elements", 2)[0]] + [top(sc, "texture", 1)[0][0][0], top(sc, "mood", 1)[0][0][0]]
        t0, t1 = g.t_of_bar(s["start"]), min(a.duration, g.t_of_bar(s["end"] + 1))
        lines.append(f"bars {s['start']}-{s['end']} ({mmss(t0)}-{mmss(t1)}): " + ", ".join(parts))

    out = {"whole_track": whole, "per_section": lines}
    if extra:
        ref_vals = np.array([track_S[idx[p]] for p in phrases])
        rows = []
        for e in extra:
            v = float(track_S[idx[e]])
            pct = int(round(100 * (ref_vals < v).mean()))
            best_sec = max(sec_ids, key=lambda k: sec_S[k][idx[e]])
            rows.append(f"'{e}': {v:.2f} (scores higher than {pct}% of the built-in tags; strongest in bars "
                        f"{secs[best_sec]['start']}-{secs[best_sec]['end']} at {sec_S[best_sec][idx[e]]:.2f})")
        out["your_phrases"] = rows
    out["windows_listened"] = f"{len(windows)} x {WIN:g} s clips across {len(sec_ids)} sections"
    out["note"] = NOTE
    return out
