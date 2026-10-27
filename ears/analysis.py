"""Measurement core: turns an audio file into numbers on an FL-style bar grid.

Nothing here "hears" anything. It measures levels, spectra, onsets and pitch-class
energy, and report.py turns those numbers into words.
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.fft
import scipy.signal
import soundfile as sf
import soxr

warnings.filterwarnings("ignore")

ANALYSIS_SR = 22050
BANDS = (("sub", 0.0, 60.0), ("low", 60.0, 250.0), ("mid", 250.0, 2000.0), ("high", 2000.0, None))
BAND_NAMES = tuple(b[0] for b in BANDS)
NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FLOOR_DB = -99.0
CLIP_LEVEL = 0.999  # |sample| at or above this counts as hitting 0 dBFS
SILENCE_DB = -60.0
ONSET_LAG_SEC = 0.040


# ----------------------------------------------------------------------------- helpers

def to_db(power, floor: float = FLOOR_DB):
    p = np.maximum(np.asarray(power, dtype=np.float64), 1e-30)
    return np.maximum(10.0 * np.log10(p), floor)


def mmss(sec: float) -> str:
    m, s = divmod(int(round(max(0.0, float(sec)))), 60)
    return f"{m}:{s:02d}"


def key_name(tonic: int, mode: str) -> str:
    return f"{NOTE_NAMES[tonic]} {mode}"


def camelot(tonic: int, mode: str) -> str:
    rel_major = tonic if mode == "major" else (tonic + 3) % 12
    n = (7 * rel_major + 8) % 12 or 12
    return f"{n}{'B' if mode == 'major' else 'A'}"


@dataclass
class Grid:
    """FL-style bar grid: bar 1 starts at 0:00, 4 beats per bar."""

    bpm: float
    duration: float

    @property
    def beat_sec(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_sec(self) -> float:
        return 4.0 * self.beat_sec

    @property
    def n_bars(self) -> int:
        """Whole bars; a final partial bar shorter than half a bar is folded into the previous one."""
        full = int(self.duration // self.bar_sec)
        rem = self.duration - full * self.bar_sec
        return max(1, full + (1 if rem >= 0.5 * self.bar_sec else 0))

    def sample_edges(self, n: int, sr: int) -> np.ndarray:
        e = np.round(np.arange(self.n_bars + 1) * self.bar_sec * sr).astype(np.int64)
        e[-1] = n
        return np.minimum(e, n)

    def bar_at(self, t: float) -> int:
        return int(max(0.0, t) // self.bar_sec) + 1

    def t_of_bar(self, bar: float) -> float:
        return (bar - 1) * self.bar_sec


@dataclass
class Analysis:
    path: str
    sr: int
    channels: int
    duration: float
    grid: Grid
    tempo: dict
    key: dict
    loud: dict
    stereo: dict | None
    bar_n: np.ndarray  # samples per bar
    bar_rms_db: np.ndarray  # (n_bars,)
    bar_band_db: np.ndarray  # (n_bars, 4)
    bar_perc_db: np.ndarray  # (n_bars,)
    bar_corr: np.ndarray | None  # (n_bars,)
    band_side_db: np.ndarray | None  # (4,) side-vs-mid per band, whole track
    sections: list = field(default_factory=list)
    novelty: np.ndarray | None = None
    edges: dict = field(default_factory=dict)  # lead/tail silence
    spec: dict = field(default_factory=dict)  # data for rendering
    beat_curve: tuple | None = None  # (times, rms_db) per beat
    bar_chroma: np.ndarray | None = None  # (n_bars, 12) for stems/bass notes

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


# ----------------------------------------------------------------------------- loading

def load_audio(path: str) -> tuple[np.ndarray, int]:
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile message is usually clear enough
        raise ValueError(f"Couldn't decode '{os.path.basename(path)}': {exc}") from exc
    if data.shape[0] < sr:
        raise ValueError("File is shorter than 1 second; nothing to analyse.")
    if data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data), int(sr)


# ----------------------------------------------------------------------------- tempo

def _fold_bpm(b: float) -> float:
    while b < 78:
        b *= 2
    while b > 185:
        b /= 2
    return b


def _snap_bpm(b: float) -> float:
    return float(round(b)) if abs(b - round(b)) <= 0.15 else round(b, 1)


def _refine_bpm(beat_times: np.ndarray, approx: float) -> float:
    """Fit a straight line through beat times (handles skipped beats)."""
    if len(beat_times) < 8:
        return approx
    period = 60.0 / approx
    steps = np.maximum(np.round(np.diff(beat_times) / period), 1)
    idx = np.concatenate([[0.0], np.cumsum(steps)])
    slope, icpt = np.polyfit(idx, beat_times, 1)
    keep = np.abs(beat_times - (slope * idx + icpt)) < 0.06
    if keep.sum() >= 8:
        slope, icpt = np.polyfit(idx[keep], beat_times[keep], 1)
    bpm = 60.0 / slope if slope > 0 else approx
    return bpm if abs(bpm / approx - 1) < 0.08 else approx


def analyze_tempo(y: np.ndarray, sr: int, bpm_hint: float | None) -> dict:
    import librosa

    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.median)
    raw = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop))[0])
    approx = _fold_bpm(raw) if raw > 0 else 120.0
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=approx)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    detected = _snap_bpm(_fold_bpm(_refine_bpm(bt, approx)))
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop, bpm=detected)
    bt = librosa.frames_to_time(beats, sr=sr, hop_length=hop)

    bpm = float(bpm_hint) if bpm_hint else detected
    beat_sec = 60.0 / bpm

    # How well do detected beats sit on a grid that starts at 0:00?
    # (onset detection lags real transients by ~40 ms; measured on a synthetic, perfectly gridded kick)
    if len(bt) >= 8:
        z = np.exp(2j * np.pi * (bt - ONSET_LAG_SEC) / beat_sec).mean()
        fit, offset_ms = float(abs(z)), float(np.angle(z) / (2 * np.pi) * beat_sec * 1000)
    else:
        fit, offset_ms = 0.0, 0.0

    # Confidence: pulse clarity (autocorrelation at the beat lag) + onset strength on beats.
    e = env - env.mean()
    denom = float(np.dot(e, e)) or 1.0
    lag = beat_sec * sr / hop
    acs = [float(np.dot(e[:-L], e[L:])) / denom for L in range(max(1, int(lag) - 1), int(lag) + 3) if L < len(e)]
    clarity = max(0.0, max(acs)) if acs else 0.0
    ratio = float(env[beats].mean() / (env.mean() + 1e-9)) if len(beats) else 1.0
    conf = 0.5 * min(1.0, clarity / 0.35) + 0.5 * min(1.0, max(0.0, ratio - 1.0) / 1.5)

    return {
        "bpm": bpm,
        "source": "given" if bpm_hint else "detected",
        "detected": detected,
        "conf": round(conf, 2),
        "grid_fit": round(fit, 2),
        "offset_ms": round(offset_ms),
        "n_beats": int(len(bt)),
    }


# ----------------------------------------------------------------------------- key

def key_candidates(chroma_vec: np.ndarray) -> list[tuple[float, int, str]]:
    v = np.asarray(chroma_vec, dtype=np.float64)
    if v.sum() <= 0 or np.ptp(v) < 1e-9:
        return []
    out = []
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            out.append((float(np.corrcoef(v, np.roll(prof, tonic))[0, 1]), tonic, mode))
    out.sort(reverse=True)
    return out


def _is_relative(a: tuple, b: tuple) -> bool:
    (_, ta, ma), (_, tb, mb) = a, b
    if ma == mb:
        return False
    maj, mnr = (ta, tb) if ma == "major" else (tb, ta)
    return (mnr + 3) % 12 == maj


def analyze_key(bar_chroma: np.ndarray, bar_harm_db: np.ndarray) -> dict:
    total = bar_chroma.sum(axis=0)
    cands = key_candidates(total)
    if not cands:
        return {"best": None, "conf": "none", "note": "No pitched content found."}
    best, second = cands[0], cands[1]
    margin = best[0] - second[0]
    if best[0] >= 0.75 and margin >= 0.08:
        level = 2
    elif best[0] >= 0.6 and margin >= 0.03:
        level = 1
    else:
        level = 0

    # Stability: does each 16-bar window (with real harmonic content) agree?
    agree = n_win = 0
    loud = bar_harm_db > (np.max(bar_harm_db) - 30)
    for a in range(0, len(bar_chroma), 16):
        sl = slice(a, a + 16)
        if loud[sl].sum() < 4:
            continue
        wc = key_candidates(bar_chroma[sl][loud[sl]].sum(axis=0))
        if not wc:
            continue
        n_win += 1
        agree += int(wc[0][1:] == best[1:])
    if n_win >= 3 and agree / n_win < 0.5:
        level = max(0, level - 1)

    out = {
        "best": f"{key_name(best[1], best[2])} ({camelot(best[1], best[2])})",
        "r": round(best[0], 2),
        "runner_up": f"{key_name(second[1], second[2])} ({camelot(second[1], second[2])})",
        "r2": round(second[0], 2),
        "conf": ("low", "medium", "high")[level],
    }
    if n_win:
        out["windows_agree"] = f"{agree}/{n_win} 16-bar windows"
    if _is_relative(best, second) and margin < 0.06:
        out["note"] = "Runner-up is the relative key (same notes); which chord feels like home decides it."
    elif level == 0:
        out["note"] = "Low confidence: treat as a guess and check by ear."
    out["_tonic"], out["_mode"] = best[1], best[2]
    return out


# ----------------------------------------------------------------------------- spectra

def _band_matrix(freqs: np.ndarray, scale: float) -> np.ndarray:
    B = np.zeros((len(BANDS), len(freqs)), dtype=np.float32)
    for i, (_, lo, hi) in enumerate(BANDS):
        hi = hi if hi is not None else freqs[-1] + 1
        B[i] = ((freqs > lo) & (freqs <= hi)) * scale
    return B


def _logfreq_matrix(freqs: np.ndarray, fmin: float, fmax: float, n_rows: int) -> np.ndarray:
    edges = np.geomspace(fmin, fmax, n_rows + 1)
    M = np.zeros((n_rows, len(freqs)), dtype=np.float32)
    for r in range(n_rows):
        idx = np.where((freqs >= edges[r]) & (freqs < edges[r + 1]))[0]
        if len(idx) == 0:
            idx = [int(np.argmin(np.abs(freqs - math.sqrt(edges[r] * edges[r + 1]))))]
        M[r, idx] = 1.0 / len(idx)
    return M


def spectral_pass(mid: np.ndarray, side: np.ndarray | None, sr: int, n_fft: int = 8192, hop: int = 2048,
                  n_rows: int = 320, fmin: float = 20.0, chunk: int = 256) -> dict:
    """One chunked STFT over mid (and side). Returns per-frame band powers and a log-freq image."""
    from numpy.lib.stride_tricks import sliding_window_view

    win = scipy.signal.get_window("hann", n_fft).astype(np.float32)
    scale = 2.0 / (n_fft * float(np.sum(win.astype(np.float64) ** 2)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fmax = min(20000.0, sr / 2.0)
    B = _band_matrix(freqs, scale)
    M = _logfreq_matrix(freqs, fmin, fmax, n_rows) * scale

    def frames(sig):
        return sliding_window_view(np.pad(sig, (n_fft // 2, n_fft // 2)), n_fft)[::hop]

    Fm = frames(mid)
    Fs = frames(side) if side is not None else None
    n = Fm.shape[0]
    band_m = np.zeros((n, len(BANDS)), dtype=np.float32)
    band_s = np.zeros_like(band_m)
    img = np.zeros((n, n_rows), dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        X = scipy.fft.rfft(Fm[a:b] * win, axis=1, workers=-1)
        P = X.real ** 2 + X.imag ** 2
        band_m[a:b] = P @ B.T
        if Fs is not None:
            Xs = scipy.fft.rfft(Fs[a:b] * win, axis=1, workers=-1)
            Ps = Xs.real ** 2 + Xs.imag ** 2
            band_s[a:b] = Ps @ B.T
            P = P + Ps
        img[a:b] = P @ M.T
    return {"times": np.arange(n) * hop / sr, "band_m": band_m, "band_s": band_s if Fs is not None else None,
            "img": img, "fmin": fmin, "fmax": fmax}


KEY_SR = 11025


def harmonic_chroma(mono: np.ndarray, sr: int, grid: Grid | None, fmin_note: str = "C2", n_octaves: int = 6,
                    hpss: bool = True) -> np.ndarray:
    """Pitch-class energy per bar: drums removed (HPSS), then constant-Q chroma (C2..C8).

    Returns (n_bars, 12), or (1, 12) when grid is None.
    """
    import librosa

    y = soxr.resample(mono, sr, KEY_SR, quality="HQ").astype(np.float32)
    if hpss:
        S = librosa.stft(y, n_fft=1024, hop_length=256)
        H, _ = librosa.decompose.hpss(S, kernel_size=(17, 31), margin=2.0)
        y = librosa.istft(H, hop_length=256, length=len(y))
        del S, H
    cq = librosa.feature.chroma_cqt(y=y, sr=KEY_SR, hop_length=1024, fmin=librosa.note_to_hz(fmin_note),
                                    n_octaves=n_octaves)
    if grid is None:
        return cq.sum(axis=1)[None, :]
    t = np.arange(cq.shape[1]) * 1024 / KEY_SR
    return _per_bar(cq.T, t, grid, how="sum")


def _per_bar(values: np.ndarray, times: np.ndarray, grid: Grid, how: str = "mean") -> np.ndarray:
    """Aggregate time-ordered frame values (n_frames, ...) into bars by frame time."""
    nb = grid.n_bars
    idx = np.clip((times // grid.bar_sec).astype(int), 0, nb - 1)
    counts = np.bincount(idx, minlength=nb)
    nz = counts > 0
    starts = np.searchsorted(idx, np.arange(nb))[nz]
    out = np.zeros((nb,) + values.shape[1:], dtype=np.float64)
    out[nz] = np.add.reduceat(values.astype(np.float64), starts, axis=0)
    if how == "mean":
        out[nz] /= counts[nz].reshape((-1,) + (1,) * (values.ndim - 1))
    return out


# ----------------------------------------------------------------------------- loudness

def loudness(x: np.ndarray, sr: int) -> dict:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(x))
    except Exception:
        lufs = float("-inf")

    # Short-term (3 s) max, using pyloudnorm's own K-weighting coefficients.
    st_max = None
    try:
        filters = getattr(meter, "_filters", None)
        if filters:
            xk = x.astype(np.float32)
            for f in filters.values():
                xk = (f.passband_gain * scipy.signal.lfilter(f.b, f.a, xk, axis=0)).astype(np.float32)
            p = np.sum(xk.astype(np.float64) ** 2, axis=1)
            w, h = int(3 * sr), int(0.1 * sr)
            if len(p) >= w:
                cs = np.concatenate([[0.0], np.cumsum(p)])
                starts = np.arange(0, len(p) - w + 1, h)
                ms = (cs[starts + w] - cs[starts]) / w
                st_max = float(-0.691 + 10 * np.log10(max(ms.max(), 1e-12)))
    except Exception:
        st_max = None

    # Sample peak, clipping and true peak (4x oversampled).
    absx = np.abs(x)
    peak = float(absx.max())
    clip_mask = absx >= CLIP_LEVEL
    clip_samples = int(clip_mask.sum())
    over = int((absx > 1.0).sum())
    runs, first_clip = 0, None
    for ch in range(x.shape[1]):
        m = clip_mask[:, ch].astype(np.int8)
        d = np.diff(np.concatenate([[0], m, [0]]))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        long_runs = starts[(ends - starts) >= 3]
        runs += len(long_runs)
        if len(long_runs):
            t = long_runs[0] / sr
            first_clip = t if first_clip is None else min(first_clip, t)
    if first_clip is None and clip_samples:
        first_clip = float(np.argmax(clip_mask.any(axis=1)) / sr)

    tp = peak
    step, pad = 1 << 18, 64
    for ch in range(x.shape[1]):
        for a in range(0, x.shape[0], step):
            seg = x[max(0, a - pad): a + step + pad, ch]
            if np.abs(seg).max() < tp * 0.5:
                continue
            tp = max(tp, float(np.abs(scipy.signal.resample_poly(seg, 4, 1)).max()))

    return {
        "lufs": lufs,
        "st_max": st_max,
        "peak_dbfs": float(20 * np.log10(max(peak, 1e-9))),
        "true_peak_dbtp": float(20 * np.log10(max(tp, 1e-9))),
        "clip_samples": clip_samples,
        "clip_runs": runs,
        "over_0dbfs": over,
        "first_clip_t": first_clip,
        "dc": float(np.abs(x.mean(axis=0)).max()),
    }


# ----------------------------------------------------------------------------- segmentation

def novelty_curve(F: np.ndarray, weights: np.ndarray, L: int = 4) -> np.ndarray:
    n = len(F)
    nov = np.zeros(n)
    wsum = math.sqrt(float(np.sum(weights ** 2)))
    for b in range(2, n - 1):
        left = F[max(0, b - L): b].mean(axis=0)
        right = F[b: min(n, b + L)].mean(axis=0)
        nov[b] = float(np.sqrt(np.sum(((left - right) * weights) ** 2))) / wsum
    return nov


def pick_boundaries(nov: np.ndarray, n_bars: int, max_sections: int = 14) -> list[int]:
    """0-based bar indices where a new section starts (bar index b => FL bar b+1)."""
    if n_bars < 8:
        return []
    bonus = np.ones(n_bars)
    bonus[np.arange(n_bars) % 4 == 0] = 1.05
    bonus[np.arange(n_bars) % 8 == 0] = 1.10
    score = nov * bonus
    thr = max(0.45, float(np.percentile(score[2:-1], 75)) * 1.5)
    cands = []
    for b in range(4, n_bars - 3):  # every section at least 4 bars long
        if score[b] >= thr and score[b] >= score[max(0, b - 3): b + 4].max():
            cands.append(b)
    chosen: list[int] = []
    for b in sorted(cands, key=lambda i: -score[i]):
        if all(abs(b - c) >= 4 for c in chosen):
            chosen.append(b)
        if len(chosen) >= max_sections - 1:
            break
    return sorted(chosen)


def _power_mean_db(db_vals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = 10 ** (np.asarray(db_vals) / 10.0)
    w = weights / max(weights.sum(), 1)
    return to_db(np.tensordot(w, p, axes=(0, 0)))


ELEMENTS = (("perc", "drums"), ("sub", "sub"), ("low", "bass"), ("mid", "mids"), ("high", "highs"))


def describe_sections(starts: list[int], a: "Analysis") -> list[dict]:
    n = a.grid.n_bars
    bounds = [0] + starts + [n]
    secs = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        w = a.bar_n[s:e].astype(float)
        band = _power_mean_db(a.bar_band_db[s:e], w)
        secs.append({
            "start": s + 1, "end": e,
            "rms": float(_power_mean_db(a.bar_rms_db[s:e], w)),
            "perc": float(_power_mean_db(a.bar_perc_db[s:e], w)),
            **{name: float(band[i]) for i, name in enumerate(BAND_NAMES)},
        })
    if not secs:
        return secs
    tmax = {k: max(s[k] for s in secs) for k, _ in ELEMENTS}
    loudest = int(np.argmax([s["rms"] for s in secs]))
    quietest = int(np.argmin([s["rms"] for s in secs]))

    def present(s, k):
        own = s["rms"] if k == "perc" else max(s[b] for b in BAND_NAMES)
        return s[k] > max(tmax[k] - 15.0, -60.0, own - (20.0 if k == "perc" else 30.0))

    for i, s in enumerate(secs):
        parts = []
        if i == 0:
            top = max(s[b] for b in BAND_NAMES)
            has = [lbl for k, lbl in ELEMENTS if k != "perc" and s[k] > top - 10 and s[k] > -65]
            if s["perc"] - s["rms"] > -8 and s["perc"] > -65:
                has.insert(0, "drums")
            rel = s["rms"] - secs[loudest]["rms"]
            parts.append("start")
            if s["rms"] < -65:
                parts.append("near silent")
            else:
                if rel <= -3:
                    parts.append(f"{rel:.0f} dB vs loudest")
                parts.append("mostly " + "+".join(has) if has else "quiet")
        else:
            p = secs[i - 1]
            ins, outs, ups, downs = [], [], [], []
            for k, lbl in ELEMENTS:
                d = s[k] - p[k]
                if present(s, k) and not present(p, k) and d >= 6:
                    ins.append(lbl)
                elif present(p, k) and not present(s, k) and d <= -6:
                    outs.append(lbl)
                elif d >= 4 and (present(s, k) or present(p, k)):
                    ups.append(lbl)
                elif d <= -4 and (present(s, k) or present(p, k)):
                    downs.append(lbl)
            for lst, word in ((ins, "in"), (outs, "out"), (ups, "up"), (downs, "down")):
                if lst:
                    parts.append("+".join(lst) + " " + word)
            dr = s["rms"] - p["rms"]
            if dr >= 2:
                parts.append(f"energy up {dr:.0f} dB")
            elif dr <= -2:
                parts.append(f"energy down {-dr:.0f} dB")
            else:
                parts.append("energy steady" if (ins or outs or ups or downs) else "same level, timbre/harmony change")
        if len(secs) > 1 and i == loudest:
            parts.append("(loudest)")
        elif len(secs) > 2 and i == quietest and i != 0:
            parts.append("(quietest)")
        s["label"] = ", ".join(parts)
    return secs


# ----------------------------------------------------------------------------- main entry

def analyze(path: str, bpm: float | None = None, with_image: bool = True) -> Analysis:
    import librosa

    from concurrent.futures import ThreadPoolExecutor

    x, sr = load_audio(path)
    n, ch = x.shape
    duration = n / sr
    pool = ThreadPoolExecutor(max_workers=1)
    loud_future = pool.submit(loudness, x, sr)  # slowest independent step; runs alongside the rest
    mid = x.mean(axis=1)
    side = (0.5 * (x[:, 0] - x[:, 1])).astype(np.float32) if ch == 2 else None

    y = soxr.resample(mid, sr, ANALYSIS_SR, quality="HQ").astype(np.float32)
    tempo = analyze_tempo(y, ANALYSIS_SR, bpm)
    grid = Grid(tempo["bpm"], duration)
    nb = grid.n_bars
    key_pool = ThreadPoolExecutor(max_workers=1)
    chroma_future = key_pool.submit(harmonic_chroma, mid, sr, grid)

    # --- time-domain levels per bar / beat
    bar_edges = grid.sample_edges(n, sr)
    bar_n = np.maximum(np.diff(bar_edges), 1)
    pw = np.mean(x * x, axis=1)
    bar_rms_db = to_db(np.add.reduceat(pw, bar_edges[:-1], dtype=np.float64) / bar_n)

    beat_starts = np.round(np.arange(math.ceil(duration / grid.beat_sec) + 1) * grid.beat_sec * sr).astype(np.int64)
    beat_starts = beat_starts[beat_starts < n]
    beat_n = np.diff(np.append(beat_starts, n))
    beat_db = to_db(np.add.reduceat(pw, beat_starts, dtype=np.float64) / beat_n, floor=-80)
    beat_curve = (beat_starts / sr, beat_db)

    stereo = None
    bar_corr = None
    if ch == 2:
        L, R = x[:, 0], x[:, 1]
        lr = np.add.reduceat(L * R, bar_edges[:-1], dtype=np.float64)
        ll = np.add.reduceat(L * L, bar_edges[:-1], dtype=np.float64)
        rr = np.add.reduceat(R * R, bar_edges[:-1], dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            bar_corr = np.where(ll * rr > 1e-12, lr / np.sqrt(ll * rr), np.nan)
        tot = float(np.sqrt(ll.sum() * rr.sum()))
        corr = float(lr.sum() / tot) if tot > 0 else 1.0
        side_p = float(np.mean(side.astype(np.float64) ** 2))
        mid_p = float(np.mean(mid.astype(np.float64) ** 2))
        stereo = {"corr": corr, "side_vs_mid_db": float(to_db(side_p) - to_db(mid_p))}

    # --- edges (silence)
    loud_idx = np.where(np.abs(mid) > 10 ** (SILENCE_DB / 20))[0]
    edges = {"lead": float(loud_idx[0] / sr) if len(loud_idx) else duration,
             "tail": float((n - 1 - loud_idx[-1]) / sr) if len(loud_idx) else duration}

    # --- native-rate spectral pass (bands + image)
    sp = spectral_pass(mid, side, sr)
    band_tot = sp["band_m"] + (sp["band_s"] if sp["band_s"] is not None else 0)
    bar_band_db = to_db(_per_bar(band_tot, sp["times"], grid))
    band_side_db = None
    if sp["band_s"] is not None:
        band_side_db = to_db(sp["band_s"].sum(axis=0)) - to_db(sp["band_m"].sum(axis=0))
        stereo["band_side_db"] = band_side_db

    # --- analysis-rate features: drums vs tonal (HPSS on mel), chroma, MFCC
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    t22 = np.arange(S.shape[1]) * 1024 / ANALYSIS_SR
    mel = librosa.feature.melspectrogram(S=S, sr=ANALYSIS_SR, n_mels=96)
    Hm, Pm = librosa.decompose.hpss(mel, kernel_size=(17, 17))
    h_bar = _per_bar(Hm.sum(axis=0), t22, grid)
    p_bar = _per_bar(Pm.sum(axis=0), t22, grid)
    perc_share = p_bar / np.maximum(h_bar + p_bar, 1e-20)
    bar_perc_db = np.maximum(bar_rms_db + 10 * np.log10(np.maximum(perc_share, 1e-6)), FLOOR_DB)
    bar_harm_db = np.maximum(bar_rms_db + 10 * np.log10(np.maximum(1 - perc_share, 1e-6)), FLOOR_DB)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)[1:]
    bar_mfcc = _per_bar(mfcc.T, t22, grid)
    del S, mel, Hm, Pm

    bar_chroma = chroma_future.result()
    key_pool.shutdown()
    key = analyze_key(bar_chroma, bar_harm_db)
    loud = loud_future.result()
    pool.shutdown()

    a = Analysis(path=path, sr=sr, channels=ch, duration=duration, grid=grid, tempo=tempo, key=key, loud=loud,
                 stereo=stereo, bar_n=bar_n, bar_rms_db=bar_rms_db, bar_band_db=bar_band_db,
                 bar_perc_db=bar_perc_db, bar_corr=bar_corr, band_side_db=band_side_db, edges=edges,
                 beat_curve=beat_curve, bar_chroma=bar_chroma)

    # --- structure
    cnorm = bar_chroma / np.maximum(bar_chroma.sum(axis=1, keepdims=True), 1e-9)
    F = np.column_stack([bar_rms_db, bar_band_db, bar_perc_db, bar_mfcc, cnorm])
    floors = np.array([3.0] * 6 + [3.0] * bar_mfcc.shape[1] + [0.05] * 12)
    weights = np.array([1.5, 1, 1, 1, 1, 1.0] + [0.3] * bar_mfcc.shape[1] + [0.25] * 12)
    Z = (F - np.median(F, axis=0)) / np.maximum(F.std(axis=0), floors)
    a.novelty = novelty_curve(Z, weights)
    a.sections = describe_sections(pick_boundaries(a.novelty, nb), a)

    if with_image:
        img = sp["img"]
        k = max(1, math.ceil(len(img) / 1800))
        m = (len(img) // k) * k
        if m and k > 1:
            img = img[:m].reshape(-1, k, img.shape[1]).mean(axis=1)
        a.spec = {"img_db": to_db(img, floor=-140).astype(np.float32), "t_end": duration,
                  "fmin": sp["fmin"], "fmax": sp["fmax"]}
    return a


def analyze_range(path: str, start: float, end: float, grid: Grid) -> dict:
    """Small stats block + a fine spectrogram for a time range."""
    info = sf.info(path)
    sr = int(info.samplerate)
    a0, a1 = int(start * sr), int(end * sr)
    x, _ = sf.read(path, start=a0, stop=a1, dtype="float32", always_2d=True)
    x = x[:, :2]
    mid = x.mean(axis=1)
    side = (0.5 * (x[:, 0] - x[:, 1])).astype(np.float32) if x.shape[1] == 2 else None
    sp = spectral_pass(mid, side, sr, n_fft=4096, hop=256, n_rows=300)
    band_tot = sp["band_m"] + (sp["band_s"] if sp["band_s"] is not None else 0)
    out = {
        "span": f"{mmss(start)}-{mmss(end)} = bars {grid.bar_at(start)}-{grid.bar_at(max(start, end - 1e-3))}",
        "rms_db": round(float(to_db(np.mean(x * x))), 1),
        "peak_dbfs": round(float(20 * np.log10(max(np.abs(x).max(), 1e-9))), 1),
        "bands_db": {b: int(round(float(v))) for b, v in zip(BAND_NAMES, to_db(band_tot.mean(axis=0)))},
    }
    if len(x) >= int(0.5 * sr):
        try:
            import pyloudnorm as pyln

            out["lufs"] = round(float(pyln.Meter(sr).integrated_loudness(x)), 1)
        except Exception:
            pass
    img = sp["img"]
    k = max(1, math.ceil(len(img) / 1800))
    m = (len(img) // k) * k
    if k > 1:
        img = img[:m].reshape(-1, k, img.shape[1]).mean(axis=1)
    out["_spec"] = {"img_db": to_db(img, floor=-140).astype(np.float32), "t0": start, "t1": end,
                    "fmin": sp["fmin"], "fmax": sp["fmax"]}
    return out
