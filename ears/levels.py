"""Loudness over time for long recordings (game captures) and per-level balance reports.

Loudness follows ITU-R BS.1770 (the LUFS standard): K-weighting, 100 ms energy blocks,
400 ms momentary windows with 75% overlap for the gated integrated value, 3 s short-term windows.
It streams, so an hour-long capture never has to sit in memory.
"""
from __future__ import annotations

import json
import os
import re
import threading

import numpy as np
import scipy.signal
import soundfile as sf

HOP_SEC = 0.1          # one energy block
SILENT_LUFS = -60.0    # below this a block counts as silence (loading screens, pauses)


def _db(ms):
    return -0.691 + 10 * np.log10(np.maximum(ms, 1e-12))


class LoudnessTracker:
    """Feed audio in any chunk size; keeps one K-weighted mean-square value and one sample peak per 100 ms."""

    def __init__(self, sr: int, channels: int):
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        self.stages = [(f.passband_gain, f.b, f.a) for f in meter._filters.values()]
        self.zi = [np.zeros((max(len(a), len(b)) - 1, channels)) for _, b, a in self.stages]
        self.hop = int(round(sr * HOP_SEC))
        self.sr, self.channels = sr, channels
        self._yk = np.zeros((0, channels))
        self._raw = np.zeros((0, channels), dtype=np.float32)
        self.ms: list[float] = []
        self.pk: list[float] = []

    def push(self, x: np.ndarray) -> None:
        if x.ndim == 1:
            x = x[:, None]
        y = x.astype(np.float64)
        for i, (g, b, a) in enumerate(self.stages):
            y, self.zi[i] = scipy.signal.lfilter(b, a, y, axis=0, zi=self.zi[i])
            y = g * y
        yk = np.concatenate([self._yk, y])
        raw = np.concatenate([self._raw, x.astype(np.float32)])
        n = len(yk) // self.hop
        if n:
            cut = n * self.hop
            e = (yk[:cut] ** 2).sum(axis=1).reshape(n, self.hop).mean(axis=1)
            p = np.abs(raw[:cut]).max(axis=1).reshape(n, self.hop).max(axis=1)
            self.ms.extend(e.tolist())
            self.pk.extend(p.tolist())
            yk, raw = yk[cut:], raw[cut:]
        self._yk, self._raw = yk, raw

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(self.ms), len(self.pk))
        return np.asarray(self.ms[:n]), np.asarray(self.pk[:n])


def windowed(ms: np.ndarray, n: int) -> np.ndarray:
    """Mean energy of the n blocks ending at each index (shorter at the very start)."""
    cs = np.concatenate([[0.0], np.cumsum(ms)])
    idx = np.arange(1, len(ms) + 1)
    lo = np.maximum(idx - n, 0)
    return (cs[idx] - cs[lo]) / (idx - lo)


def integrated(ms: np.ndarray, b0: int, b1: int) -> float | None:
    """BS.1770 gated loudness of blocks [b0, b1)."""
    seg = ms[b0:b1]
    if len(seg) < 4:
        return None
    gates = windowed(seg, 4)[3:]
    l = _db(gates)
    g = gates[l > -70]
    if not len(g):
        return None
    rel = _db(g.mean()) - 10
    g2 = g[_db(g) > rel]
    return float(_db(g2.mean())) if len(g2) else None


def bed_level(ms: np.ndarray, b0: int, b1: int) -> float | None:
    """The steady music level: median momentary (400 ms) loudness while something is playing.

    Short windows matter: a gunshot every few seconds lands in every 3 s window, but only in a few 400 ms ones,
    so the median stays on the music."""
    if b1 - b0 < 4:
        return None
    m = _db(windowed(ms[b0:b1], 4)[3:])
    m = m[m > SILENT_LUFS]
    return float(np.median(m)) if len(m) else None


def segment_stats(ms: np.ndarray, pk: np.ndarray, st: np.ndarray, b0: int, b1: int) -> dict:
    lufs = integrated(ms, b0, b1)
    bed = bed_level(ms, b0, b1)
    win = st[min(b0 + 29, b1 - 1):b1] if b1 - b0 >= 30 else windowed(ms[b0:b1], 4)
    win_db = _db(win)
    active = win_db[win_db > SILENT_LUFS]
    blk = _db(ms[b0:b1])
    peak = float(pk[b0:b1].max()) if b1 > b0 else 0.0
    sound_sec = float(np.sum(blk >= SILENT_LUFS)) * HOP_SEC
    out = {}
    if b1 - b0 < 50:
        # A few seconds (e.g. the tail after the last marker) is too short to judge a level's music.
        out["unreliable"] = f"only {(b1 - b0) * HOP_SEC:.1f} s long"
    elif b1 - b0 >= 100 and sound_sec < 0.3 * (b1 - b0) * HOP_SEC:
        # A level with only a few seconds of sound is not a music measurement (music off, game stalled...).
        out["unreliable"] = (f"only {sound_sec:.0f} s of sound in {(b1 - b0) * HOP_SEC:.0f} s "
                             "(music off or not started?)")
    return {
        **out,
        "lufs": None if lufs is None else round(lufs, 1),
        "bed": None if bed is None else round(bed, 1),
        "loud_p95": round(float(np.percentile(active, 95)), 1) if len(active) else None,
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-9)), 1),
        "silent_pct": int(round(100 * float(np.mean(blk < SILENT_LUFS)))) if len(blk) else 100,
    }


def auto_segments(ms: np.ndarray, min_gap_sec: float = 1.0, min_len_sec: float = 8.0) -> tuple[list, int]:
    """Split on silences (loading screens, menus between levels). Returns segments and how many blips were dropped."""
    silent = _db(ms) < SILENT_LUFS
    gap = int(round(min_gap_sec / HOP_SEC))
    segs, start, run = [], None, 0
    for i, s in enumerate(silent):
        if not s:
            if start is None:
                start = i
            run = 0
        else:
            run += 1
            if start is not None and run >= gap:
                segs.append((start, i - run + 1))
                start = None
    if start is not None:
        segs.append((start, len(ms)))
    keep = [s for s in segs if (s[1] - s[0]) * HOP_SEC >= min_len_sec]
    return keep, len(segs) - len(keep)


def mmss(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{t % 60:04.1f}" if t < 600 else f"{int(t // 60)}:{int(t % 60):02d}"


_TIME = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*(?:[-=,]\s*)?(.*)$")


def parse_splits(splits: list[str]) -> list[tuple[float, str]]:
    out = []
    for s in splits:
        m = _TIME.match(s or "")
        if not m:
            raise ValueError(f"Couldn't read split '{s}'. Use 'm:ss label', e.g. '2:31 level 2'.")
        t = int(m.group(1) or 0) * 60 + float(m.group(2))
        out.append((t, m.group(3).strip() or f"from {mmss(t)}"))
    return sorted(out)


def sidecar_path(audio_path: str) -> str:
    return os.path.splitext(audio_path)[0] + ".markers.json"


def read_sidecar(audio_path: str) -> dict | None:
    p = sidecar_path(audio_path)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


_mem: dict = {}
_mem_lock = threading.Lock()


def track_file(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-100 ms energy and peak for a whole file (cached by path, size and mtime)."""
    st = os.stat(path)
    key = (os.path.normcase(path), st.st_size, st.st_mtime_ns)
    with _mem_lock:
        if key in _mem:
            return _mem[key]
    with sf.SoundFile(path) as f:
        lt = LoudnessTracker(f.samplerate, f.channels)
        for block in f.blocks(blocksize=f.samplerate * 10, dtype="float32", always_2d=True):
            lt.push(block)
        dur = f.frames / f.samplerate if f.frames > 0 else len(lt.ms) * HOP_SEC
    ms, pk = lt.arrays()
    res = (ms, pk, float(dur))
    with _mem_lock:
        if len(_mem) > 8:
            _mem.clear()
        _mem[key] = res
    return res


def balance(segs: list[dict], target: float | None) -> tuple[float | None, list[str]]:
    """Fill in change_db / volume_x per segment against the target (or the median level). Returns ref and notes."""
    shaky = [s for s in segs if s.get("unreliable") and s.get("bed") is not None]
    beds = [s["bed"] for s in segs if s.get("bed") is not None and not s.get("unreliable")]
    if not beds:
        if shaky:
            return None, [f"Hardly any sound in any level ({'; '.join(s['label'] + ': ' + s['unreliable'] for s in shaky[:4])}). "
                          "That's not music: check the game's music was on and the game was running at normal "
                          "speed (a hidden or minimised browser tab can slow a web game right down)."]
        return None, ["Everything was silent: check the capture device and that the game's sound was on."]
    ref = float(target) if target is not None else float(np.median(beds))
    notes = []
    for s in shaky:
        notes.append(f"{s['label']}: {s['unreliable']}, so it's left out.")
    for s in segs:
        if s.get("bed") is None or s.get("unreliable"):
            s["change_db"] = None
            continue
        d = round(ref - s["bed"], 1)
        s["change_db"] = d
        s["volume_x"] = round(10 ** (d / 20), 2)
    if len(beds) >= 2:
        ok = [s for s in segs if s.get("change_db") is not None]
        lo = min(ok, key=lambda s: s["bed"])
        hi = max(ok, key=lambda s: s["bed"])
        spread = round(hi["bed"] - lo["bed"], 1)
        notes.append(f"Spread between quietest ({lo['label']}, {lo['bed']} LUFS) and loudest ({hi['label']}, "
                     f"{hi['bed']} LUFS): {spread} dB." + (" That's within 1.5 dB, so they're well matched."
                                                           if spread <= 1.5 else ""))
    for s in segs:
        d = s.get("change_db")
        if d is not None and abs(d) >= 1.5:
            notes.append(f"{s['label']}: {'turn down' if d < 0 else 'turn up'} by {abs(d)} dB "
                         f"(volume x{s['volume_x']}).")
    return ref, notes


def capture_report(path: str, splits: list[str] | None = None, labels: list[str] | None = None,
                   target_lufs: float | None = None) -> tuple[dict, dict]:
    """Per-level loudness for one long recording. Returns (report, plot data)."""
    ms, pk, dur = track_file(path)
    st = windowed(ms, 30)
    side = read_sidecar(path) or {}
    n = len(ms)
    how = ""
    marks: list[tuple[float, str]] = []
    if splits:
        marks, how = parse_splits(splits), "your split times"
    elif side.get("markers"):
        marks, how = [(float(m["t"]), str(m["label"])) for m in side["markers"]], "markers set during the capture"
    dropped = 0
    if marks:
        if marks[0][0] > 3.0:
            marks = [(0.0, "before first marker")] + marks
        else:
            marks[0] = (0.0, marks[0][1])
        bounds = [(int(round(t / HOP_SEC)), lab) for t, lab in marks if t < dur]
        segs_b = [(b, bounds[i + 1][0] if i + 1 < len(bounds) else n, lab) for i, (b, lab) in enumerate(bounds)]
    else:
        auto, dropped = auto_segments(ms)
        segs_b = [(b0, b1, f"part {i + 1}") for i, (b0, b1) in enumerate(auto)]
        how = "silences of 1 s or more (no markers were set)"
    if labels:
        segs_b = [(b0, b1, labels[i] if i < len(labels) and labels[i] else lab) for i, (b0, b1, lab) in enumerate(segs_b)]
    segs = []
    for b0, b1, lab in segs_b:
        if b1 - b0 < 10:
            continue
        s = {"label": lab, "from": mmss(b0 * HOP_SEC), "to": mmss(b1 * HOP_SEC)}
        s.update(segment_stats(ms, pk, st, b0, b1))
        s["_b"] = (b0, b1)
        segs.append(s)
    ref, notes = balance(segs, target_lufs)
    whole = integrated(ms, 0, n)
    rep = {
        "file": os.path.basename(path),
        "duration": mmss(dur),
        "whole_lufs": None if whole is None else round(whole, 1),
        "split_by": how,
        "reference_lufs": None if ref is None else round(ref, 1),
        "reference_is": "your target" if target_lufs is not None else "the median level across the segments",
        "levels": [{k: v for k, v in s.items() if not k.startswith("_")} for s in segs],
        "verdict": notes,
    }
    if dropped:
        rep["ignored"] = f"{dropped} short sound blips (under 8 s, e.g. menu clicks) between silences"
    rep["how_to_read"] = ("bed = median momentary (400 ms) loudness while sound is playing: the steady music level, "
                          "robust to short loud effects (not to constant ones; for the cleanest reading turn the "
                          "game's sound effects off while capturing). lufs = standard integrated loudness (loud "
                          "effects pull it up). loud_p95 = the loudest 3-second stretches. "
                          "change_db / volume_x = how much to change that level's music volume to match the reference "
                          "(volume_x is the linear multiplier, e.g. an AudioSource volume).")
    warn = []
    if side.get("other_apps"):
        warn.append("Other apps were also making sound during the capture: " + ", ".join(side["other_apps"])
                    + ". Their sound is mixed in.")
    if side.get("volume_note"):
        warn.append(side["volume_note"])
    if side.get("clipped_blocks"):
        warn.append(f"The game's output went over full scale in {side['clipped_blocks']} short blocks.")
    if warn:
        rep["warnings"] = warn
    plot = {"t": np.arange(n) * HOP_SEC, "st": _db(st), "segs": segs, "ref": ref, "title": os.path.basename(path)}
    return rep, plot


def folder_report(files: list[str], target_lufs: float | None = None) -> tuple[dict, dict]:
    """Treat each file as one level (e.g. the game's music files)."""
    segs = []
    for p in files:
        ms, pk, dur = track_file(p)
        st = windowed(ms, 30)
        s = {"label": os.path.basename(p), "length": mmss(dur)}
        s.update(segment_stats(ms, pk, st, 0, len(ms)))
        segs.append(s)
    ref, notes = balance(segs, target_lufs)
    rep = {
        "files": len(segs),
        "reference_lufs": None if ref is None else round(ref, 1),
        "reference_is": "your target" if target_lufs is not None else "the median level across the files",
        "levels": segs,
        "verdict": notes,
        "how_to_read": ("Each file is measured on its own, before any in-game volume. bed = median momentary loudness "
                        "while playing; change_db / volume_x = gain to match the reference."),
    }
    return rep, {"segs": segs, "ref": ref, "title": "per-file levels"}
