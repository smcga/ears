"""Note transcription with Spotify's Basic Pitch (ONNX model), mapped onto the FL bar grid.

Gives: notes, a chord per bar (inferred from note content + bass), a compressed chord chart,
FL-style note listings for a bar range, and tempo-mapped MIDI files.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
from dataclasses import dataclass

import numpy as np

from .analysis import NOTE_NAMES, Grid, camelot, key_candidates, key_name

_model = None
_lock = threading.Lock()

# Chord qualities as intervals above the root. Order matters only for ties (simpler first).
QUALITIES = (("", (0, 4, 7)), ("m", (0, 3, 7)), ("5", (0, 7)), ("sus2", (0, 2, 7)), ("sus4", (0, 5, 7)),
             ("7", (0, 4, 7, 10)), ("maj7", (0, 4, 7, 11)), ("m7", (0, 3, 7, 10)), ("dim", (0, 3, 6)),
             ("aug", (0, 4, 8)))
MIN_BAR_WEIGHT = 0.15  # seconds x velocity; below this a bar counts as "no notes"

LEGEND = ("Positions are beat.step (step = 1/16 note, 1.1 = downbeat); (n) = length in steps; "
          "octaves as FL Studio shows them (middle C = C5). Notes sounding together are joined with '+'.")
LIMITS = ("Basic Pitch is an automatic transcriber: heavily processed synths, dense pads, reverb tails and "
          "stem bleed cause missing notes, octave slips and ghost notes. Chords are inferred from the notes "
          "in each bar (a '?' means a weak match; '(A)' means only that note was found). Trust repeated "
          "patterns more than any single bar, and check by ear before copying anything.")


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    vel: float


def fl_name(midi: int) -> str:
    """Note name as FL Studio's piano roll shows it (MIDI 60 = C5)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12}"


# ----------------------------------------------------------------------------- Basic Pitch

def _load():
    global _model
    if _model is None:
        root = logging.getLogger()
        level = root.level
        root.setLevel(logging.ERROR)  # basic_pitch warns about TF/TFLite/CoreML being absent; we use ONNX
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from basic_pitch import ICASSP_2022_MODEL_PATH
                from basic_pitch.inference import Model

                _model = Model(ICASSP_2022_MODEL_PATH)
        finally:
            root.setLevel(level)
    return _model


def available() -> tuple[bool, str]:
    try:
        _load()
        return True, "basic-pitch (onnx)"
    except Exception as exc:  # pragma: no cover - depends on install
        return False, f"Basic Pitch isn't installed or can't load ({exc})."


def transcribe(path: str, bpm: float, fmin: float | None = None, fmax: float | None = None) -> list[Note]:
    from basic_pitch.inference import predict

    with _lock:
        model = _load()
        with contextlib.redirect_stdout(sys.stderr):  # basic_pitch prints progress; stdout is the MCP channel
            _, _, events = predict(path, model, midi_tempo=bpm, minimum_frequency=fmin, maximum_frequency=fmax)
    notes = [Note(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]
    return sorted(notes, key=lambda n: (n.start, n.pitch))


# ----------------------------------------------------------------------------- note maths

def _arrays(notes: list[Note]):
    if not notes:
        z = np.zeros(0)
        return z, z, np.zeros(0, int), np.zeros(0, int), z
    s = np.array([n.start for n in notes])
    e = np.array([n.end for n in notes])
    p = np.array([n.pitch for n in notes])
    v = np.array([n.vel for n in notes])
    return s, e, p, p % 12, v


def pc_weights(arr, t0: float, t1: float) -> np.ndarray:
    s, e, _, pc, v = arr
    if len(s) == 0:
        return np.zeros(12)
    ov = np.clip(np.minimum(e, t1) - np.maximum(s, t0), 0, None)
    return np.bincount(pc, weights=ov * v, minlength=12)


@dataclass
class Chord:
    name: str
    conf: float  # share of note weight explained by the chord tones


def best_chord(w: np.ndarray, bass: np.ndarray | None = None) -> Chord | None:
    tot = float(w.sum())
    if tot < MIN_BAR_WEIGHT:
        return None
    wn = w / tot
    strong = np.where(wn >= 0.1)[0]
    if len(strong) < 2:
        return Chord(f"({NOTE_NAMES[int(np.argmax(wn))]})", float(wn.max()))
    bn = bass / bass.sum() if bass is not None and bass.sum() > MIN_BAR_WEIGHT / 3 else None
    best, best_score = None, -9.0
    for root in range(12):
        for qname, iv in QUALITIES:
            tones = [(root + i) % 12 for i in iv]
            cov = float(wn[tones].sum())
            mean_tone = cov / len(tones)
            missing = sum(wn[t] < 0.35 * mean_tone for t in tones)
            score = cov - 0.15 * missing - 0.04 * max(0, len(tones) - 3)
            if bn is not None:
                score += 0.25 * float(bn[root])
            if score > best_score + 1e-9:
                best, best_score = (root, qname, tones, cov), score
    root, qname, tones, cov = best
    name = NOTE_NAMES[root] + qname
    if bn is not None:
        bpc = int(np.argmax(bn))
        if bn[bpc] > 0.5 and bpc != root and bpc in tones:
            name += "/" + NOTE_NAMES[bpc]
    if cov < 0.7:
        name += "?"
    return Chord(name, cov)


def bar_chords(notes: list[Note], bass: list[Note] | None, grid: Grid) -> list[str]:
    arr, barr = _arrays(notes), (_arrays(bass) if bass else None)
    out = []
    for bar in range(1, grid.n_bars + 1):
        t0, t1 = grid.t_of_bar(bar), grid.t_of_bar(bar + 1)
        tm = (t0 + t1) / 2
        bw = (lambda a, b: pc_weights(barr, a, b)) if barr else (lambda a, b: None)
        whole = best_chord(pc_weights(arr, t0, t1), bw(t0, t1))
        h1 = best_chord(pc_weights(arr, t0, tm), bw(t0, tm))
        h2 = best_chord(pc_weights(arr, tm, t1), bw(tm, t1))
        if (h1 and h2 and h1.name != h2.name and min(h1.conf, h2.conf) >= 0.75
                and (whole is None or whole.conf < 0.8)):
            out.append(f"{h1.name} {h2.name}")
        else:
            out.append(whole.name if whole else "-")
    return out


def bar_roots(bass: list[Note], grid: Grid) -> list[str]:
    arr = _arrays(bass)
    out = []
    for bar in range(1, grid.n_bars + 1):
        t0, t1 = grid.t_of_bar(bar), grid.t_of_bar(bar + 1)
        tm = (t0 + t1) / 2
        parts = []
        for a, b in ((t0, tm), (tm, t1)):
            w = pc_weights(arr, a, b)
            parts.append(NOTE_NAMES[int(np.argmax(w))] if w.sum() >= MIN_BAR_WEIGHT / 2 else "-")
        out.append(parts[0] if parts[0] == parts[1] else " ".join(parts))
    return out


def compress(seq: list[str], first_bar: int = 1, max_lines: int = 40) -> list[str]:
    """Turn per-bar labels into a chart: 'bars 31-46: | Am | F | C | G | x4'."""
    lines, i, n, loose = [], 0, len(seq), []

    def flush():
        if loose:
            a, b = loose[0][0], loose[-1][0]
            cells = [c for _, c in loose]
            label = "no notes" if all(c == "-" for c in cells) else "| " + " | ".join(cells) + " |"
            lines.append(f"bars {a + first_bar}-{b + first_bar}: {label}" if b > a else f"bar {a + first_bar}: {label}")
            loose.clear()

    while i < n:
        L_best, r_best = 1, 1
        for L in (1, 2, 3, 4, 6, 8):
            r = 1
            while i + (r + 1) * L <= n and seq[i + r * L: i + (r + 1) * L] == seq[i: i + L]:
                r += 1
            if r >= 2 and r * L > L_best * r_best:
                L_best, r_best = L, r
        if r_best == 1:
            loose.append((i, seq[i]))
            if len(loose) == 8:
                flush()
            i += 1
            continue
        flush()
        pat, a, b = seq[i: i + L_best], i + first_bar, i + L_best * r_best - 1 + first_bar
        if all(c == "-" for c in pat):
            lines.append(f"bars {a}-{b}: no notes")
        elif L_best == 1:
            lines.append(f"bars {a}-{b}: {pat[0]} (held {r_best} bars)")
        else:
            lines.append(f"bars {a}-{b}: | " + " | ".join(pat) + f" | x{r_best}")
        i += L_best * r_best
    flush()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines; ask for a bar range)"]
    return lines


def bar_activity(path: str, grid: Grid, floor_db: float = 35.0) -> np.ndarray:
    """True for bars where this audio is within `floor_db` of its loudest bar (i.e. actually playing)."""
    import soundfile as sf

    from .analysis import to_db

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    edges = grid.sample_edges(len(x), sr)
    pw = np.mean(x * x, axis=1)
    db = to_db(np.add.reduceat(pw, edges[:-1], dtype=np.float64) / np.maximum(np.diff(edges), 1))
    return db > max(float(db.max()) - floor_db, -75.0)


def gate(notes: list[Note], active: np.ndarray, grid: Grid) -> list[Note]:
    """Drop notes that start in bars where the source is near-silent (stem bleed, reverb tails)."""
    return [n for n in notes if active[min(grid.bar_at(n.start), grid.n_bars) - 1]]


MAJOR_STEPS, MINOR_STEPS = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)


def key_relation(from_notes: dict | None, from_audio: dict) -> str | None:
    """Explain a disagreement between the note-based key and the full-mix key, if any."""
    if not from_notes or from_audio.get("_tonic") is None:
        return None
    a = (from_notes["_tonic"], from_notes["_mode"])
    b = (from_audio["_tonic"], from_audio["_mode"])
    if a == b:
        return "Notes and full mix agree on the key."

    def scale(t, m):
        return {(t + i) % 12 for i in (MAJOR_STEPS if m == "major" else MINOR_STEPS)}

    shared = len(scale(*a) & scale(*b))
    if shared >= 6:
        return (f"The notes centre on {NOTE_NAMES[a[0]]} while the full mix reads {key_name(*b)}: the two keys share "
                f"{shared}/7 scale notes, so it's the same palette with a different home note (think C major vs "
                "D dorian). Which note feels like home decides it; trust your ears.")
    return "The transcribed part and the full mix point to different keys; this part may cover only some of the harmony."


def key_from_notes(notes: list[Note]) -> dict | None:
    arr = _arrays(notes)
    if len(arr[0]) == 0:
        return None
    w = np.bincount(arr[3], weights=(arr[1] - arr[0]) * arr[4], minlength=12)
    c = key_candidates(w)
    if not c:
        return None
    share = w / w.sum()
    top = np.argsort(share)[::-1][:7]
    return {"best": f"{key_name(c[0][1], c[0][2])} ({camelot(c[0][1], c[0][2])})", "r": round(c[0][0], 2),
            "runner_up": f"{key_name(c[1][1], c[1][2])}", "r2": round(c[1][0], 2),
            "pitch_classes": ", ".join(f"{NOTE_NAMES[i]} {share[i] * 100:.0f}%" for i in top if share[i] >= 0.02),
            "_tonic": c[0][1], "_mode": c[0][2]}


def detail_rows(notes: list[Note], grid: Grid, b0: int, b1: int, max_notes: int = 240) -> tuple[list[str], bool]:
    step = grid.beat_sec / 4
    rows, used, truncated = [], 0, False
    for bar in range(b0, b1 + 1):
        t0 = grid.t_of_bar(bar)
        groups: dict[int, list[tuple[int, int]]] = {}
        for n in notes:
            if t0 - step / 2 <= n.start < t0 + grid.bar_sec - step / 2:
                s = int(round((n.start - t0) / step))
                groups.setdefault(s, []).append((n.pitch, max(1, int(round((n.end - n.start) / step)))))
        parts = []
        for s in sorted(groups):
            g = sorted(set(groups[s]))
            lens = sorted({L for _, L in g})
            ln = str(lens[0]) if len(lens) == 1 else "/".join(str(L) for _, L in g)
            parts.append(f"{s // 4 + 1}.{s % 4 + 1} {'+'.join(fl_name(p) for p, _ in g)} ({ln})")
            used += len(g)
        rows.append(f"{bar}: " + (", ".join(parts) if parts else "-"))
        if used >= max_notes and bar < b1:
            truncated = True
            break
    return rows, truncated


def note_range(notes: list[Note]) -> str:
    if not notes:
        return "none"
    p = np.array([n.pitch for n in notes])
    lo, hi = int(np.percentile(p, 2)), int(np.percentile(p, 98))
    if (lo, hi) == (int(p.min()), int(p.max())):
        return f"{fl_name(lo)}-{fl_name(hi)}"
    return f"{fl_name(lo)}-{fl_name(hi)} (most notes), extremes {fl_name(int(p.min()))}-{fl_name(int(p.max()))}"


def write_midi(path: str, tracks: list[tuple[str, list[Note], int]], grid: Grid, quantize: bool = True) -> int:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=grid.bpm)
    pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))
    step = grid.beat_sec / 4
    count = 0
    for name, notes, program in tracks:
        inst = pretty_midi.Instrument(program=program, name=name)
        seen: dict[tuple[int, float], int] = {}
        for n in notes:
            s, e = n.start, n.end
            if quantize:
                s = round(s / step) * step
                e = max(s + step, round(e / step) * step)
            key = (n.pitch, round(s, 4))
            vel = int(np.clip(round(n.vel * 127), 1, 127))
            if key in seen:  # quantising can stack duplicates; keep the longest
                old = inst.notes[seen[key]]
                old.end = max(old.end, e)
                old.velocity = max(old.velocity, vel)
                continue
            seen[key] = len(inst.notes)
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=n.pitch, start=s, end=e))
        count += len(inst.notes)
        pm.instruments.append(inst)
    pm.write(path)
    return count
