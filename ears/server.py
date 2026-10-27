"""ears: an MCP server that lets Claude 'listen' to audio bounces by measuring them."""
from __future__ import annotations

import functools
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict
from typing import Annotated, Literal

import numpy as np

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .paths import allowed_roots, find_audio as _find_audio, output_dir, resolve_audio_path

INSTRUCTIONS = (
    "ears measures audio files (FL Studio bounces) under the user's music folder. "
    "Use `listen` for one file, `compare` for two versions, `stems` to split into stems (slow), "
    "`notes` to transcribe notes/chords to MIDI (Basic Pitch), `describe` for genre/mood/texture tags (CLAP), "
    "`expect` for a bar-by-bar surprise map (listening as prediction), `listen_live` to step through a track without "
    "seeing ahead (write your own prediction before each call), `sounds_like` to compare against the user's other tracks, "
    "`find_audio` to locate files by name. "
    "`capture` records what the computer is playing (e.g. a game while the user plays: start, mark each new level, "
    "status for live levels, snapshot to grab the last seconds, stop for a per-level loudness report); `levels` "
    "compares loudness per level for a recording or per file for a folder of level music. "
    "Bar numbers follow FL Studio's playlist (bar 1 = start of file); "
    "pass `bpm` when the user gives the project tempo. Results are measurements, not hearing: "
    "say so, call out low-confidence key/tempo, and look at the spectrogram image before describing the track."
)

mcp = MCPServer("ears", instructions=INSTRUCTIONS, log_level="WARNING")

_cache: "OrderedDict[tuple, object]" = OrderedDict()
_cache_lock = threading.Lock()


def _analysis(real: str, bpm: float | None):
    from .analysis import analyze

    st = os.stat(real)
    key = (os.path.normcase(real), st.st_mtime_ns, st.st_size, float(bpm) if bpm else None)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    a = analyze(real, bpm)
    with _cache_lock:
        _cache[key] = a
        while len(_cache) > 16:
            _cache.popitem(last=False)
    return a


def friendly(fn):
    """Turn exceptions into ToolErrors so Claude sees the actual message (mcp v2 hides other errors)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (PermissionError, FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            raise ToolError(f"ears hit an internal error: {type(exc).__name__}: {exc}") from exc

    return wrapper


def _dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


PathArg = Annotated[str, Field(description=r"Full path to a .wav/.mp3/.flac file, e.g. C:\Users\me\Music\My Project\my_track.mp3")]
BpmArg = Annotated[float | None, Field(description="Project tempo in BPM if known (e.g. 130). Omit to auto-detect.", ge=40, le=300)]


@mcp.tool(annotations=ToolAnnotations(title="Listen to a track", readOnlyHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def listen(
    path: PathArg,
    bpm: BpmArg = None,
    start_sec: Annotated[float | None, Field(description="Optional: start of a range to zoom into (seconds).", ge=0)] = None,
    end_sec: Annotated[float | None, Field(description="Optional: end of the zoom range (seconds).", gt=0)] = None,
):
    """Analyse an audio bounce and return a compact listening report plus a spectrogram image.

    Report: duration/format, tempo + beat confidence, key estimate (with runner-up and confidence),
    loudness (LUFS, peak, true peak, clipping), stereo width/correlation, a 4-bar table on FL's bar grid
    (RMS and sub/low/mid/high band levels), detected sections with plain-English labels, mix flags,
    and what the measurements can't tell you. Pass bpm for exact bar numbers. Give start_sec/end_sec
    (one is enough; defaults to a 30 s window) for an extra zoomed spectrogram of that range.
    """
    from . import render
    from .analysis import analyze_range
    from .report import listen_report

    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    rng = None
    if start_sec is not None or end_sec is not None:
        s = float(start_sec) if start_sec is not None else max(0.0, float(end_sec) - 30.0)
        e = float(end_sec) if end_sec is not None else s + 30.0
        e = min(e, a.duration)
        if not (0 <= s < e):
            raise ValueError(f"Bad range: start {s} s, end {e} s (track is {a.duration:.1f} s long).")
        rng = analyze_range(real, s, e, a.grid)

    out = output_dir()
    img, fmt, img_path = render.overview(a, out / f"{_stem(real)}_overview")
    report = listen_report(a, img_path, rng)
    content: list = []
    zoom = None
    if rng:
        zoom = render.zoom(a, rng, out / f"{_stem(real)}_zoom_{int(rng['_spec']['t0'])}-{int(rng['_spec']['t1'])}s")
        report["zoom_image_saved"] = zoom[2]
    report["took_sec"] = round(time.time() - t0, 1)
    content.append(_dump(report))
    content.append(Image(data=img, format=fmt))
    if zoom:
        content.append(Image(data=zoom[0], format=zoom[1]))
    return content


@mcp.tool(annotations=ToolAnnotations(title="Compare two versions", readOnlyHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def compare(
    path_a: Annotated[str, Field(description="Full path to version A (the older one), e.g. ...\\my_track.mp3")],
    path_b: Annotated[str, Field(description="Full path to version B (the newer one), e.g. ...\\my_track_v2.mp3")],
    bpm: BpmArg = None,
):
    """Compare two versions of a track (e.g. my_track.mp3 vs my_track_v2.mp3) on the same FL bar grid.

    Returns a plain-English summary of what changed (length, loudness, peaks/clipping, key, tempo,
    band balance per section, stereo, structure), then the numbers and a 4-bar delta table (B minus A),
    plus both spectrograms stacked in one image.
    """
    from . import render
    from .report import compare_report

    t0 = time.time()
    ra, rb = resolve_audio_path(path_a), resolve_audio_path(path_b)
    a = _analysis(ra, bpm)
    b = _analysis(rb, bpm)
    if not bpm and a.grid.bpm != b.grid.bpm and abs(a.grid.bpm - b.grid.bpm) <= 0.5:
        b = _analysis(rb, a.grid.bpm)  # detection jitter: put both on A's grid
    out = output_dir()
    img, fmt, img_path = render.compare_pair(a, b, out / f"compare_{_stem(ra)}_vs_{_stem(rb)}")
    rep = compare_report(a, b, img_path)
    rep["took_sec"] = round(time.time() - t0, 1)
    return [_dump(rep), Image(data=img, format=fmt)]


@mcp.tool(annotations=ToolAnnotations(title="Split into stems (slow)", readOnlyHint=False, destructiveHint=False,
                                      idempotentHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def stems(
    path: PathArg,
    bpm: BpmArg = None,
    force: Annotated[bool, Field(description="Re-run separation even if stems already exist.")] = False,
):
    """SLOW: split a track into drums / bass / other / vocals with Demucs (htdemucs) and report on each stem.

    Saves 32-bit float WAVs in a '<name>_stems' folder next to the source (re-used next time if the
    source hasn't changed). Takes roughly 15-60 s on the GPU (several minutes on CPU), so only use it
    when the user asks for stems or per-instrument detail. Per stem: active bars, level, band balance,
    main pitch classes (bass) / key guess (other, vocals), plus a bars x stems activity image.
    """
    from . import render
    from .stems import STEM_ORDER, available, separate, stem_report

    ok, info = available()
    if not ok:
        raise RuntimeError(f"Stem separation isn't available: {info}")
    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    paths, sep = separate(real, force=force)
    reports, bars = {}, []
    for name in STEM_ORDER:
        rep, bar_db = stem_report(name, paths[name], a.grid)
        reports[name] = rep
        bars.append(bar_db)
    img, fmt, img_path = render.stem_activity(list(STEM_ORDER), bars, a.grid,
                                              f"{a.name}: stem activity per bar ({a.grid.bpm:g} BPM grid)",
                                              output_dir() / f"{_stem(real)}_stems_activity")
    out = {
        "file": a.name,
        "stems_folder": os.path.dirname(paths["drums"]),
        **sep,
        "grid": f"{a.grid.bpm:g} BPM, bar 1 = 0:00",
        "stems": reports,
        "note": ("Demucs is trained on band music: on electronic tracks 'vocals' often catches leads/synths and "
                 "'other' holds pads/chords/FX. Expect some bleed between stems."),
        "image_saved": img_path,
        "took_sec": round(time.time() - t0, 1),
    }
    return [_dump(out), Image(data=img, format=fmt)]


STEM_DESCRIPTIONS = {
    "other": "Demucs 'other' stem (pads, chords, plucks, leads)",
    "vocals": "Demucs 'vocals' stem (on electronic tracks often leads or vocal chops)",
    "bass": "Demucs 'bass' stem",
    "mix": "full mix, no separation (drums can add junk notes)",
}


def _detail_range(bars: str | None, a, notes) -> tuple[int, int]:
    g, n = a.grid, a.grid.n_bars
    if bars:
        m = re.match(r"^\s*(\d+)\s*(?:(?:-|–|to)\s*(\d+))?\s*$", bars)
        if not m:
            raise ValueError("`bars` should look like '33-40' (or just '33' for 8 bars from there).")
        b0 = int(m.group(1))
        b1 = int(m.group(2)) if m.group(2) else b0 + 7
    else:
        loud = next((s for s in a.sections if "(loudest)" in s["label"]), None)
        b0 = loud["start"] if loud else 1
        if notes and not any(g.t_of_bar(b0) <= x.start < g.t_of_bar(b0 + 8) for x in notes):
            b0 = g.bar_at(notes[0].start)
        b1 = b0 + 7
    b0 = max(1, min(b0, n))
    return b0, max(b0, min(b1, n, b0 + 15))


@mcp.tool(annotations=ToolAnnotations(title="Transcribe notes & chords to MIDI", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def notes(
    path: PathArg,
    bpm: BpmArg = None,
    stem: Annotated[Literal["other", "bass", "vocals", "mix"], Field(
        description="What to transcribe: 'other' = Demucs melodic/chord stem (default), 'bass', 'vocals' "
                    "(often leads on electronic tracks), or 'mix' = whole track without separation.")] = "other",
    bars: Annotated[str | None, Field(
        description="Bar range for the detailed note list and close-up, e.g. '33-40' (max 16 bars). "
                    "Default: 8 bars from the start of the loudest section.")] = None,
    quantize: Annotated[bool, Field(description="Snap the saved MIDI to 1/16 steps.")] = True,
):
    """Transcribe notes and chords with Basic Pitch and save MIDI that lines up with FL's bars.

    Uses Demucs stems (made first if missing: slow the first time, reused after). Transcribes the chosen stem
    plus the bass stem (to anchor chord roots) and returns: a chord chart on FL's bar grid with repeats
    compressed, bassline roots per bar, a key estimate from the notes, a beat.step note list for a bar range
    (FL octave names), the saved .mid paths, and a piano-roll image (whole song + close-up).
    """
    from . import render
    from . import transcribe as tr
    from .stems import available as stems_available, separate, stem_paths

    ok, info = tr.available()
    if not ok:
        raise RuntimeError(info)
    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    g = a.grid
    used, src, bass_src, sep, fallback = stem, real, None, None, None
    if stem != "mix":
        s_ok, s_info = stems_available()
        if s_ok:
            paths, sep = separate(real)
            src = paths[stem]
            bass_src = paths["bass"] if stem != "bass" else None
        else:
            used, fallback = "mix", f"Stem separation unavailable ({s_info}); transcribed the full mix instead."

    raw = tr.transcribe(src, g.bpm, fmax=400.0 if used == "bass" else None)
    main = tr.gate(raw, tr.bar_activity(src, g), g)
    dropped = len(raw) - len(main)
    if bass_src:
        braw = tr.transcribe(bass_src, g.bpm, fmax=400.0)
        bass = tr.gate(braw, tr.bar_activity(bass_src, g), g)
        dropped += len(braw) - len(bass)
    elif used == "mix":
        bass = [n for n in main if n.pitch < 52]  # below E3 (FL E4): treat as bassline
    else:
        bass = None

    chords = tr.bar_chords(main, bass, g) if used != "bass" else None
    roots = tr.bar_roots(main if used == "bass" else bass, g) if (used == "bass" or bass) else None
    b0, b1 = _detail_range(bars, a, main)
    rows, truncated = tr.detail_rows(main, g, b0, b1)

    out_dir, _ = stem_paths(real)
    os.makedirs(out_dir, exist_ok=True)
    base = _stem(real)
    midi = []
    p = os.path.join(out_dir, f"{base}_{used}.mid")
    n_written = tr.write_midi(p, [(used, main, 33 if used == "bass" else 0)], g, quantize)
    midi.append(p)
    if bass_src and bass:
        p = os.path.join(out_dir, f"{base}_bass.mid")
        tr.write_midi(p, [("bass", bass, 33)], g, quantize)
        midi.append(p)

    kf = tr.key_from_notes(main if used != "bass" else main + (bass or []))
    tracks = [(used, main, "#2f62d8")]
    if bass and used not in ("bass", "mix"):
        tracks.append(("bass", bass, "#f26b1d"))
    img, fmt, img_path = render.piano_roll(
        a, tracks, chords or roots or [], b0, b1,
        f"{a.name}: {used} notes (Basic Pitch), {g.bpm:g} BPM grid"
        + (f"  |  key from notes: {kf['best']}" if kf else ""),
        output_dir() / f"{base}_{used}_notes")

    detail = {"bars": f"{b0}-{b1}", "legend": tr.LEGEND, "rows": rows}
    if truncated:
        detail["note"] = "Stopped early: lots of notes. Ask for a shorter bar range."
    rep = {
        "file": a.name,
        "transcribed": STEM_DESCRIPTIONS[used],
        "grid": f"{g.bpm:g} BPM, bar 1 = 0:00, {g.n_bars} bars",
        "notes_found": len(main),
        "range": tr.note_range(main),
        "key_from_notes": {k: v for k, v in kf.items() if not k.startswith("_")} if kf else None,
        "key_from_audio": a.key.get("best"),
        "key_note": tr.key_relation(kf, a.key),
        "chords_per_bar": tr.compress(chords) if chords else None,
        "bass_roots_per_bar": tr.compress(roots) if roots else None,
        "detail": detail,
        "midi_saved": midi,
        "midi_note": (f"Tempo {g.bpm:g} BPM, 4/4{', quantised to 1/16' if quantize else ''}: drop it into FL "
                      f"at bar 1 and it lines up with the bounce ({n_written} notes in the main file)."),
        "limits": tr.LIMITS,
        "image_saved": img_path,
    }
    if dropped:
        rep["ignored_notes"] = f"{dropped} notes in bars where the stem is near-silent (bleed/reverb) were left out."
    if fallback:
        rep["fallback"] = fallback
    if sep and not sep.get("reused_existing"):
        rep["stem_separation"] = sep
    rep["took_sec"] = round(time.time() - t0, 1)
    return [_dump({k: v for k, v in rep.items() if v is not None}), Image(data=img, format=fmt)]


@mcp.tool(annotations=ToolAnnotations(title="Describe the vibe (CLAP tags)", readOnlyHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def describe(
    path: PathArg,
    bpm: BpmArg = None,
    phrases: Annotated[list[str] | None, Field(
        description="Optional: your own descriptions to score, e.g. ['garage-ish', 'swung', 'gritty bass']. Max 12.")] = None,
):
    """Describe a track in words with CLAP audio-text embeddings: genre/style, groove, texture, elements and mood.

    Scores a fixed tag vocabulary for the whole track and per detected section (on FL's bar grid), and optionally
    scores your own phrases ("is this garage-ish?"). Scores are relative similarities: compare within a category.
    First call loads two models (~10 s); later calls take a few seconds plus the normal analysis.
    """
    from .describe import available, describe as _describe

    ok, info = available()
    if not ok:
        raise RuntimeError(f"Tagging isn't available: {info}")
    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    rep = {"file": a.name, "grid": f"{a.grid.bpm:g} BPM, bar 1 = 0:00", **_describe(a, phrases)}
    rep["took_sec"] = round(time.time() - t0, 1)
    return _dump(rep)


def _returns(a, E) -> list[str]:
    from .embed import section_embeddings

    secs = section_embeddings(a, E)
    out = []
    for k in range(1, len(secs)):
        s0, e0, v = secs[k]
        sims = [(float(v @ secs[j][2]), j) for j in range(k)]
        sim, j = max(sims)
        what = ("a return of" if sim >= 0.9 else "related to" if sim >= 0.8
                else "new material (nearest earlier part:")
        tail = ")" if sim < 0.8 else ""
        out.append(f"bars {s0}-{e0}: {what} bars {secs[j][0]}-{secs[j][1]} ({sim:.2f}){tail}")
    return out


@mcp.tool(annotations=ToolAnnotations(title="Listen as prediction (surprise map)", readOnlyHint=True,
                                      openWorldHint=False), structured_output=False)
@friendly
def expect(path: PathArg, bpm: BpmArg = None):
    """Listen the way prediction works: guess every bar from the bars before it, then measure the surprise.

    Returns a surprise value per bar (1 = business as usual, 2.5+ = broke expectation), the biggest moments
    explained (what was expected, what happened, whether it landed on the 4/8/16-bar phrase grid, how many bars
    it took to settle), stretches that played out as expected, which parts return, and an image: the learned
    self-similarity map (CLAP, which bars sound alike) with the surprise curve under it.
    """
    from . import expect as ex
    from . import render
    from .embed import bar_embeddings

    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    E = bar_embeddings(a)
    M = ex.measured(a)
    steps = ex.run(E, M)
    rep = {"file": a.name, "grid": f"{a.grid.bpm:g} BPM, bar 1 = 0:00, {a.grid.n_bars} bars",
           **ex.summary(a, steps, M), "returns": _returns(a, E),
           "limits": ("The predictor is a simple pattern-matcher (repeat / phrase / context), not a trained "
                      "listener: it captures structural expectation, not harmonic tension or groove feel.")}
    if a.tempo["grid_fit"] < 0.6:
        rep["grid_warning"] = (f"Beats don't sit well on the {a.grid.bpm:g} BPM grid (fit {a.tempo['grid_fit']:.2f}): "
                               "bar numbers and on/off-grid remarks are unreliable. Pass the real bpm if known.")
    img, fmt, img_path = render.structure_map(a, E, steps, output_dir() / f"{_stem(real)}_expect")
    rep["image_saved"] = img_path
    rep["took_sec"] = round(time.time() - t0, 1)
    return [_dump(rep), Image(data=img, format=fmt)]


_live: dict[str, dict] = {}


@mcp.tool(annotations=ToolAnnotations(title="Listen in time (step through)", readOnlyHint=True,
                                      openWorldHint=False), structured_output=False)
@friendly
def listen_live(
    path: PathArg,
    bpm: BpmArg = None,
    bars: Annotated[int, Field(description="How many bars to reveal per call.", ge=1, le=32)] = 8,
    my_guess: Annotated[str | None, Field(
        description="Your prediction for the bars about to be revealed (free text). It's echoed back next to "
                    "what actually happened, so you can see where your expectation was right or wrong.")] = None,
    from_bar: Annotated[int | None, Field(description="Start (or jump) to this bar.", ge=1)] = None,
    restart: Annotated[bool, Field(description="Start again from bar 1.")] = False,
):
    """Hear a track in time: each call reveals only the next few bars, never anything after them.

    Per call: the newly heard bars (level, what came in/out, which earlier bar each one sounds like, surprise vs
    a guess made from the past only), your previous guess next to the predictor's own forecast and what actually
    happened, the predictor's forecast for the next chunk, and a spectrogram of everything heard so far with the
    future greyed out. Before calling again, write down what you expect next and pass it as my_guess.
    """
    from . import expect as ex
    from . import render
    from .analysis import mmss as _mmss
    from .embed import bar_embeddings

    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    g, n = a.grid, a.grid.n_bars
    key = os.path.normcase(real)
    st = _live.get(key)
    if restart or st is None or from_bar:
        st = _live[key] = {"pos": from_bar or 1, "forecast": None}
    if st["pos"] > n:
        _live.pop(key, None)
        return _dump({"file": a.name, "done": f"Track finished at bar {n}. Call again (or restart=True) to start over."})
    E = bar_embeddings(a)
    M = ex.measured(a)
    b0, b1 = st["pos"], min(n, st["pos"] + bars - 1)
    steps = ex.run(E, M, upto=b1)

    rows = []
    for k in range(b0 - 1, b1):
        s = steps[k]
        sim = (f"like bar {s.nearest} ({s.nearest_sim:.2f})" if s.nearest and s.nearest_sim >= 0.9
               else f"new (nearest bar {s.nearest}, {s.nearest_sim:.2f})" if s.nearest else "first bar")
        sur = "warm-up" if s.bar <= ex.WARMUP else f"surprise {s.surprise:.1f}"
        rows.append(f"{s.bar} ({_mmss(g.t_of_bar(s.bar))}): {ex.what_changed(M, k)}, now {M[k, 0]:.0f} dB; {sim}; {sur}")
    evs = [e for e in ex.events(steps) if b0 - 1 <= e[0] <= b1 - 1]
    happened = ("; ".join(f"bar {steps[i].bar} broke expectation ({p:.1f}): {ex.explain(steps[i], M)}"
                          for i, _, p in evs)
                or "No big surprises: it went roughly as the past suggested.")
    nxt = b1 + 1
    fc = ex.forecast(a, E, M, steps, nxt, bars) if nxt <= n else None
    rep = {
        "file": a.name,
        "heard": f"bars {b0}-{b1} of {n} ({_mmss(g.t_of_bar(b0))}-{_mmss(min(a.duration, g.t_of_bar(b1 + 1)))})",
        "your_guess_for_these_bars": my_guess or "(none given)",
        "model_forecast_was": st["forecast"] or "(no forecast: first chunk)",
        "what_happened": happened,
        "bars": rows,
        "model_forecast_next": fc or "This was the last chunk.",
        "next": (f"Write your own expectation for bars {nxt}-{min(n, nxt + bars - 1)}, then call listen_live again "
                 "with my_guess." if nxt <= n else "Track finished."),
        "honesty": ("Only bars up to the cyan line have been analysed for this view (tempo/grid comes from the whole "
                    "file). Text and a picture of the sound, delivered in steps; not audio in real time."),
    }
    img, fmt, _ = render.live_view(a, steps, b1, output_dir() / f"{_stem(real)}_live",
                                   (nxt, min(n, nxt + bars - 1)) if nxt <= n else None)
    st["pos"], st["forecast"] = nxt, fc
    return [_dump(rep), Image(data=img, format=fmt)]


@mcp.tool(annotations=ToolAnnotations(title="Sounds like (compare to other tracks)", readOnlyHint=True,
                                      openWorldHint=False), structured_output=False)
@friendly
def sounds_like(
    path: PathArg,
    bpm: BpmArg = None,
    against: Annotated[list[str] | None, Field(description="Other audio files to compare against (full paths).")] = None,
    folder: Annotated[str | None, Field(
        description="Or: compare against the newest audio files in this folder (default: the track's own folder).")] = None,
    max_files: Annotated[int, Field(description="Max reference files (first run takes ~15 s each).", ge=1, le=16)] = 8,
):
    """Describe a track by what it sounds like among your own music, using learned CLAP fingerprints.

    For each section, finds the closest-sounding section in the reference tracks, and ranks the references
    by whole-track similarity. References are your own files, so the answer is "bars 69-132 sound most like
    my_other_track bars 53-84" rather than adjectives. Fingerprints are cached, so repeat runs are fast.
    """
    from .embed import bar_embeddings, section_embeddings
    from .paths import find_audio as _fa

    t0 = time.time()
    real = resolve_audio_path(path)
    a = _analysis(real, bpm)
    tsec = section_embeddings(a, bar_embeddings(a))
    if against:
        refs = [resolve_audio_path(p) for p in against]
    else:
        listing = _fa("", folder or os.path.dirname(real), limit=100)
        refs = [r.split("  ")[-1].strip() for r in listing["newest_first"]]
        refs = [resolve_audio_path(r) for r in refs]
    refs = [r for r in refs if os.path.normcase(r) != os.path.normcase(real)][:max_files]
    if not refs:
        raise ValueError("No other audio files to compare against. Pass `against` or a `folder`.")
    lib = []
    for r in refs:
        ra = _analysis(r, None)
        re_ = bar_embeddings(ra)
        lib.append((ra.name, section_embeddings(ra, re_), re_.mean(axis=0) / max(np.linalg.norm(re_.mean(axis=0)), 1e-9)))
    Ea = bar_embeddings(a)
    a_vec = Ea.mean(axis=0) / max(np.linalg.norm(Ea.mean(axis=0)), 1e-9)
    whole = sorted(((float(a_vec @ v), name) for name, _, v in lib), reverse=True)
    lines = []
    for s0, e0, v in tsec:
        cands = sorted(((float(v @ rv), name, rs, rend) for name, secs, _ in lib for rs, rend, rv in secs), reverse=True)[:2]
        lines.append(f"bars {s0}-{e0}: " + "; ".join(f"{nm} bars {rs}-{rend} ({sc:.2f})" for sc, nm, rs, rend in cands))
    rep = {"file": a.name, "compared_with": len(lib),
           "whole_track_closest": [f"{nm} ({sc:.2f})" for sc, nm in whole],
           "per_section_closest": lines,
           "how_to_read": ("Cosine similarity of learned CLAP fingerprints: about 0.9+ = very close in sound, "
                           "0.8 = related, below 0.7 = different. It reflects overall sound (timbre, density, "
                           "instrumentation), not melody or key."),
           "took_sec": round(time.time() - t0, 1)}
    return _dump(rep)


def _level_output(path: str, splits=None, labels=None, target_lufs=None) -> list:
    from . import render
    from .levels import capture_report

    rep, plot = capture_report(path, splits, labels, target_lufs)
    img, fmt, img_path = render.level_timeline(plot, output_dir() / f"{_stem(path)}_levels")
    rep["image"] = img_path
    return [_dump(rep), Image(data=img, format=fmt)]


@mcp.tool(annotations=ToolAnnotations(title="Listen to the computer (record a game while you play)",
                                      readOnlyHint=False, destructiveHint=False, openWorldHint=False),
          structured_output=False)
@friendly
def capture(
    action: Annotated[Literal["start", "mark", "status", "snapshot", "stop", "devices"], Field(
        description="start recording; mark a moment (e.g. a new level); status = live loudness readout; "
                    "snapshot = save the last N seconds as a file to listen/describe; stop = finish and get the "
                    "per-level report; devices = list outputs.")] = "status",
    label: Annotated[str | None, Field(description="For mark: what starts here, e.g. 'level 2'.")] = None,
    name: Annotated[str, Field(description="For start: name for the recording file, e.g. the game's name.")] = "game",
    device: Annotated[str | None, Field(description="For start: part of an output device name (default: Windows default output).")] = None,
    max_minutes: Annotated[float, Field(description="For start: stop automatically after this long.", ge=1, le=240)] = 90,
    hotkey: Annotated[str | None, Field(description="For start: global marker hotkey (default ctrl+alt+m).")] = None,
    seconds: Annotated[float, Field(description="For snapshot: how many recent seconds to save (max 60).", ge=3, le=60)] = 30,
):
    """Record what the computer is playing (system audio), e.g. a game while the user plays it.

    Typical use: start -> the user plays, pressing the hotkey (Ctrl+Alt+M) when each new level starts, or telling
    you so you can `mark` it -> `status` any time for live levels ("level 3 is running +4 dB vs the earlier levels")
    -> `snapshot` to grab the last seconds and run listen/describe on them -> stop for a per-level loudness report
    with suggested volume changes. Captures everything the output device plays, corrected for the Windows volume
    slider; other apps making sound are reported. Recordings go to <music folder>\\ears captures.
    """
    from . import capture as cap

    if action == "devices":
        return _dump(cap.list_devices())
    if action == "start":
        rec = cap.start(name, device, max_minutes, hotkey)
        time.sleep(1.2)  # let the hotkey and volume watcher report in
        st = rec.status()
        st["tell_the_user"] = (f"Recording. Play normally; press {rec.hotkey.upper()} as each new level starts "
                               "(or tell me). Close other apps that make sound. Ask for status any time; say stop "
                               "when done." if rec.hotkey_ok else
                               "Recording. The marker hotkey isn't available, so tell me when each level starts.")
        return _dump(st)
    rec = cap.current()
    if rec is None:
        raise ValueError("Nothing has been recorded in this session. Use action='start' first.")
    if action == "mark":
        if not rec.running:
            raise ValueError("The recording has stopped, so there's nothing to mark.")
        m = rec.mark(label)
        return _dump({"marked": m["label"], "at": cap.mmss(m["t"]), "markers": len(rec.markers)})
    if action == "status":
        return _dump(rec.status())
    if action == "snapshot":
        if not rec.running:
            raise ValueError("The recording has stopped; use listen on the recording file instead: " + rec.final_path())
        p = rec.snapshot(seconds)
        return _dump({"saved": p, "seconds": seconds,
                      "next": "Call listen / describe / notes on this file to hear what is playing right now."})
    # stop
    if rec.running:
        cap.stop()
    path = rec.final_path()
    if rec.error and not os.path.isfile(path):
        raise ValueError(f"The recording failed: {rec.error}")
    out = _level_output(path)
    head = json.loads(out[0])
    head = {"recording": path, "length": cap.mmss(rec.elapsed()), "markers": len(rec.markers), **head}
    if rec.error:
        head["recording_error"] = rec.error
    out[0] = _dump(head)
    return out


@mcp.tool(annotations=ToolAnnotations(title="Loudness per level (game music balance)", readOnlyHint=True,
                                      openWorldHint=False), structured_output=False)
@friendly
def levels(
    path: Annotated[str, Field(description="A capture/recording file, or a folder of music files (one per level).")],
    splits: Annotated[list[str] | None, Field(
        description="Optional level start times, 'm:ss label' each, e.g. ['0:00 menu', '1:12 level 1']. "
                    "Default: markers saved with the capture, else split on silences.")] = None,
    labels: Annotated[list[str] | None, Field(description="Optional names for the segments, in order.")] = None,
    target_lufs: Annotated[float | None, Field(description="Optional target music level in LUFS; default = match the median level.",
                                               ge=-50, le=-3)] = None,
):
    """Compare loudness between levels: one recording split by markers/silences, or each file in a folder.

    Gives each level's music level (median short-term LUFS, robust to short loud effects), integrated LUFS,
    peak, and the dB / linear volume change to match the others or a target, plus a timeline image.
    """
    from .paths import AUDIO_EXTS, resolve_folder

    p = path.strip().strip('"')
    if os.path.isdir(p):
        from . import render
        from .levels import folder_report

        folder = resolve_folder(p)[0]
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                       if os.path.splitext(f)[1].lower() in AUDIO_EXTS)[:60]
        if not files:
            raise ValueError(f"No audio files in {folder}.")
        rep, plot = folder_report([resolve_audio_path(f) for f in files], target_lufs)
        img, fmt, img_path = render.level_bars(plot, output_dir() / f"{os.path.basename(folder)}_levels")
        rep["image"] = img_path
        return [_dump(rep), Image(data=img, format=fmt)]
    return _level_output(resolve_audio_path(p), splits, labels, target_lufs)


@mcp.tool(annotations=ToolAnnotations(title="Find audio files", readOnlyHint=True, openWorldHint=False),
          structured_output=False)
@friendly
def find_audio(
    query: Annotated[str, Field(description="Part of the file name, e.g. 'my_track'. Empty = any.")] = "",
    folder: Annotated[str | None, Field(description="Optional folder to search (must be inside the music folder).")] = None,
    limit: Annotated[int, Field(description="Max results.", ge=1, le=100)] = 20,
):
    """Find audio files (wav/mp3/flac/aiff/ogg) by name in the music folder, newest first."""
    res = _find_audio(query, folder, limit)
    res["searched"] = folder or "; ".join(allowed_roots())
    return _dump(res)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
