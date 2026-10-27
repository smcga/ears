"""ears: local audio analysis for any MCP-compatible AI client."""
from __future__ import annotations

import functools
import importlib.util
import io
import json
import os
import platform
import re
import secrets
import sys
import threading
import time
import traceback
from collections import OrderedDict
from typing import Annotated, Any, Literal

import numpy as np

from mcp.server.mcpserver import Audio, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from .paths import (
    allowed_roots,
    find_audio as _find_audio,
    output_dir,
    resolve_audio_path,
    save_outputs_enabled,
)
from .perception import PerceptionError

INSTRUCTIONS = (
    "ears measures audio files under folders the user has allowed. "
    "Use `listen` for one file, `compare` for two versions, `stems` to split into stems (slow), "
    "`notes` to transcribe notes/chords to MIDI (Basic Pitch), `describe` for genre/mood/texture tags (CLAP), "
    "`expect` for a bar-by-bar surprise map (listening as prediction), `listen_live` to step through a track without "
    "seeing ahead (write your own prediction before each call), `sounds_like` to compare against the user's other tracks, "
    "`find_audio` to locate files by name, and `capabilities` to check optional features. "
    "Use `perceive` when the user asks what arbitrary audio actually sounds like: it sends an immutable recent "
    "capture window or file excerpt to a configured audio-native provider and returns grounded text, measurements, "
    "uncertainty and a memory handle for follow-up questions. Because perception may upload source audio, call it "
    "only when the user asks Ears to hear, identify, critique or compare sound. `auditory_memory` searches prior results. "
    "`capture` records what the computer is playing (e.g. a game while the user plays: start, mark each new level, "
    "status for live levels, snapshot to grab the last seconds, stop for a per-level loudness report); `levels` "
    "compares loudness per level for a recording or per file for a folder of level music. "
    "Bar numbers follow FL Studio's playlist (bar 1 = start of file); "
    "pass `bpm` when the user gives the project tempo. Results are measurements, not human hearing: "
    "say so and call out low-confidence key/tempo. Use returned spectrograms when the client exposes images; "
    "otherwise rely on the text report and say that visual inspection was unavailable. Only call `audio_clip` "
    "when the user explicitly asks to send a short excerpt to an audio-capable model."
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
    """Turn expected failures into messages any calling model can act on."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (PermissionError, FileNotFoundError, ValueError, PerceptionError) as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            raise ToolError(f"ears hit an internal error: {type(exc).__name__}: {exc}") from exc

    return wrapper


def _dump(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


ReportResult = Annotated[CallToolResult, dict[str, Any]]


def _result(report: dict[str, Any], images: list[tuple[bytes, str]] | None = None,
            audio: tuple[bytes, str] | None = None) -> CallToolResult:
    """Return the same report for models and applications, plus optional standard MCP media."""
    content = [TextContent(type="text", text=_dump(report))]
    for data, fmt in images or []:
        content.append(Image(data=data, format=fmt).to_image_content())
    if audio:
        content.append(Audio(data=audio[0], format=audio[1]).to_audio_content())
    return CallToolResult(content=content, structured_content=report)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


PathArg = Annotated[str, Field(description="Absolute path to an allowed .wav/.mp3/.flac/.aiff/.ogg file.")]
BpmArg = Annotated[float | None, Field(description="Project tempo in BPM if known (e.g. 130). Omit to auto-detect.", ge=40, le=300)]


@mcp.tool(annotations=ToolAnnotations(title="Listen to a track", readOnlyHint=not save_outputs_enabled(),
                                      destructiveHint=False, idempotentHint=True, openWorldHint=False))
@friendly
def listen(
    path: PathArg,
    bpm: BpmArg = None,
    start_sec: Annotated[float | None, Field(description="Optional: start of a range to zoom into (seconds).", ge=0)] = None,
    end_sec: Annotated[float | None, Field(description="Optional: end of the zoom range (seconds).", gt=0)] = None,
) -> ReportResult:
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
    zoom = None
    if rng:
        zoom = render.zoom(a, rng, out / f"{_stem(real)}_zoom_{int(rng['_spec']['t0'])}-{int(rng['_spec']['t1'])}s")
        if zoom[2]:
            report["zoom_image_saved"] = zoom[2]
    report["took_sec"] = round(time.time() - t0, 1)
    images = [(img, fmt)] + ([(zoom[0], zoom[1])] if zoom else [])
    return _result(report, images)


@mcp.tool(annotations=ToolAnnotations(title="Compare two versions", readOnlyHint=not save_outputs_enabled(),
                                      destructiveHint=False, idempotentHint=True, openWorldHint=False))
@friendly
def compare(
    path_a: Annotated[str, Field(description="Full path to version A (the older one), e.g. ...\\my_track.mp3")],
    path_b: Annotated[str, Field(description="Full path to version B (the newer one), e.g. ...\\my_track_v2.mp3")],
    bpm: BpmArg = None,
) -> ReportResult:
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
    return _result(rep, [(img, fmt)])


@mcp.tool(annotations=ToolAnnotations(title="Split into stems (slow)", readOnlyHint=False, destructiveHint=True,
                                      idempotentHint=True, openWorldHint=True))
@friendly
def stems(
    path: PathArg,
    bpm: BpmArg = None,
    force: Annotated[bool, Field(description="Re-run separation even if stems already exist.")] = False,
) -> ReportResult:
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
        "took_sec": round(time.time() - t0, 1),
    }
    if img_path:
        out["image_saved"] = img_path
    return _result(out, [(img, fmt)])


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
                                      destructiveHint=True, idempotentHint=True, openWorldHint=False))
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
) -> ReportResult:
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
    }
    if img_path:
        rep["image_saved"] = img_path
    if dropped:
        rep["ignored_notes"] = f"{dropped} notes in bars where the stem is near-silent (bleed/reverb) were left out."
    if fallback:
        rep["fallback"] = fallback
    if sep and not sep.get("reused_existing"):
        rep["stem_separation"] = sep
    rep["took_sec"] = round(time.time() - t0, 1)
    return _result({k: v for k, v in rep.items() if v is not None}, [(img, fmt)])


@mcp.tool(annotations=ToolAnnotations(title="Describe the vibe (CLAP tags)", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=True, openWorldHint=True))
@friendly
def describe(
    path: PathArg,
    bpm: BpmArg = None,
    phrases: Annotated[list[str] | None, Field(
        description="Optional: your own descriptions to score, e.g. ['garage-ish', 'swung', 'gritty bass']. Max 12.")] = None,
) -> ReportResult:
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
    return _result(rep)


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


@mcp.tool(annotations=ToolAnnotations(title="Listen as prediction (surprise map)", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=True, openWorldHint=True))
@friendly
def expect(path: PathArg, bpm: BpmArg = None) -> ReportResult:
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
    if img_path:
        rep["image_saved"] = img_path
    rep["took_sec"] = round(time.time() - t0, 1)
    return _result(rep, [(img, fmt)])


_live: dict[str, dict] = {}
_live_lock = threading.Lock()


@mcp.tool(annotations=ToolAnnotations(title="Listen in time (step through)", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=False, openWorldHint=True))
@friendly
def listen_live(
    path: PathArg,
    bpm: BpmArg = None,
    session_id: Annotated[str | None, Field(
        description="Handle returned by the first call. Pass it on every later call for this listening session.")] = None,
    bars: Annotated[int, Field(description="How many bars to reveal per call.", ge=1, le=32)] = 8,
    my_guess: Annotated[str | None, Field(
        description="Your prediction for the bars about to be revealed (free text). It's echoed back next to "
                    "what actually happened, so you can see where your expectation was right or wrong.")] = None,
    from_bar: Annotated[int | None, Field(description="Start (or jump) to this bar.", ge=1)] = None,
    restart: Annotated[bool, Field(description="Start again from bar 1.")] = False,
) -> ReportResult:
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
    path_key = os.path.normcase(real)
    with _live_lock:
        if session_id:
            st = _live.get(session_id)
            if st is None:
                raise ValueError("Unknown or finished listen_live session. Omit session_id to start a new one.")
            if st["path"] != path_key:
                raise ValueError("That listen_live session belongs to a different audio file.")
        else:
            session_id = secrets.token_urlsafe(12)
            st = _live[session_id] = {"path": path_key, "pos": from_bar or 1, "forecast": None,
                                      "lock": threading.Lock()}
    with st["lock"]:
        if restart or from_bar:
            st["pos"], st["forecast"] = from_bar or 1, None
        if st["pos"] > n:
            with _live_lock:
                _live.pop(session_id, None)
            return _result({"file": a.name, "session_id": session_id,
                            "done": f"Track finished at bar {n}. Start a new session to listen again."})
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
            rows.append(f"{s.bar} ({_mmss(g.t_of_bar(s.bar))}): {ex.what_changed(M, k)}, "
                        f"now {M[k, 0]:.0f} dB; {sim}; {sur}")
        evs = [e for e in ex.events(steps) if b0 - 1 <= e[0] <= b1 - 1]
        happened = ("; ".join(f"bar {steps[i].bar} broke expectation ({p:.1f}): {ex.explain(steps[i], M)}"
                              for i, _, p in evs)
                    or "No big surprises: it went roughly as the past suggested.")
        nxt = b1 + 1
        fc = ex.forecast(a, E, M, steps, nxt, bars) if nxt <= n else None
        rep = {
            "file": a.name,
            "session_id": session_id,
            "heard": f"bars {b0}-{b1} of {n} ({_mmss(g.t_of_bar(b0))}-{_mmss(min(a.duration, g.t_of_bar(b1 + 1)))})",
            "your_guess_for_these_bars": my_guess or "(none given)",
            "model_forecast_was": st["forecast"] or "(no forecast: first chunk)",
            "what_happened": happened,
            "bars": rows,
            "model_forecast_next": fc or "This was the last chunk.",
            "next": (f"Write your own expectation for bars {nxt}-{min(n, nxt + bars - 1)}, then call "
                     f"listen_live again with session_id='{session_id}' and my_guess."
                     if nxt <= n else "Track finished."),
            "honesty": ("Only bars up to the cyan line have been analysed for this view (tempo/grid comes from "
                        "the whole file). Text and a picture of the sound, delivered in steps; not audio in real time."),
        }
        img, fmt, _ = render.live_view(a, steps, b1, output_dir() / f"{_stem(real)}_live",
                                       (nxt, min(n, nxt + bars - 1)) if nxt <= n else None)
        st["pos"], st["forecast"] = nxt, fc
        if nxt > n:
            with _live_lock:
                _live.pop(session_id, None)
        return _result(rep, [(img, fmt)])


@mcp.tool(annotations=ToolAnnotations(title="Sounds like (compare to other tracks)", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=True, openWorldHint=True))
@friendly
def sounds_like(
    path: PathArg,
    bpm: BpmArg = None,
    against: Annotated[list[str] | None, Field(description="Other audio files to compare against (full paths).")] = None,
    folder: Annotated[str | None, Field(
        description="Or: compare against the newest audio files in this folder (default: the track's own folder).")] = None,
    max_files: Annotated[int, Field(description="Max reference files (first run takes ~15 s each).", ge=1, le=16)] = 8,
) -> ReportResult:
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
        refs = [r["path"] for r in listing["files"]]
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
    return _result(rep)


def _level_output(path: str, splits=None, labels=None, target_lufs=None) -> CallToolResult:
    from . import render
    from .levels import capture_report

    rep, plot = capture_report(path, splits, labels, target_lufs)
    img, fmt, img_path = render.level_timeline(plot, output_dir() / f"{_stem(path)}_levels")
    if img_path:
        rep["image_saved"] = img_path
    return _result(rep, [(img, fmt)])


def _module_available(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


@mcp.tool(annotations=ToolAnnotations(title="Show ears capabilities", readOnlyHint=True, openWorldHint=False))
@friendly
def capabilities() -> ReportResult:
    """Show installed features, platform constraints, allowed folders and persistence settings."""
    from .perception import provider_status

    perception = provider_status()
    perception["available"] = any(p.get("configured") for p in perception["providers"].values())
    perception["recent_source_available"] = (
        sys.platform == "win32" and _module_available("soundcard", "pycaw", "comtypes")
    )
    perception["network"] = "Provider-dependent; Gemini uploads only the selected immutable audio window."
    features = {
        "listen_compare_levels": {"available": True, "network": False},
        "audio_clip": {"available": True, "network": False, "max_seconds": {"mono": 15, "stereo": 6}},
        "perception": perception,
        "stems": {
            "available": _module_available("torch", "demucs"),
            "network": "May download pretrained Demucs weights on first use if not cached.",
        },
        "notes": {"available": _module_available("basic_pitch", "onnxruntime"), "network": False},
        "describe_expect_sounds_like": {
            "available": _module_available("torch", "transformers"),
            "network": "Downloads pinned CLAP model weights from Hugging Face on first use if not cached.",
        },
        "capture": {
            "available": sys.platform == "win32" and _module_available("soundcard", "pycaw", "comtypes"),
            "platform": "Windows 10/11",
            "network": False,
        },
    }
    return _result({
        "server": "ears",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "allowed_roots": allowed_roots(),
        "features": features,
        "persistence": {
            "save_generated_images": save_outputs_enabled(),
            "output_dir": str(output_dir()) if save_outputs_enabled() else None,
            "persistent_embedding_cache": os.environ.get("EARS_CACHE_DIR") or None,
            "note": "Reports and media are returned inline. Disk persistence is opt-in except for stems, MIDI and captures.",
        },
    })


@mcp.tool(annotations=ToolAnnotations(title="Send a short audio excerpt", readOnlyHint=True, openWorldHint=False))
@friendly
def audio_clip(
    path: PathArg,
    start_sec: Annotated[float, Field(description="Excerpt start in seconds.", ge=0)] = 0,
    end_sec: Annotated[float | None, Field(
        description="Excerpt end in seconds. Default: 12 seconds after start. Max: 15 s mono / 6 s stereo.",
        gt=0)] = None,
    channels: Annotated[Literal["mono", "stereo"], Field(
        description="Mono is smaller and most compatible; stereo preserves spatial information.")] = "mono",
    sample_rate: Annotated[Literal[16000, 24000], Field(
        description="Output sample rate. 24 kHz preserves more musical detail; 16 kHz is smaller.")] = 24000,
) -> ReportResult:
    """Return a short WAV excerpt as standard MCP AudioContent plus a text fallback.

    Only use this when the user explicitly asks an audio-capable model to hear an excerpt. The MCP client may
    transmit the returned audio to its model provider. Longer excerpts should be analysed with listen instead.
    """
    import soundfile as sf
    import soxr

    real = resolve_audio_path(path)
    with sf.SoundFile(real) as src:
        duration = len(src) / src.samplerate
        start = min(float(start_sec), duration)
        default_duration = 6.0 if channels == "stereo" else 12.0
        end = min(float(end_sec) if end_sec is not None else start + default_duration, duration)
        if not start < end:
            raise ValueError(f"Bad excerpt range: {start:g}-{end:g} s (track length {duration:.1f} s).")
        max_duration = 6.0 if channels == "stereo" else 15.0
        if end - start > max_duration:
            raise ValueError(f"{channels.title()} audio excerpts are limited to {max_duration:g} seconds. "
                             "Use a shorter range or mono output.")
        src.seek(round(start * src.samplerate))
        frames = round((end - start) * src.samplerate)
        data = src.read(frames, dtype="float32", always_2d=True)
        source_rate = src.samplerate
    if channels == "mono":
        data = data.mean(axis=1)
    else:
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]
    if source_rate != sample_rate:
        data = soxr.resample(data, source_rate, sample_rate, quality="HQ")
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    payload = buf.getvalue()
    report = {
        "file": os.path.basename(real),
        "range_sec": {"start": round(start, 3), "end": round(end, 3), "duration": round(end - start, 3)},
        "format": {"container": "wav", "sample_rate": sample_rate, "channels": channels, "bytes": len(payload)},
        "privacy": "The MCP client may send this excerpt to its configured model provider.",
        "fallback": "If audio is not exposed to the model, use listen for a text and spectrogram report.",
    }
    return _result(report, audio=(payload, "wav"))


def _perception_window(source: str, seconds: float, label: str):
    from .perception import AudioWindow, window_from_file

    if source.strip().lower() == "recent":
        if sys.platform != "win32":
            raise ValueError("The recent system-audio source is currently available on Windows 10/11 only. "
                             "Pass an audio file path on this platform.")
        from . import capture as cap

        rec = cap.current()
        if rec is None:
            raise ValueError("No recent system audio exists. Start capture first, then play the sound and call perceive.")
        samples, sample_rate, meta = rec.recent_audio(seconds)
        warnings = []
        if meta.get("error"):
            warnings.append(f"Capture device error after this audio was recorded: {meta['error']}")
        if not meta.get("running"):
            warnings.append("Capture is no longer running; this is the final retained window.")
        if meta.get("muted_during_session"):
            warnings.append("Windows output was muted during part of this capture session.")
        return AudioWindow(
            label=label,
            samples=samples,
            sample_rate=sample_rate,
            source="recent",
            origin_start_sec=float(meta["session_start_sec"]),
            origin_end_sec=float(meta["session_end_sec"]),
            warnings=tuple(warnings),
        )
    real = resolve_audio_path(source)
    return window_from_file(real, seconds, label)


@mcp.tool(annotations=ToolAnnotations(title="Perceive arbitrary audio", readOnlyHint=False,
                                      destructiveHint=False, idempotentHint=False, openWorldHint=True))
@friendly
def perceive(
    source: Annotated[str, Field(
        description="'recent' for the current capture buffer, or an absolute path to an allowed audio file.")] = "recent",
    seconds: Annotated[float, Field(
        description="How much audio to hear, taken from the end of the source (max 60 seconds).", ge=3, le=60)] = 30,
    question: Annotated[str, Field(
        description="Open-ended question about what is audible, e.g. 'What happened?' or 'Does this mix work?'.",
        min_length=1, max_length=2000)] = "What do you hear?",
    provider: Annotated[str, Field(
        description="Audio-native backend. 'auto' uses EARS_PERCEPTION_PROVIDER (default: Gemini).",
        min_length=1, max_length=64)] = "auto",
    against: Annotated[str | None, Field(
        description="Optional second allowed audio file (or 'recent') for an A/B listening comparison.")] = None,
    memory_id: Annotated[str | None, Field(
        description="Returned by an earlier perceive call. Reuses that exact immutable audio for a follow-up question.")] = None,
) -> ReportResult:
    """Ask an audio-native model an open-ended question about recent system audio or a file.

    The result is text and structured data, so this works even when the MCP client cannot forward audio. The selected
    audio window is sent directly to the configured provider; with Gemini, that means it leaves the computer. Ears
    returns model observations, deterministic measurements, recommendations and uncertainty as separate layers.
    Use memory_id for follow-ups so Ears listens to exactly the same waveform instead of silently recapturing it.
    """
    from .perception import perceive_windows, recall

    if memory_id:
        # Resolve before any capture/file work so a follow-up cannot accidentally listen to new audio.
        item = recall(memory_id)
        windows = item.windows
    else:
        windows = (_perception_window(source, seconds, "A" if against else "audio"),)
        if against:
            if seconds > 30:
                raise ValueError("A/B perception is limited to 30 seconds per source to bound upload size and cost.")
            windows += (_perception_window(against, seconds, "B"),)
    report = perceive_windows(windows, question, provider, memory_id)
    report["privacy"] = (
        "The selected audio is sent to the configured perception provider unless the local silence guard handles it. "
        "No recording file is created for a recent-audio question."
    )
    return _result(report)


@mcp.tool(annotations=ToolAnnotations(title="Search auditory memory", readOnlyHint=True,
                                      destructiveHint=False, idempotentHint=True, openWorldHint=False))
@friendly
def auditory_memory(
    query: Annotated[str, Field(
        description="Optional word or phrase to find in prior perception observations.", max_length=500)] = "",
    limit: Annotated[int, Field(description="Maximum recent perception windows to return.", ge=1, le=20)] = 10,
) -> ReportResult:
    """Search bounded in-process perception history by its observed summaries and events.

    This is an event index over audio that has already been perceived, not continuous background interpretation.
    Use a returned memory_id with perceive to listen again and ask a follow-up about the original immutable audio.
    """
    from .perception import memory_index

    rows = memory_index(query, limit)
    return _result({
        "query": query,
        "memories": rows,
        "bounded": True,
        "persistence": "Memory is process-local, size-bounded and disappears when Ears restarts.",
        "proactive_limit": "MCP clients decide when tools run; Ears cannot inject observations into an arbitrary chat.",
    })


@mcp.tool(annotations=ToolAnnotations(title="Listen to the computer (record a game while you play)",
                                      readOnlyHint=False, destructiveHint=False, idempotentHint=False,
                                      openWorldHint=False))
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
) -> ReportResult:
    """Record what the computer is playing (system audio), e.g. a game while the user plays it.

    Typical use: start -> the user plays, pressing the hotkey (Ctrl+Alt+M) when each new level starts, or telling
    you so you can `mark` it -> `status` any time for live levels ("level 3 is running +4 dB vs the earlier levels")
    -> `snapshot` to grab the last seconds and run listen/describe on them -> stop for a per-level loudness report
    with suggested volume changes. Captures everything the output device plays. Raw loopback audio is retained for
    recording/perception; a separate temporary volume-corrected signal is used for measurements. Other apps making
    sound are reported. Recordings go to <music folder>\\ears captures.
    """
    if sys.platform != "win32":
        raise ValueError("System-audio capture is currently available on Windows 10/11 only.")
    from . import capture as cap

    if action == "devices":
        return _result(cap.list_devices())
    if action == "start":
        rec = cap.start(name, device, max_minutes, hotkey)
        time.sleep(1.2)  # let the hotkey and volume watcher report in
        st = rec.status()
        st["tell_the_user"] = (f"Recording. Play normally; press {rec.hotkey.upper()} as each new level starts "
                               "(or tell me). Close other apps that make sound. Ask what I heard at any time; "
                               "perceive reads the recent buffer without saving it. Say stop when done."
                               if rec.hotkey_ok else
                               "Recording. The marker hotkey isn't available, so tell me when each level starts. "
                               "You can still ask what I heard at any time.")
        return _result(st)
    rec = cap.current()
    if rec is None:
        raise ValueError("Nothing has been recorded in this session. Use action='start' first.")
    if action == "mark":
        if not rec.running:
            raise ValueError("The recording has stopped, so there's nothing to mark.")
        m = rec.mark(label)
        return _result({"marked": m["label"], "at": cap.mmss(m["t"]), "markers": len(rec.markers)})
    if action == "status":
        return _result(rec.status())
    if action == "snapshot":
        if not rec.running:
            raise ValueError("The recording has stopped; use listen on the recording file instead: " + rec.final_path())
        p = rec.snapshot(seconds)
        return _result({"saved": p, "seconds": seconds,
                        "next": "Call listen / describe / notes on this file to hear what is playing right now."})
    # stop
    if rec.running:
        cap.stop()
    path = rec.final_path()
    if rec.error and not os.path.isfile(path):
        rec.cleanup_measurement()
        raise ValueError(f"The recording failed: {rec.error}")
    try:
        out = _level_output(rec.measurement_path())
    finally:
        rec.cleanup_measurement()
    head = dict(out.structured_content)
    head = {"recording": path, "length": cap.mmss(rec.elapsed()), "markers": len(rec.markers), **head}
    if rec.error:
        head["recording_error"] = rec.error
    out.structured_content = head
    out.content[0] = TextContent(type="text", text=_dump(head))
    return out


@mcp.tool(annotations=ToolAnnotations(title="Loudness per level (game music balance)",
                                      readOnlyHint=not save_outputs_enabled(), destructiveHint=False,
                                      idempotentHint=True, openWorldHint=False))
@friendly
def levels(
    path: Annotated[str, Field(description="A capture/recording file, or a folder of music files (one per level).")],
    splits: Annotated[list[str] | None, Field(
        description="Optional level start times, 'm:ss label' each, e.g. ['0:00 menu', '1:12 level 1']. "
                    "Default: markers saved with the capture, else split on silences.")] = None,
    labels: Annotated[list[str] | None, Field(description="Optional names for the segments, in order.")] = None,
    target_lufs: Annotated[float | None, Field(description="Optional target music level in LUFS; default = match the median level.",
                                               ge=-50, le=-3)] = None,
) -> ReportResult:
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
        if img_path:
            rep["image_saved"] = img_path
        return _result(rep, [(img, fmt)])
    return _level_output(resolve_audio_path(p), splits, labels, target_lufs)


@mcp.tool(annotations=ToolAnnotations(title="Find audio files", readOnlyHint=True, openWorldHint=False))
@friendly
def find_audio(
    query: Annotated[str, Field(description="Part of the file name, e.g. 'my_track'. Empty = any.")] = "",
    folder: Annotated[str | None, Field(description="Optional folder to search (must be inside the music folder).")] = None,
    limit: Annotated[int, Field(description="Max results.", ge=1, le=100)] = 20,
) -> ReportResult:
    """Find audio files (wav/mp3/flac/aiff/ogg) by name in the music folder, newest first."""
    res = _find_audio(query, folder, limit)
    res["searched"] = folder or os.pathsep.join(allowed_roots())
    return _result(res)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
