"""ears self-test.

1. Builds a synthetic track with known answers (130 BPM kick, 45 Hz sub, G major chord,
   quiet 8-bar intro then loud section, one clipped peak) and checks listen()'s report.
2. Checks MP3/FLAC decoding, compare(), and the folder guard.
3. Starts the real MCP server over stdio and calls it like Claude would.

Run:  .venv\\Scripts\\python.exe tests\\selftest.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "selftest"
TEST_DIR.mkdir(exist_ok=True)
ROOTS = str(TEST_DIR)
os.environ["EARS_ALLOWED_ROOTS"] = ROOTS
sys.path.insert(0, str(ROOT))

SR, BPM = 44100, 130.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
RESULTS: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> bool:
    RESULTS.append((bool(ok), what))
    print(("  PASS  " if ok else "  FAIL  ") + what)
    return bool(ok)


def synth(path: Path, intro_bars=8, main_bars=16, extra_bars=0, sub_gain=1.0, chord_gain=1.0, clip=True):
    n_bars = intro_bars + main_bars + extra_bars
    n = int(round(n_bars * BAR * SR))
    t = np.arange(n) / SR
    intro_end = intro_bars * BAR
    L = np.zeros(n)
    R = np.zeros(n)

    # Kick on every beat: pitch-swept sine, quiet in the intro
    kl = int(0.35 * SR)
    tk = np.arange(kl) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-tk / 0.03)) / SR) * np.exp(-tk / 0.12)
    for b in range(n_bars * 4):
        s = int(round(b * BEAT * SR))
        g = 0.1 if b * BEAT < intro_end else 0.9
        e = min(n, s + kl)
        L[s:e] += g * kick[: e - s]
        R[s:e] += g * kick[: e - s]

    # 45 Hz sub, only in the loud section (10 ms fade-in)
    sub_env = np.clip((t - intro_end) / 0.01, 0, 1)
    sub = 0.35 * sub_gain * np.sin(2 * np.pi * 45 * t) * sub_env
    L += sub
    R += sub

    # G major chord voiced G2 D3 G3 B3 D4 (4 harmonics each), slightly detuned in R for some width
    cl, cr = np.zeros(n), np.zeros(n)
    for f0 in (98.00, 146.83, 196.00, 246.94, 293.66):
        for h, amp in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
            cl += amp * np.sin(2 * np.pi * f0 * h * t)
            cr += amp * np.sin(2 * np.pi * f0 * h * 1.002 * t + 0.3)
    env = np.where(t < intro_end, 0.1, 1.0) * 0.06 * chord_gain
    L += cl * env
    R += cr * env

    x = np.stack([L, R], axis=1)
    x *= 0.89 / np.abs(x).max()  # loud section peaks at about -1 dBFS
    if clip:  # one deliberately clipped peak: bar 17, on the kick
        s = int(round(16 * BAR * SR)) + int(0.004 * SR)
        x[s: s + 60] *= 3.0
    x = np.clip(x, -1.0, 1.0)
    sf.write(str(path), x, SR, subtype="PCM_16")
    return x


PROGRESSION = [("Am", [57, 60, 64], 45), ("F", [53, 57, 60], 41), ("C", [60, 64, 67], 48), ("G", [55, 59, 62], 43)] * 2


def synth_chords(path: Path):
    """8 bars at 130 BPM: Am | F | C | G twice, one chord per bar, with a root-note bassline."""
    n = int(round(len(PROGRESSION) * BAR * SR))
    x = np.zeros(n)
    for i, (_, chord, bass) in enumerate(PROGRESSION):
        s, e = int(round(i * BAR * SR)), int(round(((i + 1) * BAR - 0.05) * SR))
        tt = np.arange(e - s) / SR
        env = np.minimum(1, tt / 0.01) * np.minimum(1, (tt[-1] - tt) / 0.03)
        for m in chord:
            f = 440 * 2 ** ((m - 69) / 12)
            x[s:e] += 0.12 * env * sum(a * np.sin(2 * np.pi * f * h * tt) for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)))
        fb = 440 * 2 ** ((bass - 69) / 12)
        x[s:e] += env * (0.25 * np.sin(2 * np.pi * fb * tt) + 0.08 * np.sin(4 * np.pi * fb * tt))
    x *= 0.8 / np.abs(x).max()
    sf.write(str(path), np.stack([x, x], axis=1), SR, subtype="PCM_16")


def run_notes_test():
    import pretty_midi

    from ears import server
    from ears import transcribe as tr
    from ears.analysis import Grid

    print("\n== 7. notes(): synthetic Am | F | C | G progression (x2), transcribed with Basic Pitch")
    wav = TEST_DIR / "synthetic_chords_130.wav"
    synth_chords(wav)
    t0 = time.time()
    res = server.notes(str(wav), bpm=130, stem="mix", bars="1-4")
    rep = json.loads(res[0])
    print(f"  notes() took {time.time() - t0:.1f} s; JSON {len(res[0])} chars")
    for k in ("notes_found", "range", "key_from_notes", "chords_per_bar", "bass_roots_per_bar"):
        print(f"    {k}: {json.dumps(rep.get(k), ensure_ascii=False)}")
    for row in rep["detail"]["rows"]:
        print(f"    detail {row}")

    notes = tr.transcribe(str(wav), 130.0)
    per_bar = tr.bar_chords(notes, [n for n in notes if n.pitch < 52], Grid(130.0, len(PROGRESSION) * BAR))
    got = [c.split("/")[0].rstrip("?") for c in per_bar]
    want = [c for c, _, _ in PROGRESSION]
    hits = sum(g == w for g, w in zip(got, want))
    check(hits >= 7, f"chords per bar {hits}/8 right (got {per_bar})")
    check(rep["key_from_notes"]["best"].split(" (")[0] in ("C major", "A minor"),
          f"key from notes C major / A minor (got {rep['key_from_notes']['best']})")
    row1 = rep["detail"]["rows"][0]
    check("C5" in row1 and "E5" in row1 and ("A4" in row1 or "A3" in row1),
          f"bar 1 note list has the Am notes in FL octave names ({row1})")
    check(rep["bass_roots_per_bar"] == ["bars 1-8: | A | F | C | G | x2"], f"bass roots {rep['bass_roots_per_bar']}")
    mid = rep["midi_saved"][0]
    pm = pretty_midi.PrettyMIDI(mid)
    pitches = {n.pitch for inst in pm.instruments for n in inst.notes}
    wanted = {p for _, ch, b in PROGRESSION for p in ch + [b]}
    first_bar = [n for inst in pm.instruments for n in inst.notes if n.start < BAR - 0.01]
    tempo = float(pm.get_tempo_changes()[1][0])
    check(len(wanted & pitches) >= len(wanted) - 1 and abs(tempo - 130) < 0.01,
          f"MIDI saved at {tempo:g} BPM with the right pitches ({len(wanted & pitches)}/{len(wanted)} expected)")
    check(all(abs(n.start / (BEAT / 4) - round(n.start / (BEAT / 4))) < 1e-3 for n in first_bar),
          "MIDI notes sit on the 1/16 grid")


def run_describe_test():
    from ears import server

    print("\n== 8. describe(): CLAP tags (synthetic audio is unlike real records, so check relative behaviour)")
    kick_phrase = "four-on-the-floor techno kick"

    def score(rep):
        row = next(r for r in rep["your_phrases"] if r.startswith(f"'{kick_phrase}'"))
        return float(row.split(": ")[1].split(" ")[0])

    t0 = time.time()
    drums = json.loads(server.describe(str(TEST_DIR / "synthetic_130_Gmaj.wav"), bpm=130, phrases=[kick_phrase]))
    chords = json.loads(server.describe(str(TEST_DIR / "synthetic_chords_130.wav"), bpm=130, phrases=[kick_phrase]))
    print(f"  two describe() calls took {time.time() - t0:.1f} s")
    print(f"    chords-only track: {chords['per_section'][0]}")
    top_groove = chords["whole_track"]["groove"]["top"][0]
    check(top_groove.startswith("no drums"), f"chords-only track: top groove tag is 'no drums' ({top_groove})")
    check(score(drums) - score(chords) >= 0.1,
          f"kick phrase scores higher on the kick track ({score(drums):.2f}) than chords-only ({score(chords):.2f})")


def run_time_tests():
    import numpy as np

    from ears import expect as ex
    from ears import server
    from ears.embed import bar_embeddings

    print("\n== 9. expect / listen_live / sounds_like: listening as prediction")
    wav = str(TEST_DIR / "synthetic_130_Gmaj.wav")
    rep = json.loads(server.expect(wav, bpm=130)[0])
    print(f"    events at bars {rep['surprise_events_at_bars']}; moments: {rep['big_moments'][:1]}")
    check(9 in rep["surprise_events_at_bars"], "expect: the quiet-intro -> loud change at bar 9 is a surprise")

    a = server._analysis(wav, 130.0)
    E, M = bar_embeddings(a), ex.measured(a)
    full = [s.surprise for s in ex.run(E, M)]
    part = [s.surprise for s in ex.run(E, M, upto=8)]
    check(np.allclose(full[:8], part), "surprise is causal: bars 1-8 score the same whether or not later bars exist")

    r1 = json.loads(server.listen_live(wav, bpm=130, bars=8, restart=True)[0])
    r2 = json.loads(server.listen_live(wav, bpm=130, bars=8, my_guess="it gets louder at bar 9")[0])
    check(r1["heard"].startswith("bars 1-8") and r2["heard"].startswith("bars 9-16")
          and r2["your_guess_for_these_bars"] == "it gets louder at bar 9" and "bar 9" in r2["what_happened"],
          "listen_live steps 1-8 then 9-16 and reports the bar-9 surprise")

    sl = json.loads(server.sounds_like(wav, bpm=130, against=[str(TEST_DIR / "synthetic_130_Gmaj_v2.wav"),
                                                                str(TEST_DIR / "synthetic_chords_130.wav")]))
    check(sl["whole_track_closest"][0].startswith("synthetic_130_Gmaj_v2"),
          f"sounds_like ranks the near-copy first ({sl['whole_track_closest']})")


def _bed(seconds: float, gain_db: float, seed: int, sfx_every: float = 0.0) -> np.ndarray:
    """Steady 'music' (chord + soft noise) at a known level, optionally with short loud 'gunshots'."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 48000)) / 48000
    x = sum(np.sin(2 * np.pi * f * t) for f in (196.0, 246.9, 293.7)) / 3 * 0.5
    x = x + 0.05 * rng.standard_normal(len(t))
    x = x * 10 ** (gain_db / 20)
    if sfx_every:
        for s0 in np.arange(1.0, seconds - 0.2, sfx_every):
            i = int(s0 * 48000)
            x[i:i + 4800] += 0.6 * rng.standard_normal(4800) * np.exp(-np.arange(4800) / 900)
    return np.stack([x, x], axis=1).astype(np.float32)


def run_game_tests():
    import pyloudnorm as pyln

    from ears import server
    from ears.levels import track_file, integrated

    print("\n== 10. Game audio: per-level loudness (levels) and system-audio capture")
    wav = str(TEST_DIR / "synthetic_130_Gmaj.wav")
    x, sr = sf.read(wav, always_2d=True)
    ref = pyln.Meter(sr).integrated_loudness(x)
    ms, _, _ = track_file(wav)
    mine = integrated(ms, 0, len(ms))
    check(abs(mine - ref) < 0.3, f"streaming loudness matches pyloudnorm ({mine:.2f} vs {ref:.2f} LUFS)")

    gap = np.zeros((96000, 2), np.float32)
    game = np.concatenate([_bed(20, -30, 1, sfx_every=3.0), gap, _bed(20, -24, 2), gap, _bed(20, -30, 3)])
    gpath = TEST_DIR / "fake_game_capture.flac"
    sf.write(gpath, game, 48000, subtype="PCM_24")
    side = TEST_DIR / "fake_game_capture.markers.json"
    side.unlink(missing_ok=True)
    rep = json.loads(server.levels(str(gpath))[0])
    lv = rep["levels"]
    print(f"    auto split: {[(l['label'], l['from'], l['bed'], l['lufs'], l.get('change_db')) for l in lv]}")
    check(len(lv) == 3 and "silences" in rep["split_by"], "levels splits a capture on the silences between levels")
    check(len(lv) == 3 and abs((lv[1]["bed"] - lv[2]["bed"]) - 6) < 0.5 and lv[1]["change_db"] < -5,
          "level 2 is found 6 dB louder and told to turn down")
    check(len(lv) == 3 and abs(lv[0]["bed"] - lv[2]["bed"]) < 1.0 and lv[0]["lufs"] - lv[0]["bed"] > 1.0,
          "music level ignores loud sound effects (integrated LUFS doesn't)")
    side.write_text(json.dumps({"markers": [{"t": 0.0, "label": "forest"}, {"t": 22.0, "label": "cave"},
                                            {"t": 44.0, "label": "castle"}]}))
    rep = json.loads(server.levels(str(gpath))[0])
    check([l["label"] for l in rep["levels"]] == ["forest", "cave", "castle"], "levels uses saved capture markers")
    import soundfile as _sf
    dur = _sf.info(str(gpath)).duration
    side.write_text(json.dumps({"markers": [{"t": 0.0, "label": "forest"}, {"t": 22.0, "label": "cave"},
                                            {"t": 44.0, "label": "castle"}, {"t": dur - 1.0, "label": "tail"}]}))
    rep = json.loads(server.levels(str(gpath))[0])
    tail = rep["levels"][-1]
    check(tail["label"] == "tail" and tail.get("unreliable") and tail["change_db"] is None
          and not any(v.startswith("tail:") and "turn" in v for v in rep["verdict"]),
          "a 1-second tail after the last marker is left out of the balance")
    side.unlink()

    fdir = TEST_DIR / "level_music"
    fdir.mkdir(exist_ok=True)
    for name, g in (("level1.wav", -20), ("level2.wav", -14), ("level3.wav", -20)):
        sf.write(fdir / name, _bed(10, g, 7), 48000)
    rep = json.loads(server.levels(str(fdir))[0])
    ch = {l["label"]: l["change_db"] for l in rep["levels"]}
    check(abs(ch["level2.wav"] + 6) < 0.3 and abs(ch["level1.wav"]) < 0.3,
          f"folder mode: one file per level, gain to match ({ch})")

    if not HAVE_CAPTURE:
        skip("capture test (install requirements-capture.txt)")
        return
    import shutil

    shutil.rmtree(TEST_DIR / "captures", ignore_errors=True)
    os.environ["EARS_CAPTURE_DIR"] = str(TEST_DIR / "captures")
    try:
        st = json.loads(server.capture(action="start", name="selftest", max_minutes=1, hotkey="ctrl+alt+f12"))
    except Exception as exc:
        check(False, f"capture start (needs an audio output device): {exc}")
        return
    time.sleep(1.5)
    mk = json.loads(server.capture(action="mark", label="level 2"))
    time.sleep(1.0)
    st2 = json.loads(server.capture(action="status"))
    out = server.capture(action="stop")
    rep = json.loads(out[0])
    print(f"    capture: started {st['elapsed']}, marker at {mk['at']}, status {st2['elapsed']}, hotkey: {st2['hotkey']}")
    check(st2["recording"] and os.path.isfile(rep["recording"]) and rep["recording"].endswith(".flac")
          and rep["markers"] == 1 and len(out) == 2,
          f"capture start/mark/status/stop records a FLAC with the marker ({rep['length']})")
    os.environ.pop("EARS_CAPTURE_DIR")


def run_unit_tests():
    from ears import server
    from ears.paths import resolve_audio_path

    print("\n== 1. Synthetic track: 130 BPM, G major, quiet 8-bar intro, sub, one clipped peak")
    wav = TEST_DIR / "synthetic_130_Gmaj.wav"
    synth(wav)
    t0 = time.time()
    res = server.listen(str(wav))  # no bpm given: tempo must be detected
    took = time.time() - t0
    rep = json.loads(res[0])
    print(f"  listen() took {took:.1f} s; JSON {len(res[0])} chars; content = {[type(c).__name__ for c in res]}")
    show = {k: rep[k] for k in ("tempo", "key", "loudness", "stereo", "flags", "sections")}
    print("  report (excerpt):")
    for k, v in show.items():
        print(f"    {k}: {json.dumps(v, ensure_ascii=False)}")
    print("    bars4 rows: " + "; ".join(f"{r[0]}: rms {r[2]} sub {r[3]} mid {r[5]}" for r in rep["bars4"]["rows"]))

    bpm = rep["tempo"]["bpm"]
    check(abs(bpm - 130) <= 1, f"tempo ~130 (got {bpm}, beat_conf {rep['tempo']['beat_conf']})")
    check(str(rep["key"]["best"]).startswith("G major"), f"key G major (got {rep['key']['best']}, conf {rep['key']['conf']})")
    starts = [int(s.split()[1].split("-")[0]) for s in rep["sections"][1:]]
    check(any(8 <= s <= 10 for s in starts), f"section change near bar 9 (changes at bars {starts})")
    check(rep["loudness"]["clipped_samples"] > 0 and any("CLIPPING" in f for f in rep["flags"]),
          f"clipping flagged ({rep['loudness']['clipped_samples']} samples, first {rep['loudness'].get('first_clip')})")
    check("bar 17" in str(rep["loudness"].get("first_clip")), "clip located at bar 17")
    check(any(f.startswith("Sub much louder") for f in rep["flags"]), "sub-heavy balance flagged")
    lufs, st = rep["loudness"]["lufs"], rep["loudness"]["short_term_max_lufs"]
    check(lufs is not None and st is not None and st >= lufs and -30 < lufs < 0, f"loudness sane (LUFS {lufs}, short-term max {st})")
    print(f"  (beat offset vs grid on a perfectly gridded kick: {server._analysis(str(wav), None).tempo['offset_ms']} ms)")

    print("\n== 2. Given bpm + zoom range")
    res = server.listen(str(wav), bpm=130, start_sec=12, end_sec=22)
    rep2 = json.loads(res[0])
    check(len(res) == 3 and rep2["tempo"]["source"] == "given", f"zoom image returned ({rep2['range']['span']})")

    print("\n== 3. MP3 + FLAC decoding")
    x, _ = sf.read(str(wav), dtype="float32")
    for ext, kw in ((".flac", {}), (".mp3", {"format": "MP3", "subtype": "MPEG_LAYER_III"})):
        p = TEST_DIR / f"synthetic_130_Gmaj{ext}"
        try:
            sf.write(str(p), x * 0.9, SR, **kw)
        except Exception as exc:
            check(False, f"could write test {ext}: {exc}")
            continue
        r = json.loads(server.listen(str(p), bpm=130)[0])
        check(r["format"]["sr"] == SR and abs(r["tempo"]["detected"] - 130) <= 1, f"{ext} decodes (detected {r['tempo']['detected']} BPM)")

    print("\n== 4. compare(): B has +sub, +chord, no clip, 4 extra bars")
    wav_b = TEST_DIR / "synthetic_130_Gmaj_v2.wav"
    synth(wav_b, extra_bars=4, sub_gain=1.6, chord_gain=1.4, clip=False)
    res = server.compare(str(wav), str(wav_b), bpm=130)
    cmp_ = json.loads(res[0])
    for line in cmp_["summary"]:
        print("    - " + line)
    s = " ".join(cmp_["summary"])
    check("longer" in s, "compare notices B is longer")
    check("clipping fixed" in s, "compare notices clipping fixed")
    nb = cmp_["numbers"]
    dsub = nb["b"]["bands"]["sub"] - nb["a"]["bands"]["sub"]
    dlow = nb["b"]["bands"]["low"] - nb["a"]["bands"]["low"]
    check(dsub - dlow >= 1, f"compare sees more sub relative to the rest (sub {dsub:+} dB vs low {dlow:+} dB)")

    print("\n== 5. Folder guard")
    for bad in (r"C:\Windows\win.ini", str(TEST_DIR / ".." / "secret.wav"), r"relative\path.wav"):
        try:
            resolve_audio_path(bad)
            check(False, f"rejects {bad}")
        except (PermissionError, ValueError) as exc:
            check(True, f"rejects {bad} -> {type(exc).__name__}")


async def run_mcp_test():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    print("\n== 6. End-to-end over MCP stdio (how Claude will call it)")
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "ears_server.py")],
                                   env={**os.environ, "EARS_ALLOWED_ROOTS": ROOTS,
                                        "EARS_CAPTURE_DIR": str(TEST_DIR / "captures")})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = sorted(t.name for t in (await session.list_tools()).tools)
            check(tools == ["capture", "compare", "describe", "expect", "find_audio", "levels", "listen",
                            "listen_live", "notes", "sounds_like", "stems"], f"tools listed: {tools}")
            t0 = time.time()
            res = await session.call_tool("listen", {"path": str(TEST_DIR / "synthetic_130_Gmaj.wav"), "bpm": 130})
            kinds = [c.type for c in res.content]
            img = next((c for c in res.content if c.type == "image"), None)
            check(not getattr(res, "is_error", False) and kinds[:2] == ["text", "image"],
                  f"listen over MCP returns text + image ({kinds}, {time.time() - t0:.1f} s, image {len(img.data) * 3 // 4 // 1024 if img else 0} KB)")
            res = await session.call_tool("listen", {"path": r"C:\Windows\win.ini"})
            msg = res.content[0].text if res.content else ""
            check(getattr(res, "is_error", False), f"outside path rejected over MCP: {msg[:90]}")
            if HAVE_NOTES:
                res = await session.call_tool("notes", {"path": str(TEST_DIR / "synthetic_chords_130.wav"),
                                                        "bpm": 130, "stem": "mix"})
                txt = res.content[0].text if res.content else ""
                check(not getattr(res, "is_error", False) and [c.type for c in res.content] == ["text", "image"]
                      and "| Am | F | C | G | x2" in txt, "notes over MCP returns the chord chart + piano roll")
            res = await session.call_tool("find_audio", {"query": "synthetic", "folder": str(TEST_DIR)})
            check("synthetic_130_Gmaj" in res.content[0].text, "find_audio finds the test files")
            if not HAVE_CAPTURE:
                return
            res = await session.call_tool("capture", {"action": "start", "name": "mcp selftest", "hotkey": "ctrl+alt+f11"})
            ok = not getattr(res, "is_error", False)
            await asyncio.sleep(1.5)
            await session.call_tool("capture", {"action": "mark", "label": "level 2"})
            await asyncio.sleep(1.0)
            res = await session.call_tool("capture", {"action": "stop"})
            txt = res.content[0].text if res.content else ""
            check(ok and not getattr(res, "is_error", False) and '"markers":1' in txt
                  and [c.type for c in res.content] == ["text", "image"],
                  f"capture start/mark/stop over MCP ({txt[:80]})")


def _have(*mods: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in mods)


# Optional tools: their checks are skipped when their packages aren't installed.
HAVE_NOTES = _have("basic_pitch", "pretty_midi", "onnxruntime")
HAVE_CLAP = _have("torch", "transformers")
HAVE_CAPTURE = _have("soundcard", "pycaw")


def skip(what: str):
    print(f"\n  SKIP  {what}")


def main():
    run_unit_tests()
    if HAVE_NOTES:
        run_notes_test()
    else:
        skip("notes tests (install requirements-notes.txt + basic-pitch)")
    if HAVE_CLAP:
        run_describe_test()
        run_time_tests()
    else:
        skip("describe / expect / listen_live / sounds_like tests (install requirements-stems.txt + requirements-describe.txt)")
    run_game_tests()
    asyncio.run(run_mcp_test())
    passed = sum(ok for ok, _ in RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
