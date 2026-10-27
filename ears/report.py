"""Turns Analysis numbers into a compact, plain-English report (JSON-ready dicts)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .analysis import BAND_NAMES, Analysis, _power_mean_db, mmss

LIMITS = (
    "This is measurement, not hearing: ears can't judge vibe, groove, sound choice or whether the mix 'works'. "
    "Band levels are energy, not perceived loudness. Key/tempo are estimates (check low-confidence ones by ear). "
    "Section labels describe level/band changes, not musical intent. Bar numbers assume the file starts at FL bar 1."
)
UNITS = ("dB = RMS dBFS of the full stereo signal; bands sub<60, low 60-250, mid 250-2k, high>2k Hz "
         "(band 'low' = kick body + bassline); drums = percussive energy (HPSS)")


def _r(x, nd=1):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), nd) if nd else int(round(float(x)))


def blocks4(a: Analysis) -> list[dict]:
    """Power-averaged 4-bar blocks."""
    out = []
    for s in range(0, a.grid.n_bars, 4):
        e = min(a.grid.n_bars, s + 4)
        w = a.bar_n[s:e].astype(float)
        band = _power_mean_db(a.bar_band_db[s:e], w)
        out.append({"s": s + 1, "e": e, "rms": float(_power_mean_db(a.bar_rms_db[s:e], w)),
                    **{b: float(band[i]) for i, b in enumerate(BAND_NAMES)},
                    "corr": float(np.nanmean(a.bar_corr[s:e])) if a.bar_corr is not None else None})
    return out


def _bars_label(s: int, e: int) -> str:
    return f"{s}-{e}" if e > s else str(s)


def main_part_mask(a: Analysis) -> np.ndarray:
    """Bars within 6 dB of the loudest 4-bar block: 'the loud part'."""
    bl = blocks4(a)
    top = max(b["rms"] for b in bl)
    mask = np.zeros(a.grid.n_bars, bool)
    for b in bl:
        if b["rms"] >= top - 6:
            mask[b["s"] - 1: b["e"]] = True
    return mask


def band_balance(a: Analysis, mask: np.ndarray | None = None) -> dict:
    m = mask if mask is not None else np.ones(a.grid.n_bars, bool)
    band = _power_mean_db(a.bar_band_db[m], a.bar_n[m].astype(float))
    return {b: float(band[i]) for i, b in enumerate(BAND_NAMES)}


def mix_flags(a: Analysis) -> list[str]:
    f: list[str] = []
    L, g = a.loud, a.grid

    # Clipping / peaks
    if L["clip_samples"]:
        where = ""
        if L["first_clip_t"] is not None:
            where = f", first at {mmss(L['first_clip_t'])} (bar {g.bar_at(L['first_clip_t'])})"
        if L["clip_runs"]:
            f.append(f"CLIPPING: {L['clip_runs']} hard-clip run(s) (3+ samples stuck at full scale), "
                     f"{L['clip_samples']} samples in total{where}. Audible distortion likely there.")
        else:
            over = f"; {L['over_0dbfs']} go over 0 dBFS (MP3 decode overs)" if L["over_0dbfs"] else ""
            f.append(f"{L['clip_samples']} isolated samples hit 0 dBFS{over}{where}: probably a limiter ceiling at 0 dB; "
                     "set the ceiling to -1 dB for exports.")
    if L["true_peak_dbtp"] > -1.0:
        f.append(f"True peak {L['true_peak_dbtp']:+.1f} dBTP: above the usual -1 dBTP streaming ceiling; may distort after encoding.")
    lufs = L["lufs"]
    if math.isfinite(lufs):
        if lufs > -7:
            f.append(f"Very loud ({lufs:.1f} LUFS): heavily limited; check transients/kick punch aren't squashed.")
        elif lufs < -16:
            f.append(f"Quiet ({lufs:.1f} LUFS): fine for a pre-master; club masters are typically -6 to -9 LUFS.")

    # Band balance in the loud part
    bb = band_balance(a, main_part_mask(a))
    sub_mid, low_mid, high_mid = bb["sub"] - bb["mid"], bb["low"] - bb["mid"], bb["high"] - bb["mid"]
    if sub_mid > 6:
        f.append(f"Sub much louder than mids (+{sub_mid:.0f} dB in the loud sections): huge on a club rig, "
                 "may sound thin on laptop/phone speakers; check the sub/kick level.")
    elif sub_mid < -12:
        f.append(f"Light sub ({sub_mid:.0f} dB vs mids in the loud sections): may lack weight on big systems.")
    if low_mid > 8:
        f.append(f"60-250 Hz is heavy (+{low_mid:.0f} dB vs mids): possible boom/mud in kick body or bass.")
    if high_mid < -18:
        f.append(f"Dark top end (highs {high_mid:.0f} dB vs mids).")
    elif high_mid > 2:
        f.append(f"Bright top end (highs +{high_mid:.0f} dB vs mids): check hats/air for harshness.")

    # Level jumps between 4-bar blocks
    bl = blocks4(a)
    jumps = []
    for p, c in zip(bl[:-1], bl[1:]):
        d = c["rms"] - p["rms"]
        if abs(d) >= 6 and max(p["rms"], c["rms"]) > -50:
            jumps.append((abs(d), d, c["s"]))
    for _, d, bar in sorted(jumps, reverse=True)[:3]:
        f.append(f"Level jump {d:+.0f} dB into bar {bar} (fine if it's a drop/breakdown, check if not).")

    # Stereo
    st = a.stereo
    if st:
        corr = st["corr"]
        if corr < 0:
            f.append(f"Stereo correlation is negative ({corr:.2f}): parts will cancel in mono. Check phase.")
        bs = st.get("band_side_db")
        if bs is not None:
            low_side, hi_side = float(max(bs[0], bs[1])), float(max(bs[2], bs[3]))
            if low_side > -12:
                f.append(f"Low end is wide (side {low_side:.0f} dB vs mid below 250 Hz): sub/bass may lose power in mono/clubs; "
                         "consider mono below ~120 Hz.")
            if hi_side > -3:
                f.append(f"Mids/highs are very wide (side {hi_side:.0f} dB vs mid, i.e. L and R barely correlated there): "
                         "sounds big, but wide synths can thin out or vanish in mono. Check a mono fold-down.")
            elif hi_side < -20:
                f.append(f"Mids/highs are nearly mono (side {hi_side:.0f} dB vs mid): narrow stereo image.")
        if a.bar_corr is not None:
            bl_neg = [b for b in bl if b["corr"] is not None and b["corr"] < 0 and b["rms"] > -50]
            if bl_neg:
                worst = min(bl_neg, key=lambda b: b["corr"])
                f.append(f"Negative L/R correlation in bars {_bars_label(worst['s'], worst['e'])} ({worst['corr']:.2f}): "
                         "those bars will partly cancel in mono.")

    # Housekeeping
    if L["dc"] > 0.005:
        f.append(f"DC offset ({L['dc']:.3f}): add a high-pass / DC filter.")
    if a.edges["lead"] > 0.3:
        f.append(f"Starts with {a.edges['lead']:.1f} s of silence: if the bounce didn't start at bar 1, bar numbers shift.")
    t = a.tempo
    if t["n_beats"] >= 16 and t["conf"] >= 0.3:
        if t["grid_fit"] < 0.6:
            f.append(f"Beats drift against a {g.bpm:g} BPM grid starting at 0:00 (fit {t['grid_fit']:.2f}): "
                     "tempo may differ/vary, so bar numbers may be off.")
        elif abs(t["offset_ms"]) > 60:
            f.append(f"Beats sit {t['offset_ms']:+d} ms off the bar grid: the bounce may not start exactly on bar 1 "
                     "(or the tracker locked to off-beats).")
    if t["source"] == "given" and t["detected"]:
        ratio = t["detected"] / g.bpm
        if not any(abs(ratio - r) < 0.015 for r in (0.5, 1.0, 2.0, 2 / 3, 1.5)):
            f.append(f"You said {g.bpm:g} BPM but onsets suggest ~{t['detected']:g} BPM: double-check the project tempo.")
    return f


def section_lines(a: Analysis) -> list[str]:
    g = a.grid
    out = []
    for s in a.sections:
        t0, t1 = g.t_of_bar(s["start"]), min(a.duration, g.t_of_bar(s["end"] + 1))
        out.append(f"bars {_bars_label(s['start'], s['end'])} ({mmss(t0)}-{mmss(t1)}): {s['label']}")
    return out


def bars4_table(a: Analysis) -> dict:
    rows = []
    for b in blocks4(a):
        rows.append([_bars_label(b["s"], b["e"]), mmss(a.grid.t_of_bar(b["s"])), _r(b["rms"])] +
                    [_r(b[k], 0) for k in BAND_NAMES])
    return {"cols": ["bars", "t", "rms", "sub", "low", "mid", "high"], "rows": rows}


def tempo_block(a: Analysis) -> dict:
    t, g = a.tempo, a.grid
    out = {"bpm": g.bpm, "source": t["source"]}
    if t["source"] == "given":
        out["detected"] = t["detected"]
    out["beat_conf"] = t["conf"]
    out["grid"] = f"bar 1 = 0:00, 1 bar = {g.bar_sec:.3f} s, {g.n_bars} bars"
    out["grid_fit"] = t["grid_fit"]
    if abs(t["offset_ms"]) > 25:
        out["beat_offset_ms"] = t["offset_ms"]
    return out


def loudness_block(a: Analysis) -> dict:
    L, g = a.loud, a.grid
    out = {
        "lufs": _r(L["lufs"]),
        "short_term_max_lufs": _r(L["st_max"]),
        "peak_dbfs": _r(L["peak_dbfs"]),
        "true_peak_dbtp": _r(L["true_peak_dbtp"]),
        "clipped_samples": L["clip_samples"],
    }
    if L["clip_samples"]:
        out["clip_runs"] = L["clip_runs"]
        if L["first_clip_t"] is not None:
            out["first_clip"] = f"{mmss(L['first_clip_t'])} (bar {g.bar_at(L['first_clip_t'])})"
    bb = band_balance(a, main_part_mask(a))
    out["band_balance_loud_part_db"] = {k: _r(v, 0) for k, v in bb.items()}
    return out


def stereo_block(a: Analysis) -> dict | str:
    st = a.stereo
    if not st:
        return "mono file"
    out = {"corr": _r(st["corr"], 2), "side_vs_mid_db": _r(st["side_vs_mid_db"])}
    if st.get("band_side_db") is not None:
        out["side_vs_mid_by_band_db"] = {b: _r(v, 0) for b, v in zip(BAND_NAMES, st["band_side_db"])}
    worst = None
    if a.bar_corr is not None:
        bl = [b for b in blocks4(a) if b["corr"] is not None and np.isfinite(b["corr"]) and b["rms"] > -50]
        if bl:
            worst = min(bl, key=lambda b: b["corr"])
            out["lowest_corr"] = f"bars {_bars_label(worst['s'], worst['e'])}: {worst['corr']:.2f}"
    out["mono_ok"] = not (st["corr"] < 0 or (worst is not None and worst["corr"] < 0))
    return out


def key_block(a: Analysis) -> dict:
    return {k: v for k, v in a.key.items() if not k.startswith("_")}


def listen_report(a: Analysis, image_path: str | None = None, rng: dict | None = None) -> dict:
    rep = {
        "file": a.name,
        "format": {"duration": f"{mmss(a.duration)} ({a.duration:.1f} s)", "sr": a.sr, "ch": a.channels},
        "tempo": tempo_block(a),
        "key": key_block(a),
        "loudness": loudness_block(a),
        "stereo": stereo_block(a),
        "flags": mix_flags(a) or ["Nothing stands out from the measurements."],
        "sections": section_lines(a),
        "bars4": bars4_table(a),
    }
    if rng:
        rep["range"] = {k: v for k, v in rng.items() if not k.startswith("_")}
    rep["units"] = UNITS
    rep["limits"] = LIMITS
    if image_path:
        rep["image_saved"] = image_path
    return rep


# ----------------------------------------------------------------------------- compare

def _delta(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}"


def _section_starts(a: Analysis) -> list[int]:
    return [s["start"] for s in a.sections[1:]]


def _range_stats(a: Analysis, s: int, e: int) -> dict | None:
    """Power-averaged stats for bars s..e (1-based, inclusive), clipped to the track."""
    s0, e0 = max(1, s), min(a.grid.n_bars, e)
    if s0 > e0:
        return None
    sl = slice(s0 - 1, e0)
    w = a.bar_n[sl].astype(float)
    band = _power_mean_db(a.bar_band_db[sl], w)
    return {"rms": float(_power_mean_db(a.bar_rms_db[sl], w)), **{k: float(band[i]) for i, k in enumerate(BAND_NAMES)}}


def compare_report(a: Analysis, b: Analysis, image_path: str | None = None) -> dict:
    summ: list[str] = []
    A, B = a.loud, b.loud

    # Length / tempo / key
    dd = b.duration - a.duration
    if abs(dd) >= 0.5:
        summ.append(f"B is {mmss(abs(dd))} {'longer' if dd > 0 else 'shorter'} ({mmss(a.duration)} vs {mmss(b.duration)}; "
                    f"{a.grid.n_bars} vs {b.grid.n_bars} bars).")
    else:
        summ.append(f"Same length ({mmss(a.duration)}, {a.grid.n_bars} bars).")
    if a.grid.bpm != b.grid.bpm:
        summ.append(f"Tempo changed: {a.grid.bpm:g} -> {b.grid.bpm:g} BPM.")
    ka, kb = a.key.get("best"), b.key.get("best")
    if ka != kb:
        summ.append(f"Key estimate changed: {ka} -> {kb} (confidence {a.key.get('conf')} / {b.key.get('conf')}).")

    # Loudness / peaks
    if math.isfinite(A["lufs"]) and math.isfinite(B["lufs"]):
        d = B["lufs"] - A["lufs"]
        if abs(d) >= 0.3:
            summ.append(f"B is {abs(d):.1f} LU {'louder' if d > 0 else 'quieter'} ({A['lufs']:.1f} -> {B['lufs']:.1f} LUFS).")
        else:
            summ.append(f"Integrated loudness about the same ({A['lufs']:.1f} vs {B['lufs']:.1f} LUFS).")
    dp = B["true_peak_dbtp"] - A["true_peak_dbtp"]
    if abs(dp) >= 0.3:
        summ.append(f"True peak {A['true_peak_dbtp']:+.1f} -> {B['true_peak_dbtp']:+.1f} dBTP.")
    if bool(A["clip_samples"]) != bool(B["clip_samples"]) or abs(A["clip_samples"] - B["clip_samples"]) > 10:
        summ.append(f"Clipped samples {A['clip_samples']} -> {B['clip_samples']}"
                    + (" (clipping fixed)." if A["clip_samples"] and not B["clip_samples"] else
                       " (new clipping!)." if B["clip_samples"] and not A["clip_samples"] else "."))

    # Overall band balance
    ba, bb_ = band_balance(a), band_balance(b)
    top = max(max(ba.values()), max(bb_.values()))
    bd = {k: bb_[k] - ba[k] for k in BAND_NAMES if max(ba[k], bb_[k]) > top - 30}  # skip near-empty bands
    moved = [f"{k} {_delta(v)} dB" for k, v in bd.items() if abs(v) >= 1.0]
    if moved:
        summ.append("Whole-track band levels in B: " + ", ".join(moved) + ".")

    # Stereo
    if a.stereo and b.stereo:
        dc = b.stereo["corr"] - a.stereo["corr"]
        dw = b.stereo["side_vs_mid_db"] - a.stereo["side_vs_mid_db"]
        if abs(dc) >= 0.05 or abs(dw) >= 1.5:
            summ.append(f"Stereo: correlation {a.stereo['corr']:.2f} -> {b.stereo['corr']:.2f}, "
                        f"side level {_delta(dw)} dB ({'wider' if dw > 0 else 'narrower'}).")

    # Arrangement shift: do A's later section changes reappear in B a fixed number of bars earlier/later?
    sa, sb = _section_starts(a), _section_starts(b)
    shift, shift_from, pairs = 0, None, []
    offs = Counter(y - x for x in sa for y in sb if y != x and abs(y - x) <= 32)
    if offs:
        d, _ = offs.most_common(1)[0]
        pairs = [(x, y) for x in sa for y in sb if abs(y - (x + d)) <= 1 and all(abs(x - z) > 1 for z in sb)]
        if len(pairs) >= 2:
            shift, shift_from = d, min(y for _, y in pairs)
            summ.append(f"Arrangement: from bar {shift_from} on, B's sections are A's moved {shift:+d} bars "
                        f"({', '.join(f'A {x}->B {y}' for x, y in pairs)}), so about {abs(shift)} bars were "
                        f"{'cut' if shift < 0 else 'added'} before that point. Bar-by-bar deltas below are re-aligned for it.")

    # Per-4-bar changes (re-aligned after a detected shift), merged into spans with the same kind of change
    rows, sigs = [], []
    for s in range(1, b.grid.n_bars + 1, 4):
        e = min(b.grid.n_bars, s + 3)
        off = shift if shift_from and s >= shift_from else 0
        x, y = _range_stats(a, s - off, e - off), _range_stats(b, s, e)
        label = _bars_label(s, e) + (f" (A {_bars_label(s - off, e - off)})" if off else "")
        if x is None:
            rows.append([label, "only in B"])
            sigs.append(None)
            continue
        d = {k: y[k] - x[k] for k in ("rms",) + BAND_NAMES}
        if max(abs(v) for v in d.values()) >= 1.0:
            rows.append([label, _r(d["rms"])] + [_r(d[k], 0) for k in BAND_NAMES])
        btop = max(max(x[k], y[k]) for k in BAND_NAMES)
        sig = tuple(sorted((k, v > 0) for k, v in d.items()
                           if k != "rms" and abs(v) >= 2.5 and max(x[k], y[k]) > btop - 30))
        sigs.append((sig, d, s, e) if sig else None)
    a_last = b.grid.n_bars - (shift if shift_from else 0)
    if a_last < a.grid.n_bars:
        rows.append([f"A {a_last + 1}-{a.grid.n_bars}", "only in A"])

    spans = []
    for sg in sigs:
        if sg and spans and spans[-1]["sig"] == sg[0] and spans[-1]["e"] + 1 == sg[2]:
            spans[-1]["e"] = sg[3]
            spans[-1]["ds"].append(sg[1])
        elif sg:
            spans.append({"sig": sg[0], "s": sg[2], "e": sg[3], "ds": [sg[1]]})
    for sp in spans:
        sp["mag"] = max(abs(np.mean([d[k] for d in sp["ds"]])) for k, _ in sp["sig"]) * len(sp["ds"]) ** 0.5
    for sp in sorted(spans, key=lambda s: -s["mag"])[:5]:
        parts = [f"{k} {_delta(np.mean([d[k] for d in sp['ds']]), 0)} dB" for k, _ in sp["sig"]]
        summ.append(f"Bars {_bars_label(sp['s'], sp['e'])}: " + ", ".join(parts) + " in B.")

    # Structure (changes not explained by the shift)
    moved_a = {x for x, _ in pairs} if shift_from else set()
    moved_b = {y for _, y in pairs} if shift_from else set()
    new = [s for s in sb if s not in moved_b and all(abs(s - t) > 1 for t in sa)]
    gone = [s for s in sa if s not in moved_a and all(abs(s - t) > 1 for t in sb)]
    if new or gone:
        msg = "Structure: "
        if new:
            msg += "new section change(s) at bar " + ", ".join(map(str, new)) + " in B"
        if gone:
            msg += ("; " if new else "") + "A's change(s) at bar " + ", ".join(map(str, gone)) + " no longer detected"
        summ.append(msg + ".")
    elif not shift_from:
        summ.append("Structure: same section boundaries detected.")

    fa, fb = set(mix_flags(a)), set(mix_flags(b))
    flags_new = sorted(fb - fa)
    flags_gone = sorted(fa - fb)

    out = {
        "a": a.name, "b": b.name,
        "summary": summ,
        "flags_new_in_b": flags_new or None,
        "flags_gone_in_b": flags_gone or None,
        "numbers": {
            "a": {"dur": round(a.duration, 1), "bpm": a.grid.bpm, "key": ka, "lufs": _r(A["lufs"]),
                  "tp": _r(A["true_peak_dbtp"]), "clip": A["clip_samples"], "corr": _r(a.stereo["corr"], 2) if a.stereo else None,
                  "bands": {k: _r(v, 0) for k, v in ba.items()}},
            "b": {"dur": round(b.duration, 1), "bpm": b.grid.bpm, "key": kb, "lufs": _r(B["lufs"]),
                  "tp": _r(B["true_peak_dbtp"]), "clip": B["clip_samples"], "corr": _r(b.stereo["corr"], 2) if b.stereo else None,
                  "bands": {k: _r(v, 0) for k, v in bb_.items()}},
        },
        "sections_a": section_lines(a),
        "sections_b": section_lines(b),
        "delta_4bar": {"cols": ["bars", "d_rms", "d_sub", "d_low", "d_mid", "d_high"],
                        "note": "B minus A, dB; rows within +-1 dB omitted", "rows": rows},
    }
    if image_path:
        out["image_saved"] = image_path
    return {k: v for k, v in out.items() if v is not None}
