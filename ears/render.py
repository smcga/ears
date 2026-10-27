"""Spectrogram / activity images. Uses matplotlib's object API (thread-safe, no pyplot)."""
from __future__ import annotations

import io
import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .analysis import Analysis, Grid, mmss

TICK_HZ = [30, 60, 100, 250, 500, 1000, 2000, 5000, 10000, 16000]
SECTION_COLOR = "#38e0ff"
MAX_BYTES = 900_000


def _hz(f: float) -> str:
    return f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"


def _save(fig: Figure, out_path) -> tuple[bytes, str, str]:
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data, fmt = buf.getvalue(), "png"
    if len(data) > MAX_BYTES:
        buf = io.BytesIO()
        fig.savefig(buf, format="jpeg", pil_kwargs={"quality": 88})
        data, fmt = buf.getvalue(), "jpeg"
    path = str(out_path.with_suffix(".png" if fmt == "png" else ".jpg"))
    with open(path, "wb") as fh:
        fh.write(data)
    return data, fmt, path


def _spec_axes(ax, spec: dict, x0: float, x1: float, vrange: float = 80.0):
    img = spec["img_db"]
    lo, hi = math.log10(spec["fmin"]), math.log10(spec["fmax"])
    vmax = float(np.percentile(img, 99.7))
    ax.imshow(img.T, origin="lower", aspect="auto", extent=[x0, x1, lo, hi], cmap="magma",
              vmin=vmax - vrange, vmax=vmax, interpolation="antialiased")
    ticks = [f for f in TICK_HZ if spec["fmin"] <= f <= spec["fmax"]]
    ax.set_yticks([math.log10(f) for f in ticks])
    ax.set_yticklabels([_hz(f) for f in ticks])
    ax.set_ylabel("Hz")
    for f, name in ((60, "sub"), (250, "low"), (2000, "mid")):
        ax.axhline(math.log10(f), color="white", lw=0.6, ls=":", alpha=0.45)
    ax.set_ylim(lo, hi)


def _time_axis(ax, grid: Grid, step_sec: float):
    sec = ax.secondary_xaxis("top", functions=(lambda b: (b - 1) * grid.bar_sec, lambda s: s / grid.bar_sec + 1))
    sec.xaxis.set_major_locator(MultipleLocator(step_sec))
    sec.xaxis.set_major_formatter(FuncFormatter(lambda s, _: mmss(s)))
    sec.tick_params(labelsize=10, colors="#444")


def _bar_lines(axes, first: int, last: int, every: int, **kw):
    for ax in axes:
        for bar in range(first, last + 1, every):
            ax.axvline(bar, **kw)


def overview(a: Analysis, out_path) -> tuple[bytes, str, str]:
    g = a.grid
    x0, x1 = 1.0, 1.0 + a.duration / g.bar_sec
    fig = Figure(figsize=(18, 7), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.06, left=0.04, right=0.995, top=0.86, bottom=0.08)
    axl = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=axl)

    t, db = a.beat_curve
    xb = 1 + t / g.bar_sec
    axl.fill_between(xb, -60, db, step="post", color="#3b6ea5", alpha=0.85, lw=0)
    top = max(-1.0, float(np.max(db)) + 2)
    axl.set_ylim(max(-60.0, top - 45), top)
    axl.set_ylabel("RMS dB", fontsize=9)
    axl.tick_params(labelbottom=False, labelsize=9)
    axl.grid(axis="y", alpha=0.3)

    _spec_axes(ax, a.spec, x0, x1)
    ax.set_xlim(x0, x1)
    every = 8 if g.n_bars <= 200 else 16
    _bar_lines([ax], 1, g.n_bars + 1, every, color="white", lw=0.8, alpha=0.4)
    _bar_lines([axl], 1, g.n_bars + 1, every, color="#888", lw=0.6, alpha=0.5)
    ax.set_xticks(list(range(1, g.n_bars + 2, every)))
    ax.tick_params(labelsize=11)
    ax.set_xlabel("bar (FL playlist numbering; white lines every %d bars)" % every, fontsize=11)
    _time_axis(axl, g, 30 if a.duration > 120 else 10)

    lo, hi = ax.get_ylim()
    for s in a.sections[1:]:
        for axx in (ax, axl):
            axx.axvline(s["start"], color=SECTION_COLOR, lw=1.6, ls="--", alpha=0.95)
        ax.text(s["start"] + 0.3, hi - 0.04 * (hi - lo), str(s["start"]), color=SECTION_COLOR, fontsize=11,
                fontweight="bold", va="top")
    for f, name in ((40, "sub"), (120, "low"), (700, "mid"), (6000, "high")):
        if a.spec["fmin"] < f < a.spec["fmax"]:
            ax.text(x0 + 0.004 * (x1 - x0), math.log10(f), name, color="white", fontsize=9, alpha=0.75, va="center")

    k = a.key.get("best") or "?"
    lufs = a.loud["lufs"]
    fig.suptitle(f"{a.name}  |  {g.bpm:g} BPM ({a.tempo['source']})  |  key {k} [{a.key.get('conf')}]  |  "
                 f"{lufs:.1f} LUFS, peak {a.loud['peak_dbfs']:.1f} dBFS   (cyan dashed = detected section changes)",
                 fontsize=13, y=0.985)
    return _save(fig, out_path)


def zoom(a: Analysis, rng: dict, out_path) -> tuple[bytes, str, str]:
    g, spec = a.grid, rng["_spec"]
    x0, x1 = 1 + spec["t0"] / g.bar_sec, 1 + spec["t1"] / g.bar_sec
    fig = Figure(figsize=(18, 6), dpi=100)
    ax = fig.add_axes([0.04, 0.1, 0.955, 0.76])
    _spec_axes(ax, spec, x0, x1)
    ax.set_xlim(x0, x1)
    b0, b1 = math.floor(x0), math.ceil(x1)
    for bar in range(b0, b1 + 1):
        ax.axvline(bar, color="white", lw=1.0, alpha=0.6)
        for q in (0.25, 0.5, 0.75):
            ax.axvline(bar + q, color="white", lw=0.5, alpha=0.2, ls=":")
    nb = b1 - b0
    step = 1 if nb <= 32 else 2 if nb <= 64 else 4
    ax.set_xticks([b for b in range(b0, b1 + 1, step) if x0 <= b <= x1])
    ax.tick_params(labelsize=11)
    ax.set_xlabel("bar (solid = bar, dotted = beat)", fontsize=11)
    _time_axis(ax, g, 5 if spec["t1"] - spec["t0"] <= 60 else 15)
    fig.suptitle(f"{a.name}  zoom {rng['span']}  |  rms {rng['rms_db']} dB, peak {rng['peak_dbfs']} dBFS",
                 fontsize=13, y=0.985)
    return _save(fig, out_path)


def compare_pair(a: Analysis, b: Analysis, out_path) -> tuple[bytes, str, str]:
    nb = max(a.grid.n_bars, b.grid.n_bars)
    fig = Figure(figsize=(18, 9), dpi=100)
    gs = fig.add_gridspec(2, 1, hspace=0.18, left=0.04, right=0.995, top=0.93, bottom=0.07)
    every = 8 if nb <= 200 else 16
    ax0 = None
    for i, (x, tag) in enumerate(((a, "A"), (b, "B"))):
        ax = fig.add_subplot(gs[i], sharex=ax0) if ax0 else fig.add_subplot(gs[i])
        ax0 = ax0 or ax
        x1 = 1 + x.duration / x.grid.bar_sec
        _spec_axes(ax, x.spec, 1.0, x1)
        _bar_lines([ax], 1, nb + 1, every, color="white", lw=0.8, alpha=0.4)
        for s in x.sections[1:]:
            ax.axvline(s["start"], color=SECTION_COLOR, lw=1.6, ls="--")
        ax.set_title(f"{tag}: {x.name}  ({x.loud['lufs']:.1f} LUFS, {x.grid.bpm:g} BPM, {x.grid.n_bars} bars)",
                     fontsize=12, loc="left")
        ax.set_xticks(list(range(1, nb + 2, every)))
        ax.tick_params(labelsize=10)
    ax0.set_xlim(1, 1 + nb)
    ax.set_xlabel("bar (FL playlist numbering)", fontsize=11)
    return _save(fig, out_path)


def _surprise_axes(ax, steps, n_bars: int):
    s = np.array([st.surprise if st.bar > 4 else 0.0 for st in steps])
    colors = ["#c62828" if v >= 2.5 else "#ef9a9a" if v >= 1.8 else "#90a4ae" for v in s]
    ax.bar(np.arange(1, len(s) + 1) + 0.5, np.minimum(s, 10), width=0.9, color=colors)
    ax.axhline(2.5, color="#c62828", lw=0.8, ls="--")
    ax.axhline(1.0, color="#555", lw=0.6, ls=":")
    ax.set_xlim(1, n_bars + 1)
    ax.set_ylim(0, max(4.0, float(min(10, s.max() if len(s) else 1)) + 0.5))
    ax.set_ylabel("surprise", fontsize=10)


def structure_map(a: Analysis, E: np.ndarray, steps, out_path):
    """Learned self-similarity (which bars sound alike to CLAP) + per-bar surprise underneath."""
    n = a.grid.n_bars
    G = E @ E.T
    fig = Figure(figsize=(13, 14.5), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[10, 2.2], hspace=0.08, left=0.07, right=0.98, top=0.94, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    axs = fig.add_subplot(gs[1], sharex=ax)
    lo = float(np.percentile(G, 5))
    ax.imshow(G, cmap="magma", vmin=lo, vmax=1.0, extent=[1, n + 1, n + 1, 1], interpolation="nearest", aspect="auto")
    ax.set_xlim(1, n + 1)
    every = 8 if n <= 200 else 16
    ticks = list(range(1, n + 2, every))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=10, labelbottom=False)
    for s in a.sections[1:]:
        ax.axvline(s["start"], color=SECTION_COLOR, lw=1.0, ls="--", alpha=0.8)
        ax.axhline(s["start"], color=SECTION_COLOR, lw=1.0, ls="--", alpha=0.8)
    ax.set_ylabel("bar", fontsize=11)
    ax.set_title("Learned similarity (CLAP, per bar): bright = these two bars sound alike. Bright blocks off the "
                 "diagonal = a part comes back.", fontsize=11, loc="left")
    _surprise_axes(axs, steps, n)
    axs.set_xticks(ticks)
    axs.set_xlabel("bar (FL numbering).  Surprise: each bar guessed from earlier bars only; red = broke expectation",
                   fontsize=10)
    for s in a.sections[1:]:
        axs.axvline(s["start"], color=SECTION_COLOR, lw=1.0, ls="--", alpha=0.8)
    fig.suptitle(f"{a.name}: structure as a learned model hears it ({a.grid.bpm:g} BPM grid)", fontsize=13, y=0.985)
    return _save(fig, out_path)


def live_view(a: Analysis, steps, heard_to_bar: int, out_path, forecast_bars: tuple[int, int] | None = None):
    """Spectrogram of what has been heard so far (future greyed out) + surprise per heard bar."""
    import matplotlib

    g, spec = a.grid, a.spec
    n = g.n_bars
    t_end = min(a.duration, g.t_of_bar(heard_to_bar + 1))
    img = spec["img_db"].astype(float).copy()
    ncols = img.shape[0]
    cut = int(np.ceil(ncols * t_end / spec["t_end"]))
    heard = img[:cut]
    vmax = float(np.percentile(heard, 99.7)) if cut else 0.0
    img[cut:] = np.nan
    cmap = matplotlib.colormaps["magma"].copy()
    cmap.set_bad("#e6e6e6")
    fig = Figure(figsize=(18, 7), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1.4], hspace=0.08, left=0.04, right=0.995, top=0.9, bottom=0.08)
    ax, axs = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    lo, hi = np.log10(spec["fmin"]), np.log10(spec["fmax"])
    ax.imshow(img.T, origin="lower", aspect="auto", extent=[1, 1 + spec["t_end"] / g.bar_sec, lo, hi], cmap=cmap,
              vmin=vmax - 80, vmax=vmax, interpolation="antialiased")
    ticks = [f for f in TICK_HZ if spec["fmin"] <= f <= spec["fmax"]]
    ax.set_yticks([np.log10(f) for f in ticks])
    ax.set_yticklabels([_hz(f) for f in ticks])
    ax.set_xlim(1, n + 1)
    every = 8 if n <= 200 else 16
    ax.set_xticks(list(range(1, n + 2, every)))
    ax.tick_params(labelbottom=False, labelsize=10)
    for bar in range(1, n + 2, every):
        ax.axvline(bar, color="white", lw=0.7, alpha=0.4)
    ax.axvline(heard_to_bar + 1, color="#00e5ff", lw=2.0)
    ax.text(heard_to_bar + 1.5, (lo + hi) / 2, "not heard yet", color="#777", fontsize=14, va="center")
    if forecast_bars:
        ax.axvspan(forecast_bars[0], forecast_bars[1] + 1, color="#00e5ff", alpha=0.08, lw=0)
    _surprise_axes(axs, steps, n)
    axs.set_xticks(list(range(1, n + 2, every)))
    axs.set_xlabel("bar (FL numbering).  Surprise of each heard bar vs a guess from the bars before it", fontsize=10)
    fig.suptitle(f"{a.name}: heard up to bar {heard_to_bar} ({mmss(t_end)} of {mmss(a.duration)})", fontsize=13, y=0.985)
    return _save(fig, out_path)


BLACK_KEYS = {1, 3, 6, 8, 10}


def piano_roll(a: Analysis, tracks: list, chords: list[str], b0: int, b1: int, title: str, out_path):
    """Top: whole-song piano roll. Bottom: close-up of bars b0..b1 with chord names.

    tracks: list of (label, notes, color); notes have .start/.end (s), .pitch (MIDI), .vel (0-1).
    """
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Rectangle

    from .transcribe import fl_name

    g = a.grid
    nb = g.n_bars
    fig = Figure(figsize=(18, 9), dpi=100)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.22, left=0.045, right=0.995, top=0.93, bottom=0.06)
    ax, ad = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    allp = np.array([n.pitch for _, ns, _ in tracks for n in ns]) if any(ns for _, ns, _ in tracks) else np.array([60])
    lo, hi = int(np.percentile(allp, 1)) - 2, int(np.percentile(allp, 99)) + 2

    def pitch_ticks(axx, plo, phi, every_note):
        ticks = [p for p in range(plo, phi + 1) if every_note or p % 12 == 0]
        axx.set_yticks(ticks)
        axx.set_yticklabels([fl_name(p) for p in ticks], fontsize=8 if every_note else 10)
        for p in range(plo, phi + 1):
            if p % 12 in BLACK_KEYS:
                axx.axhspan(p - 0.5, p + 0.5, color="#000", alpha=0.045, lw=0)

    # Overview
    lw = max(1.0, min(4.0, 220 / max(1, hi - lo)))
    for label, ns, color in tracks:
        segs = [((1 + n.start / g.bar_sec, n.pitch), (1 + n.end / g.bar_sec, n.pitch)) for n in ns if lo <= n.pitch <= hi]
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=0.85, label=f"{label} ({len(ns)} notes)"))
    ax.set_xlim(1, nb + 1)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    pitch_ticks(ax, lo, hi, False)
    every = 8 if nb <= 200 else 16
    ax.set_xticks(list(range(1, nb + 2, every)))
    for bar in range(1, nb + 2, every):
        ax.axvline(bar, color="#999", lw=0.6, alpha=0.6)
    for s in a.sections[1:]:
        ax.axvline(s["start"], color=SECTION_COLOR, lw=1.4, ls="--")
    ax.axvspan(b0, b1 + 1, color="#ffc400", alpha=0.18, lw=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax.set_title("whole track (bars, FL numbering; cyan = section changes; yellow = close-up below)", fontsize=10, loc="left")

    # Close-up
    t0, t1 = g.t_of_bar(b0), g.t_of_bar(b1 + 1)
    inview = [(lbl, [n for n in ns if n.end > t0 and n.start < t1], c) for lbl, ns, c in tracks]
    dp = np.array([n.pitch for _, ns, _ in inview for n in ns] or [60])
    dlo, dhi = int(np.percentile(dp, 3)) - 1, int(np.percentile(dp, 97)) + 1  # stray ghost notes fall off-screen
    if dhi - dlo > 48:
        med = int(np.median(dp))
        dlo, dhi = max(dlo, med - 24), min(dhi, med + 24)
    for _, ns, color in inview:
        rects = [Rectangle((1 + n.start / g.bar_sec, n.pitch - 0.4), (n.end - n.start) / g.bar_sec, 0.8) for n in ns]
        alphas = [0.35 + 0.65 * min(1.0, n.vel / 0.8) for n in ns]
        pc = PatchCollection(rects, facecolors=[to_rgba(color, al) for al in alphas], edgecolor="#222", linewidth=0.4)
        ad.add_collection(pc)
    ad.set_xlim(b0, b1 + 1)
    ad.set_ylim(dlo - 0.5, dhi + 2.6)
    pitch_ticks(ad, dlo, dhi, dhi - dlo <= 40)
    for bar in range(b0, b1 + 2):
        ad.axvline(bar, color="#444", lw=1.0)
        for q in (0.25, 0.5, 0.75):
            ad.axvline(bar + q, color="#888", lw=0.5, ls=":")
    ad.set_xticks(list(range(b0, b1 + 2)))
    for bar in range(b0, b1 + 1):
        if bar - 1 < len(chords) and chords[bar - 1] not in ("-",):
            parts = chords[bar - 1].split(" ")
            for k, ch in enumerate(parts):
                ad.text(bar + 0.03 + k * 0.5, dhi + 1.7, ch, fontsize=12, fontweight="bold", va="center", color="#b00020",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    ad.set_xlabel("bar (solid = bar, dotted = beat); red = chord inferred per bar", fontsize=10)
    fig.suptitle(title, fontsize=13, y=0.985)
    return _save(fig, out_path)


def stem_activity(names: list[str], bar_db: list[np.ndarray], grid: Grid, title: str, out_path):
    nb = max(len(v) for v in bar_db)
    M = np.full((len(names), nb), -80.0)
    for i, v in enumerate(bar_db):
        M[i, : len(v)] = v
    vmax = float(M.max())
    fig = Figure(figsize=(18, 3.6), dpi=100)
    ax = fig.add_axes([0.06, 0.2, 0.93, 0.62])
    ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmax - 45, vmax=vmax, interpolation="nearest",
              extent=[1, nb + 1, len(names) - 0.5, -0.5])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    every = 8 if nb <= 200 else 16
    for bar in range(1, nb + 2, every):
        ax.axvline(bar, color="white", lw=0.8, alpha=0.5)
    ax.set_xticks(list(range(1, nb + 2, every)))
    ax.set_xlabel("bar (brighter = louder; per-bar RMS of each stem)", fontsize=11)
    fig.suptitle(title, fontsize=13, y=0.97)
    return _save(fig, out_path)


def _dev_color(d):
    if d is None:
        return "#999999"
    d = abs(d)
    return "#2e9d4f" if d < 1.5 else ("#e0a000" if d < 3 else "#d43c2c")


def level_timeline(plot: dict, out_path):
    """Short-term loudness over a capture, one shaded span per level with its music level and the gain to match."""
    t, st, segs, ref = plot["t"], plot["st"], plot["segs"], plot["ref"]
    dur = float(t[-1]) if len(t) else 1.0
    fig = Figure(figsize=(18, 5.2), dpi=100)
    ax = fig.add_axes([0.05, 0.13, 0.93, 0.72])
    y = np.clip(st, -60, 0)
    ax.plot(t, y, color="#5a6b8c", lw=0.7, alpha=0.9)
    for i, s in enumerate(segs):
        b0, b1 = s["_b"]
        x0, x1 = b0 * 0.1, b1 * 0.1
        ax.axvspan(x0, x1, color="#000000", alpha=0.035 if i % 2 else 0.0)
        ax.axvline(x0, color="#333333", lw=0.8, alpha=0.6)
        if s.get("bed") is not None:
            c = _dev_color(s.get("change_db"))
            ax.hlines(s["bed"], x0, x1, color=c, lw=3)
            d = s.get("change_db")
            txt = f"{s['label']}\n{s['bed']:.1f}" + ("" if d is None else f" ({-d + 0.0:+.1f})")
            ax.text((x0 + x1) / 2, min(s["bed"] + 2.5, -1), txt, ha="center", va="bottom", fontsize=10, color=c,
                    fontweight="bold", clip_on=True)
    if ref is not None:
        ax.axhline(ref, color="#222222", ls="--", lw=1)
        ax.text(dur * 0.995, ref - 0.6, f"reference {ref:.1f}", va="top", ha="right", fontsize=9)
    lo = min(-45.0, float(np.percentile(y[y > -60], 2)) - 3 if np.any(y > -60) else -45.0)
    beds = [s["bed"] for s in segs if s.get("bed") is not None]
    ax.set_ylim(lo, min(0.0, max(beds, default=-20.0) + 9))
    ax.set_xlim(0, dur)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _: mmss(s)))
    ax.set_ylabel("short-term LUFS (3 s)")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{plot['title']}: music level per level (thick line; in brackets = dB above(+)/below(-) the reference)",
                 fontsize=12, y=0.97)
    return _save(fig, out_path)


def level_bars(plot: dict, out_path):
    segs, ref = plot["segs"], plot["ref"]
    names = [s["label"] for s in segs]
    vals = [s["bed"] if s.get("bed") is not None else -70 for s in segs]
    fig = Figure(figsize=(12, max(2.5, 0.45 * len(segs) + 1.2)), dpi=100)
    ax = fig.add_axes([0.32, 0.12, 0.63, 0.78])
    ypos = np.arange(len(segs))
    ax.barh(ypos, [v + 70 for v in vals], left=-70, color=[_dev_color(s.get("change_db")) for s in segs])
    for i, s in enumerate(segs):
        if s.get("bed") is not None:
            ax.text(s["bed"] + 0.3, i, f"{s['bed']:.1f}" + ("" if s.get("change_db") is None else f" ({-s['change_db'] + 0.0:+.1f})"),
                    va="center", fontsize=9)
    if ref is not None:
        ax.axvline(ref, color="#222222", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    good = [v for v in vals if v > -70]
    ax.set_xlim(min(good, default=-40) - 6, max(good, default=-10) + 5)
    ax.set_xlabel("music level (median short-term LUFS)")
    fig.suptitle(plot["title"], fontsize=12)
    return _save(fig, out_path)
