"""Listening as prediction: before each bar, guess it from the bars already heard, then measure the surprise.

Every guess is "this bar will be like an earlier bar". Experts:
  - last bar           (nothing changes)
  - 4 / 8 / 16 back    (the phrase repeats)
  - context match      (the last 4 bars sounded like a moment earlier on, so what followed that comes next)
Experts that have been right lately get more weight. Surprise = how wrong the weighted guess was, relative
to how wrong guesses have typically been so far in this track (1 = business as usual).
Only bars before the current one are ever used (bar embeddings are per-bar audio, no look-ahead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import Analysis, mmss

EXPERT_NAMES = ("last bar", "4 bars back", "8 bars back", "16 bars back", "context match")
LAGS = (1, 4, 8, 16)
SURPRISE_EVENT = 2.5


@dataclass
class Step:
    bar: int  # 1-based
    expert: str  # most trusted expert when the guess was made
    ref: int | None  # 1-based bar that expert copied
    ctx_sim: float  # context-match similarity (0-1)
    surprise: float
    d_learned: float
    d_level: float
    nearest: int | None  # most similar earlier bar (learned), 1-based
    nearest_sim: float


def measured(a: Analysis) -> np.ndarray:
    """Per-bar measured features in dB: rms, sub, low, mid, high, drums."""
    return np.column_stack([a.bar_rms_db, a.bar_band_db, a.bar_perc_db]).astype(float)


def _context_match(G: np.ndarray, t: int, k: int = 4) -> tuple[int | None, float]:
    """Best earlier bar j (< t, 0-based) whose k preceding bars resemble the k bars before t."""
    if t < k + 1:
        return None, 0.0
    best, best_s = None, -9.0
    for j in range(k, t):
        s = float(np.mean([G[t - 1 - i, j - 1 - i] for i in range(k)]))
        s += 0.01 if (t - j) % 8 == 0 else 0.0
        if s > best_s + 1e-9 or (abs(s - best_s) <= 1e-9 and j > (best or -1)):
            best, best_s = j, s
    return best, best_s


def run(E: np.ndarray, M: np.ndarray, upto: int | None = None) -> list[Step]:
    """Causal pass over bars 1..upto. Never looks at bar t when predicting bar t."""
    n = len(E) if upto is None else min(upto, len(E))
    G = E @ E.T
    err = np.ones(len(EXPERT_NAMES))
    hist_L, hist_M = [], []
    steps: list[Step] = []
    for t in range(n):
        refs: list[int | None] = [t - lag if t - lag >= 0 else None for lag in LAGS]
        j, csim = _context_match(G, t)
        refs.append(j)
        avail = [i for i, r in enumerate(refs) if r is not None]
        if not avail:
            steps.append(Step(t + 1, "none", None, 0.0, 1.0, 0.0, 0.0, None, 0.0))
            continue
        w = np.exp(-3.0 * err[avail])
        w /= w.sum()
        pe = sum(wi * E[refs[i]] for wi, i in zip(w, avail))
        pe /= max(np.linalg.norm(pe), 1e-9)
        pm = sum(wi * M[refs[i]] for wi, i in zip(w, avail))
        d_l = 1.0 - float(pe @ E[t])
        d_m = float(np.mean(np.abs(np.clip(pm, -80, 0) - np.clip(M[t], -80, 0))))
        med_l = max(float(np.median(hist_L)) if len(hist_L) >= 4 else 0.0, 0.02)
        med_m = max(float(np.median(hist_M)) if len(hist_M) >= 4 else 0.0, 1.5)
        surprise = 0.5 * (d_l / med_l + d_m / med_m)
        top = avail[int(np.argmax(w))]
        near, nsim = (None, 0.0)
        if t > 0:
            jj = int(np.argmax(G[t, :t]))
            near, nsim = jj + 1, float(G[t, jj])
        steps.append(Step(t + 1, EXPERT_NAMES[top], refs[top] + 1, csim if j is not None else 0.0,
                          surprise, d_l, d_m, near, nsim))
        hist_L.append(d_l)
        hist_M.append(d_m)
        for i in avail:  # update each expert's track record with its own normalised error
            e_i = 0.5 * ((1.0 - float(E[refs[i]] @ E[t])) / med_l
                         + float(np.mean(np.abs(np.clip(M[refs[i]], -80, 0) - np.clip(M[t], -80, 0)))) / med_m)
            err[i] = 0.7 * err[i] + 0.3 * min(e_i, 5.0)
    return steps


# ----------------------------------------------------------------------------- words

def grid_note(bar: int) -> str:
    if (bar - 1) % 16 == 0:
        return "on the 16-bar grid"
    if (bar - 1) % 8 == 0:
        return "on the 8-bar grid"
    if (bar - 1) % 4 == 0:
        return "on the 4-bar grid"
    off = (bar - 1) % 8
    near = bar - off if off <= 4 else bar + (8 - off)
    return f"off the phrase grid ({abs(bar - near)} bar{'s' if abs(bar - near) > 1 else ''} {'after' if near < bar else 'before'} bar {near})"


def what_changed(M: np.ndarray, t: int) -> str:
    """Plain words for bar t (0-based) vs bar t-1."""
    if t == 0:
        return "start"
    d = M[t] - M[t - 1]
    names = ("level", "sub", "bass", "mids", "highs", "drums")
    ins, outs = [], []
    for i in range(1, 6):
        if d[i] >= 6 and M[t, i] > -60:
            ins.append(names[i])
        elif d[i] <= -6 and M[t - 1, i] > -60:
            outs.append(names[i])
    parts = []
    if ins:
        parts.append("+".join(ins) + " in/up")
    if outs:
        parts.append("+".join(outs) + " out/down")
    if abs(d[0]) >= 1.5:
        parts.append(f"level {d[0]:+.0f} dB")
    elif not parts:
        parts.append("level steady")
    return ", ".join(parts)


def explain(step: Step, M: np.ndarray) -> str:
    t = step.bar - 1
    exp = f"expected {step.expert}" + (f" (like bar {step.ref})" if step.ref else "")
    if step.nearest and step.nearest_sim >= 0.9:
        heard = f"sounded like bar {step.nearest} ({step.nearest_sim:.2f}): a return"
    elif step.nearest:
        heard = f"sounded new (closest earlier bar {step.nearest}, {step.nearest_sim:.2f})"
    else:
        heard = ""
    got = what_changed(M, t)
    if got == "level steady":
        got = "a change in the sound itself (same level, no part in/out)"
    return f"{exp}; got {got}; {heard}; {grid_note(step.bar)}".replace("; ;", ";")


WARMUP = 4  # first bars: nothing heard yet, so surprise isn't meaningful


def events(steps: list[Step], thr: float = SURPRISE_EVENT) -> list[tuple[int, int, float]]:
    """Merge consecutive surprising bars into (first_bar_index, n_bars_to_settle, peak) events."""
    out, i = [], WARMUP
    while i < len(steps):
        if steps[i].surprise >= thr:
            j, peak = i, steps[i].surprise
            while j + 1 < len(steps) and steps[j + 1].surprise >= 1.8:
                j += 1
                peak = max(peak, steps[j].surprise)
            out.append((i, j - i + 1, peak))
            i = j + 1
        else:
            i += 1
    return out


def summary(a: Analysis, steps: list[Step], M: np.ndarray, top_n: int = 8) -> dict:
    g = a.grid
    s = np.array([st.surprise for st in steps])
    s[:WARMUP] = np.nan
    ev = events(steps)
    order = sorted(sorted(ev, key=lambda e: -e[2])[:top_n])
    moments = []
    for i, settle, peak in order:
        st = steps[i]
        tail = f" (took {settle} bars to settle)" if settle > 1 else ""
        moments.append(f"bar {st.bar} ({mmss(g.t_of_bar(st.bar))}) surprise {peak:.1f}{tail}: {explain(st, M)}")

    # Predictable stretches: runs of >= 8 bars with surprise < 1.2
    runs, start = [], None
    for i, v in enumerate(list(np.nan_to_num(s, nan=99.0)) + [99]):
        if v < 1.2 and start is None:
            start = i
        elif v >= 1.2 and start is not None:
            if i - start >= 8:
                refs = [steps[k].ref for k in range(start, i) if steps[k].expert == "context match" and steps[k].ref]
                how = ""
                if refs:
                    how = f", replaying earlier material (e.g. bar {start + 1} ~ bar {refs[0]})"
                runs.append(f"bars {start + 1}-{i}: played out as expected{how}")
            start = None

    on_grid = sum(1 for i, _, _ in ev if (steps[i].bar - 1) % 4 == 0)
    return {
        "how_to_read": ("Surprise 1 = as predictable as this track usually is; 2.5+ = a real break in expectation. "
                        f"Each bar was guessed only from bars before it; bars 1-{WARMUP} are warm-up (null)."),
        "surprise_per_bar": [None if np.isnan(v) else round(float(v), 1) for v in s],
        "big_moments": moments,
        "predictable_stretches": runs[:8],
        "timing": (f"{on_grid}/{len(ev)} surprises land on the 4-bar grid; the rest break the phrase grid"
                   if ev else "no big surprises"),
        "surprise_events_at_bars": [steps[i].bar for i, _, _ in ev],
        "mean_surprise": round(float(np.nanmean(s)), 2) if np.isfinite(s).any() else None,
    }


def forecast(a: Analysis, E: np.ndarray, M: np.ndarray, steps: list[Step], nxt: int, n: int) -> str:
    """Words for what the predictor expects in bars nxt..nxt+n-1, using only bars < nxt."""
    t = nxt - 1
    if t <= 0:
        return "Nothing heard yet, so no expectation."
    # Context match on the newest bars: find earlier j whose 4 predecessors match bars t-4..t-1 (0-based)
    best, best_s = None, -9.0
    if t >= 6:
        GG = E[:t] @ E[:t].T
        for jj in range(4, t - 1):  # leave at least 1 heard bar after the match to project from
            sc = float(np.mean([GG[t - 1 - i, jj - 1 - i] for i in range(4)])) + (0.01 if (t - jj) % 8 == 0 else 0.0)
            if sc > best_s:
                best, best_s = jj, sc
    changes = [steps[i].bar for i, _, _ in events(steps[:t])]
    last_change = changes[-1] if changes else 1
    lens = [last_change + L for L in (8, 16, 32) if last_change + L > nxt - 1]
    phrase = (f"Current part started around bar {last_change}; on 8/16/32-bar phrasing the next change would come "
              f"at bar {', '.join(map(str, lens[:3]))}.") if lens else ""
    if best is not None and best_s >= 0.85:
        m = min(n, t - best)
        fut = M[best: best + m]
        dl = float(fut[:, 0].mean() - M[t - 1, 0])
        return (f"The last 4 bars sound like bars {best - 3}-{best} ({best_s:.2f}), so it expects bars {nxt}-{nxt + m - 1} "
                f"to go like bars {best + 1}-{best + m} did: {what_changed(np.vstack([M[t - 1:t], fut[-1:]]), 1)} "
                f"by then, average level {dl:+.0f} dB vs now. {phrase}").strip()
    return (f"Nothing earlier matches the last 4 bars closely (best {best_s:.2f}), so it expects more of the same "
            f"for now. {phrase}").strip()
