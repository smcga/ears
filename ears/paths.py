"""Path safety: ears only reads audio files under the allowed root folder(s).

Set EARS_ALLOWED_ROOTS to change it (separate several folders with ';').
"""
from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_ROOTS = str(Path.home() / "Music")
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg"}
PROJECT_DIR = Path(__file__).resolve().parent.parent


def allowed_roots() -> list[str]:
    raw = os.environ.get("EARS_ALLOWED_ROOTS", DEFAULT_ROOTS)
    roots = []
    for part in raw.split(";"):
        part = part.strip().strip('"')
        if part:
            roots.append(os.path.realpath(part))
    return roots


def _under(path: str, root: str) -> bool:
    p, r = os.path.normcase(path), os.path.normcase(root)
    try:
        return os.path.commonpath([p, r]) == r
    except ValueError:  # different drives
        return False


def _clean(raw: str) -> str:
    p = (raw or "").strip().strip('"').strip("'").strip()
    return os.path.expanduser(p)


def resolve_audio_path(raw: str) -> str:
    """Return the real path of an allowed audio file, or raise a clear error."""
    p = _clean(raw)
    roots = allowed_roots()
    if not p:
        raise ValueError(f"No path given. Example: {os.path.join(roots[0], 'My Project', 'my_track.mp3')}")
    if not os.path.isabs(p):
        raise ValueError(f"Please give a full path (e.g. {roots[0]}\\...\\track.mp3), not '{raw}'.")
    real = os.path.realpath(p)
    if not any(_under(real, r) for r in roots):
        raise PermissionError(
            f"ears may only read files under: {'; '.join(roots)}. '{p}' is outside that. "
            "(Change EARS_ALLOWED_ROOTS in the MCP config to allow other folders.)"
        )
    ext = os.path.splitext(real)[1].lower()
    if ext not in AUDIO_EXTS:
        raise ValueError(f"'{os.path.basename(real)}' isn't a supported audio file ({', '.join(sorted(AUDIO_EXTS))}).")
    if not os.path.isfile(real):
        hint = ""
        folder = os.path.dirname(real)
        if os.path.isdir(folder):
            near = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in AUDIO_EXTS)[:12]
            if near:
                hint = " Audio files in that folder: " + ", ".join(near)
        raise FileNotFoundError(f"File not found: {real}.{hint}")
    return real


def resolve_folder(raw: str | None) -> list[str]:
    if not raw:
        return allowed_roots()
    p = os.path.realpath(_clean(raw))
    if not any(_under(p, r) for r in allowed_roots()):
        raise PermissionError(f"ears may only look inside: {'; '.join(allowed_roots())}.")
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Folder not found: {p}")
    return [p]


def find_audio(query: str = "", folder: str | None = None, limit: int = 20, budget_sec: float = 10.0) -> dict:
    """Newest-first list of audio files whose name contains `query`.

    Breadth-first, with project folders searched before everything else (sample libraries can be huge).
    """
    from collections import deque

    q = (query or "").lower().strip()
    hits: list[tuple[float, int, str]] = []
    t0 = time.monotonic()
    truncated = False
    first, later = deque(), deque(resolve_folder(folder))
    projects_only = False
    while first or later:
        if time.monotonic() - t0 > budget_sec:
            truncated = True
            break
        if not first and hits and not folder and time.monotonic() - t0 > 0.2:
            projects_only = True  # project folders done and we have matches: skip the sample libraries
            break
        in_project = bool(first)
        d = first.popleft() if first else later.popleft()
        try:
            with os.scandir(d) as it:
                for e in it:
                    name = e.name.lower()
                    if e.is_dir(follow_symlinks=False):
                        if not name.endswith("_stems") and name != "backup":
                            (first if in_project or "project" in name else later).append(e.path)
                    elif os.path.splitext(name)[1] in AUDIO_EXTS and q in name:
                        st = e.stat()
                        hits.append((st.st_mtime, st.st_size, e.path))
        except OSError:
            continue
    hits.sort(reverse=True)
    rows = [
        f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(m))}  {s / 1e6:6.1f} MB  {p}"
        for m, s, p in hits[: max(1, min(limit, 100))]
    ]
    out = {"matches": len(hits), "newest_first": rows}
    if projects_only:
        out["note"] = "Searched project folders only (matches found); pass `folder` to search everywhere."
    elif truncated:
        out["note"] = f"Search stopped after {budget_sec:g} s (big folder); pass `folder` to search somewhere specific."
    return out


def output_dir() -> Path:
    d = Path(os.environ.get("EARS_OUTPUT_DIR", PROJECT_DIR / "output"))
    d.mkdir(parents=True, exist_ok=True)
    return d
