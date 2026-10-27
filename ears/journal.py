"""Persistent auditory events, bounded clips and personal preference examples."""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .perception import AudioWindow


@dataclass(frozen=True)
class JournalConfig:
    root: Path
    hours: float = 24
    max_bytes: int = 512 * 1024 * 1024
    max_events: int = 20000
    segment_seconds: float = 12
    block_seconds: float = 2
    device: str | None = None
    classifier: bool = True
    whisper: bool = True
    clap: bool = True
    captions: bool = False

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())
        if not 0 < self.hours <= 8760 or self.max_bytes < 1024 or self.max_events < 1:
            raise ValueError("Journal retention and budget must be positive (retention <= 365 days).")
        if not 0.1 <= self.block_seconds <= self.segment_seconds <= 30:
            raise ValueError("Require 0.1 <= block_seconds <= segment_seconds <= 30.")
        ratio = self.segment_seconds / self.block_seconds
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("segment_seconds must be a whole multiple of block_seconds (normally 2 seconds).")

    @classmethod
    def from_env(cls):
        default = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "state")) / "ears" / "journal"
        flag = lambda key, default: os.environ.get(key, str(int(default))).lower() in {"1", "true", "yes"}
        return cls(
            Path(os.environ.get("EARS_JOURNAL_DIR", default)),
            hours=float(os.environ.get("EARS_JOURNAL_HOURS", "24")),
            max_bytes=int(float(os.environ.get("EARS_JOURNAL_MAX_MB", "512")) * 1024 * 1024),
            max_events=int(os.environ.get("EARS_JOURNAL_MAX_EVENTS", "20000")),
            segment_seconds=float(os.environ.get("EARS_JOURNAL_SEGMENT_SECONDS", "12")),
            device=os.environ.get("EARS_JOURNAL_DEVICE") or None,
            classifier=flag("EARS_JOURNAL_CLASSIFIER", True),
            whisper=flag("EARS_JOURNAL_WHISPER", True),
            clap=flag("EARS_JOURNAL_CLAP", True),
            captions=flag("EARS_JOURNAL_CAPTIONS", False),
        )


def parse_time(value: str | float | None, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Timestamp must be finite.")
        return number
    except (TypeError, ValueError):
        pass
    text = str(value).strip().lower()
    if text == "now":
        return time.time()
    match = re.fullmatch(r"(?:last\s+)?(\d+(?:\.\d+)?)\s*(s|m|h|d)(?:\s+ago)?", text)
    if match:
        return time.time() - float(match[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match[2]]
    stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("ISO timestamps need a timezone, e.g. 2026-10-27T12:00:00Z.")
    return stamp.timestamp()


def iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds")


class Journal:
    def __init__(self, config: JournalConfig | None = None):
        self.config = config or JournalConfig.from_env()
        self.root = self.config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self.clips = self.root / "clips"
        self.clips.mkdir(exist_ok=True)
        if self.clips.resolve().parent != self.root:
            raise PermissionError("Journal clips directory must remain inside the configured store.")
        self.db = sqlite3.connect(self.root / "journal.sqlite3", timeout=15)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=15000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL,
                kind TEXT NOT NULL, text TEXT NOT NULL, tags TEXT NOT NULL,
                apps TEXT NOT NULL, details TEXT NOT NULL, clip TEXT,
                bytes INTEGER NOT NULL DEFAULT 0, vector BLOB, vector_model TEXT
            );
            CREATE INDEX IF NOT EXISTS event_time ON events(start);
            CREATE TABLE IF NOT EXISTS ratings (
                event_id TEXT PRIMARY KEY, rating REAL NOT NULL, note TEXT NOT NULL,
                vector BLOB, vector_model TEXT, measured TEXT NOT NULL, created REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS daemon_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def state(self, **values):
        if values:
            with self.db:
                for key, value in values.items():
                    self.db.execute("INSERT OR REPLACE INTO daemon_state VALUES (?,?)", (key, json.dumps(value)))
        return {r["key"]: json.loads(r["value"]) for r in self.db.execute("SELECT * FROM daemon_state")}

    def _clip_path(self, name: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}\.flac", name):
            raise ValueError("Invalid journal clip name.")
        path = (self.clips / name).resolve()
        if path.parent != self.clips.resolve():
            raise PermissionError("Journal clip escaped its store.")
        return path

    def retention_hours(self):
        # Readers use the writer's policy even when the MCP client only sets EARS_JOURNAL_DIR.
        row = self.db.execute("SELECT value FROM daemon_state WHERE key='retention_hours'").fetchone()
        return float(json.loads(row[0])) if row else self.config.hours

    def add(self, start: float, end: float, result: dict, samples=None, sr=48000, apps=()) -> str:
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("Invalid event interval.")
        event_id = uuid.uuid4().hex
        clip, size = None, 0
        if samples is not None:
            clip = event_id + ".flac"
            target = self._clip_path(clip)
            temporary = target.with_suffix(".part")
            try:
                sf.write(temporary, samples, sr, format="FLAC", subtype="PCM_24")
                temporary.replace(target)
                size = target.stat().st_size
            finally:
                temporary.unlink(missing_ok=True)
        vector = result.get("vector")
        blob = None if vector is None else np.asarray(vector, dtype="<f4").tobytes()
        details = {k: v for k, v in result.items() if k != "vector"}
        try:
            with self.db:
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    event_id, start, end, result.get("kind", "activity"), result.get("text", ""),
                    json.dumps(result.get("tags", [])), json.dumps(sorted(set(apps))),
                    json.dumps(details, allow_nan=False), clip, size, blob, result.get("vector_model"),
                ))
        except Exception:
            if clip:
                self._clip_path(clip).unlink(missing_ok=True)
            raise
        self.prune()
        return event_id

    def prune(self, now: float | None = None):
        cutoff = (time.time() if now is None else now) - self.config.hours * 3600
        rows = list(self.db.execute("SELECT id,clip,bytes,start FROM events ORDER BY start DESC"))
        total = sum(r["bytes"] for r in rows)
        remove = {r["id"] for i, r in enumerate(rows) if r["start"] < cutoff or i >= self.config.max_events}
        for row in rows:
            if row["id"] in remove:
                total -= row["bytes"]
        for row in reversed(rows):
            if total <= self.config.max_bytes:
                break
            if row["id"] not in remove:
                remove.add(row["id"])
                total -= row["bytes"]
        # Commit metadata removal before deleting files; a crash can leave an orphan, never a live row with a
        # partially overwritten clip. The daemon reconciles orphans while it owns the writer lock.
        with self.db:
            self.db.executemany("DELETE FROM events WHERE id=?", ((key,) for key in remove))
        for row in rows:
            if row["id"] in remove and row["clip"]:
                self._clip_path(row["clip"]).unlink(missing_ok=True)

    def reconcile(self):
        referenced = {r[0] for r in self.db.execute("SELECT clip FROM events WHERE clip IS NOT NULL")}
        for path in self.clips.iterdir():
            if (path.is_file() and re.fullmatch(r"[0-9a-f]{32}\.(flac|part)", path.name)
                    and ((path.suffix == ".flac" and path.name not in referenced) or path.suffix == ".part")):
                if path.resolve().parent == self.clips.resolve():
                    path.unlink()

    def _public(self, row):
        return {
            "event_id": row["id"], "start": iso(row["start"]), "end": iso(row["end"]),
            "start_unix": row["start"], "end_unix": row["end"], "kind": row["kind"],
            "text": row["text"], "tags": json.loads(row["tags"]), "active_apps": json.loads(row["apps"]),
            "details": json.loads(row["details"]),
            "clip_available": bool(row["clip"] and self._clip_path(row["clip"]).is_file()),
            "perceive_source": "journal:" + row["id"],
        }

    def search(self, since=None, until=None, query="", limit=50):
        now = time.time()
        lower, upper = parse_time(since, now - 3600), parse_time(until, now)
        if lower > upper:
            raise ValueError("since must precede until.")
        lower = max(lower, now - self.retention_hours() * 3600)
        if lower > upper:
            return []
        # User search is a literal substring, never SQL or a query-language expression.
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self.db.execute("""SELECT * FROM events WHERE end >= ? AND start <= ?
            AND (text || tags || apps || details) LIKE ? ESCAPE '\\'
            ORDER BY start DESC LIMIT ?""", (lower, upper, "%" + escaped + "%", min(max(int(limit), 1), 200)))
        return [self._public(row) for row in rows]

    def get(self, event_id: str):
        row = self.db.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        if row is None or row["start"] < time.time() - self.retention_hours() * 3600:
            raise ValueError("Journal event is missing or expired under the retention policy.")
        return row

    def window(self, event_id: str, label="audio") -> AudioWindow:
        row = self.get(event_id)
        if not row["clip"] or not self._clip_path(row["clip"]).is_file():
            raise ValueError("This journal event has no retained audio clip.")
        # A concurrent retention pass may remove a clip first; report expiration rather than replacing
        # the requested sound with a different one.
        try:
            samples, sr = sf.read(self._clip_path(row["clip"]), dtype="float32", always_2d=True)
        except (OSError, sf.LibsndfileError) as exc:
            raise ValueError("The journal clip expired during retrieval; query the timeline again.") from exc
        return AudioWindow(label, samples, sr, "journal:" + event_id, row["start"], row["end"])

    def rate(self, event_id: str, rating: float, note=""):
        if not math.isfinite(rating) or not 1 <= rating <= 5:
            raise ValueError("Taste ratings must be between 1 and 5.")
        existing = self.db.execute("SELECT * FROM ratings WHERE event_id=?", (event_id,)).fetchone()
        if existing is not None:
            with self.db:
                self.db.execute("UPDATE ratings SET rating=?,note=?,created=? WHERE event_id=?",
                                (rating, note[:2000], time.time(), event_id))
            return {"event_id": event_id, "rating": rating, "note": note[:2000]}
        row = self.get(event_id)
        if row["kind"] != "music" and "music" not in json.loads(row["tags"]):
            raise ValueError("Taste ratings require a journal event labelled music.")
        details = json.loads(row["details"])
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO ratings VALUES (?,?,?,?,?,?,?)", (
                event_id, rating, note[:2000], row["vector"], row["vector_model"],
                json.dumps(details.get("measured", {})), time.time(),
            ))
            self.db.execute("DELETE FROM ratings WHERE event_id NOT IN (SELECT event_id FROM ratings ORDER BY created DESC LIMIT 5000)")
        return {"event_id": event_id, "rating": rating, "note": note[:2000]}

    def taste(self, event_id: str | None = None):
        ratings = list(self.db.execute("SELECT * FROM ratings ORDER BY created DESC LIMIT 5000"))
        out = {"rated_examples": len(ratings), "scale": "1=dislike, 5=love; personal preference, not quality"}
        if event_id is None:
            out["recent"] = [{"event_id": r["event_id"], "rating": r["rating"], "note": r["note"]}
                             for r in ratings[:20]]
            return out
        target = self.get(event_id)
        target_measured = json.loads(target["details"]).get("measured", {})
        neighbours = []
        if target["vector"]:
            v = np.frombuffer(target["vector"], dtype="<f4")
            for row in ratings:
                if row["event_id"] == event_id or not row["vector"] or row["vector_model"] != target["vector_model"]:
                    continue
                other = np.frombuffer(row["vector"], dtype="<f4")
                if other.shape != v.shape:
                    continue
                similarity = float(np.dot(v, other) / max(np.linalg.norm(v) * np.linalg.norm(other), 1e-9))
                measured = json.loads(row["measured"])
                differences = []
                for key, scale in (("rms_dbfs", 12), ("peak_dbfs", 12), ("crest_db", 8), ("stereo_correlation", 1)):
                    a, b = target_measured.get(key), measured.get(key)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        differences.append(abs(a - b) / scale)
                measurement_similarity = math.exp(-sum(differences) / len(differences)) if differences else None
                # Audio character dominates; measurements provide a modest, explicit tie-break rather than
                # treating loudness or stereo width as inherently good.
                weight = max(similarity, 0) ** 4 * (0.75 + 0.25 * measurement_similarity
                                                   if measurement_similarity is not None else 1)
                neighbours.append({"event_id": row["event_id"], "similarity": round(similarity, 4),
                                   "rating": row["rating"], "note": row["note"],
                                   "measurement_similarity": measurement_similarity,
                                   "preference_weight": weight, "measured": measured})
        neighbours.sort(key=lambda row: row["similarity"], reverse=True)
        out["neighbours"] = [n for n in neighbours[:5] if n["similarity"] > 0.2]
        out["target_measured"] = target_measured
        out["estimated_preference"] = None
        if len(out["neighbours"]) >= 3:
            weights = [n["preference_weight"] for n in out["neighbours"]]
            out["estimated_preference"] = round(sum(n["rating"] * w for n, w in zip(out["neighbours"], weights))
                                                 / sum(weights), 2)
        else:
            out["note"] = "Need at least three similar rated examples with matching CLAP vectors for an estimate."
        return out
