"""Deterministic journal integration tests; no audio hardware, downloads or paid APIs."""
from __future__ import annotations

import dataclasses
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from ears.journal import Journal, JournalConfig, parse_time
from ears.journal_daemon import Segmenter, WriterLock, daemon_status, run, write_config
from ears.journal_models import LocalModels
from ears.local_perception import QwenBackend, _NoRedirect
from ears.perception import PerceptionError


class FakeModels:
    def analyse(self, samples, sr):
        return {"kind": "music", "tags": ["music", "impact"], "text": "Someone shouted hello over music",
                "vector": np.array([1, 0, 0], dtype=np.float32), "vector_model": "test-clap@1",
                "measured": {"peak_dbfs": -6}}


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = JournalConfig(self.root, classifier=False, whisper=False, clap=False)
        self.now = time.time()
        self.audio = np.full((8000, 2), 0.1, dtype=np.float32)

    def tearDown(self):
        self.temp.cleanup()

    def add(self, journal, start=None, **kwargs):
        start = self.now - 10 if start is None else start
        return journal.add(start, start + 1, FakeModels().analyse(self.audio, 8000), self.audio, 8000, **kwargs)

    def test_persistent_search_transcripts_tags_and_apps(self):
        with Journal(self.config) as store:
            key = self.add(store, apps=["game.exe"])
        with Journal(self.config) as store:
            for query in ("shouted", "impact", "game.exe"):
                rows = store.search("1h ago", query=query)
                self.assertEqual(rows[0]["event_id"], key)
                self.assertTrue(rows[0]["clip_available"])
            self.assertFalse(store.search(query="' OR 1=1 --"))
            self.assertFalse(store.search(query="%"))
            self.assertFalse(store.search(query="_wildcard_"))
            self.assertFalse(store.search(since=str(self.now + 100), until=str(self.now + 200)))

    def test_clip_retrieval_and_traversal_rejection(self):
        with Journal(self.config) as store:
            key = self.add(store)
            window = store.window(key)
            self.assertEqual(window.source, "journal:" + key)
            self.assertAlmostEqual(float(window.samples[0, 0]), 0.1, places=5)
            self.assertFalse(window.samples.flags.writeable)
            with self.assertRaises(ValueError):
                store.window("../../secret")
            with self.assertRaises(ValueError):
                store._clip_path("../secret.flac")

    def test_time_and_row_retention(self):
        config = dataclasses.replace(self.config, max_events=2)
        with Journal(config) as store:
            old = self.add(store, self.now - 90000)
            with self.assertRaises(ValueError):
                store.get(old)
            keys = [self.add(store, self.now - i) for i in (30, 20, 10)]
            self.assertEqual([r[0] for r in store.db.execute("SELECT id FROM events ORDER BY start")], keys[1:])
            self.assertEqual(len(list(store.clips.glob("*.flac"))), 2)

    def test_reader_honours_daemon_retention_policy(self):
        config = dataclasses.replace(self.config, hours=48)
        with Journal(config) as store:
            key = self.add(store, self.now - 90000)
            store.state(retention_hours=48)
        with Journal(self.config) as reader:
            self.assertEqual(reader.get(key)["id"], key)
            self.assertTrue(reader.search(since="2d ago", query="shouted"))

    def test_disk_budget_keeps_newest_and_recovers_orphans(self):
        rng = np.random.default_rng(5)
        noise = rng.uniform(-0.5, 0.5, (8000, 2)).astype(np.float32)
        with Journal(self.config) as store:
            first = store.add(self.now - 20, self.now - 19, {"text": "noise"}, noise, 8000)
            size = store.get(first)["bytes"]
        with Journal(dataclasses.replace(self.config, max_bytes=size + 100)) as store:
            second = store.add(self.now - 10, self.now - 9, {"text": "noise"}, noise, 8000)
            self.assertIsNotNone(store.get(second))
            with self.assertRaises(ValueError):
                store.get(first)
            self.assertLessEqual(sum(p.stat().st_size for p in store.clips.glob("*.flac")), size + 100)
            orphan = store.clips / ("0" * 32 + ".flac")
            orphan.write_bytes(b"orphan")
            store.reconcile()
            self.assertFalse(orphan.exists())

    def test_taste_uses_matching_vectors_and_keeps_ratings_after_clip_expiration(self):
        with Journal(self.config) as store:
            keys = [self.add(store, self.now - i) for i in (50, 40, 30, 20)]
            for key, rating in zip(keys, (4, 5, 3)):
                store.rate(key, rating, "I like this groove")
            self.assertEqual(store.window(keys[0]).source, "journal:" + keys[0])
            taste = store.taste(keys[-1])
            self.assertEqual(taste["estimated_preference"], 4)
            self.assertEqual(len(taste["neighbours"]), 3)
            with self.assertRaises(ValueError):
                store.rate(keys[0], 8)
            store.prune(now=self.now + 90000)
            self.assertEqual(store.taste()["rated_examples"], 3)
            self.assertFalse(list(store.clips.glob("*.flac")))
            store.rate(keys[0], 2, "Changed my mind")
            self.assertEqual(store.taste()["recent"][0]["rating"], 2)

    def test_segmenter_silence_transients_and_gaps(self):
        with Journal(self.config) as store:
            segmenter = Segmenter(store, FakeModels(), sample_rate=8000)
            silent = np.zeros((16000, 2), dtype=np.float32)
            transient = silent.copy()
            transient[8000:8010] = 0.1
            start = self.now - 30
            segmenter.push(silent, start, start + 2)
            segmenter.push(transient, start + 2, start + 4, ["game.exe"])
            segmenter.push(silent, start + 8, start + 10)
            segmenter.finish()
            kinds = [r[0] for r in store.db.execute("SELECT kind FROM events ORDER BY start")]
            self.assertEqual(kinds, ["silence", "music", "gap", "silence"])
            row = store.search(query="shouted")[0]
            self.assertEqual(row["active_apps"], ["game.exe"])
            self.assertTrue(row["clip_available"])

    def test_single_writer_lock(self):
        with WriterLock(self.root):
            with self.assertRaises(RuntimeError):
                with WriterLock(self.root):
                    pass
        with WriterLock(self.root):
            pass

    def test_daemon_runs_to_stop_and_persists(self):
        stop = threading.Event()

        class FakeSource:
            sample_rate = 48000
            name = "test output"

            def blocks(self, event):
                for _ in range(3):
                    yield np.full((96000, 2), 0.1, dtype=np.float32)
                event.set()

        with patch("ears.audio_source.active_apps", return_value=["test.exe"]):
            run(self.config, source_factory=FakeSource, models=FakeModels(), stop=stop)
        state = daemon_status(self.config)
        self.assertFalse(state["running"])
        self.assertEqual(state["status"], "stopped")
        with Journal(self.config) as store:
            self.assertTrue(store.search(query="test.exe"))

    def test_empty_search_does_not_create_store_and_config_excludes_cloud_secrets(self):
        self.assertEqual(daemon_status(self.config)["status"], "not_started")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "should-not-save", "EARS_JOURNAL_DIR": str(self.root)}):
            values = json.loads(write_config(self.config).read_text())
            self.assertNotIn("GEMINI_API_KEY", values)
            self.assertEqual(values["EARS_JOURNAL_DIR"], str(self.root.resolve()))

    def test_timestamp_parser(self):
        self.assertAlmostEqual(parse_time("1h ago", 0), time.time() - 3600, delta=1)
        self.assertEqual(parse_time("2026-01-01T00:00:00Z", 0), 1767225600)
        with self.assertRaises(ValueError):
            parse_time("2026-01-01T00:00:00", 0)

    def test_startup_manifest_only_touches_requested_test_location(self):
        from ears.journal_daemon import startup
        with patch.dict(os.environ, {"APPDATA": str(self.root)}):
            with patch.object(Path, "home", return_value=self.root):
                with patch("ears.journal_daemon.command", return_value=["C:/test space/python", "-m", "ears.journal_daemon", "run"]):
                    with patch("subprocess.run"):
                        result = startup(self.config)
                        target = Path(result["startup_file"])
                        self.assertTrue(target.is_relative_to(self.root))
                        content = target.read_text(encoding="utf-8")
                        self.assertIn("ears.journal_daemon", content)
                        startup(self.config, remove=True)
                        self.assertFalse(target.exists())

    def test_default_output_change_reconnects_and_macos_requires_virtual_input(self):
        from ears.audio_source import DeviceChanged, SystemAudioSource
        fake_sc = SimpleNamespace(default_speaker=lambda: SimpleNamespace(id="new-default"))
        with patch("ears.audio_source.soundcard_module", return_value=fake_sc):
            with patch("ears.audio_source.sys.platform", "darwin"):
                with self.assertRaisesRegex(ValueError, "virtual loopback"):
                    next(SystemAudioSource().blocks(threading.Event()))
        class Recorder:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        fake_mic = SimpleNamespace(name="monitor", recorder=lambda **kw: Recorder())
        calls = iter([SimpleNamespace(id="old-default"), SimpleNamespace(id="new-default")])
        fake_sc = SimpleNamespace(default_speaker=lambda: next(calls), get_microphone=lambda **kw: fake_mic)
        with patch("ears.audio_source.soundcard_module", return_value=fake_sc):
            with patch("ears.audio_source.sys.platform", "linux"):
                with self.assertRaises(DeviceChanged):
                    next(SystemAudioSource().blocks(threading.Event()))

    def test_local_models_degrade_honestly_when_disabled(self):
        result = LocalModels(self.config).analyse(self.audio, 8000)
        self.assertEqual(result["kind"], "activity")
        self.assertEqual(result["text"], "Unclassified audio activity")
        self.assertNotIn("vector", result)

    def test_classifier_and_whisper_adapter_preserve_speech_timestamps(self):
        config = dataclasses.replace(self.config, classifier=True, whisper=True)
        models = LocalModels(config)
        transcript = SimpleNamespace(start=0.1, end=0.5, text=" Hello! ", no_speech_prob=0.01, avg_logprob=-0.2)
        rejected = SimpleNamespace(start=0.6, end=0.8, text=" hallucination ", no_speech_prob=0.95, avg_logprob=-2)
        fake_whisper = SimpleNamespace(transcribe=lambda *a, **k: (iter([transcript, rejected]), SimpleNamespace(language="en")))
        with patch.object(models, "classify", return_value=[{"label": "Shout", "score": 0.8}]):
            with patch.object(models, "whisper", return_value=fake_whisper):
                result = models.analyse(self.audio, 8000)
        self.assertEqual(result["kind"], "speech")
        self.assertEqual(result["text"], "Hello!")
        self.assertEqual(result["transcript"][0]["start_sec"], 0.1)
        with patch.object(models, "classify", side_effect=RuntimeError("missing weights")):
            with patch.object(models, "whisper", side_effect=RuntimeError("also missing")):
                result = models.analyse(self.audio, 8000)
        self.assertEqual(result["kind"], "activity")
        self.assertEqual(len(result["errors"]), 2)

    def test_qwen_contract_is_local_and_attaches_waveform(self):
        payload = {"choices": [{"message": {"content": '{"answer":"tone","summary":"a tone"}'}}]}
        class Response(io.BytesIO):
            pass
        with Journal(self.config) as store:
            window = store.window(self.add(store))
        with patch.dict(os.environ, {"EARS_QWEN_URL": "http://127.0.0.1:8901/v1"}):
            with patch("urllib.request.OpenerDirector.open", return_value=Response(json.dumps(payload).encode())) as call:
                result = QwenBackend().perceive((window,), "What is audible?", [], [])
                request = call.call_args.args[0]
                content = json.loads(request.data)["messages"][0]["content"]
                self.assertTrue(content[-1]["audio_url"]["url"].startswith("data:audio/wav;base64,"))
                self.assertEqual(result["result"]["answer"], "tone")
        with patch.dict(os.environ, {"EARS_QWEN_URL": "https://external.example/v1"}):
            with self.assertRaises(PerceptionError):
                QwenBackend()
        with self.assertRaises(PerceptionError):
            _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://external.example")


if __name__ == "__main__":
    unittest.main()
