"""Offline tests for provider-neutral perception, memory and raw capture windows."""
from __future__ import annotations

import collections
import threading
from types import SimpleNamespace

import numpy as np

from ears import capture
from ears.perception import (
    AudioWindow,
    PerceptionError,
    memory_index,
    perceive_windows,
    register_backend,
)


class FakeBackend:
    name = "fake"
    model = "fake-audio-1"
    network = True
    audio_leaves_computer = True
    calls = 0
    prior_counts: list[int] = []
    measured_counts: list[int] = []

    def perceive(self, windows, question, prior_turns, measured):
        self.__class__.calls += 1
        self.__class__.prior_counts.append(len(prior_turns))
        self.__class__.measured_counts.append(len(measured))
        return {
            "result": {
                "answer": f"Heard {len(windows)} source(s): {question}",
                "summary": "A steady tone followed by an impact.",
                "audible": ["tone", "impact"],
                "events": [{
                    "source": windows[0].label,
                    "start_sec": 0.5,
                    "end_sec": 0.8,
                    "label": "impact",
                    "evidence": "short broadband transient",
                    "confidence": 0.8,
                }],
                "perceptual_observations": [{
                    "claim": "The impact interrupts the tone.",
                    "evidence": "audible transient",
                    "confidence": 0.8,
                }],
                "recommendations": ["Reduce the impact if it should sit behind the tone."],
                "uncertainty": ["The source of the impact is unknown."],
            },
            "usage": {"fake_tokens": 1},
        }


class FailingBackend:
    name = "failing"
    model = "broken"

    def perceive(self, windows, question, prior_turns, measured):
        raise PerceptionError("provider disconnected")


def tone_window(label="audio", start=20.0):
    sr = 24000
    t = np.arange(sr * 2) / sr
    x = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return AudioWindow(label, x, sr, "test", start, start + 2)


def test_perception_and_follow_up() -> None:
    register_backend("fake", FakeBackend)
    first = perceive_windows((tone_window(),), "What happened?", "fake")
    assert first["answer"].startswith("Heard 1 source")
    assert first["heard"]["events"][0]["origin_start_sec"] == 20.5
    assert first["measured"][0]["rms_dbfs"] > -30
    assert first["provenance"]["audio_uploaded"] is True
    assert first["recommendations"]

    second = perceive_windows((), "Listen again: was it abrupt?", "fake", first["memory_id"])
    assert second["memory_id"] == first["memory_id"]
    assert FakeBackend.prior_counts[-1] == 1
    assert memory_index("impact", 10)[0]["memory_id"] == first["memory_id"]


def test_silence_guard_does_not_call_provider() -> None:
    before = FakeBackend.calls
    silent = AudioWindow("silence", np.zeros((24000, 2), np.float32), 24000, "test")
    result = perceive_windows((silent,), "What do you hear?", "fake")
    assert result["heard"]["summary"] == "Effectively silent audio."
    assert result["provenance"]["audio_uploaded"] is False
    assert FakeBackend.calls == before

    sparse = np.zeros((24000 * 30, 1), np.float32)
    sparse[12000:14400] = 0.5  # a 100 ms event must not be discarded as silence
    result = perceive_windows((AudioWindow("impact", sparse, 24000, "test"),), "What was that?", "fake")
    assert result["provenance"]["audio_uploaded"] is True
    assert FakeBackend.calls == before + 1


def test_comparison_and_provider_failure_are_explicit() -> None:
    result = perceive_windows((tone_window("A"), tone_window("B")), "Which is clearer?", "fake")
    assert len(result["measured"]) == 2
    assert FakeBackend.measured_counts[-1] == 2
    assert result["answer"].startswith("Heard 2 source")

    register_backend("failing", FailingBackend)
    try:
        perceive_windows((tone_window(),), "What happened?", "failing")
    except PerceptionError as exc:
        assert "disconnected" in str(exc)
    else:
        raise AssertionError("provider failure was turned into invented observations")


def test_raw_capture_window_and_measurement_separation() -> None:
    raw = np.full((4800, 2), 0.1, np.float32)
    corrected = capture.Recorder.measurement_audio(raw, 6.0)
    assert np.allclose(raw, 0.1)
    assert float(corrected[0, 0]) > 0.19
    assert not np.shares_memory(raw, corrected)

    rec = capture.Recorder.__new__(capture.Recorder)
    rec.ring = collections.deque([raw])
    rec.ring_frames = len(raw)
    rec.frames = len(raw)
    rec._lock = threading.Lock()
    rec._done = threading.Event()
    rec.speaker = SimpleNamespace(name="fake output")
    rec.error = None
    rec.muted_sec = 0
    recent, sr, meta = rec.recent_audio(1)
    assert sr == capture.SR
    assert recent.flags.writeable is False
    assert not np.shares_memory(recent, raw)
    assert meta["signal"].startswith("raw loopback")

    rec.ring.clear()
    rec.ring_frames = 0
    rec.error = "device disconnected"
    try:
        rec.recent_audio(1)
    except ValueError as exc:
        assert "device disconnected" in str(exc)
    else:
        raise AssertionError("empty disconnected capture should fail explicitly")


if __name__ == "__main__":
    test_perception_and_follow_up()
    test_silence_guard_does_not_call_provider()
    test_comparison_and_provider_failure_are_explicit()
    test_raw_capture_window_and_measurement_separation()
    print("Perception tests: ok")
