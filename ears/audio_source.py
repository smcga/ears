"""Streaming loopback capture without session files, hotkeys or time limits."""
from __future__ import annotations

import json
import subprocess
import sys


class DeviceChanged(RuntimeError):
    pass


def soundcard_module():
    if sys.platform == "win32":
        from .capture import _sc
        return _sc()
    import soundcard
    return soundcard


class SystemAudioSource:
    def __init__(self, device=None, sample_rate=48000, block_seconds=2):
        self.device = device
        self.sample_rate = sample_rate
        self.block_seconds = block_seconds
        self.name = ""

    def blocks(self, stop):
        sc = soundcard_module()
        if sys.platform == "darwin":
            if not self.device:
                raise ValueError("macOS needs EARS_JOURNAL_DEVICE set to a routed virtual loopback input, e.g. BlackHole.")
            mic = sc.get_microphone(self.device, include_loopback=True)
            speaker_id = None
        else:
            speaker = sc.get_speaker(self.device) if self.device else sc.default_speaker()
            if speaker is None:
                raise ValueError("No output device is available.")
            speaker_id = speaker.id
            mic = sc.get_microphone(id=speaker.id, include_loopback=True)
        self.name = mic.name
        with mic.recorder(samplerate=self.sample_rate, channels=2) as recorder:
            while not stop.is_set():
                if not self.device and speaker_id is not None:
                    current = sc.default_speaker()
                    if current is None or current.id != speaker_id:
                        raise DeviceChanged("Default output changed; reconnecting to its monitor.")
                yield recorder.record(numframes=round(self.sample_rate * self.block_seconds))


def active_apps():
    """Apps contributing to the mix, not a claim that one app caused one event."""
    if sys.platform == "win32":
        from .capture import _com_init
        _com_init()
        from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
        apps = []
        for session in AudioUtilities.GetAllSessions():
            try:
                if session._ctl.QueryInterface(IAudioMeterInformation).GetPeakValue() > 0.001:
                    apps.append(session.Process.name() if session.Process else "system")
            except Exception:
                continue
        return sorted(set(apps))
    if sys.platform.startswith("linux"):
        # pactl also works with the PipeWire PulseAudio compatibility server.
        run = subprocess.run(["pactl", "--format=json", "list", "sink-inputs"],
                             capture_output=True, text=True, timeout=2, check=True)
        inputs = json.loads(run.stdout)
        return sorted({row.get("properties", {}).get("application.name", "unknown") for row in inputs
                       if not row.get("corked", False) and not row.get("mute", False)})
    return []
