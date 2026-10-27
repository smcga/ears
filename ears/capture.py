"""Record what the computer is playing (WASAPI loopback), e.g. a game while you play it.

- Captures the output device's mix, corrected for the Windows volume slider so levels are what the game
  actually sends (the slider is applied before the loopback point on software-volume devices).
- Global hotkey (default Ctrl+Alt+M) drops a marker without leaving the game.
- Writes a WAV while recording (header refreshed every few seconds, so a crash still leaves a usable file),
  then converts it to FLAC on stop. Markers and notes go in <name>.markers.json next to it.
- Notes which other apps were making sound, so a stray YouTube tab doesn't get blamed on the game.
"""
from __future__ import annotations

import atexit
import collections
import ctypes
import json
import os
import sys
import threading
import time
import warnings
from ctypes import wintypes
from datetime import datetime

import numpy as np
import soundfile as sf

from .levels import HOP_SEC, LoudnessTracker, _db, bed_level, integrated, mmss, sidecar_path
from .paths import _under, allowed_roots

SR = 48000
BLOCK = SR // 10
RING_SEC = 60
DEFAULT_HOTKEY = "ctrl+alt+m"
_IGNORE_APPS = {"system", "audiodg.exe", "python.exe", "pythonw.exe"}


def capture_dir() -> str:
    raw = os.environ.get("EARS_CAPTURE_DIR") or os.path.join(allowed_roots()[0], "ears captures")
    real = os.path.realpath(raw)
    if not any(_under(real, r) for r in allowed_roots()):
        raise PermissionError(f"EARS_CAPTURE_DIR ({real}) must be inside the allowed folders "
                              f"({'; '.join(allowed_roots())}) so ears can read the recordings back.")
    os.makedirs(real, exist_ok=True)
    return real


def _com_init() -> None:
    """Make sure this thread has COM set up (WASAPI needs it; tool calls arrive on arbitrary worker threads)."""
    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # MTA; harmless if already initialised either way
    except Exception:
        pass


_sc_mod = None


def _sc():
    """soundcard, imported once on a fresh thread (its import fails on a thread where COM is already set up),
    then COM initialised on the calling thread so its calls work wherever we are."""
    global _sc_mod
    if _sc_mod is None:
        box: dict = {}

        def imp():
            try:
                warnings.filterwarnings("ignore", module="soundcard")
                import soundcard

                box["m"] = soundcard
            except Exception as exc:
                box["e"] = exc

        t = threading.Thread(target=imp, name="ears-soundcard-import")
        t.start()
        t.join()
        if "e" in box:
            raise RuntimeError(f"Couldn't load the Windows audio library: {box['e']}")
        _sc_mod = box["m"]
    _com_init()
    return _sc_mod


def list_devices() -> dict:
    sc = _sc()
    default = sc.default_speaker()
    return {"outputs": [s.name + ("  (default)" if s.id == default.id else "") for s in sc.all_speakers()],
            "hint": "Pass part of a name as `device` to record that output instead of the default."}


def _pick_speaker(device: str | None):
    sc = _sc()
    if not device:
        return sc.default_speaker()
    want = device.lower()
    for s in sc.all_speakers():
        if want in s.name.lower():
            return s
    raise ValueError(f"No output device matching '{device}'. Outputs: " + ", ".join(s.name for s in sc.all_speakers()))


# ---- hotkey -------------------------------------------------------------------------------------------------------

_MODS = {"ctrl": 0x2, "control": 0x2, "alt": 0x1, "shift": 0x4, "win": 0x8}


def parse_hotkey(text: str) -> tuple[int, int]:
    mods, vk = 0, None
    for part in text.lower().replace(" ", "").split("+"):
        if part in _MODS:
            mods |= _MODS[part]
        elif len(part) == 1 and part.isalnum():
            vk = ord(part.upper())
        elif part.startswith("f") and part[1:].isdigit() and 1 <= int(part[1:]) <= 24:
            vk = 0x6F + int(part[1:])
        else:
            raise ValueError(f"Can't read hotkey part '{part}'. Use e.g. 'ctrl+alt+m' or 'ctrl+f9'.")
    if vk is None:
        raise ValueError("Hotkey needs a key, e.g. 'ctrl+alt+m'.")
    return mods, vk


# ---- volume / app sessions (pycaw, in its own COM thread) ---------------------------------------------------------

def _endpoint(speaker_id: str):
    from pycaw.pycaw import AudioUtilities

    sp = AudioUtilities.GetSpeakers()
    if sp.id == speaker_id:
        return sp, True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in AudioUtilities.GetAllDevices():
            if d.id == speaker_id:
                return d, False
    return None, False


class Recorder:
    def __init__(self, name: str, device: str | None, max_minutes: float, hotkey: str):
        self.speaker = _pick_speaker(device)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "game")).strip() or "game"
        base = os.path.join(capture_dir(), f"{safe}_{stamp}")
        i = 2
        while os.path.exists(base + ".flac") or os.path.exists(base + ".wav"):
            base = os.path.join(capture_dir(), f"{safe}_{stamp}_{i}")
            i += 1
        self.base = base
        self.wav_path, self.flac_path = base + ".wav", base + ".flac"
        self.max_frames = int(max_minutes * 60 * SR)
        self.hotkey = hotkey
        self.hotkey_ok: bool | None = None
        self.markers: list[dict] = []
        self.frames = 0
        self.gain_db = 0.0
        self.volume_hist: list[float] = []
        self.muted_sec = 0
        self.hw_volume = False
        self.apps: collections.Counter = collections.Counter()
        self.app_volume: dict[str, float] = {}
        self.clipped_blocks = 0
        self.error: str | None = None
        self.started_at = time.time()
        self.lt = LoudnessTracker(SR, 2)
        self.ring: collections.deque = collections.deque()
        self.ring_frames = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._done = threading.Event()
        self._hk_tid = None
        self.stopped_reason = ""

    # -- lifecycle --
    def start(self) -> None:
        self._threads = [threading.Thread(target=self._record, daemon=True, name="ears-capture"),
                         threading.Thread(target=self._watch, daemon=True, name="ears-volume"),
                         threading.Thread(target=self._hotkey_loop, daemon=True, name="ears-hotkey")]
        for t in self._threads:
            t.start()
        if not self._ready.wait(5):
            self._stop.set()
            raise RuntimeError("The audio device didn't start recording within 5 s.")
        if self.error:
            raise RuntimeError(self.error)

    def elapsed(self) -> float:
        return self.frames / SR

    def mark(self, label: str | None) -> dict:
        with self._lock:
            n = len(self.markers) + 1
            m = {"t": round(self.elapsed(), 1), "label": (label or "").strip() or f"mark {n}"}
            self.markers.append(m)
        self._write_sidecar()
        return m

    def stop(self, reason: str = "stopped") -> None:
        if not self.stopped_reason:
            self.stopped_reason = reason
        self._stop.set()
        self._quit_hotkey()
        self._done.wait(15)
        for t in getattr(self, "_threads", []):
            t.join(3)
        self._finalize()

    @property
    def running(self) -> bool:
        return not self._done.is_set()

    # -- threads --
    def _record(self) -> None:
        wav = None
        try:
            sc = _sc()
            mic = sc.get_microphone(id=self.speaker.id, include_loopback=True)
            wav = sf.SoundFile(self.wav_path, "w", samplerate=SR, channels=2, subtype="PCM_24", format="WAV")
            last_sync = time.time()
            with mic.recorder(samplerate=SR, channels=2) as rec:
                self._ready.set()
                while not self._stop.is_set():
                    x = rec.record(numframes=BLOCK)
                    if x.ndim == 1:
                        x = np.repeat(x[:, None], 2, axis=1)
                    elif x.shape[1] == 1:
                        x = np.repeat(x, 2, axis=1)
                    if self.gain_db:
                        x = x * np.float32(10 ** (self.gain_db / 20))
                    if np.abs(x).max() > 1.0:
                        self.clipped_blocks += 1
                    wav.write(np.clip(x, -1, 1))
                    self.lt.push(x)
                    with self._lock:
                        self.frames += len(x)
                        self.ring.append(x)
                        self.ring_frames += len(x)
                        while self.ring_frames - len(self.ring[0]) >= RING_SEC * SR:
                            self.ring_frames -= len(self.ring.popleft())
                    if time.time() - last_sync > 5:
                        _update_header(wav)
                        last_sync = time.time()
                    if self.frames >= self.max_frames:
                        self.stopped_reason = self.stopped_reason or "reached the time limit"
                        break
        except Exception as exc:  # surfaced through status/stop
            self.error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            if wav is not None:
                wav.close()
            self._stop.set()
            self._quit_hotkey()
            self._done.set()
            if self.stopped_reason == "reached the time limit":
                self._finalize()

    def _quit_hotkey(self) -> None:
        if self._hk_tid:
            ctypes.windll.user32.PostThreadMessageW(self._hk_tid, 0x0012, 0, 0)  # WM_QUIT

    def _watch(self) -> None:
        """Every second: Windows volume (to undo it) and which apps are making sound."""
        try:
            import comtypes

            try:  # comtypes may already have set this thread up (on first import); either mode works here
                comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            except OSError:
                pass
            from pycaw.pycaw import AudioUtilities, IAudioMeterInformation
        except Exception as exc:
            print(f"ears: volume watcher unavailable: {exc}", file=sys.stderr)
            return
        try:
            dev, is_default = _endpoint(self.speaker.id)
            vol = dev.EndpointVolume if dev is not None else None
            if vol is not None:
                self.hw_volume = bool(vol.QueryHardwareSupport() & 1)
        except Exception:
            vol, is_default = None, False
        while not self._stop.is_set():
            try:
                if vol is not None:
                    muted = bool(vol.GetMute())
                    db = float(vol.GetMasterVolumeLevel())
                    self.volume_hist.append(round(db, 1))
                    if muted:
                        self.muted_sec += 1
                    # Software volume is applied before the loopback point: undo it.
                    self.gain_db = 0.0 if (self.hw_volume or muted) else min(-db, 40.0)
                if is_default:
                    for s in AudioUtilities.GetAllSessions():
                        try:
                            name = s.Process.name() if s.Process else "system"
                            peak = s._ctl.QueryInterface(IAudioMeterInformation).GetPeakValue()
                            if peak > 0.001:
                                self.apps[name] += 1
                                self.app_volume[name] = round(float(s.SimpleAudioVolume.GetMasterVolume()), 2)
                        except Exception:
                            continue
            except Exception as exc:
                print(f"ears: volume watcher: {exc}", file=sys.stderr)
            self._stop.wait(1.0)

    def _hotkey_loop(self) -> None:
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        try:
            mods, vk = parse_hotkey(self.hotkey)
        except ValueError:
            self.hotkey_ok = False
            return
        self._hk_tid = kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None, 1, mods | 0x4000, vk):  # MOD_NOREPEAT
            self.hotkey_ok = False
            return
        self.hotkey_ok = True
        msg = wintypes.MSG()
        try:
            while not self._stop.is_set() and user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == 0x0312:  # WM_HOTKEY
                    self.mark(None)
        finally:
            user32.UnregisterHotKey(None, 1)

    # -- outputs --
    def _notes(self) -> dict:
        out: dict = {}
        others = [f"{a} ({n} s)" for a, n in self.apps.most_common() if a.lower() not in _IGNORE_APPS]
        if len(others) > 1:
            out["other_apps"] = others[1:]
        out["apps_heard"] = others
        if self.app_volume:
            out["app_mixer_volume"] = {a: v for a, v in self.app_volume.items() if a.lower() not in _IGNORE_APPS}
        if self.muted_sec:
            out["volume_note"] = f"Windows output was muted for {self.muted_sec} s, so that stretch is silent."
        elif self.volume_hist and (max(self.volume_hist) - min(self.volume_hist)) > 0.5:
            out["volume_note"] = ("The Windows volume changed during the capture; ears corrected for it about once a "
                                  "second, so the moment of each change may be off.")
        if self.clipped_blocks:
            out["clipped_blocks"] = self.clipped_blocks
        return out

    def _write_sidecar(self, final: bool = False) -> None:
        data = {"device": self.speaker.name, "started": datetime.fromtimestamp(self.started_at).isoformat(timespec="seconds"),
                "seconds": round(self.elapsed(), 1), "markers": list(self.markers),
                "windows_volume_db": self.volume_hist[-1] if self.volume_hist else None,
                "volume_corrected": not self.hw_volume, "final": final}
        data.update(self._notes())
        target = self.flac_path if final else self.wav_path
        try:
            with open(sidecar_path(target), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=1)
        except OSError as exc:
            print(f"ears: couldn't write markers: {exc}", file=sys.stderr)

    def _finalize(self) -> None:
        with self._lock:
            if getattr(self, "_finalized", False):
                return
            self._finalized = True
        if not os.path.isfile(self.wav_path):
            return
        try:
            with sf.SoundFile(self.wav_path) as src, \
                    sf.SoundFile(self.flac_path, "w", samplerate=SR, channels=2, subtype="PCM_24", format="FLAC") as dst:
                for block in src.blocks(blocksize=SR * 30, dtype="float32", always_2d=True):
                    dst.write(block)
            os.remove(self.wav_path)
            self._write_sidecar(final=True)  # same .markers.json name as during recording; now marked final
        except Exception as exc:  # keep the WAV if conversion fails
            print(f"ears: FLAC conversion failed, keeping WAV: {exc}", file=sys.stderr)
            self.flac_path = self.wav_path
            self._write_sidecar(final=True)

    def final_path(self) -> str:
        return self.flac_path if os.path.isfile(self.flac_path) else self.wav_path

    def snapshot(self, seconds: float) -> str:
        with self._lock:
            blocks = list(self.ring)
            now = self.elapsed()
        if not blocks:
            raise ValueError("Nothing recorded yet.")
        x = np.concatenate(blocks)[-int(min(seconds, RING_SEC) * SR):]
        stamp = f"{int(now // 60):02d}-{int(now % 60):02d}"
        out = f"{self.base}_snap_{stamp}.flac"
        sf.write(out, np.clip(x, -1, 1), SR, subtype="PCM_24", format="FLAC")
        return out

    def status(self) -> dict:
        ms, _ = self.lt.arrays()
        n = len(ms)
        el = self.elapsed()
        st = {"recording": self.running, "file": self.final_path() if not self.running else self.wav_path,
              "elapsed": mmss(el), "device": self.speaker.name}
        if self.error:
            st["error"] = self.error
        if n >= 30:
            st["now_3s_lufs"] = round(float(_db(ms[-30:].mean())), 1)
        if n >= 4:
            last30 = integrated(ms, max(0, n - 300), n)
            st["last_30s_lufs"] = None if last30 is None else round(last30, 1)
        # Running level per marker segment, so you can compare this level to earlier ones while playing.
        bounds = [0] + [int(round(m["t"] / HOP_SEC)) for m in self.markers] + [n]
        labels = ["before first marker"] + [m["label"] for m in self.markers]
        segs = []
        for i in range(len(bounds) - 1):
            b0, b1 = bounds[i], bounds[i + 1]
            bed = bed_level(ms, b0, b1) if b1 - b0 >= 20 else None
            if bed is not None:
                segs.append({"label": labels[i], "from": mmss(b0 * HOP_SEC), "bed": round(bed, 1)})
        b_last = bounds[-2]
        if n - b_last >= 200:
            frac = float(np.mean(_db(ms[b_last:n]) >= -60))
            if frac < 0.3:
                st["warning"] = (f"Only {frac * (n - b_last) * HOP_SEC:.0f} s of sound in the last "
                                 f"{(n - b_last) * HOP_SEC:.0f} s. Is the music on, and is the game running at "
                                 "normal speed? (A hidden browser tab slows web games down.)")
        if segs:
            st["levels_so_far"] = segs
            if len(segs) >= 2:
                cur, prev = segs[-1], segs[:-1]
                ref = float(np.median([s["bed"] for s in prev]))
                st["current_vs_earlier"] = f"{cur['label']} is running {cur['bed'] - ref:+.1f} dB vs the earlier levels"
        st["markers"] = len(self.markers)
        st["hotkey"] = (f"{self.hotkey} drops a marker" if self.hotkey_ok
                        else f"hotkey {self.hotkey} unavailable (in use by another app?); mark via chat instead")
        if not self.hw_volume and self.volume_hist:
            st["windows_volume_db"] = self.volume_hist[-1]
        st.update({k: v for k, v in self._notes().items() if k in ("other_apps", "volume_note")})
        if self.stopped_reason and not self.running:
            st["stopped"] = self.stopped_reason
        return st


def _update_header(f: sf.SoundFile) -> None:
    """Rewrite the WAV header now so the file is valid even if the process dies mid-recording."""
    try:
        from soundfile import _snd

        _snd.sf_command(f._file, 0x1060, sf._ffi.NULL, 0)  # SFC_UPDATE_HEADER_NOW
    except Exception:
        f.flush()


_current: Recorder | None = None
_cur_lock = threading.Lock()


def current() -> Recorder | None:
    return _current


def start(name: str, device: str | None, max_minutes: float, hotkey: str | None) -> Recorder:
    global _current
    with _cur_lock:
        if _current is not None and _current.running:
            raise ValueError(f"Already recording to {_current.wav_path} ({mmss(_current.elapsed())} so far). "
                             "Stop it first.")
        rec = Recorder(name, device, max_minutes, hotkey or os.environ.get("EARS_MARK_HOTKEY", DEFAULT_HOTKEY))
        rec.start()
        _current = rec
        return rec


def stop() -> Recorder:
    with _cur_lock:
        rec = _current
        if rec is None:
            raise ValueError("Nothing is being recorded. Start with action='start'.")
        rec.stop()
        return rec


@atexit.register
def _shutdown() -> None:
    rec = _current
    if rec is not None and rec.running:
        rec.stop("ears was shut down")
