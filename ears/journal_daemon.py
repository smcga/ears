"""Independent auditory journal service and login-startup management."""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .journal import Journal, JournalConfig
from .journal_models import LocalModels


class WriterLock:
    """An OS-released process lock; a stale PID never prevents restarting after a crash."""
    def __init__(self, root):
        self.path = root / "writer.lock"

    def __enter__(self):
        self.file = self.path.open("a+b")
        self.file.seek(0, 2)
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            raise RuntimeError("The journal daemon already owns this store.") from exc
        return self

    def __exit__(self, *args):
        self.file.close()


class Segmenter:
    """Check activity every block; retain active runs up to a bounded clip duration."""
    def __init__(self, store, models, sample_rate=48000):
        self.store, self.models, self.sr = store, models, sample_rate
        self.blocks, self.apps = [], set()
        self.start = self.end = None
        self.last_end = None
        self.silent_start = self.silent_end = None

    def flush_silence(self):
        if self.silent_start is not None:
            self.store.add(self.silent_start, self.silent_end, {"kind": "silence", "text": "Silence", "tags": []})
            self.silent_start = self.silent_end = None

    def flush(self):
        if not self.blocks:
            return
        samples = np.concatenate(self.blocks)
        try:
            result = self.models.analyse(samples, self.sr)
        except Exception as exc:
            result = {"kind": "activity", "text": "Local analysis failed; raw audio retained.",
                      "tags": [], "errors": [f"{type(exc).__name__}: {exc}"]}
        for segment in result.get("transcript", []):
            segment["start_unix"] = self.start + segment["start_sec"]
            segment["end_unix"] = self.start + segment["end_sec"]
        self.store.add(self.start, self.end, result, samples, self.sr, self.apps)
        self.store.state(last_event=self.end, analysis_errors=result.get("errors", []))
        self.blocks, self.apps = [], set()
        self.start = self.end = None

    def push(self, samples, start, end, apps=()):
        if self.last_end is not None and start - self.last_end > 0.25:
            self.flush()
            self.flush_silence()
            self.store.add(self.last_end, start, {"kind": "gap", "text": "Capture gap or processing backlog.", "tags": []})
        self.last_end = end
        # Peak guard preserves brief effects; RMS guard detects sustained low-level activity.
        x = np.asarray(samples)
        active = float(np.max(np.abs(x))) > 0.003 or float(np.sqrt(np.mean(x.astype(float) ** 2))) > 0.001
        if not active:
            self.flush()
            if self.silent_start is None:
                self.silent_start = start
            self.silent_end = end
            if end - self.silent_start >= 30:
                self.flush_silence()
            return
        self.flush_silence()
        if self.start is None:
            self.start = start
        self.end = end
        self.blocks.append(np.array(samples, dtype=np.float32, copy=True))
        self.apps.update(apps)
        if sum(len(block) for block in self.blocks) >= self.store.config.segment_seconds * self.sr:
            self.flush()

    def finish(self):
        self.flush()
        self.flush_silence()


def run(config=None, source_factory=None, models=None, stop=None):
    from .audio_source import SystemAudioSource, active_apps
    config = config or JournalConfig.from_env()
    config.root.mkdir(parents=True, exist_ok=True)
    source_factory = source_factory or (lambda: SystemAudioSource(config.device, block_seconds=config.block_seconds))
    stop = stop or threading.Event()
    stop_file = config.root / "stop.request"
    with WriterLock(config.root):
        stop_file.unlink(missing_ok=True)
        audio = queue.Queue(maxsize=16)

        def capture_loop():
            dropped = 0
            with Journal(config) as status:
                while not stop.is_set():
                    try:
                        source = source_factory()
                        status.state(status="connecting", pid=os.getpid(), heartbeat=time.time())
                        cursor = None
                        for raw in source.blocks(stop):
                            if stop_file.exists():
                                stop.set()
                                break
                            samples = np.array(raw, dtype=np.float32, copy=True)
                            if samples.ndim == 1:
                                samples = samples[:, None]
                            if samples.ndim != 2 or not len(samples) or not np.isfinite(samples).all():
                                raise ValueError("Audio device returned an empty or invalid block; reconnecting.")
                            duration = len(samples) / source.sample_rate
                            if cursor is None:
                                cursor = time.time() - duration
                            start, cursor = cursor, cursor + duration
                            try:
                                apps = active_apps()
                                app_error = None
                            except Exception as exc:
                                apps, app_error = [], str(exc)
                            try:
                                audio.put_nowait((samples, start, cursor, apps))
                            except queue.Full:
                                dropped += 1
                            status.state(status="listening", heartbeat=time.time(), device=source.name,
                                         last_capture_end=cursor, dropped_blocks=dropped, app_attribution_error=app_error,
                                         error=None)
                        if not stop.is_set():
                            raise RuntimeError("Audio source ended unexpectedly.")
                    except Exception as exc:
                        status.state(status="reconnecting", heartbeat=time.time(), error=f"{type(exc).__name__}: {exc}")
                        for _ in range(20):
                            if stop.is_set() or stop_file.exists():
                                stop.set()
                                break
                            stop.wait(0.25)

        with Journal(config) as store:
            store.reconcile()
            store.prune()
            store.state(status="starting", pid=os.getpid(), started=time.time(),
                        retention_hours=config.hours, clip_budget_bytes=config.max_bytes)
            processor = Segmenter(store, models or LocalModels(config))
            def capture_runner():
                try:
                    capture_loop()
                finally:
                    stop.set()

            thread = threading.Thread(target=capture_runner, name="ears-journal-capture", daemon=True)
            thread.start()
            previous_handlers = {}
            last_prune = time.monotonic()
            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
            try:
                while not stop.is_set() or not audio.empty():
                    if stop_file.exists():
                        stop.set()
                    try:
                        processor.push(*audio.get(timeout=0.25))
                    except queue.Empty:
                        if time.monotonic() - last_prune >= 30:
                            store.prune()
                            last_prune = time.monotonic()
                processor.finish()
            finally:
                stop.set()
                thread.join(5)
                store.state(status="stopped", heartbeat=time.time())
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)


def daemon_status(config=None):
    config = config or JournalConfig.from_env()
    if not (config.root / "journal.sqlite3").is_file():
        return {"status": "not_started", "root": str(config.root)}
    with Journal(config) as store:
        state = store.state()
    state["root"] = str(config.root)
    state["running"] = state.get("status") != "stopped" and time.time() - state.get("heartbeat", 0) < 30
    return state


def write_config(config):
    """Persist only journal/local model settings, never cloud API credentials."""
    values = {key: value for key, value in os.environ.items()
              if key.startswith(("EARS_JOURNAL_", "EARS_QWEN_"))}
    values["EARS_JOURNAL_DIR"] = str(config.root)
    path = config.root / "service-config.json"
    config.root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def command(config):
    executable = Path(sys.executable)
    if os.name == "nt" and executable.with_name("pythonw.exe").is_file():
        executable = executable.with_name("pythonw.exe")
    return [str(executable), "-m", "ears.journal_daemon", "run", "--config", str(write_config(config))]


def startup(config, remove=False):
    args = None if remove else command(config)
    if os.name == "nt":
        folder = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
        target = folder / "EarsJournal.vbs"
        if not remove:
            cmd = subprocess.list2cmdline(args).replace('"', '""')
            # Restart after an unexpected crash; a requested stop exits with code zero.
            content = f'Set sh = CreateObject("WScript.Shell")\nDo\n  code = sh.Run("{cmd}", 0, True)\n  If code = 0 Then Exit Do\n  WScript.Sleep 5000\nLoop\n'
    elif sys.platform == "darwin":
        import plistlib
        target = Path.home() / "Library/LaunchAgents/io.ears.journal.plist"
        if not remove:
            content = plistlib.dumps({"Label": "io.ears.journal", "ProgramArguments": args,
                                     "RunAtLoad": True, "KeepAlive": {"SuccessfulExit": False}}).decode()
    else:
        target = Path.home() / ".config/systemd/user/ears-journal.service"
        if not remove:
            escaped = " ".join('"' + arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"' for arg in args)
            content = ("[Unit]\nDescription=Ears auditory journal\nAfter=default.target\n[Service]\n"
                       f"ExecStart={escaped}\nRestart=on-failure\nRestartSec=5\n[Install]\nWantedBy=default.target\n")
    if remove:
        if sys.platform.startswith("linux"):
            subprocess.run(["systemctl", "--user", "disable", "ears-journal.service"], check=True)
        target.unlink(missing_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        if sys.platform.startswith("linux"):
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "ears-journal.service"], check=True)
    return {"startup_file": str(target), "installed": not remove, "takes_effect": "next login"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "start", "stop", "status", "devices", "prepare-models", "install-startup", "remove-startup"])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        for key, value in values.items():
            if key.startswith(("EARS_JOURNAL_", "EARS_QWEN_")) and isinstance(value, str):
                os.environ[key] = value
    config = JournalConfig.from_env()
    if args.action == "devices":
        from .audio_source import soundcard_module
        sc = soundcard_module()
        print(json.dumps({"outputs": [{"id": s.id, "name": s.name} for s in sc.all_speakers()],
                          "inputs_and_monitors": [{"id": s.id, "name": s.name} for s in sc.all_microphones(include_loopback=True)]}, indent=2))
        return
    if args.action == "run":
        run(config)
        return
    if args.action == "prepare-models":
        LocalModels(config, download=True).prepare()
        print("Local journal models cached. Listening will use cached weights only.")
        return
    if args.action == "start":
        state = daemon_status(config)
        if state.get("running"):
            print(json.dumps(state))
            return
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        subprocess.Popen(command(config), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **options)
        print(json.dumps({"status": "starting", "root": str(config.root)}))
    elif args.action == "stop":
        config.root.mkdir(parents=True, exist_ok=True)
        (config.root / "stop.request").touch()
        print(json.dumps({"status": "stop_requested"}))
    elif args.action == "status":
        print(json.dumps(daemon_status(config), indent=2))
    else:
        print(json.dumps(startup(config, remove=args.action == "remove-startup")))


if __name__ == "__main__":
    main()
