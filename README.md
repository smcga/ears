# ears

Audio analysis and open-ended auditory perception for any AI client that supports local [Model Context Protocol](https://modelcontextprotocol.io/) tools over stdio.

Ears turns audio into measurements, musical structure, learned descriptions and visualisations that a language model can reason about. Its optional perception layer can also send an immutable audio window directly to a configured audio-native model and return the interpretation as text, so the chat client itself does not need audio support.

Every perception result separates what the model heard, what Ears measured from the original waveform, and what the model recommends. Perceptual and aesthetic claims can still be wrong; measurements provide evidence rather than a universal score for whether music is good.

The optional **auditory journal** listens independently of your chat: it records local event labels, speech transcripts,
music embeddings and short FLAC clips in a persistent SQLite timeline. Search what happened later, then ask your
chosen perception model to hear the exact retained clip. Background processing uses local models, with no Gemini calls.

## Compatibility

| Capability | Requirement |
|---|---|
| Text and structured reports | Any local stdio MCP client |
| Spectrograms and piano rolls | Client/model support for MCP `ImageContent`; text remains available without it |
| Short source-audio excerpts | Client/model support for MCP `AudioContent`; only sent when `audio_clip` is called |
| Open-ended audio perception | Any text-capable MCP client; a configured audio-native provider such as Gemini |
| Core file analysis | Windows, macOS or Linux; Python 3.12 or 3.13 |
| Journal capture | Windows WASAPI; Linux PulseAudio/PipeWire monitor; macOS with a routed virtual loopback input |
| Game-session `capture` tool | Windows 10/11 only |
| FL Studio bar/note conventions | Optional workflow convention, not a protocol requirement |

## Tools

| Tool | What it does | Speed |
|---|---|---|
| `listen(path, bpm?, start_sec?, end_sec?)` | Tempo, key, LUFS/peak/clipping, stereo, a 4-bar table on FL's bar grid, detected sections, mix flags, plus a spectrogram (and a zoomed one if you give a range) | ~10 s for a 5-min track |
| `compare(path_a, path_b, bpm?)` | What changed between two versions: loudness, peaks, band balance per section, arrangement shifts, key, length | ~10–20 s |
| `stems(path, bpm?, force?)` | Demucs split into drums/bass/other/vocals, saved to `<name>_stems\` next to the file, plus a short report per stem | ~20 s on a recent NVIDIA GPU (much slower on CPU) + ~10 s analysis; reused next time |
| `notes(path, bpm?, stem?, bars?, quantize?)` | Basic Pitch transcription of a stem (default `other` = chords/melody; also `bass`, `vocals`, `mix`): chord chart per bar, bass roots, key from the notes, a beat.step note list for a bar range, a piano-roll image, and tempo-mapped `.mid` files in `<name>_stems\` | ~20 s (+ stem split the first time) |
| `describe(path, bpm?, phrases?)` | CLAP tags in words: genre/style, groove, texture, elements, mood, for the whole track and per section; optionally scores your own phrases ("garage-ish", "swung") | ~10 s (+ ~10 s model load on first call) |
| `expect(path, bpm?)` | Listening as prediction: guesses every bar from the bars before it and measures the surprise. Biggest moments (what was expected, what happened, on/off the phrase grid, bars to settle), predictable stretches, which parts return, plus a learned similarity map with the surprise curve | ~10 s |
| `listen_live(path, bpm?, session_id?, bars?, my_guess?)` | Step through a track a few bars at a time without seeing ahead. The first call returns an explicit session handle for later calls | a few s per step |
| `sounds_like(path, against? or folder?)` | Describes each section by the closest-sounding section in your other tracks (learned CLAP fingerprints) instead of adjectives | ~15 s per new reference, then cached |
| `audio_clip(path, start_sec?, end_sec?)` | Returns an explicitly requested excerpt of at most 15 seconds as standard MCP audio, with a text fallback and privacy warning | <1 s |
| `perceive(source, seconds?, question?, provider?, against?, memory_id?)` | Asks an audio-native model what is actually audible in a recent capture or file. Returns timestamped events, uncertainty, original-waveform measurements and recommendations as text/JSON. Supports A/B comparison and exact-audio follow-ups | provider-dependent |
| `auditory_memory(query?, limit?)` | Searches the bounded, in-process event index created by prior `perceive` calls; returned handles can be listened to again | <1 s |
| `timeline(since?, until?, query?, limit?)` | Searches the persistent journal by time, transcript, labels and active apps. Returns `journal:EVENT_ID` references for listening again | <1 s normally |
| `taste(event_id?, rating?, note?)` | Records your explicit 1–5 music ratings, or finds similar rated CLAP examples and their waveform measurements | <1 s normally |
| `capabilities()` | Reports installed optional features, platform support, allowed folders and persistence settings | <1 s |
| `capture(action, ...)` | Records what Windows is playing (Windows only). `start`, then press **Ctrl+Alt+M** as each new level starts; `status` = live readout; `snapshot` = optionally save recent audio; `stop` = per-level loudness report + timeline. `perceive(source="recent")` reads the buffer without saving | live; report in a few s |
| `levels(path, splits?, labels?, target_lufs?)` | Loudness per level: a capture split by its markers (or by silences, or your times), or a **folder** of music files (one per level). Gives each level's music level, the dB change and linear volume (e.g. for a Unity AudioSource) to match the others or a target | ~1 s per minute of audio |
| `find_audio(query, folder?)` | Finds bounces by name, newest first | <1 s |

Bar numbers match FL Studio's playlist **if the bounce starts at bar 1**. Pass `bpm` (e.g. 130) for exact bars.

## Things to ask your AI client

- "Listen to C:\Users\me\Music\My Project\my_track.mp3 at 130 bpm"
- "Compare my_track.mp3 and my_track_v2.mp3 in my Music folder: what changed?"
- "Zoom into 0:50–1:05 of my_track at 130 bpm. Does the drop hit cleanly?"
- "Split my_track into stems and tell me where the bass plays and what notes it uses"
- "Transcribe the chords in my_track at 130 bpm and show me the notes in bars 53-60"
- "Get me the bassline of my_track as MIDI"
- "Describe my_track. Does it come across as garage-ish or more techno? Is the groove swung?"
- "Run expect on my_track at 130 bpm: where does it break expectations?"
- "Listen to my_track live, 8 bars at a time, and tell me what you expect before each step"
- "Send bars 33-40 as a short audio clip so you can listen directly" (audio-capable clients only)
- "Start monitoring my computer audio. What do you hear in the last 30 seconds?"
- "Did you hear that strange sound just now? Roughly when did it happen?"
- "Listen to these two mixes. Which sounds clearer, and does the waveform evidence support that?"
- "Listen to that exact clip again: did the vocal sound natural or processed?" (uses the returned auditory-memory handle)
- "What does my_new_idea sound like among my other bounces?"
- "Find my latest bounce and check it for clipping"
- "Listen to my game while I play. I'll press Ctrl+Alt+M at each new level" ... "how's this level compared to the others?" ... "stop: which levels need turning up or down?"
- "Grab the last 30 seconds of the game and tell me what the music sounds like"
- "Compare the loudness of the music files in my Game music folder"

Note names follow FL Studio's piano roll (middle C = C5). The saved MIDI is at your BPM in 4/4 and snapped to 1/16, so dragging it into FL at bar 1 lines it up with the bounce.

`listen_live` returns a `session_id`. Clients must pass that handle into each later call so independent conversations cannot affect one another. `perceive` similarly returns a `memory_id`; passing it back asks another question about the exact immutable audio rather than a newly captured window.

## Perception setup

The first built-in backend is [Gemini's on-demand audio-understanding API](https://ai.google.dev/gemini-api/docs/generate-content/audio). Set an API key in the MCP server environment:

```json
{
  "EARS_ALLOWED_ROOTS": "C:\\Users\\me\\Music",
  "GEMINI_API_KEY": "your-key",
  "EARS_PERCEPTION_PROVIDER": "gemini",
  "EARS_GEMINI_MODEL": "gemini-3.8-flash"
}
```

Then use `capture(action="start")`, play the sound, and call `perceive(source="recent", seconds=30, question="What happened?")`. Ears copies the selected raw buffer in memory and sends only that immutable window. It does not create a snapshot file. Files can also be perceived directly, and `against` adds a second labelled clip for A/B listening.

The provider boundary is deliberately small. Local or alternative cloud integrations can implement the `PerceptionBackend` protocol and call `register_backend(name, factory)` in `ears.perception`; the MCP tool and result schema do not depend on Gemini. A Qwen or other local adapter can therefore be added without changing clients.

Perception is on demand. Prior calls form a bounded event index searchable with `auditory_memory`; continuous local
history comes from the separate journal below. The MCP client decides when to ask questions or show commentary.

## Always-on auditory journal

Install the optional models and capture dependencies in the same environment as the server:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[journal]"
.venv\Scripts\ears-journal.exe prepare-models
.venv\Scripts\ears-journal.exe start
.venv\Scripts\ears-journal.exe status
```

On macOS/Linux, use `.venv/bin/python` and `.venv/bin/ears-journal`. `prepare-models` explicitly downloads the
configured weights once. The listener loads cached weights only: missing packages or models appear in each event's
`errors` and in daemon status, while the raw clip is retained. This degraded mode does not invent semantic labels.

The listener has no session duration cap. It checks activity every two seconds, groups active audio into clips of up
to twelve seconds, and stores silence as metadata without a clip. Inference runs behind a bounded capture queue.
If inference cannot keep up, dropped-block counts and timeline gaps expose the missing history. CPU speed depends on
the hardware and selected models; this is not a claim that every laptop can run every model in real time.

Background inference uses an [AudioSet AST classifier](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
for speech/music/effects/notification candidates, [tiny Whisper through faster-whisper](https://github.com/SYSTRAN/faster-whisper)
on CPU with int8 computation for speech, and the first pinned CLAP music model for music vectors. Labels, transcripts
and source estimates can be wrong. The journal stores their provenance and errors with the waveform evidence.

### Start at login and stop

```powershell
.venv\Scripts\ears-journal.exe install-startup
.venv\Scripts\ears-journal.exe stop
.venv\Scripts\ears-journal.exe remove-startup
```

`install-startup` creates a hidden Windows login launcher, a Linux systemd user unit, or a macOS LaunchAgent. It takes
effect at the next login; `start` starts the current session. `stop` requests a clean flush and shutdown;
`remove-startup` removes the login integration without deleting history. Installation does not require an administrator.
Keep the installed environment at its original path. `run` runs in the foreground for troubleshooting.

Startup saves the journal/local-model settings to `service-config.json` inside the journal directory. Cloud API keys
are excluded. Re-run `install-startup` after changing journal settings. Point your MCP client at the same
`EARS_JOURNAL_DIR` as the daemon; the default is `%LOCALAPPDATA%/ears/journal` on Windows and
`~/.local/state/ears/journal` elsewhere.

### Cross-platform audio routing

- Windows uses WASAPI output loopback. The journal keeps raw audio and does not create game-session recordings.
- Linux uses a PulseAudio monitor, including PipeWire's PulseAudio compatibility server. Install the system's
  PulseAudio client libraries and `pactl` for active-app attribution. No microphone is selected by default.
- macOS needs a virtual input such as BlackHole and routing of system audio into it. Set `EARS_JOURNAL_DEVICE` to
  that input's name; Ears refuses to silently substitute the default microphone. Active-app attribution is unavailable.

Use `ears-journal devices` to list output devices and monitor/virtual inputs. Without a pinned device, Windows/Linux
follow default output changes and reconnect after failures. Connection errors remain visible through `timeline` and
`ears-journal status`. `active_apps` means apps active in the mixed output, not proof that a particular app produced
a particular sound.

### Search, listen again, and learn your taste

```python
timeline(since="1h ago", query="shout")
perceive(source="journal:EVENT_ID", question="What was that noise?", provider="gemini")
perceive(source="journal:EVENT_ID", question="Does the music work?", provider="qwen")
taste(event_id="EVENT_ID", rating=5, note="Love the swung drums and restrained bass")
taste(event_id="ANOTHER_EVENT_ID")
```

Times accept ISO timestamps with a timezone, Unix seconds, or relative forms such as `1h ago` and `2d ago`.
Search is a case-insensitive literal substring over transcripts, tags, apps and event details. Returned clips preserve
their original timestamps and can be perceived by any text-capable MCP client. A missing/expired clip produces an
explicit error. Unlike the `seconds` selection for files, a journal reference retrieves its entire bounded event clip.

Taste calibration saves explicit user ratings with their CLAP vector and measured evidence. It finds neighbours only
within the same model/revision, excludes the target's own rating, and estimates preference only after three similar
rated examples. Similar examples, user notes and measurements accompany the estimate; this is personal taste rather
than a track-quality score. `perceive` automatically includes available preference examples for journal clips.
Ratings survive clip expiration, are editable, and are capped at the latest 5,000 examples.

### Local Qwen perception

Run a compatible local [Qwen3-Omni vLLM server](https://github.com/QwenLM/Qwen3-Omni#vllm-serve-usage), then set:

```text
EARS_QWEN_URL=http://127.0.0.1:8901/v1
EARS_QWEN_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
```

Use `provider="qwen"` for on-demand perception. Set `EARS_JOURNAL_CAPTIONS=1` to add richer local captions to each
active journal segment. This option needs substantial inference hardware and can increase processing backlog.
The adapter uses loopback HTTP with proxies and redirects disabled; it does not send source audio to an external
provider. Hosting the Qwen model is a separate setup step and is not included in the lightweight journal install.

### Retention and budgets

The defaults retain at most 24 hours, 512 MiB of clips, and 20,000 event rows, evicting oldest events first.
Whichever limit is reached first wins, so loud/compressed content may retain less than a day. Retention runs during
recording and idle periods. SQLite pages are reused; model caches, ratings and small metadata files are separate from
the clip budget. A clip being written can temporarily exceed the budget by one bounded segment. Crash recovery
removes unreferenced clips. Audio cannot be recovered after retention removes it.

| Setting | Default / purpose |
|---|---|
| `EARS_JOURNAL_DIR` | Persistent journal and clips directory; use the same path in the daemon and MCP client |
| `EARS_JOURNAL_HOURS` | `24`, rolling retention in hours |
| `EARS_JOURNAL_MAX_MB` | `512`, clip budget in MiB |
| `EARS_JOURNAL_MAX_EVENTS` | `20000`, maximum event metadata rows |
| `EARS_JOURNAL_SEGMENT_SECONDS` | `12`, maximum grouped active clip length (up to 30, multiples of 2) |
| `EARS_JOURNAL_DEVICE` | Optional fixed output / required macOS virtual input |
| `EARS_JOURNAL_CLASSIFIER` | `1`; set `0` to disable AudioSet classification |
| `EARS_JOURNAL_WHISPER` | `1`; set `0` to disable transcription |
| `EARS_JOURNAL_CLAP` | `1`; set `0` to disable music vectors |
| `EARS_JOURNAL_WHISPER_MODEL` | `tiny`; model name or local faster-whisper directory |
| `EARS_JOURNAL_CLASSIFIER_MODEL` | `MIT/ast-finetuned-audioset-10-10-0.4593` or a local model directory |
| `EARS_JOURNAL_CPU_THREADS` | `2`, limits CPU inference threads |
| `EARS_JOURNAL_CAPTIONS` | `0`, opt-in local Qwen captions |

### Game captures, what to know

- It records **everything** the output device plays, so close YouTube/Spotify/Discord first (ears lists any other apps it heard).
- The saved recording and perception buffer preserve raw Windows loopback audio. A separate temporary signal undoes the Windows volume slider for loudness measurements, so correction cannot distort what the audio model hears. In-game volume settings and the per-app mixer *do* count.
- The "music level" ignores short loud sound effects. Constant effects (non-stop gunfire) still mix in; for the cleanest reading turn the game's SFX off while capturing.
- Browser games: keep the game's tab visible. A hidden or minimised tab can slow the game right down, and the music may never start.
- Recordings go to `ears captures\` in your music folder as FLAC (roughly 10-20 MB a minute), with a `.markers.json` beside each. Recording stops on its own after 90 minutes.

## Install

Clone the repository, then install the core server into a virtual environment:

```powershell
# Windows
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

```bash
# macOS / Linux
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
```

This installs the `ears-mcp` command. MP3 decoding uses libsndfile through `soundfile`, so ffmpeg is not required.

### Optional features

```powershell
# CPU stems and CLAP descriptions
.venv\Scripts\python.exe -m pip install -e ".[stems-cpu,describe-cpu]"

# NVIDIA CUDA 12.4 stems, then CLAP descriptions
.venv\Scripts\python.exe -m pip install -r requirements-stems.txt
.venv\Scripts\python.exe -m pip install -r requirements-describe.txt

# Basic Pitch note/MIDI transcription
.venv\Scripts\python.exe -m pip install -r requirements-notes.txt
.venv\Scripts\python.exe -m pip install --no-deps basic-pitch==0.4.0

# Windows system-audio capture
.venv\Scripts\python.exe -m pip install -e ".[capture]"
```

`basic-pitch` is installed with `--no-deps` because its declared TensorFlow dependency is incompatible with the supported Windows/Python combination; Ears uses its bundled ONNX model. `requirements.lock.txt` records the complete environment used during development.

The first CLAP call downloads approximately 1.3 GB of pinned model weights from Hugging Face. Later analysis is local. Use `capabilities()` to see what is available on the current installation.

## Connect an MCP client

Ears speaks standard MCP over stdio. Configure your client to run `ears-mcp` and set the folders it may read:

```json
{
  "mcpServers": {
    "ears": {
      "command": "C:\\path\\to\\ears\\.venv\\Scripts\\ears-mcp.exe",
      "env": {
        "EARS_ALLOWED_ROOTS": "C:\\Users\\me\\Music",
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

On macOS/Linux, the command is normally `/path/to/ears/.venv/bin/ears-mcp`. Client configuration formats differ, but the command, environment and stdio transport are the same. Claude Desktop, Claude Code, Codex, Cursor, Gemini CLI, VS Code and other MCP hosts can all use the server when they support local stdio tools.

Clients that do not forward images or audio to their model still receive the complete text and structured report.

## Settings

- `EARS_ALLOWED_ROOTS`: folders Ears may read. Separate paths with `;` on Windows or `:` on macOS/Linux. A JSON array of paths is also accepted. Default: the current user's `Music` folder.
- `EARS_SAVE_OUTPUTS`: set to `1` to save generated images as well as returning them inline. Default: off.
- `EARS_OUTPUT_DIR`: saved-image folder when `EARS_SAVE_OUTPUTS=1`. Default: `output/` in the repository.
- `EARS_CACHE_DIR`: opt-in persistent cache for learned CLAP fingerprints. Without it, the cache is memory-only.
- `EARS_CAPTURE_DIR`: Windows capture folder. Default: `<first allowed folder>/ears captures`; it must remain inside an allowed root.
- `EARS_MARK_HOTKEY`: Windows in-game marker hotkey. Default: `ctrl+alt+m`.
- `EARS_PERCEPTION_PROVIDER`: perception backend used by `provider="auto"`. Default: `gemini`.
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: enables the built-in Gemini perception backend.
- `EARS_GEMINI_MODEL`: Gemini model ID. Default: `gemini-3.8-flash`.
- `EARS_PERCEPTION_TIMEOUT`: provider request timeout in seconds. Default: `90`.

## Privacy and side effects

- Core analysis is local and does not upload source audio.
- The journal persists system audio, transcripts and active-app names locally until its retention limits apply.
  Starting it is explicit; login startup is installed only by `install-startup`. It does not call Gemini in the background.
- `perceive` sends only the selected immutable audio window to its configured provider. Gemini calls use its API directly and return text/JSON to any MCP client. Effectively silent windows are handled locally and are not uploaded.
- Perception memory is bounded, process-local and never written to disk. It disappears when Ears restarts.
- `describe`, `expect`, `listen_live` and `sounds_like` may download pinned CLAP weights from Hugging Face on first use.
- `audio_clip` returns source audio to the MCP client. The client may transmit it to its configured model provider, so the tool should only be called at the user's explicit request.
- `stems` writes WAV files next to the source; `notes` writes MIDI; `capture` writes recordings and marker files.
- Recent perception reads the capture buffer directly. `capture(action="snapshot")` is the explicit way to save a recent excerpt.
- Generated images are returned inline and are only saved when `EARS_SAVE_OUTPUTS=1`.

## Self-test

```powershell
.venv\Scripts\python.exe tests\selftest.py
.venv\Scripts\python.exe tests\contract_test.py
.venv\Scripts\python.exe tests\perception_test.py
.venv\Scripts\python.exe tests\journal_test.py
```

The synthetic-audio suite checks tempo, key, sections, clipping, MP3/FLAC decoding, comparison, path safety and game-level loudness. Perception tests cover silence, A/B input, provider failure, exact-audio follow-ups, event memory and separation of raw listening audio from corrected measurements. Optional model-backed checks run when their packages are installed. The contract test starts the real stdio server and verifies structured results and media contracts.

The device/provider acceptance checklist in [tests/perception_acceptance.md](tests/perception_acceptance.md) covers real music, mixed dialogue/effects, A/B judgement, text-only clients and failure/disconnection behaviour.

GitHub Actions runs the core suite on Windows, Linux and macOS with Python 3.12 and 3.13. Journal tests exercise
SQLite persistence, literal/time search, retention and budget eviction, immutable clip retrieval, singleton locking,
segmentation, shutdown, taste neighbours, and the local-model HTTP contract without recording live system audio.

## Licence

MIT. See [LICENSE](LICENSE).
