# ears

Local audio analysis for any AI client that supports local [Model Context Protocol](https://modelcontextprotocol.io/) tools over stdio.

Ears turns audio files into measurements, musical structure, learned descriptions and visualisations that a language model can reason about. Every tool returns a text report plus structured MCP data. Image-capable clients also receive spectrograms; audio-capable clients can receive an explicitly requested short excerpt.

Ears is a sensory layer, not an LLM. Most tools measure or classify audio locally; they do not reproduce human hearing or decide whether music is good.

## Compatibility

| Capability | Requirement |
|---|---|
| Text and structured reports | Any local stdio MCP client |
| Spectrograms and piano rolls | Client/model support for MCP `ImageContent`; text remains available without it |
| Short source-audio excerpts | Client/model support for MCP `AudioContent`; only sent when `audio_clip` is called |
| Core file analysis | Windows, macOS or Linux; Python 3.12 or 3.13 |
| System-audio capture | Windows 10/11 only |
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
| `capabilities()` | Reports installed optional features, platform support, allowed folders and persistence settings | <1 s |
| `capture(action, ...)` | Records what Windows is playing (Windows only), e.g. a game while you play it. `start`, then press **Ctrl+Alt+M** in-game as each level starts (or ask your assistant to `mark`); `status` = live readout; `snapshot` = save the last 30 s for analysis; `stop` = per-level loudness report + timeline | live; report in a few s |
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
- "What does my_new_idea sound like among my other bounces?"
- "Find my latest bounce and check it for clipping"
- "Listen to my game while I play. I'll press Ctrl+Alt+M at each new level" ... "how's this level compared to the others?" ... "stop: which levels need turning up or down?"
- "Grab the last 30 seconds of the game and tell me what the music sounds like"
- "Compare the loudness of the music files in my Game music folder"

Note names follow FL Studio's piano roll (middle C = C5). The saved MIDI is at your BPM in 4/4 and snapped to 1/16, so dragging it into FL at bar 1 lines it up with the bounce.

`listen_live` returns a `session_id`. Clients must pass that handle into each later call so independent conversations cannot affect one another.

### Game captures, what to know

- It records **everything** the output device plays, so close YouTube/Spotify/Discord first (ears lists any other apps it heard).
- It undoes the Windows volume slider, so numbers are what the game sends, not how loud your speakers are. In-game volume settings and the per-app mixer *do* count, so keep them where players will have them.
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
        "EARS_ALLOWED_ROOTS": "C:\\Users\\me\\Music"
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

## Privacy and side effects

- Core analysis is local and does not upload source audio.
- `describe`, `expect`, `listen_live` and `sounds_like` may download pinned CLAP weights from Hugging Face on first use.
- `audio_clip` returns source audio to the MCP client. The client may transmit it to its configured model provider, so the tool should only be called at the user's explicit request.
- `stems` writes WAV files next to the source; `notes` writes MIDI; `capture` writes recordings and marker files.
- Generated images are returned inline and are only saved when `EARS_SAVE_OUTPUTS=1`.

## Self-test

```powershell
.venv\Scripts\python.exe tests\selftest.py
.venv\Scripts\python.exe tests\contract_test.py
```

The synthetic-audio suite checks tempo, key, sections, clipping, MP3/FLAC decoding, comparison, path safety and game-level loudness. Optional model-backed checks run when their packages are installed. The contract test starts the real stdio server, verifies every tool has an output schema, checks structured results and exercises standard MCP audio content.

GitHub Actions runs the core suite on Windows and Linux with Python 3.12 and 3.13.

## Licence

MIT. See [LICENSE](LICENSE).
