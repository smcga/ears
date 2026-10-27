# ears: lets Claude "listen" to your bounces

A local MCP server that measures audio files in your music folder and reports back in plain English, plus a spectrogram image Claude can look at. It measures; it doesn't hear.

## Tools

| Tool | What it does | Speed |
|---|---|---|
| `listen(path, bpm?, start_sec?, end_sec?)` | Tempo, key, LUFS/peak/clipping, stereo, a 4-bar table on FL's bar grid, detected sections, mix flags, plus a spectrogram (and a zoomed one if you give a range) | ~10 s for a 5-min track |
| `compare(path_a, path_b, bpm?)` | What changed between two versions: loudness, peaks, band balance per section, arrangement shifts, key, length | ~10–20 s |
| `stems(path, bpm?, force?)` | Demucs split into drums/bass/other/vocals, saved to `<name>_stems\` next to the file, plus a short report per stem | ~20 s on a recent NVIDIA GPU (much slower on CPU) + ~10 s analysis; reused next time |
| `notes(path, bpm?, stem?, bars?, quantize?)` | Basic Pitch transcription of a stem (default `other` = chords/melody; also `bass`, `vocals`, `mix`): chord chart per bar, bass roots, key from the notes, a beat.step note list for a bar range, a piano-roll image, and tempo-mapped `.mid` files in `<name>_stems\` | ~20 s (+ stem split the first time) |
| `describe(path, bpm?, phrases?)` | CLAP tags in words: genre/style, groove, texture, elements, mood, for the whole track and per section; optionally scores your own phrases ("garage-ish", "swung") | ~10 s (+ ~10 s model load on first call) |
| `expect(path, bpm?)` | Listening as prediction: guesses every bar from the bars before it and measures the surprise. Biggest moments (what was expected, what happened, on/off the phrase grid, bars to settle), predictable stretches, which parts return, plus a learned similarity map with the surprise curve | ~10 s |
| `listen_live(path, bpm?, bars?, my_guess?)` | Step through a track a few bars at a time without seeing ahead. Each step shows what was just heard, your previous guess next to the predictor's forecast and what happened, and a spectrogram with the future greyed out | a few s per step |
| `sounds_like(path, against? or folder?)` | Describes each section by the closest-sounding section in your other tracks (learned CLAP fingerprints) instead of adjectives | ~15 s per new reference, then cached |
| `capture(action, ...)` | Records what Windows is playing (Windows only), e.g. a game while you play it. `start`, then press **Ctrl+Alt+M** in-game as each level starts (or tell Claude to `mark`); `status` = live readout ("level 3 is running +4 dB vs the earlier levels"); `snapshot` = save the last 30 s so Claude can `listen`/`describe` what's playing now; `stop` = per-level loudness report + timeline | live; report in a few s |
| `levels(path, splits?, labels?, target_lufs?)` | Loudness per level: a capture split by its markers (or by silences, or your times), or a **folder** of music files (one per level). Gives each level's music level, the dB change and linear volume (e.g. for a Unity AudioSource) to match the others or a target | ~1 s per minute of audio |
| `find_audio(query, folder?)` | Finds bounces by name, newest first | <1 s |

Bar numbers match FL Studio's playlist **if the bounce starts at bar 1**. Pass `bpm` (e.g. 130) for exact bars.

## Things to say to Claude

- "Listen to C:\Users\me\Music\My Project\my_track.mp3 at 130 bpm"
- "Compare my_track.mp3 and my_track_v2.mp3 in my Music folder: what changed?"
- "Zoom into 0:50–1:05 of my_track at 130 bpm. Does the drop hit cleanly?"
- "Split my_track into stems and tell me where the bass plays and what notes it uses"
- "Transcribe the chords in my_track at 130 bpm and show me the notes in bars 53-60"
- "Get me the bassline of my_track as MIDI"
- "Describe my_track. Does it come across as garage-ish or more techno? Is the groove swung?"
- "Run expect on my_track at 130 bpm: where does it break expectations?"
- "Listen to my_track live, 8 bars at a time, and tell me what you expect before each step"
- "What does my_new_idea sound like among my other bounces?"
- "Find my latest bounce and check it for clipping"
- "Listen to my game while I play. I'll press Ctrl+Alt+M at each new level" ... "how's this level compared to the others?" ... "stop: which levels need turning up or down?"
- "Grab the last 30 seconds of the game and tell me what the music sounds like"
- "Compare the loudness of the music files in my Game music folder"

Note names follow FL Studio's piano roll (middle C = C5). The saved MIDI is at your BPM in 4/4 and snapped to 1/16, so dragging it into FL at bar 1 lines it up with the bounce.

### Game captures, what to know

- It records **everything** the output device plays, so close YouTube/Spotify/Discord first (ears lists any other apps it heard).
- It undoes the Windows volume slider, so numbers are what the game sends, not how loud your speakers are. In-game volume settings and the per-app mixer *do* count, so keep them where players will have them.
- The "music level" ignores short loud sound effects. Constant effects (non-stop gunfire) still mix in; for the cleanest reading turn the game's SFX off while capturing.
- Browser games: keep the game's tab visible. A hidden or minimised tab can slow the game right down, and the music may never start.
- Recordings go to `ears captures\` in your music folder as FLAC (roughly 10-20 MB a minute), with a `.markers.json` beside each. Recording stops on its own after 90 minutes.

## Install

Windows 10/11 and Python 3.12. From the folder you cloned ears into:

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-stems.txt      # optional: stems tool, ~2.5 GB
.venv\Scripts\python.exe -m pip install -r requirements-notes.txt      # optional: notes tool
.venv\Scripts\python.exe -m pip install --no-deps basic-pitch==0.4.0
.venv\Scripts\python.exe -m pip install -r requirements-describe.txt   # optional: describe tool
.venv\Scripts\python.exe -m pip install -r requirements-capture.txt    # optional: capture tool
```

`basic-pitch` must go in with `--no-deps`: it declares an old TensorFlow that doesn't exist for Python 3.12 on Windows, and ears uses its bundled ONNX model instead (see `requirements-notes.txt`). `requirements.lock.txt` lists the exact versions ears was tested with.

MP3 decoding uses libsndfile (bundled with `soundfile`), so ffmpeg isn't needed.

## Register with Claude

Replace `C:\path\to\ears` with where you cloned ears, and `C:\Users\me\Music` with the folder that holds your audio.

Claude Code:

```
claude mcp add ears --scope user -e EARS_ALLOWED_ROOTS=C:\Users\me\Music -- C:\path\to\ears\.venv\Scripts\python.exe C:\path\to\ears\ears_server.py
```

Claude Desktop (Settings > Developer > Edit Config, then restart Claude Desktop):

```json
"mcpServers": {
  "ears": {
    "command": "C:\\path\\to\\ears\\.venv\\Scripts\\python.exe",
    "args": ["C:\\path\\to\\ears\\ears_server.py"],
    "env": { "EARS_ALLOWED_ROOTS": "C:\\Users\\me\\Music" }
  }
}
```

## Settings (env vars in the config above)

- `EARS_ALLOWED_ROOTS`: folders ears may read, separated by `;` (default: the `Music` folder in your user profile). Anything else is refused.
- `EARS_OUTPUT_DIR`: where images are saved (default `output\` in the ears folder).
- `EARS_CAPTURE_DIR`: where game captures go (default `<first allowed folder>\ears captures`; must be inside an allowed folder).
- `EARS_MARK_HOTKEY`: the in-game marker hotkey (default `ctrl+alt+m`).

## Self-test

```
.venv\Scripts\python.exe tests\selftest.py
```

Builds a synthetic 130 BPM track (kick, 45 Hz sub, G major chord, quiet 8-bar intro, one clipped peak) and checks tempo, key, the bar-9 section change, clipping and the sub flag. It also checks MP3/FLAC decoding, `compare`, the folder guard, `notes` on a synthetic Am-F-C-G progression (chords, bass roots, MIDI), per-level loudness on a fake three-level game capture (including one with loud sound effects), a short silent system-audio capture, and a real MCP round trip. The optional tools' checks are skipped if their packages aren't installed.
