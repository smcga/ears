# Perception acceptance checklist

Run this checklist with the Windows capture extra installed and a perception provider configured. Record the model,
provider, audio device and date with the result; audio-model behaviour can change independently of Ears.

## 1. Recent music

1. Start `capture` and play at least 30 seconds of a known track.
2. Call `perceive(source="recent", seconds=30, question="What is actually audible, and is the music working?")`.
3. Pass when the result names concrete audible elements, gives useful uncertainty, contains original-waveform
   measurements, creates no snapshot file, and does not claim events outside the selected window.

## 2. Dialogue, music and an effect

1. Play a 30-second fixture containing separated dialogue, music and one distinctive effect at known times.
2. Ask Ears to identify each event and approximately when it occurs.
3. Pass when all present categories are identified, timestamps are within the agreed fixture tolerance, absent events
   are not invented, and both clip-relative and capture-session timestamps are returned.

## 3. A/B music judgement with evidence

1. Call `perceive` with mix A as `source` and mix B as `against`.
2. Ask which sounds clearer and why.
3. Pass when the answer identifies A and B, explains audible evidence, keeps recommendations separate, and uses or
   explicitly qualifies the deterministic measurements rather than treating taste as a numerical fact.

## 4. Text-only MCP client

1. Disable or use a client without MCP audio rendering.
2. Repeat case 1.
3. Pass when the client receives the complete answer and structured observation through text without `AudioContent`.

## 5. Silence and failures

- Silence or muted output: pass when Ears reports effective silence locally and uploads no audio.
- Device disconnect: pass when retained audio is labelled with a warning, or an empty buffer fails explicitly.
- Invalid/expired API key, timeout or provider disconnection: pass when the tool returns an actionable error and no
  invented observation.
- Follow-up: pass when using `memory_id` reuses the exact original waveform; an expired handle must fail explicitly.

The automated `tests/perception_test.py` and `tests/contract_test.py` cover these invariants without making paid API
calls. This checklist is the required provider/device benchmark before calling a backend production-ready.
