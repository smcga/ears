"""Lazy CPU models for local journal enrichment. Never download while listening."""
from __future__ import annotations

import os

import numpy as np
import soxr

from .perception import AudioWindow, signal_evidence

AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


class LocalModels:
    def __init__(self, config, download=False):
        self.config = config
        self.local_only = not download
        self._classifier = self._whisper = self._clap = None

    def classifier(self):
        if self._classifier is None:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            torch.set_num_threads(max(1, int(os.environ.get("EARS_JOURNAL_CPU_THREADS", "2"))))
            model_id = os.environ.get("EARS_JOURNAL_CLASSIFIER_MODEL", AST_MODEL)
            kwargs = {"local_files_only": self.local_only}
            extractor = AutoFeatureExtractor.from_pretrained(model_id, **kwargs)
            model = AutoModelForAudioClassification.from_pretrained(model_id, use_safetensors=True, **kwargs)
            self._classifier = (extractor, model.eval().cpu())
        return self._classifier

    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel
            self._whisper = WhisperModel(
                os.environ.get("EARS_JOURNAL_WHISPER_MODEL", "tiny"), device="cpu", compute_type="int8",
                cpu_threads=max(1, int(os.environ.get("EARS_JOURNAL_CPU_THREADS", "2"))),
                local_files_only=self.local_only,
            )
        return self._whisper

    def clap(self):
        if self._clap is None:
            from transformers import ClapModel, ClapProcessor
            from .describe import MODELS
            model_id, revision = MODELS[0]
            kwargs = {"revision": revision, "local_files_only": self.local_only}
            self._clap = (
                ClapProcessor.from_pretrained(model_id, **kwargs),
                ClapModel.from_pretrained(model_id, use_safetensors=True, **kwargs).eval().cpu(),
            )
        return self._clap

    def prepare(self):
        for enabled, load in ((self.config.classifier, self.classifier), (self.config.whisper, self.whisper),
                              (self.config.clap, self.clap)):
            if enabled:
                load()

    def classify(self, audio, sr):
        import torch
        extractor, model = self.classifier()
        mono = soxr.resample(audio.mean(axis=1), sr, 16000).astype(np.float32)
        with torch.inference_mode():
            inputs = extractor(mono, sampling_rate=16000, return_tensors="pt")
            # AudioSet is multi-label; softmax would wrongly force speech/music/effects to compete.
            probabilities = model(**inputs).logits.sigmoid()[0].cpu().numpy()
        indexes = np.argsort(probabilities)[-8:][::-1]
        return [{"label": model.config.id2label[int(i)], "score": round(float(probabilities[i]), 4)}
                for i in indexes if probabilities[i] >= 0.15]

    def analyse(self, samples, sr):
        result = {"kind": "activity", "tags": [], "text": "", "errors": [],
                  "measured": signal_evidence(AudioWindow("journal", samples, sr, "capture"))}
        labels = []
        if self.config.classifier:
            try:
                labels = self.classify(samples, sr)
                result["classification"] = labels
                result["classifier_model"] = os.environ.get("EARS_JOURNAL_CLASSIFIER_MODEL", AST_MODEL)
            except Exception as exc:
                result["errors"].append(f"classifier: {type(exc).__name__}: {exc}")
        names = " ".join(row["label"].lower() for row in labels)
        if any(word in names for word in ("music", "singing", "instrument", "guitar", "piano", "drum")):
            result["tags"].append("music")
            result["kind"] = "music"
        speech = any(word in names for word in ("speech", "conversation", "narration", "shout", "speak"))
        if speech:
            result["tags"].append("speech")
            if result["kind"] == "activity":
                result["kind"] = "speech"
        if any(word in names for word in ("beep", "bell", "alarm", "telephone", "notification")):
            result["tags"].append("notification")
        if labels and not result["tags"]:
            result["tags"].append("sound_effect")
        result["tags"] += [row["label"] for row in labels[:5]]
        if self.config.whisper and (speech or not labels):
            try:
                mono = soxr.resample(samples.mean(axis=1), sr, 16000).astype(np.float32)
                segments, info = self.whisper().transcribe(
                    mono, beam_size=1, vad_filter=True, condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                )
                transcript = [{"start_sec": float(s.start), "end_sec": float(s.end), "text": s.text.strip()}
                              for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
                result["transcript"] = transcript
                result["language"] = info.language
                result["text"] = " ".join(s["text"] for s in transcript)
                if transcript and "speech" not in result["tags"]:
                    result["tags"].append("speech")
                    if result["kind"] == "activity":
                        result["kind"] = "speech"
            except Exception as exc:
                result["errors"].append(f"whisper: {type(exc).__name__}: {exc}")
        if self.config.clap and "music" in result["tags"]:
            try:
                import torch
                from .describe import MODELS
                proc, model = self.clap()
                mono = soxr.resample(samples.mean(axis=1), sr, 48000).astype(np.float32)
                with torch.inference_mode():
                    inputs = proc(audio=[mono], sampling_rate=48000, return_tensors="pt")
                    vector = model.get_audio_features(**inputs).cpu().numpy()[0]
                result["vector"] = vector / max(float(np.linalg.norm(vector)), 1e-9)
                result["vector_model"] = "@".join(MODELS[0])
            except Exception as exc:
                result["errors"].append(f"clap: {type(exc).__name__}: {exc}")
        if self.config.captions:
            try:
                from .local_perception import QwenBackend
                backend = QwenBackend()
                window = AudioWindow("journal", samples, sr, "capture")
                response = backend.perceive((window,), "Describe the audible events concisely.", [], [result["measured"]])
                result["caption"] = response["result"].get("summary", "")
            except Exception as exc:
                result["errors"].append(f"local_caption: {type(exc).__name__}: {exc}")
        result["text"] = " ".join(filter(None, [result["text"], result.get("caption", "")]))
        if not result["text"]:
            result["text"] = ", ".join(result["tags"]) or "Unclassified audio activity"
        return result
