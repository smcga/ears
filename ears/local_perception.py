"""Qwen audio via a user-operated, loopback-only vLLM compatible server."""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .perception import PerceptionError, _prompt, _wav_bytes


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PerceptionError("The local audio backend redirected its request; refusing to forward audio.")


class QwenBackend:
    name = "qwen"
    network = True  # HTTP transport, but no external provider.
    audio_leaves_computer = False

    def __init__(self):
        self.base_url = os.environ.get("EARS_QWEN_URL", "http://127.0.0.1:8901/v1").rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise PerceptionError("EARS_QWEN_URL must use a local loopback address.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PerceptionError("EARS_QWEN_URL must be a plain loopback API base URL.")
        self.model = os.environ.get("EARS_QWEN_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")

    def perceive(self, windows, question, prior_turns, measured):
        content = [{"type": "text", "text": _prompt(question, windows, prior_turns, measured)}]
        for window in windows:
            content += [{"type": "text", "text": "Audio label: " + window.label},
                        {"type": "audio_url", "audio_url": {
                            "url": "data:audio/wav;base64," + base64.b64encode(_wav_bytes(window)).decode("ascii")}}]
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps({
            "model": self.model, "messages": [{"role": "user", "content": content}],
            "modalities": ["text"], "temperature": 0.2, "max_tokens": 2048,
        }).encode(), headers={"Content-Type": "application/json"})
        # Disable environment proxies as well as redirects: this backend is strictly local.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=float(os.environ.get("EARS_PERCEPTION_TIMEOUT", "90"))) as response:
                payload = json.load(response)
            text = payload["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.strip("`").removeprefix("json").strip()
            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                result = {"answer": text, "summary": text, "uncertainty": ["Unstructured local-model response."]}
            if not isinstance(result, dict):
                raise ValueError("Expected an object from the local model.")
            return {"result": result, "usage": payload.get("usage", {})}
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as exc:
            raise PerceptionError(f"Local Qwen perception failed: {exc}") from exc
