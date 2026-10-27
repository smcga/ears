"""Small, client-neutral MCP contract test for CI and local development."""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ears-contract-") as raw:
        folder = Path(raw)
        path = folder / "tone.wav"
        silence_path = folder / "silence.wav"
        sr = 24000
        t = np.arange(sr * 3) / sr
        sf.write(path, 0.1 * np.sin(2 * np.pi * 440 * t), sr)
        sf.write(silence_path, np.zeros(sr * 3, dtype=np.float32), sr)
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "ears_server.py")],
            env={**os.environ, "EARS_ALLOWED_ROOTS": str(folder)},
        )
        async with Client(params, mode="legacy", read_timeout_seconds=20) as client:
            tools = (await client.list_tools()).tools
            names = {tool.name for tool in tools}
            assert {"listen", "audio_clip", "capabilities", "find_audio", "perceive", "auditory_memory"} <= names
            assert all(tool.output_schema for tool in tools)
            by_name = {tool.name: tool for tool in tools}
            assert by_name["listen"].annotations.read_only_hint is True
            assert by_name["stems"].annotations.destructive_hint is True
            assert by_name["capture"].annotations.read_only_hint is False

            result = await client.call_tool("capabilities", {})
            assert not result.is_error
            assert result.structured_content["features"]["audio_clip"]["available"]

            result = await client.call_tool(
                "audio_clip",
                {"path": str(path), "start_sec": 0, "end_sec": 3},
            )
            assert not result.is_error
            assert [block.type for block in result.content] == ["text", "audio"]
            assert result.structured_content["range_sec"]["duration"] == 3

            result = await client.call_tool("listen", {"path": str(folder / "missing.wav")})
            assert result.is_error

            result = await client.call_tool("perceive", {
                "source": str(silence_path),
                "seconds": 3,
                "question": "What do you hear?",
            })
            assert not result.is_error
            assert result.structured_content["heard"]["summary"] == "Effectively silent audio."
            assert result.structured_content["provenance"]["audio_uploaded"] is False

    print("MCP contract: ok")


if __name__ == "__main__":
    asyncio.run(main())
