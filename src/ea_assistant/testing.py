from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .config import LLMCallConfig, STTConfig
from .domain import FACT_SECTIONS
from .models import Segment


class FakeAudio:
    def __init__(self) -> None:
        self.probed: list[str] = []
        self.normalized: list[tuple[str, str, str | None, str | None]] = []

    def probe(self, path: str) -> float:
        self.probed.append(path)
        return 3.0

    def normalize(
        self,
        path: str,
        output: str,
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        from pathlib import Path

        self.normalized.append((path, output, start, end))
        Path(output).write_bytes(b"fake wav")


class FakeSensors:
    def __init__(self, temp: float | None = 40, ac: bool | None = True) -> None:
        self.temp, self.ac = temp, ac

    def gpu_temperature(self) -> float | None:
        return self.temp

    def on_ac(self) -> bool | None:
        return self.ac


class FakeSpeechToText:
    def __init__(
        self, segments: list[Segment] | None = None, events: list[str] | None = None
    ) -> None:
        self.segments = segments or [Segment("s0001", 0, 1, "مرحبا", -0.1, 0.01, 1.0)]
        self.events = events if events is not None else []
        self.calls = 0
        self.configs: list[STTConfig] = []
        self.model_identifier = "fake-stt-revision"
        self.detected_language: str | None = "ar"
        self.language_probability: float | None = 0.97

    def validate_compute_type(self, config: STTConfig) -> None:
        if config.compute_type not in {"int8", "int8_float32", "float32"}:
            raise ValueError("unsupported fake compute type")

    def transcribe(
        self, path: str, config: STTConfig
    ) -> tuple[list[Segment], str, str]:
        self.events.append("stt-load")
        self.calls += 1
        self.configs.append(config)
        return self.segments, config.model, config.compute_type

    def release(self) -> None:
        self.events.append("stt-release")


class FakeLLM:
    def __init__(
        self,
        events: list[str] | None = None,
        responder: Callable[[str, dict[str, str]], str] | None = None,
    ) -> None:
        self.events = events if events is not None else []
        self.calls = 0
        self.instructions: list[str] = []
        self.contents: list[str] = []
        self.configs: list[LLMCallConfig] = []
        self.english_transcript_calls = 0
        self.extract_calls = 0
        self.responder = responder
        self.loaded = False
        self.digests = {"fake-llm": "llm-digest", "gemma4:12b": "llm-digest"}

    def ensure_gpu_free(self) -> None:
        self.events.append("llm-gpu-free-check")
        if self.loaded:
            raise RuntimeError("Fake LLM is still resident on GPU")

    def chat(
        self,
        instructions: str,
        content: str,
        config: LLMCallConfig,
        schema: dict[str, Any] | None = None,
    ) -> str:
        if not self.loaded:
            self.events.append("llm-load")
            self.loaded = True
        self.events.append("llm-call")
        self.calls += 1
        self.instructions.append(instructions)
        self.contents.append(content)
        self.configs.append(config)
        obj = json.loads(content) if content.startswith("{") else {}
        if instructions.startswith("Translate"):
            self.english_transcript_calls += 1
        if instructions.startswith("Extract"):
            self.extract_calls += 1
        if self.responder:
            return self.responder(instructions, obj)
        if instructions.startswith("Translate"):
            return json.dumps({key: "Hello" for key in obj})
        if instructions.startswith("Extract"):
            return json.dumps({section.value: [] for section in FACT_SECTIONS})
        return "Meeting summary."

    def unload(self, model: str) -> None:
        if self.loaded:
            self.events.append("llm-unload")
            self.loaded = False

    def model_digest(self, model: str) -> str:
        return self.digests.get(model, "digest-" + model)
