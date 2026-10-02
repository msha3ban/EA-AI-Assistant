from __future__ import annotations

from typing import Any, Protocol

from .config import LLMCallConfig, STTConfig
from .models import Segment


class Audio(Protocol):
    def probe(self, path: str) -> float: ...
    def normalize(
        self,
        path: str,
        output: str,
        start: str | None = None,
        end: str | None = None,
    ) -> None: ...


class SpeechToText(Protocol):
    model_identifier: str

    def validate_compute_type(self, config: STTConfig) -> None: ...
    def transcribe(
        self, path: str, config: STTConfig
    ) -> tuple[list[Segment], str, str]: ...
    def release(self) -> None: ...


class LLM(Protocol):
    def chat(
        self,
        instructions: str,
        content: str,
        config: LLMCallConfig,
        schema: dict[str, Any] | None = None,
    ) -> str: ...
    def unload(self, model: str) -> None: ...
    def ensure_gpu_free(self) -> None: ...
    def model_digest(self, model: str) -> str: ...


class Sensors(Protocol):
    def gpu_temperature(self) -> float | None: ...
    def on_ac(self) -> bool | None: ...
