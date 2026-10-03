from __future__ import annotations

import ipaddress
import math
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from .domain import StageName
from .models import Segment

DEFAULT_VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
    "max_speech_duration_s": 30.0,
}


def valid_detection_value(value: Any, *, probability: bool) -> bool:
    if isinstance(value, bool):
        return False
    if probability:
        return (
            isinstance(value, (int, float))
            and 0 <= value <= 1
            and math.isfinite(value)
        )
    return isinstance(value, int) and value > 0


@dataclass(frozen=True)
class STTConfig:
    engine: str = "faster-whisper"
    model: str = "large-v3"
    backend: str = "auto"
    device: str = "cuda"
    compute_type: str = "int8"
    language: str = "ar"
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_VAD_PARAMETERS.copy()
    )
    download_root: str | None = None
    initial_prompt: str | None = None
    hotwords: str | None = None
    vocabulary: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.engine not in {"faster-whisper", "transcribe-cpp"}:
            raise ValueError(f"Unknown speech-to-text engine {self.engine!r}")
        if self.engine == "transcribe-cpp":
            if self.language == "auto" or not self.language:
                raise ValueError(
                    "transcribe-cpp language must be set; auto is unsupported"
                )
            if not self.model.endswith(".gguf"):
                raise ValueError("transcribe-cpp model must be a .gguf path")
            if self.backend not in {"auto", "cpu", "cuda", "vulkan"}:
                raise ValueError(
                    f"Unsupported transcribe-cpp backend {self.backend!r}"
                )

    def adapter_values(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("vocabulary")
        return values


@dataclass(frozen=True)
class LLMCallConfig:
    model: str
    num_ctx: int
    num_predict: int
    think: bool = False
    temperature: float = 0.0
    timeout: float = 120.0


@dataclass(frozen=True)
class StageConfig:
    num_ctx: int
    num_predict: int
    think: bool = False
    temperature: float = 0.0
    timeout: float = 120.0

    def adapter_values(self, model: str) -> LLMCallConfig:
        return LLMCallConfig(
            model,
            self.num_ctx,
            self.num_predict,
            self.think,
            self.temperature,
            self.timeout,
        )


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "gemma4:12b"
    timeout: float = 120
    translate: StageConfig = field(
        default_factory=lambda: StageConfig(8192, 4096, timeout=600)
    )
    extract: StageConfig = field(
        default_factory=lambda: StageConfig(8192, 4096, timeout=600)
    )
    summary: StageConfig = field(
        default_factory=lambda: StageConfig(8192, 512, timeout=300)
    )
    unload_timeout: float = 15.0
    unload_poll_interval: float = 0.25

    def stage(self, name: StageName) -> StageConfig:
        return cast(StageConfig, getattr(self, name.value))


@dataclass(frozen=True)
class ThermalConfig:
    hard_stop_c: float = 95
    require_ac: bool = True


@dataclass(frozen=True)
class DetectionConfig:
    compression_ratio: float = 2.4
    no_speech_prob: float = 0.7
    avg_logprob: float = -1.0
    word_probability: float = 0.5
    min_low_confidence_words: int = 1
    avg_logprob_only: float | None = None

    def __post_init__(self) -> None:
        if not valid_detection_value(self.word_probability, probability=True):
            raise ValueError("word_probability must be between 0 and 1")
        if not valid_detection_value(
            self.min_low_confidence_words, probability=False
        ):
            raise ValueError("min_low_confidence_words must be a positive integer")
        if self.avg_logprob_only is not None and (
            isinstance(self.avg_logprob_only, bool)
            or not isinstance(self.avg_logprob_only, (int, float))
            or not math.isfinite(self.avg_logprob_only)
        ):
            raise ValueError("avg_logprob_only must be a number or omitted")

    def detector_rules(self) -> tuple[tuple[str, Callable[[Segment], bool]], ...]:
        rules: list[tuple[str, Callable[[Segment], bool]]] = [
            (
                "low-confidence words",
                lambda segment: segment.confidence_signals
                and sum(
                    probability < self.word_probability
                    for probability in segment.word_probabilities
                ) >= self.min_low_confidence_words,
            ),
        ]
        if self.avg_logprob_only is not None:
            rules.append(
                (
                    "low average log probability",
                    lambda segment: (
                        self.avg_logprob_only is not None
                        and segment.confidence_signals
                        and segment.avg_logprob < self.avg_logprob_only
                    ),
                )
            )
        rules.extend(
            [
                (
                    "repetition loop",
                    lambda segment: segment.compression_ratio > self.compression_ratio,
                ),
                (
                    "likely text over silence",
                    lambda segment: (
                        segment.confidence_signals
                        and segment.no_speech_prob > self.no_speech_prob
                        and segment.avg_logprob < self.avg_logprob
                    ),
                ),
            ]
        )
        return tuple(rules)

    def active_detectors(self) -> tuple[str, ...]:
        return tuple(reason for reason, _ in self.detector_rules())


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = field(
        default_factory=lambda: Path("~/.local/share/ea-assistant").expanduser()
    )
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    allow_remote_endpoint: bool = False


def _defaults() -> dict[str, Any]:
    defaults = AppConfig()
    return {
        "data_dir": str(defaults.data_dir),
        "stt": asdict(defaults.stt),
        "llm": {
            "endpoint": defaults.llm.endpoint,
            "model": defaults.llm.model,
            "timeout": defaults.llm.timeout,
            "unload_timeout": defaults.llm.unload_timeout,
            "unload_poll_interval": defaults.llm.unload_poll_interval,
            "stages": {
                name.value: asdict(defaults.llm.stage(name))
                for name in (StageName.TRANSLATE, StageName.EXTRACT, StageName.SUMMARY)
            },
        },
        "thermal": asdict(defaults.thermal),
        "detection": asdict(defaults.detection),
        "allow_remote_endpoint": defaults.allow_remote_endpoint,
    }


def load_config(path: str | None = None) -> AppConfig:
    raw = _defaults()
    chosen = (
        Path(path).expanduser()
        if path
        else Path(str(raw["data_dir"])).expanduser() / "config.toml"
    )
    if chosen.exists():
        with chosen.open("rb") as f:
            user_values = tomllib.load(f)
        if "vocabulary" in user_values.get("stt", {}):
            raise ValueError("[stt] vocabulary is managed by the application")
        _merge(raw, user_values)
    llmraw = raw["llm"]
    stages = llmraw["stages"]
    config = AppConfig(
        Path(str(raw["data_dir"])).expanduser(),
        STTConfig(**raw["stt"]),
        LLMConfig(
            endpoint=str(llmraw["endpoint"]),
            model=str(llmraw["model"]),
            timeout=float(llmraw["timeout"]),
            translate=StageConfig(**stages[StageName.TRANSLATE.value]),
            extract=StageConfig(**stages[StageName.EXTRACT.value]),
            summary=StageConfig(**stages[StageName.SUMMARY.value]),
            unload_timeout=float(llmraw["unload_timeout"]),
            unload_poll_interval=float(llmraw["unload_poll_interval"]),
        ),
        ThermalConfig(**raw["thermal"]),
        DetectionConfig(**raw["detection"]),
        bool(raw["allow_remote_endpoint"]),
    )
    if not config.allow_remote_endpoint and not _is_loopback(config.llm.endpoint):
        raise ValueError(
            "Non-loopback LLM endpoint is disabled; set "
            "allow_remote_endpoint = true explicitly"
        )
    if config.stt.engine == "faster-whisper" and config.stt.compute_type not in {
        "int8",
        "int8_float32",
        "float32",
    }:
        raise ValueError(
            f"Unsupported Pascal GPU compute type {config.stt.compute_type!r}; "
            "use int8, int8_float32, or float32"
        )
    return config


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(dst.get(key), dict) and isinstance(value, dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def _is_loopback(endpoint: str) -> bool:
    try:
        host = urlsplit(endpoint).hostname
        return host is not None and (
            host == "localhost"
            or host.endswith(".localhost")
            or ipaddress.ip_address(host).is_loopback
        )
    except ValueError:
        return False
