from __future__ import annotations

import ipaddress
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

from .domain import StageName

DEFAULT_VAD_PARAMETERS: dict[str, Any] = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 200,
    "max_speech_duration_s": 30.0,
}


@dataclass(frozen=True)
class STTConfig:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "int8"
    language: str = "ar"
    beam_size: int = 5
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = field(
        default_factory=lambda: DEFAULT_VAD_PARAMETERS.copy()
    )
    download_root: str | None = None

    def adapter_values(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LLMCallConfig:
    model: str
    num_ctx: int
    num_predict: int
    think: bool = False
    temperature: float = 0.0


@dataclass(frozen=True)
class StageConfig:
    num_ctx: int
    num_predict: int
    think: bool = False
    temperature: float = 0.0

    def adapter_values(self, model: str) -> LLMCallConfig:
        return LLMCallConfig(
            model, self.num_ctx, self.num_predict, self.think, self.temperature
        )


@dataclass(frozen=True)
class LLMConfig:
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen3:8b"
    timeout: float = 120
    translate: StageConfig = field(default_factory=lambda: StageConfig(8192, 4096))
    extract: StageConfig = field(default_factory=lambda: StageConfig(8192, 4096))
    summary: StageConfig = field(default_factory=lambda: StageConfig(4096, 512))
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
            _merge(raw, tomllib.load(f))
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
            "Non-loopback LLM endpoint is disabled; set allow_remote_endpoint = true explicitly"
        )
    if config.stt.compute_type not in {"int8", "int8_float32", "float32"}:
        raise ValueError(
            f"Unsupported Pascal GPU compute type {config.stt.compute_type!r}; use int8, int8_float32, or float32"
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
