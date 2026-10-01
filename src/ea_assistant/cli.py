from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import math
import re
import shutil
import subprocess
import tempfile
import wave
from array import array
from dataclasses import replace
from datetime import date
from pathlib import Path

from .application import Application
from .config import AppConfig, load_config
from .ollama import OllamaClient
from .real_adapters import FasterWhisper, FfmpegAudio, MachineSensors


def build_application(config: AppConfig) -> Application:
    return Application(
        FfmpegAudio(),
        FasterWhisper(),
        OllamaClient(
            config.llm.endpoint,
            config.llm.timeout,
            unload_timeout=config.llm.unload_timeout,
            poll_interval=config.llm.unload_poll_interval,
        ),
        MachineSensors(),
        config,
    )


def main(argv: list[str] | None = None, app: Application | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ea")
    parser.add_argument("--config", default=None)
    subs = parser.add_subparsers(dest="command", required=True)
    process = subs.add_parser("process")
    process.add_argument("--config", default=argparse.SUPPRESS)
    process.add_argument("recording")
    process.add_argument("--title", required=True)
    process.add_argument("--date", required=True)
    process.add_argument("--topic")
    pre = subs.add_parser("preflight")
    pre.add_argument("--config", default=argparse.SUPPRESS)
    setup = subs.add_parser("setup-models")
    setup.add_argument("--config", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "setup-models":
        try:
            from faster_whisper import download_model

            download_model(
                config.stt.model,
                local_files_only=False,
                cache_dir=config.stt.download_root,
            )
        except ImportError as exc:
            raise RuntimeError("Install the stt extra before setup-models") from exc
        return 0
    if args.command == "preflight":
        return preflight(config)
    application = app or build_application(config)
    meeting = application.create_meeting(
        args.recording, args.title, date.fromisoformat(args.date), args.topic
    )
    application.process(meeting)
    print(meeting.folder)
    return 0


def preflight(config: AppConfig) -> int:
    import ctranslate2
    import faster_whisper

    stt = FasterWhisper()
    llm = OllamaClient(
        config.llm.endpoint,
        config.llm.timeout,
        unload_timeout=config.llm.unload_timeout,
        poll_interval=config.llm.unload_poll_interval,
    )
    report: dict[str, object] = {
        "ctranslate2": ctranslate2.__version__,
        "faster_whisper": faster_whisper.__version__,
        "cuda_device_count": ctranslate2.get_cuda_device_count(),
        "supported_compute_types_cuda": [],
        "configured_compute_type": config.stt.compute_type,
        "effective_compute_type": None,
        "ffmpeg": shutil.which("ffmpeg"),
        "ollama_reachable": False,
        "model_present": False,
        "model_digest": "unknown",
        "gpu_temperature_c": MachineSensors().gpu_temperature(),
        "on_ac": MachineSensors().on_ac(),
        "driver_version": None,
        "gpu_name": None,
        "compute_capability": None,
        "cuda_runtime_version": _cuda_runtime_version(),
        "cuda_version_reported_by_nvidia_smi": None,
        "cudnn_version": _cudnn_version(),
    }
    try:
        report["supported_compute_types_cuda"] = sorted(
            ctranslate2.get_supported_compute_types("cuda")
        )
    except RuntimeError:
        report["supported_compute_types_cuda"] = []
    try:
        digest = llm.model_digest(config.llm.model)
        report["model_digest"] = digest
        report["model_present"] = digest != "unknown"
        report["ollama_reachable"] = True
    except (OSError, RuntimeError, ValueError):
        pass
    _gpu_information(report)
    try:
        stt.validate_compute_type(config.stt)
        llm.ensure_gpu_free()
        with tempfile.TemporaryDirectory(prefix="ea-preflight-") as directory:
            sample = Path(directory) / "tone.wav"
            _write_tone(sample)
            _, _, compute_type = stt.transcribe(
                str(sample), replace(config.stt, vad_filter=False)
            )
            report["effective_compute_type"] = compute_type
    except (OSError, RuntimeError, ValueError) as exc:
        report["stt_inference_error"] = str(exc)
    finally:
        stt.release()
    if report["effective_compute_type"] is None:
        report["effective_compute_type"] = "unavailable"
    print(json.dumps(report, indent=2))
    return 0


def _write_tone(path: Path) -> None:
    samples = array(
        "h",
        (
            int(1000 * math.sin(2 * math.pi * 440 * index / 16000))
            for index in range(16000)
        ),
    )
    if samples.itemsize != 2:
        samples.byteswap()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())


def _gpu_information(report: dict[str, object]) -> None:
    try:
        query = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,compute_cap",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[0]
            .split(",")
        )
        report["driver_version"] = query[0].strip()
        report["gpu_name"] = query[1].strip()
        report["compute_capability"] = query[2].strip()
        output = subprocess.run(
            ["nvidia-smi"], check=True, capture_output=True, text=True
        ).stdout
        match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
        if match:
            report["cuda_version_reported_by_nvidia_smi"] = match.group(1)
    except (OSError, subprocess.SubprocessError, IndexError):
        pass


def _cudnn_version() -> str | None:
    library = ctypes.util.find_library("cudnn")
    if not library:
        return None
    try:
        cudnn = ctypes.CDLL(library)
        version = cudnn.cudnnGetVersion
        version.restype = ctypes.c_size_t
        value = int(version())
        return f"{value // 1000}.{value % 1000 // 100}.{value % 100}"
    except (OSError, AttributeError):
        return None


def _cuda_runtime_version() -> str | None:
    library = ctypes.util.find_library("cudart")
    if not library:
        return None
    try:
        cudart = ctypes.CDLL(library)
        runtime_version = ctypes.c_int()
        status = cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
        if status != 0:
            return None
        value = runtime_version.value
        return f"{value // 1000}.{value % 1000 // 10}"
    except (OSError, AttributeError):
        return None
