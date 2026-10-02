from __future__ import annotations

import importlib.util
import json
import math
import shutil
import urllib.request
import wave
from array import array
from pathlib import Path

import pytest


@pytest.mark.accuracy
def test_example_accuracy_suite_with_local_models(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    try:
        import ctranslate2
    except ImportError:
        pytest.skip("faster-whisper/CTranslate2 are not installed")
    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("faster-whisper/CTranslate2 are not installed")
    if ctranslate2.get_cuda_device_count() < 1:
        pytest.skip("no CUDA device is available")

    try:
        request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(request, timeout=1) as response:
            models = json.load(response).get("models", [])
    except (OSError, ValueError):
        pytest.skip("Ollama is not available")
    model_names = {str(model.get("name")) for model in models}
    if "qwen3:8b" not in model_names:
        pytest.skip("example LLM model qwen3:8b is not installed")
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    stt_exists = any(cache.glob("models--Systran--faster-whisper-large-v3*"))
    if not stt_exists:
        pytest.skip("example STT model large-v3 is not cached locally")

    project = Path(__file__).resolve().parents[1]
    example = project / "docs" / "accuracy-example"
    for name in ("suite.toml", "reference.txt", "facts.toml", "vocabulary.toml"):
        shutil.copy2(example / name, tmp_path / name)
    samples = array(
        "h",
        (int(1200 * math.sin(2 * math.pi * 440 * n / 16000)) for n in range(3 * 16000)),
    )
    with wave.open(str(tmp_path / "demo.wav"), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())
    from ea_assistant.cli import main

    assert main(["accuracy", str(tmp_path / "suite.toml")]) == 0
    assert list((tmp_path / "results").glob("smoke-*/report.json"))
