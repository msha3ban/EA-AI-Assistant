from __future__ import annotations

import math
import sys
import types
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ea_assistant.accuracy.suite import merge_config
from ea_assistant.cli import build_application
from ea_assistant.config import AppConfig, STTConfig, load_config
from ea_assistant.real_adapters import TranscribeCppSpeechToText


def test_config_and_suite_engine_selection(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f'data_dir="{tmp_path}"\n'
        '[stt]\nengine="transcribe-cpp"\n'
        'model="/tmp/model.gguf"\nbackend="cpu"\n'
    )
    config = load_config(str(path))
    assert isinstance(build_application(config).stt, TranscribeCppSpeechToText)
    assert config.stt.backend == "cpu"
    suite_config = merge_config(
        AppConfig(),
        {
            "stt": {
                "engine": "transcribe-cpp",
                "model": "/tmp/model.gguf",
                "backend": "cpu",
            }
        },
        tmp_path,
    )
    assert isinstance(build_application(suite_config).stt, TranscribeCppSpeechToText)
    assert suite_config.stt.model == "/tmp/model.gguf"
    with pytest.raises(ValueError, match="language.*auto"):
        replace(config.stt, language="auto")
    with pytest.raises(ValueError, match="engine"):
        replace(config.stt, engine="other")
    path.write_text('[stt]\nvocabulary=["private term"]\n')
    with pytest.raises(ValueError, match=r"\[stt\] vocabulary"):
        load_config(str(path))
    with pytest.raises(ValueError, match=r"\[run.stt\] vocabulary"):
        merge_config(AppConfig(), {"stt": {"vocabulary": ["private term"]}}, tmp_path)
    assert "vocabulary" not in config.stt.adapter_values()


def _wav(path: Path, seconds: int = 70) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 16000 * seconds)


def test_chunks_signals_vocabulary_truncation_and_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "recording.wav"
    model_path = tmp_path / "model.gguf"
    _wav(wav)
    model_path.write_bytes(b"fake model")
    calls: list[dict[str, object]] = []
    closed: list[str] = []
    supported = True

    class OutputTruncated(Exception):
        def __init__(self) -> None:
            self.partial_result = SimpleNamespace(text="partial", tokens=())

    class Model:
        def __init__(self, path: str, backend: str) -> None:
            assert path == str(model_path) and backend == "cpu"

        def supports(self, feature: str) -> bool:
            assert feature == "vocabulary"
            return supported

        def session(self) -> Model:
            return self

        def run(self, pcm: object, **kwargs: object) -> object:
            calls.append({"length": len(pcm), **kwargs})  # type: ignore[arg-type]
            if not supported:
                return SimpleNamespace(text="مرحبا", tokens=())
            if len(calls) == 2:
                return SimpleNamespace(text=" ", tokens=())
            if len(calls) == 3:
                raise OutputTruncated()
            return SimpleNamespace(
                text="مرحبا",
                tokens=(SimpleNamespace(p=0.5), SimpleNamespace(p=0.25)),
            )

        def close(self) -> None:
            closed.append("close")

    fake = types.ModuleType("transcribe_cpp")
    fake.Model = Model  # type: ignore[attr-defined]
    fake.OutputTruncated = OutputTruncated  # type: ignore[attr-defined]
    fake.OutputRepetition = type(  # type: ignore[attr-defined]
        "OutputRepetition", (Exception,), {}
    )
    monkeypatch.setitem(sys.modules, "transcribe_cpp", fake)
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = (  # type: ignore[attr-defined]
        lambda **kwargs: SimpleNamespace(**kwargs)
    )
    vad.get_speech_timestamps = lambda audio, options: [  # type: ignore[attr-defined]
        {"start": 16000, "end": 16000 * 33},
        {"start": 16000 * 34, "end": 16000 * 37},
        {"start": 16000 * 40, "end": 16000 * 65},
    ]
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)
    adapter = TranscribeCppSpeechToText()
    config = STTConfig(
        engine="transcribe-cpp",
        model=str(model_path),
        backend="cpu",
        vocabulary=("Kafka", "SAP"),
        initial_prompt="Vocabulary: wrong term",
        hotwords="wrong hotword",
    )
    segments, _, backend = adapter.transcribe(str(wav), config)
    assert backend == "cpu"
    assert all(int(call["length"]) <= 30 * 16000 for call in calls)
    assert [(s.start, s.end) for s in segments] == [(1, 31), (40, 65)]
    assert len(calls) == 3
    assert segments[0].avg_logprob == pytest.approx(
        (math.log(0.5) + math.log(0.25)) / 2
    )
    assert segments[0].confidence_signals is True
    assert any(s.text == "partial" and "repetition loop" in s.flags for s in segments)
    assert next(s for s in segments if s.text == "partial").confidence_signals is False
    assert all(call["vocabulary"] == ["Kafka", "SAP"] for call in calls)
    assert adapter.vocabulary_applied is True
    assert adapter.model_identifier.startswith("sha256:")
    adapter.release()
    assert len(closed) == 2

    supported = False
    calls.clear()
    unsupported_segments, _, _ = adapter.transcribe(str(wav), config)
    assert all(call["vocabulary"] is None for call in calls)
    assert adapter.vocabulary_applied is False
    assert unsupported_segments
    assert all(not segment.confidence_signals for segment in unsupported_segments)
    adapter.release()


@pytest.mark.parametrize(
    ("regions", "sample_count", "expected"),
    [
        ([{"start": 0, "end": 16000}, {"start": 32001, "end": 48000}],
         64000, [(0, 16000), (32001, 48000)]),
        ([{"start": 0, "end": 16000}, {"start": 32000, "end": 48000}],
         64000, [(0, 48000)]),
        ([{"start": 0, "end": 31 * 16000}],
         31 * 16000, [(0, 30 * 16000), (30 * 16000, 31 * 16000)]),
        ([{"start": -100, "end": 32000},
          {"start": 100000, "end": 200000}],
         64000, [(0, 32000)]),
    ],
)
def test_chunk_boundaries(
    regions: list[dict[str, int]], sample_count: int,
    expected: list[tuple[int, int]],
) -> None:
    assert TranscribeCppSpeechToText._chunks(regions, sample_count) == expected


def test_missing_binding_error_is_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    original = builtins.__import__

    def missing(name: str, *args: object, **kwargs: object) -> object:
        if name == "transcribe_cpp":
            raise ModuleNotFoundError("private recording name", name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    wav = tmp_path / "recording.wav"
    _wav(wav, 1)
    with pytest.raises(RuntimeError, match="PYTHONPATH.*TRANSCRIBE_LIBRARY") as error:
        TranscribeCppSpeechToText().transcribe(
            str(wav), STTConfig(engine="transcribe-cpp", model="/tmp/model.gguf")
        )
    assert "private recording name" not in str(error.value)
