from __future__ import annotations

import json
import sys
import types
import wave
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ea_assistant.application import Application
from ea_assistant.config import AppConfig, LLMCallConfig, STTConfig, load_config
from ea_assistant.models import Segment
from ea_assistant.ollama import OllamaClient
from ea_assistant.real_adapters import FasterWhisper
from ea_assistant.testing import FakeAudio, FakeLLM, FakeSensors, FakeSpeechToText


def make_config(path: Path) -> AppConfig:
    return replace(load_config(), data_dir=path)


def test_rerun_uses_recording_database_path(tmp_path: Path) -> None:
    recording_path = tmp_path / "voice.m4a"
    recording_path.write_bytes(b"a")
    audio = FakeAudio()
    app = Application(
        audio,
        FakeSpeechToText(),
        FakeLLM(),
        FakeSensors(),
        make_config(tmp_path / "data"),
    )
    meeting = app.create_meeting(str(recording_path), "T", date(2026, 10, 1))
    app.process(meeting)
    app.process(meeting)
    assert (meeting.folder / "voice.m4a").exists()
    assert audio.probed[-1] == str(meeting.folder / "voice.m4a")


def test_ollama_unload_timeout_is_configurable_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[llm]\nunload_timeout = 2.5\nunload_poll_interval = 0.05\n",
        encoding="utf-8",
    )
    config = load_config(str(config_path))
    assert config.llm.unload_timeout == 2.5
    assert config.llm.unload_poll_interval == 0.05


def test_english_transcript_is_batched(tmp_path: Path) -> None:
    recording_path = tmp_path / "v.m4a"
    recording_path.write_bytes(b"a")
    segments = [Segment(f"s{i:04}", i, i + 1, "word") for i in range(1, 4)]
    llm = FakeLLM()
    app = Application(
        FakeAudio(),
        FakeSpeechToText(segments),
        llm,
        FakeSensors(),
        make_config(tmp_path / "d"),
    )
    meeting = app.create_meeting(str(recording_path), "T", date(2026, 10, 1))
    app.process(meeting)
    assert llm.english_transcript_calls == 1


def test_ollama_chat_does_not_unload_each_call_and_uses_get_for_ps() -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        payload = json.loads(data) if data else None
        calls.append((method, url, payload))
        if url.endswith("/api/chat"):
            return json.dumps({"message": {"content": "ok"}, "done": True}).encode()
        if url.endswith("/api/tags"):
            return b'{"models":[{"name":"m","digest":"sha256:abc"}]}'
        if url.endswith("/api/ps"):
            return b'{"models":[]}'
        return b"{}"

    client = OllamaClient("http://127.0.0.1:11434", transport=transport)
    client.chat(
        "instruction",
        "data",
        LLMCallConfig("m", 200, 10, False),
    )
    assert len([x for x in calls if x[1].endswith("/api/chat")]) == 1
    assert not any(x[1].endswith("/api/ps") for x in calls)
    client.unload("m")
    assert ("GET", "http://127.0.0.1:11434/api/ps", None) in calls
    assert client.models["m"] == "sha256:abc"


def test_ollama_stage_timeout_is_passed_to_transport() -> None:
    assert AppConfig().llm.translate.timeout == 600
    assert AppConfig().llm.extract.timeout == 600
    assert AppConfig().llm.summary.timeout == 300
    assert AppConfig().llm.summary.num_ctx == 8192
    assert AppConfig().llm.summary.num_predict == 512
    observed: list[float] = []

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        observed.append(timeout)
        if url.endswith("/api/tags"):
            return b'{"models":[{"name":"m","digest":"d"}]}'
        if url.endswith("/api/chat"):
            raise TimeoutError("transport timed out")
        return b"{}"

    client = OllamaClient("http://127.0.0.1:11434", timeout=120, transport=transport)
    with pytest.raises(TimeoutError):
        client.chat("rules", "content", LLMCallConfig("m", 500, 20, timeout=600))
    assert observed == [120, 600]


def test_faster_whisper_receives_decoded_samples_not_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """faster-whisper 1.2.1 decodes paths with an API PyAV 19 removed; we hand it samples."""
    np = pytest.importorskip("numpy")
    wav = tmp_path / "normalized.wav"
    frames = [0, 16384, -32768, 32767]
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"".join(f.to_bytes(2, "little", signed=True) for f in frames))
    received: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.model = types.SimpleNamespace(compute_type="int8")

        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[Any], Any]:
            received["audio"] = audio
            return [], object()

    fake = types.ModuleType("faster_whisper")
    fake.__dict__["WhisperModel"] = FakeModel
    fake.__dict__["download_model"] = lambda model, **kwargs: str(tmp_path)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    FasterWhisper().transcribe(str(wav), STTConfig(model="large-v3"))

    audio = received["audio"]
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


def test_faster_whisper_aligns_decoded_tokens_without_changing_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    np = pytest.importorskip("numpy")
    wav = tmp_path / "normalized.wav"
    with wave.open(str(wav), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * 32000)

    calls: dict[str, Any] = {"aligned": [], "encoded": []}

    class FakeTokenizer:
        eot = 100

        def __init__(self, hf: object, multilingual: bool, *, task: str, language: str):
            assert hf == "hf" and multilingual and task == "transcribe"
            assert language == "ar"

    class FakeExtractor:
        sampling_rate = 16000
        n_samples = 480000
        nb_max_frames = 3000

        def __call__(self, audio: Any) -> Any:
            assert len(audio) in (32000, 16000)
            return np.zeros((2, len(audio) // 160 + 1), dtype=np.float32)

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.model = types.SimpleNamespace(
                compute_type="int8_float32", is_multilingual=True
            )
            self.hf_tokenizer = "hf"
            self.feature_extractor = FakeExtractor()

        def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
            calls["decode"] = kwargs
            segments = [
                types.SimpleNamespace(
                    start=start, end=end, text=text, tokens=tokens,
                    avg_logprob=-0.1, no_speech_prob=0.01,
                    compression_ratio=1.0,
                )
                for start, end, text, tokens in (
                    (0.0, 1.0, "first", [101, 4, 5, 102]),
                    (1.0, 2.0, "second", [101, 6, 102]),
                    (1.0, 2.0, "third", [101, 6, 102]),
                )
            ]
            return iter(segments), types.SimpleNamespace(
                language="ar", language_probability=0.97
            )

        def encode(self, features: Any) -> object:
            calls["encoded"].append(features.shape)
            return object()

        def find_alignment(
            self, tokenizer: Any, text_tokens: Any, encoder_output: object,
            num_frames: int,
        ) -> Any:
            calls["aligned"].append((text_tokens, num_frames))
            if text_tokens == [[6]]:
                raise RuntimeError("private recording text")
            return [[{"probability": 0.43216}, {"probability": 0.75}]]

    fake = types.ModuleType("faster_whisper")
    fake.__path__ = []  # type: ignore[attr-defined]
    fake.WhisperModel = FakeModel  # type: ignore[attr-defined]
    fake.download_model = lambda model, **kwargs: str(tmp_path)  # type: ignore[attr-defined]
    tokenizer_module = types.ModuleType("faster_whisper.tokenizer")
    tokenizer_module.Tokenizer = FakeTokenizer  # type: ignore[attr-defined]
    audio_module = types.ModuleType("faster_whisper.audio")

    def pad_or_trim(features: Any, length: int) -> Any:
        return np.pad(features, ((0, 0), (0, length - features.shape[-1])))

    audio_module.pad_or_trim = pad_or_trim  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setitem(sys.modules, "faster_whisper.tokenizer", tokenizer_module)
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio_module)

    segments, _, _ = FasterWhisper().transcribe(
        str(wav), STTConfig(model=str(tmp_path), language="ar")
    )

    assert "word_timestamps" not in calls["decode"]
    assert [(s.start, s.end, s.text) for s in segments] == [
        (0.0, 1.0, "first"), (1.0, 2.0, "second"),
        (1.0, 2.0, "third"),
    ]
    assert calls["aligned"] == [([[4, 5]], 200), ([[6]], 100), ([[6]], 100)]
    assert calls["encoded"] == [(2, 3000), (2, 3000), (2, 3000)]
    assert segments[0].word_probabilities == (0.4322, 0.75)
    assert segments[0].confidence_signals is True
    assert segments[1].word_probabilities == ()
    assert segments[1].confidence_signals is False
    assert segments[2].word_probabilities == ()
    assert segments[2].confidence_signals is False
    assert caplog.text.count("Word probability alignment unavailable") == 1
    assert "private recording text" not in caplog.text


def test_faster_whisper_rejects_wav_that_is_not_16khz_mono(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    wav = tmp_path / "stereo.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="16 kHz mono"):
        FasterWhisper().transcribe(str(wav), STTConfig(model="large-v3"))
