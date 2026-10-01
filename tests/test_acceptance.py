from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
import sys
import types
import wave
import weakref
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from ea_assistant.adapters import Sensors
from ea_assistant.application import Application
from ea_assistant.cli import main
from ea_assistant.config import AppConfig, LLMCallConfig, STTConfig, load_config
from ea_assistant.domain import FACT_SECTIONS, FactStatus
from ea_assistant.models import Meeting, Provenance, Segment
from ea_assistant.ollama import OllamaClient
from ea_assistant.real_adapters import FasterWhisper, FfmpegAudio
from ea_assistant.render import SECTION_ORDER
from ea_assistant.schemas import FACT_SCHEMA, english_transcript_schema, validate_object
from ea_assistant.testing import FakeAudio, FakeLLM, FakeSensors, FakeSpeechToText


def cfg(path: Path) -> AppConfig:
    return replace(load_config(), data_dir=path)


def app_for(
    path: Path,
    segments: list[Segment] | None = None,
    llm: FakeLLM | None = None,
    sensors: Sensors | None = None,
    events: list[str] | None = None,
) -> tuple[Application, FakeSpeechToText, FakeLLM]:
    stt = FakeSpeechToText(segments, events)
    model = llm or FakeLLM(events)
    return (
        Application(FakeAudio(), stt, model, sensors or FakeSensors(), cfg(path)),
        stt,
        model,
    )


def recording(tmp_path: Path) -> Path:
    p = tmp_path / "meeting.m4a"
    p.write_bytes(b"audio")
    return p


def rich_responder(instructions: str, content: dict[str, str]) -> str:
    ids = list(content)
    if instructions.startswith("Translate"):
        return json.dumps(
            {
                k: ("up to 5,000 EGP; not approved" if i else "translated")
                for i, k in enumerate(ids)
            }
        )
    if instructions.startswith("Extract"):
        decisions = []
        actions = []
        technical = []
        purpose = []
        discussion = []
        for i, sid in enumerate(ids):
            status = (
                FactStatus.PROPOSED.value
                if sid == "s0001"
                else FactStatus.REJECTED.value
            )
            decisions.append(
                {
                    "statement": "Use the gateway",
                    "rationale": None,
                    "status": status,
                    "evidence": [sid],
                }
            )
            actions.append(
                {
                    "item": "Review options",
                    "owner": None,
                    "due_date": None,
                    "evidence": [sid],
                }
            )
            technical.append(
                {
                    "subject": "Cost ceiling",
                    "statement": "up to 5,000 EGP; not approved",
                    "status": status,
                    "units": "EGP",
                    "evidence": [sid],
                }
            )
            purpose.append({"statement": "Choose an integration", "evidence": [sid]})
            discussion.append({"statement": "Gateway discussed", "evidence": [sid]})
        return json.dumps(
            {
                "decisions": decisions,
                "action_items": actions,
                "technical_details": technical,
                "purpose_agenda": purpose,
                "discussion_points": discussion,
                "open_questions_risks": [],
                "next_steps": [],
            }
        )
    return "A local gateway was discussed."


def test_mom_structure_reconciliation_evidence_and_missing_values(
    tmp_path: Path,
) -> None:
    p = recording(tmp_path)
    segments = [Segment("s0001", 0, 1, "proposal"), Segment("s0002", 1, 2, "rejection")]
    llm = FakeLLM(responder=rich_responder)
    app, _, _ = app_for(tmp_path / "data", segments, llm)
    m = app.create_meeting(str(p), "Integration", date(2026, 10, 1))
    result = app.process(m)
    mom = result.mom.markdown
    sections = [line for line in mom.splitlines() if line.startswith("## ")]
    assert sections == SECTION_ORDER
    assert mom.count("D1:") == 1 and "D2:" not in mom and "status: rejected" in mom
    assert (
        "| # | Item | Owner | Due date | Evidence |" in mom
        and "| 1 | Review options | TBD | not stated | s0001 |" in mom
    )
    assert "rationale: not stated" in mom and "up to 5,000 EGP; not approved" in mom
    assert all(x in mom for x in ("## Technical Details", "Evidence"))
    import re

    ids = set(re.findall(r"\[(s\d{4})\]", mom))
    transcript_ids = set(re.findall(r"\[(s\d{4})\]", result.transcript))
    assert ids and ids <= transcript_ids
    assert "State: Draft" in mom and "Duration: 00:03" in mom


def test_english_transcript_exact_ids_and_retry_failure(tmp_path: Path) -> None:
    p = recording(tmp_path)
    calls = 0

    def bad(instructions: str, content: dict[str, str]) -> str:
        nonlocal calls
        calls += 1
        return (
            json.dumps({"added": "oops"})
            if instructions.startswith("Translate")
            else rich_responder(instructions, content)
        )

    llm = FakeLLM(responder=bad)
    app, _, _ = app_for(tmp_path / "data", [Segment("s0001", 0, 1, "x")], llm)
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    with pytest.raises(ValueError, match="after one retry"):
        app.process(m)
    assert calls == 2 and not (m.folder / "english-transcript.md").exists()


def test_gpu_exclusive_and_llm_unload_failure(tmp_path: Path) -> None:
    events: list[str] = []
    p = recording(tmp_path)
    app, _, _llm = app_for(
        tmp_path / "data", [Segment("s0001", 0, 1, "x")], events=events
    )
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    app.process(m)
    assert events[0:3] == ["llm-gpu-free-check", "stt-load", "stt-release"]
    active = False
    for event in events:
        if event == "stt-load":
            assert not active
        if event == "llm-load":
            assert not active
            active = True
        if event == "llm-unload":
            active = False
    assert not active

    class Broken(FakeLLM):
        def unload(self, model: str) -> None:
            raise RuntimeError("unload verification failed")

    app2, _, _ = app_for(tmp_path / "other", [Segment("s0001", 0, 1, "x")], Broken())
    m2 = app2.create_meeting(str(p), "T", date(2026, 10, 1))
    with pytest.raises(RuntimeError, match="unload verification failed"):
        app2.process(m2)


def transport_client(
    response: dict[str, Any] | None = None,
) -> tuple[OllamaClient, list[tuple[str, str, dict[str, Any] | None]]]:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        payload = json.loads(data) if data else None
        calls.append((method, url, payload))
        if url.endswith("/api/tags"):
            return b'{"models":[{"name":"model","digest":"sha256:abc"}]}'
        if url.endswith("/api/chat"):
            return json.dumps(
                response
                or {
                    "message": {
                        "content": '{"s0001":"thinking visible?"}',
                        "thinking": "private thoughts",
                    },
                    "done": True,
                }
            ).encode()
        if url.endswith("/api/ps"):
            return b'{"models":[]}'
        return b"{}"

    return OllamaClient("http://127.0.0.1:11434", transport=transport), calls


def test_ollama_budget_truncation_options_schema_and_stage_unload() -> None:
    client, calls = transport_client()
    schema = english_transcript_schema(["s0001"])
    out = client.chat(
        "rules",
        "hello",
        LLMCallConfig("model", 500, 20, False, 0),
        schema,
    )
    request = next(c[2] for c in calls if c[1].endswith("/api/chat"))
    assert (
        request
        and request["options"]["num_ctx"] == 500
        and request["think"] is False
        and request["format"] == schema
        and out == '{"s0001":"thinking visible?"}'
        and "private thoughts" not in out
    )
    assert len([c for c in calls if c[1].endswith("/api/chat")]) == 1
    with pytest.raises(ValueError, match="token budget"):
        client.chat("x" * 100, "y" * 100, LLMCallConfig("model", 20, 10))
    assert len(calls) == 2
    truncated, _ = transport_client(
        {"done_reason": "length", "message": {"content": "partial"}, "done": True}
    )
    with pytest.raises(RuntimeError, match="incomplete"):
        truncated.chat("rules", "content", LLMCallConfig("model", 500, 20))
    client.unload("model")
    unload_payload = next(
        c[2]
        for c in calls
        if c[1].endswith("/api/chat") and c[2] and c[2].get("keep_alive") == 0
    )
    assert unload_payload["keep_alive"] == 0
    assert calls[-1][0] == "GET" and calls[-1][1].endswith("/api/ps")


def test_english_transcript_json_schema_is_strict() -> None:
    schema = english_transcript_schema(["s0001", "s0002"])
    assert validate_object({"s0001": "a", "s0002": "b"}, schema)
    assert not validate_object({"s0001": "a", "s0002": "b", "extra": "c"}, schema)
    assert set(FACT_SCHEMA["properties"]) == {
        section.value for section in FACT_SECTIONS
    }
    assert FACT_SCHEMA["properties"]["technical_details"]["items"]["properties"][
        "status"
    ]["enum"] == [status.value for status in FactStatus]


def test_thermal_all_failures_and_resume(tmp_path: Path) -> None:
    p = recording(tmp_path)
    for temp, ac, match in [
        (95, True, "hard stop"),
        (None, True, "unavailable"),
        (40, False, "off or unknown"),
        (40, None, "off or unknown"),
    ]:
        app, stt, _ = app_for(tmp_path / f"{temp}-{ac}", sensors=FakeSensors(temp, ac))
        m = app.create_meeting(str(p), "T", date(2026, 10, 1))
        with pytest.raises(RuntimeError, match=match):
            app.process(m)
        assert stt.calls == 0
    app, stt, llm = app_for(tmp_path / "resume")
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    app.process(m)
    counts = (stt.calls, llm.calls)
    app.process(m)
    assert (stt.calls, llm.calls) == counts


def test_endpoint_and_compute_validation(tmp_path: Path) -> None:
    remote = tmp_path / "remote.toml"
    remote.write_text('[llm]\nendpoint="http://example.com:11434"\n')
    with pytest.raises(ValueError, match="Non-loopback"):
        load_config(str(remote))
    fp16 = tmp_path / "fp.toml"
    fp16.write_text('[stt]\ncompute_type="float16"\n')
    with pytest.raises(ValueError, match="Pascal GPU"):
        load_config(str(fp16))


def test_content_marker_never_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "CONFIDENTIAL-MARKER-992"
    p = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "data", [Segment("s0001", 0, 1, marker)])
    with caplog.at_level(logging.INFO):
        m = app.create_meeting(str(p), "Secret title", date(2026, 10, 1))
        app.process(m)
    assert marker not in caplog.text and "Secret title" not in caplog.text


def test_atomic_publish_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "data")
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    target = m.folder / "old.md"
    target.write_text("previous")

    def fail(src: Any, dst: Any) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("ea_assistant.application.os.replace", fail)
    with pytest.raises(OSError):
        app._publish(m, "old", target, "new", Provenance())
    assert target.read_text() == "previous"


def test_provenance_stage_models_decoding_and_inputs(tmp_path: Path) -> None:
    p = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "data")
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    result = app.process(m)
    prov = result.provenance
    assert prov["transcript"].models["large-v3"]["digest"] == "fake-stt-revision"
    assert prov["transcript"].compute_type == "int8"
    for name in ("english_transcript", "summary", "mom"):
        item = prov[name]
        assert item.models["qwen3:8b"]["digest"] == "llm-digest"
        assert item.vocabulary == [] and item.decoding
    assert prov["english_transcript"].input_revisions == {"transcript": 1}
    assert prov["summary"].input_revisions == {"english_transcript": 1}
    assert set(prov["mom"].decoding) == {"extract", "summary"}
    assert prov["mom"].decoding["summary"]["num_ctx"] == app.config.llm.summary.num_ctx


def test_cli_injection_creates_meeting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = recording(tmp_path)
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'data_dir = "{tmp_path / "cli-data"}"\n')
    app, _, _ = app_for(tmp_path / "inject")
    assert (
        main(
            [
                "--config",
                str(config_path),
                "process",
                str(p),
                "--title",
                "CLI",
                "--date",
                "2026-10-01",
            ],
            app=app,
        )
        == 0
    )
    assert "transcript.md" in capsys.readouterr().out or list(
        (tmp_path / "inject" / "meetings").glob("*/transcript.md")
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_ffmpeg_audio_contract_3s_m4a_to_16khz_mono(tmp_path: Path) -> None:
    recording_path = tmp_path / "tone.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            "3",
            "-c:a",
            "aac",
            str(recording_path),
        ],
        check=True,
    )
    audio = FfmpegAudio()
    duration = audio.probe(str(recording_path))
    wav = tmp_path / "normalized.wav"
    audio.normalize(str(recording_path), str(wav))
    with wave.open(str(wav), "rb") as f:
        assert f.getframerate() == 16000 and f.getnchannels() == 1
    assert 2.9 <= duration <= 3.1


def test_faster_whisper_contract_is_lazy_local_and_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numpy")
    local_wav = tmp_path / "local.wav"
    with wave.open(str(local_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    observed: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            observed["init"] = (args, kwargs)
            observed["model_ref"] = weakref.ref(self)
            self.model = types.SimpleNamespace(compute_type="int8_float32")

        def transcribe(self, path: str, **kwargs: Any) -> tuple[list[Any], Any]:
            observed["transcribe"] = kwargs

            class Info:
                pass

            class Item:
                start = 0.0
                end = 1.0
                text = "hello"
                avg_logprob = -0.1
                no_speech_prob = 0.01
                compression_ratio = 1.0

            return [Item()], Info()

    fake = types.ModuleType("faster_whisper")
    fake.__dict__["__path__"] = []
    fake.__dict__["WhisperModel"] = FakeModel

    def resolve_model(model: str, **kwargs: Any) -> str:
        observed["download_args"] = kwargs
        return "/models/snapshots/revision-123"

    fake.__dict__["download_model"] = resolve_model
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    gc_calls = 0
    real_collect = gc.collect

    def collect() -> int:
        nonlocal gc_calls
        gc_calls += 1
        return real_collect()

    monkeypatch.setattr(gc, "collect", collect)
    adapter = FasterWhisper()
    config = STTConfig(
        model="large-v3",
        device="cuda",
        compute_type="int8",
        language="auto",
        beam_size=3,
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
            "max_speech_duration_s": 30.0,
        },
    )
    segments, model, compute_type = adapter.transcribe(str(local_wav), config)
    adapter.release()
    assert observed["model_ref"]() is None and gc_calls >= 2
    assert (
        observed["init"][1]["local_files_only"] is True
        and observed["init"][1]["compute_type"] == "int8"
    )
    assert observed["transcribe"] == {
        "task": "transcribe",
        "language": None,
        "beam_size": 3,
        "vad_filter": True,
        "vad_parameters": config.vad_parameters,
    }
    assert (
        segments[0].id == "s0001"
        and model == "large-v3"
        and compute_type == "int8_float32"
    )
    assert adapter.model_identifier == "revision-123"
    assert observed["download_args"] == {"local_files_only": True, "cache_dir": None}


def test_chunking_both_stages_and_extract_retry(tmp_path: Path) -> None:
    p = recording(tmp_path)
    segments = [Segment("s0001", 0, 1, "A" * 6000), Segment("s0002", 1, 2, "B" * 6000)]

    def large_english_transcript(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Translate"):
            return json.dumps({k: v for k, v in content.items()})
        return rich_responder(instructions, content)

    llm = FakeLLM(responder=large_english_transcript)
    app, _, _ = app_for(tmp_path / "chunks", segments, llm)
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    result = app.process(m)
    assert llm.english_transcript_calls == 2 and llm.extract_calls == 2
    assert "D1: Use the gateway (status: rejected" in result.mom.markdown
    assert "D2:" not in result.mom.markdown
    assert "[s0001, s0002]" in result.mom.markdown
    calls = 0

    def retry(instructions: str, content: dict[str, str]) -> str:
        nonlocal calls
        if instructions.startswith("Extract"):
            calls += 1
            if calls == 1:
                return "{bad json"
        return rich_responder(instructions, content)

    retry_llm = FakeLLM(responder=retry)
    app2, _, _ = app_for(tmp_path / "retry", [Segment("s0001", 0, 1, "one")], retry_llm)
    m2 = app2.create_meeting(str(p), "T", date(2026, 10, 1))
    app2.process(m2)
    assert calls == 2


def test_extracted_facts_persist_when_render_stage_fails(tmp_path: Path) -> None:
    p = recording(tmp_path)
    summary_calls = 0

    def intermittent(instructions: str, content: dict[str, str]) -> str:
        nonlocal summary_calls
        if instructions.startswith("Write a short"):
            summary_calls += 1
            if summary_calls == 1:
                raise RuntimeError("summary interruption")
        return rich_responder(instructions, content)

    llm = FakeLLM(responder=intermittent)
    app, _stt, _ = app_for(tmp_path / "persist", [Segment("s0001", 0, 1, "x")], llm)
    m = app.create_meeting(str(p), "T", date(2026, 10, 1))
    with pytest.raises(RuntimeError, match="summary interruption"):
        app.process(m)
    extraction_calls = llm.extract_calls
    result = app.process(m)
    assert (
        llm.extract_calls == extraction_calls
        and "Use the gateway" in result.mom.markdown
    )


def test_unload_verification_failure_stops_stage() -> None:
    calls = []

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        calls.append((method, url))
        if url.endswith("/api/tags"):
            return b'{"models":[{"name":"m","digest":"d"}]}'
        if url.endswith("/api/chat"):
            return b'{"message":{"content":"ok"},"done":true}'
        if url.endswith("/api/ps"):
            return b'{"models":[{"name":"m"}]}'
        return b"{}"

    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    llm = OllamaClient(
        "http://127.0.0.1",
        transport=transport,
        clock=clock,
        sleep=sleep,
        unload_timeout=0.2,
        poll_interval=0.1,
    )
    llm.chat("rules", "content", LLMCallConfig("m", 200, 10))
    with pytest.raises(RuntimeError, match="still loaded"):
        llm.unload("m")
    assert calls[-1] == ("GET", "http://127.0.0.1/api/ps")


def test_setup_models_is_explicit_download_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = []
    fake = types.ModuleType("faster_whisper")
    fake.__dict__["download_model"] = lambda model, **kwargs: called.append(
        (model, kwargs)
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    toml = tmp_path / "config.toml"
    toml.write_text(f'data_dir = "{tmp_path / "data"}"\n')
    assert main(["--config", str(toml), "setup-models"]) == 0
    assert called == [("large-v3", {"local_files_only": False, "cache_dir": None})]


def test_config_is_frozen_typed_dataclass(tmp_path: Path) -> None:
    from dataclasses import FrozenInstanceError

    from ea_assistant.config import AppConfig

    config = cfg(tmp_path / "typed")
    assert isinstance(config, AppConfig)
    with pytest.raises(FrozenInstanceError):
        config.allow_remote_endpoint = True  # type: ignore[misc]


def test_extraction_invalid_schema_fails_after_one_retry(tmp_path: Path) -> None:
    path = recording(tmp_path)

    def invalid(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Extract"):
            return json.dumps({"decisions": "not-an-array"})
        return rich_responder(instructions, content)

    llm = FakeLLM(responder=invalid)
    app, _, _ = app_for(
        tmp_path / "invalid-extract", [Segment("s0001", 0, 1, "x")], llm
    )
    meeting = app.create_meeting(str(path), "T", date(2026, 10, 1))
    with pytest.raises(ValueError, match="extract response failed.*retry"):
        app.process(meeting)
    assert llm.extract_calls == 2


def test_invalid_json_from_ollama_is_rejected() -> None:
    client, _ = transport_client({"message": {"content": "not json"}, "done": True})
    with pytest.raises(RuntimeError, match="invalid JSON"):
        client.chat(
            "rules",
            "content",
            LLMCallConfig("model", 500, 20),
            {"type": "object"},
        )


def test_atomic_publish_database_failure_restores_existing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "atomic-db")
    meeting = app.create_meeting(str(path), "T", date(2026, 10, 1))
    target = meeting.folder / "artifact.md"
    target.write_text("previous", encoding="utf-8")

    def fail(*args: Any, **kwargs: Any) -> int:
        raise OSError("database commit failed")

    monkeypatch.setattr(app.store, "artefact", fail)
    with pytest.raises(OSError, match="database commit failed"):
        app._publish(meeting, "artifact", target, "replacement", Provenance())
    assert target.read_text(encoding="utf-8") == "previous"


def test_application_boundary_delegates_storage_render_and_thermal(
    tmp_path: Path,
) -> None:
    from ea_assistant.storage import Store
    from ea_assistant.thermal import ThermalGuard

    app, _, _ = app_for(tmp_path / "seams")
    assert isinstance(app.store, Store)
    assert isinstance(app.guard, ThermalGuard)
    assert not hasattr(app, "_guard")
    assert all(
        callable(getattr(app, name))
        for name in (
            "_ingest",
            "_transcribe",
            "_translate",
            "_extract",
            "_render",
        )
    )


def test_unload_polls_until_ollama_model_disappears() -> None:
    calls = 0
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    def sleep(seconds: float) -> None:
        nonlocal clock_value
        clock_value += seconds

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        nonlocal calls
        if url.endswith("/api/ps"):
            calls += 1
            return b'{"models":[{"name":"m"}]}' if calls <= 2 else b'{"models":[]}'
        return b"{}"

    client = OllamaClient(
        "http://localhost",
        transport=transport,
        clock=clock,
        sleep=sleep,
        unload_timeout=1,
        poll_interval=0.1,
    )
    client.unload("m")
    assert calls == 3


def test_unload_poll_times_out_if_model_stays_loaded() -> None:
    clock_value = 0.0

    def clock() -> float:
        return clock_value

    def sleep(seconds: float) -> None:
        nonlocal clock_value
        clock_value += seconds

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        return b'{"models":[{"name":"m"}]}' if url.endswith("/api/ps") else b"{}"

    client = OllamaClient(
        "http://localhost",
        transport=transport,
        clock=clock,
        sleep=sleep,
        unload_timeout=0.25,
        poll_interval=0.1,
    )
    with pytest.raises(RuntimeError, match="still loaded after 0.25"):
        client.unload("m")


def test_ensure_gpu_free_unloads_every_resident_model() -> None:
    requests: list[tuple[str, str, dict[str, Any] | None]] = []
    loaded = {"configured:tag", "other:tag"}

    def transport(method: str, url: str, data: bytes | None, timeout: float) -> bytes:
        payload = json.loads(data) if data else None
        requests.append((method, url, payload))
        if url.endswith("/api/ps"):
            return json.dumps({"models": [{"name": name} for name in loaded]}).encode()
        if url.endswith("/api/chat") and payload and payload.get("keep_alive") == 0:
            loaded.discard(payload["model"])
        return b"{}"

    client = OllamaClient("http://localhost", transport=transport)
    client.ensure_gpu_free()
    assert loaded == set()
    assert {
        request[2]["model"]
        for request in requests
        if request[2] and request[2].get("keep_alive") == 0
    } == {"configured:tag", "other:tag"}


def test_original_thermal_error_survives_unload_failure(tmp_path: Path) -> None:
    path = recording(tmp_path)

    class SequenceSensors:
        def __init__(self) -> None:
            self.values = iter([40, 95])

        def gpu_temperature(self) -> float | None:
            return next(self.values)

        def on_ac(self) -> bool | None:
            return True

    class FailingUnload(FakeLLM):
        def unload(self, model: str) -> None:
            raise RuntimeError("unload failed")

    app, _, _ = app_for(
        tmp_path / "thermal-chain",
        [Segment("s0001", 0, 1, "x")],
        FailingUnload(),
        SequenceSensors(),
    )
    meeting = app.create_meeting(str(path), "T", date(2026, 10, 1))
    with pytest.raises(RuntimeError, match="Thermal guard stopped processing"):
        app.process(meeting)


def test_facts_without_valid_evidence_are_retained_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ea_assistant.render import mom_markdown, reconcile

    meeting = Meeting("m", "T", date(2026, 10, 1), Path("."))
    facts = [
        {
            "decisions": [
                {
                    "statement": "Keep this fact",
                    "rationale": None,
                    "status": "agreed",
                    "evidence": ["invalid"],
                }
            ]
        }
    ]
    with caplog.at_level(logging.WARNING):
        reconciled = reconcile(facts, [Segment("s0001", 0, 1, "spoken")])
    text = mom_markdown(meeting, 12, "summary", reconciled)
    assert "Keep this fact" in text and "Evidence: not stated" in text
    assert "1 extracted facts without valid Evidence" in caplog.text


def test_resume_uses_summary_text_from_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "summary-db", [Segment("s0001", 0, 1, "x")])
    meeting = app.create_meeting(str(path), "T", date(2026, 10, 1))
    publish = app._publish

    def fail_mom(
        meeting_arg: Meeting,
        name: str,
        artifact_path: Path,
        text: str,
        provenance: Provenance,
    ) -> None:
        if name == "mom":
            raise OSError("interrupt before Mom publication")
        publish(meeting_arg, name, artifact_path, text, provenance)

    monkeypatch.setattr(app, "_publish", fail_mom)
    with pytest.raises(OSError, match="interrupt"):
        app.process(meeting)
    (meeting.folder / "summary.md").write_text(
        "tampered disk summary", encoding="utf-8"
    )
    monkeypatch.setattr(app, "_publish", publish)
    result = app.process(meeting)
    assert "Meeting summary." in result.mom.markdown
    assert "tampered disk summary" not in result.mom.markdown


def test_response_prompt_eval_count_cannot_silently_truncate() -> None:
    client, _ = transport_client(
        {
            "message": {"content": "partial"},
            "done": True,
            "prompt_eval_count": 490,
            "eval_count": 10,
        }
    )
    with pytest.raises(RuntimeError, match="exhausted num_ctx"):
        client.chat("rules", "content", LLMCallConfig("model", 500, 20))


def test_explicit_vad_defaults_are_recorded_in_transcript_provenance(
    tmp_path: Path,
) -> None:
    app, _, _ = app_for(tmp_path / "vad")
    meeting = app.create_meeting(str(recording(tmp_path)), "T", date(2026, 10, 1))
    result = app.process(meeting)
    assert app.config.stt.vad_parameters == {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
        "max_speech_duration_s": 30.0,
    }
    assert (
        result.provenance["transcript"].decoding["vad_parameters"]
        == app.config.stt.vad_parameters
    )


def test_process_validates_cuda_compute_type_before_ingest(tmp_path: Path) -> None:
    from dataclasses import replace

    path = recording(tmp_path)
    app, _, _ = app_for(tmp_path / "unsupported")
    app.config = replace(
        app.config, stt=replace(app.config.stt, compute_type="float16")
    )
    meeting = app.create_meeting(str(path), "T", date(2026, 10, 1))
    with pytest.raises(ValueError, match="unsupported fake compute type"):
        app.process(meeting)
    assert not (meeting.folder / "normalized.wav").exists()


def test_preflight_runs_one_second_injected_local_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from ea_assistant import cli

    observed: dict[str, Any] = {}

    class FakeSTT:
        def validate_compute_type(self, config: STTConfig) -> None:
            observed["configured"] = config.compute_type

        def transcribe(
            self, path: str, config: STTConfig
        ) -> tuple[list[Segment], str, str]:
            with wave.open(path, "rb") as sample:
                observed["duration"] = sample.getnframes() / sample.getframerate()
            observed["vad_filter"] = config.vad_filter
            return [], config.model, "int8_float32"

        def release(self) -> None:
            observed["released"] = True

    class FakeLLM:
        def __init__(
            self,
            endpoint: str,
            timeout: float,
            *,
            unload_timeout: float,
            poll_interval: float,
        ) -> None:
            observed["endpoint"] = endpoint
            observed["unload_timeout"] = unload_timeout
            observed["poll_interval"] = poll_interval

        def model_digest(self, model: str) -> str:
            observed["model"] = model
            return "sha256:test"

        def ensure_gpu_free(self) -> None:
            observed["gpu_free_checked"] = True

    class FakeSensors:
        def gpu_temperature(self) -> int:
            return 45

        def on_ac(self) -> bool:
            return True

    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__dict__["__version__"] = "4.8.2"
    fake_ct2.__dict__["get_cuda_device_count"] = lambda: 1
    fake_ct2.__dict__["get_supported_compute_types"] = lambda device: {
        "int8",
        "int8_float32",
        "float32",
    }
    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.__dict__["__version__"] = "1.2.1"
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)
    monkeypatch.setattr(cli, "FasterWhisper", FakeSTT)
    monkeypatch.setattr(cli, "OllamaClient", FakeLLM)
    monkeypatch.setattr(cli, "MachineSensors", FakeSensors)
    monkeypatch.setattr(
        cli,
        "_gpu_information",
        lambda report: report.update({"driver_version": "580.178.04"}),
    )

    assert cli.preflight(cfg(tmp_path / "preflight")) == 0
    report = json.loads(capsys.readouterr().out)
    assert observed["duration"] == 1 and observed["vad_filter"] is False
    assert observed["configured"] == "int8" and observed["released"] is True
    assert observed["gpu_free_checked"] is True
    assert observed["unload_timeout"] == 15 and observed["poll_interval"] == 0.25
    assert report["effective_compute_type"] == "int8_float32"
    assert report["driver_version"] == "580.178.04"
    assert report["model_digest"] == "sha256:test"
