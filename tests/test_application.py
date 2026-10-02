import hashlib
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

from ea_assistant.application import Application
from ea_assistant.config import AppConfig, load_config
from ea_assistant.domain import Pipeline, VocabularyPromptMode
from ea_assistant.models import Segment, VocabularyTerm
from ea_assistant.prompts import PromptSettings
from ea_assistant.testing import FakeAudio, FakeLLM, FakeSensors, FakeSpeechToText


def config(tmp_path: Path) -> AppConfig:
    return replace(load_config(), data_dir=tmp_path)


def test_pipeline_publishes_artifacts_and_draft_mom(tmp_path: Path) -> None:
    recording = tmp_path / "recording.m4a"
    recording.write_bytes(b"recording")
    app = Application(
        FakeAudio(),
        FakeSpeechToText([Segment("s0001", 0, 1, "مرحبا")]),
        FakeLLM(),
        FakeSensors(),
        config(tmp_path / "data"),
    )
    meeting = app.create_meeting(str(recording), "Planning", date(2026, 10, 1))
    result = app.process(meeting)
    for name in ("transcript.md", "english-transcript.md", "summary.md", "mom.md"):
        assert (meeting.folder / name).is_file()
    assert "## Summary" in result.mom.markdown
    assert "State: Draft" in result.mom.markdown
    assert "s0001" in result.transcript
    assert "s0001" in result.english_transcript
    assert set(result.provenance) == {
        "transcript",
        "english_transcript",
        "summary",
        "mom",
    }
    assert "Speaker 1" in result.transcript
    with sqlite3.connect(tmp_path / "data" / "ea.sqlite") as db:
        assert db.execute("SELECT count(*) FROM meetings").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM recordings").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM transcript_segments").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM english_segments").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM artefacts").fetchone()[0] == 4


def test_detection_flags_segments(tmp_path: Path) -> None:
    recording = tmp_path / "r.m4a"
    recording.write_bytes(b"r")
    seg = Segment("s0001", 0, 1, "loop", -2, 0.9, 3)
    app = Application(
        FakeAudio(),
        FakeSpeechToText([seg]),
        FakeLLM(),
        FakeSensors(),
        config(tmp_path / "data"),
    )
    meeting = app.create_meeting(str(recording), "T", date(2026, 10, 1))
    app.process(meeting)
    transcript = (meeting.folder / "transcript.md").read_text()
    assert "repetition loop" in transcript and "likely text over silence" in transcript


def test_thermal_guard_stops_before_transcription(tmp_path: Path) -> None:
    recording = tmp_path / "r.m4a"
    recording.write_bytes(b"r")
    cfg = config(tmp_path / "data")
    app = Application(
        FakeAudio(), FakeSpeechToText(), FakeLLM(), FakeSensors(temp=None), cfg
    )
    meeting = app.create_meeting(str(recording), "T", date(2026, 10, 1))
    try:
        app.process(meeting)
    except RuntimeError as exc:
        assert "temperature unavailable" in str(exc)
    else:
        raise AssertionError("expected thermal stop")


def test_direct_provenance_references_supplied_transcript_and_stage_vocabulary(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = "مرحبا Kafka"
    settings = PromptSettings(
        pipeline=Pipeline.DIRECT,
        vocabulary=(VocabularyTerm("Kafka", ("كافكا",)),),
        vocabulary_prompt=VocabularyPromptMode.HOTWORDS,
        input_digest=hashlib.sha256(reference.encode()).hexdigest(),
    )
    app = Application(
        FakeAudio(),
        FakeSpeechToText(),
        FakeLLM(),
        FakeSensors(),
        config(tmp_path / "data"),
        prompt_settings=settings,
    )
    meeting = app.create_meeting(str(recording), "T", date(2026, 10, 1))
    result = app.process_transcript(meeting, [Segment("s0001", 0, 1, reference)], 1.0)
    assert "english_transcript" not in result.provenance
    assert result.provenance["summary"].input_revisions == {}
    assert result.provenance["mom"].input_revisions == {}
    assert result.provenance["summary"].input_digests == {
        "reference_transcript": hashlib.sha256(reference.encode()).hexdigest()
    }
    assert result.provenance["transcript"].vocabulary == []
    assert result.provenance["summary"].vocabulary == ["Kafka (aliases: كافكا)"]
