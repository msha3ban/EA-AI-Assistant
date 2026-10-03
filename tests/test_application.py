import hashlib
import json
import math
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from ea_assistant.adapters import RetryableLLMResponseError
from ea_assistant.application import Application
from ea_assistant.config import AppConfig, DetectionConfig, LLMCallConfig, load_config
from ea_assistant.domain import FACT_SECTIONS, Pipeline, StageName, VocabularyPromptMode
from ea_assistant.models import Segment, VocabularyTerm
from ea_assistant.prompts import SUMMARY_COMBINE, PromptSettings
from ea_assistant.storage import Store
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


def test_word_probability_detection_and_storage(tmp_path: Path) -> None:
    app = Application(FakeAudio(), FakeSpeechToText(), FakeLLM(), FakeSensors(),
                      config(tmp_path / "data"))
    segment = Segment("s1", 0, 1, "words", word_probabilities=(0.49, 0.5))
    app._mark_segment_flags(segment)
    assert segment.flags == ["low-confidence words"]
    at_threshold = Segment("s2", 0, 1, "ok", word_probabilities=(0.5,))
    app._mark_segment_flags(at_threshold)
    assert at_threshold.flags == []
    app.config = replace(app.config, detection=replace(app.config.detection, min_low_confidence_words=2))
    only_one = Segment("s3", 0, 1, "ok", word_probabilities=(0.49, 0.5))
    app._mark_segment_flags(only_one)
    assert only_one.flags == []
    meeting = app.create_meeting(str(_recording(tmp_path)), "T", date(2026, 10, 1))
    app.store.save_segments(meeting.id, [segment])
    assert app.store.segments(meeting.id)[0].word_probabilities == (0.49, 0.5)
    signals = app.store.db.execute("SELECT signals FROM transcript_segments").fetchone()[0]
    assert "words" not in signals
    assert "0.49" in signals


def _recording(tmp_path: Path) -> Path:
    path = tmp_path / "r.wav"
    path.write_bytes(b"r")
    return path


def test_old_segment_signals_load_without_word_probabilities(tmp_path: Path) -> None:
    store = Store(tmp_path / "old.sqlite")
    store.db.execute("INSERT INTO transcript_segments(meeting_id,id,start,end,text,signals,flags) VALUES(?,?,?,?,?,?,?)",
                     ("m", "s", 0, 1, "old", '{"avg_logprob": -0.2, "no_speech_prob": 0.1, "compression_ratio": 1.0}', "[]"))
    store.db.commit()
    assert store.segments("m")[0].word_probabilities == ()
    assert store.segments("m")[0].confidence_signals is True


def test_active_detector_names_follow_configured_rules() -> None:
    default = DetectionConfig()
    assert "low average log probability" not in default.active_detectors()
    configured = replace(default, avg_logprob_only=-0.5)
    segment = Segment("s", 0, 1, "words", avg_logprob=-0.6)
    matched = [name for name, rule in configured.detector_rules() if rule(segment)]
    assert matched == ["low average log probability"]
    assert set(matched) <= set(configured.active_detectors())


def test_missing_confidence_signals_skip_confidence_rules_and_survive_storage(
    tmp_path: Path,
) -> None:
    app = Application(
        FakeAudio(), FakeSpeechToText(), FakeLLM(), FakeSensors(),
        replace(config(tmp_path / "data"), detection=DetectionConfig(avg_logprob_only=-0.5)),
    )
    segment = Segment(
        "s1", 0, 1, "words", -2, 0.9, 3,
        word_probabilities=(0.1,), confidence_signals=False,
    )
    app._mark_segment_flags(segment)
    assert segment.flags == ["repetition loop"]
    meeting = app.create_meeting(str(_recording(tmp_path)), "T", date(2026, 10, 1))
    app.store.save_segments(meeting.id, [segment])
    assert app.store.segments(meeting.id)[0].confidence_signals is False
    signals = json.loads(
        app.store.db.execute("SELECT signals FROM transcript_segments").fetchone()[0]
    )
    assert signals["confidence_signals"] is False


def test_default_vocabulary_prompt_reaches_stt_for_normal_processing(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "r.m4a"
    recording.write_bytes(b"r")
    stt = FakeSpeechToText(word_probabilities=(0.4,))
    app = Application(
        FakeAudio(),
        stt,
        FakeLLM(),
        FakeSensors(),
        config(tmp_path / "data"),
        prompt_settings=PromptSettings(
            vocabulary=(VocabularyTerm("Kafka", ("كافكا",)),)
        ),
    )
    meeting = app.create_meeting(str(recording), "T", date(2026, 10, 1))
    app.process(meeting)
    assert stt.configs[0].hotwords == "Kafka"
    assert stt.configs[0].initial_prompt is None
    assert "low-confidence words" in app.store.segments(meeting.id)[0].flags


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


def test_json_call_retries_retryable_llm_response_failure(tmp_path: Path) -> None:
    responses = iter([RetryableLLMResponseError("Ollama returned invalid JSON"), '{"ok": true}'])

    def responder(_instructions: str, _content: dict[str, str]) -> str:
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    app = Application(
        FakeAudio(), FakeSpeechToText(), FakeLLM(responder=responder), FakeSensors(), config(tmp_path)
    )
    result = app._json_call(
        "Instructions", {}, StageName.EXTRACT, {}, lambda value: value == {"ok": True}
    )
    assert result == {"ok": True}
    assert app.llm.calls == 2


def test_json_call_propagates_second_retryable_failure(tmp_path: Path) -> None:
    llm = FakeLLM(responder=lambda _instructions, _content: _raise_retryable())
    app = Application(FakeAudio(), FakeSpeechToText(), llm, FakeSensors(), config(tmp_path))
    try:
        app._run_llm_stage(
            StageName.EXTRACT,
            lambda: app._json_call(
                "Instructions", {}, StageName.EXTRACT, {}, lambda _value: False
            ),
        )
    except RetryableLLMResponseError as exc:
        assert str(exc) == "Ollama returned invalid JSON"
        assert llm.calls == 2
        assert app.last_failure_reason == "extract: RetryableLLMResponseError"
    else:
        raise AssertionError("expected retryable response failure")


def _raise_retryable() -> str:
    raise RetryableLLMResponseError("Ollama returned invalid JSON")


def _facts(count: int) -> list[str]:
    return [f"Fact {index} " + "x" * 100 for index in range(count)]


def _fits_budget(instructions: str, content: str, config: LLMCallConfig) -> bool:
    prompt = instructions + "\n<meeting-content>\n" + content + "\n</meeting-content>"
    return math.ceil(len(prompt) / 2.5) + config.num_predict <= config.num_ctx


def _summary_app(
    tmp_path: Path,
    statements: list[str],
    partial_summary: str = "Partial Summary.",
    settings: PromptSettings | None = None,
) -> tuple[Application, FakeLLM]:
    def responder(instructions: str, _content: dict[str, str]) -> str:
        if instructions.startswith("Extract"):
            facts = {section.value: [] for section in FACT_SECTIONS}
            facts["discussion_points"] = [
                {"statement": statement, "evidence": ["s0001"]}
                for statement in statements
            ]
            return json.dumps(facts)
        if instructions.startswith(SUMMARY_COMBINE):
            return "Combined Summary."
        return partial_summary

    llm = FakeLLM(responder=responder)
    cfg = config(tmp_path / "data")
    cfg = replace(
        cfg,
        llm=replace(
            cfg.llm,
            summary=replace(cfg.llm.summary, num_ctx=300, num_predict=64),
        ),
    )
    app = Application(
        FakeAudio(), FakeSpeechToText(), llm, FakeSensors(), cfg, settings
    )
    return app, llm


def _process_summary(app: Application, tmp_path: Path) -> str:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    meeting = app.create_meeting(str(recording), "Planning", date(2026, 10, 1))
    result = app.process_transcript(meeting, [Segment("s0001", 0, 1, "Meeting")], 1.0)
    return result.summary


def test_summary_chunks_reconciled_facts_and_combines(tmp_path: Path) -> None:
    app, llm = _summary_app(tmp_path, _facts(12))

    assert _process_summary(app, tmp_path) == "Combined Summary.\n"
    summary_calls = [
        (instructions, content, call_config)
        for instructions, content, call_config in zip(
            llm.instructions, llm.contents, llm.configs
        )
        if instructions.startswith(("Write", SUMMARY_COMBINE))
    ]
    partials = [
        content
        for instructions, content, _ in summary_calls
        if instructions.startswith("Write")
    ]
    assert len(partials) > 1
    assert (
        sum(
            instructions.startswith(SUMMARY_COMBINE)
            for instructions, _, _ in summary_calls
        )
        == 1
    )
    assert [
        entry["statement"]
        for content in partials
        for entry in json.loads(content)["discussion_points"]
    ] == _facts(12)
    for instructions, content, call_config in summary_calls:
        assert _fits_budget(instructions, content, call_config)


def test_summary_combine_uses_llm_vocabulary(tmp_path: Path) -> None:
    settings = PromptSettings(
        pipeline=Pipeline.DIRECT,
        vocabulary=(VocabularyTerm("Kafka", ("كافكا",)),),
    )
    app, llm = _summary_app(tmp_path, _facts(12), settings=settings)

    assert _process_summary(app, tmp_path) == "Combined Summary.\n"
    combine_calls = [
        (instructions, content, call_config)
        for instructions, content, call_config in zip(
            llm.instructions, llm.contents, llm.configs
        )
        if instructions.startswith(SUMMARY_COMBINE)
    ]
    assert len(combine_calls) == 1
    vocabulary_instruction = "Spell these Vocabulary terms exactly: Kafka, كافكا."
    for instructions, content, call_config in combine_calls:
        assert instructions == settings.stage_prompt("summary_combine")
        assert vocabulary_instruction in instructions
        assert _fits_budget(instructions, content, call_config)


def test_summary_fitting_facts_keep_one_call(tmp_path: Path) -> None:
    app, llm = _summary_app(tmp_path, ["A short fact"])

    assert _process_summary(app, tmp_path) == "Partial Summary.\n"
    assert (
        sum(instructions.startswith("Write") for instructions in llm.instructions) == 1
    )
    assert SUMMARY_COMBINE not in llm.instructions
    summary_index = next(
        index
        for index, instructions in enumerate(llm.instructions)
        if instructions.startswith("Write")
    )
    expected_facts = {section.value: [] for section in FACT_SECTIONS}
    expected_facts["discussion_points"] = [
        {"statement": "A short fact", "evidence": ["s0001"]}
    ]
    assert llm.instructions[summary_index] == app.prompt_settings.stage_prompt(
        "summary"
    )
    assert llm.contents[summary_index] == json.dumps(expected_facts, ensure_ascii=False)


def test_summary_combines_partial_summaries_in_multiple_passes(tmp_path: Path) -> None:
    app, llm = _summary_app(tmp_path, _facts(60))

    assert _process_summary(app, tmp_path) == "Combined Summary.\n"
    assert (
        sum(
            instructions.startswith(SUMMARY_COMBINE)
            for instructions in llm.instructions
        )
        > 1
    )
    for instructions, content, call_config in zip(
        llm.instructions, llm.contents, llm.configs
    ):
        if instructions.startswith(SUMMARY_COMBINE):
            assert _fits_budget(instructions, content, call_config)


def test_summary_rejects_single_oversized_fact_without_content(tmp_path: Path) -> None:
    statement = "private-meeting-content-" + "x" * 1000
    app, _llm = _summary_app(tmp_path, [statement])

    with pytest.raises(
        ValueError, match="^Single fact entry exceeds configured LLM token budget$"
    ) as error:
        _process_summary(app, tmp_path)
    assert statement not in str(error.value)
    assert app.last_failure_reason is not None
    assert app.last_failure_reason.startswith("summary: ValueError")


def test_summary_stops_when_combine_groups_cannot_shrink(tmp_path: Path) -> None:
    app, llm = _summary_app(
        tmp_path,
        _facts(12),
        partial_summary="x" * 200,
    )

    with pytest.raises(ValueError, match="Partial Summaries cannot be combined"):
        _process_summary(app, tmp_path)
    assert not any(
        instructions.startswith(SUMMARY_COMBINE) for instructions in llm.instructions
    )
    assert app.last_failure_reason == "summary: ValueError"


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
