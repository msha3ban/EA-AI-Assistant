from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, TypeVar, cast

from .adapters import LLM, Audio, RetryableLLMResponseError, Sensors, SpeechToText
from .config import AppConfig
from .domain import StageName
from .models import Meeting, MeetingResult, Mom, Provenance, Segment
from .prompts import PROMPT_VERSION, PromptSettings
from .render import mom_markdown, reconcile, transcript_markdown
from .schemas import FACT_SCHEMA, english_transcript_schema, validate_object
from .storage import Store
from .thermal import ThermalGuard

log = logging.getLogger(__name__)
T = TypeVar("T")


class Application:
    def __init__(
        self,
        audio: Audio,
        stt: SpeechToText,
        llm: LLM,
        sensors: Sensors,
        config: AppConfig,
        prompt_settings: PromptSettings | None = None,
    ) -> None:
        self.audio, self.stt, self.llm = audio, stt, llm
        self.config = config
        self.prompt_settings = prompt_settings or PromptSettings()
        self.last_failure_reason: str | None = None
        self.effective_language: str | None = None
        self.language_probability: float | None = None
        self.effective_compute_type: str | None = None
        self.guard = ThermalGuard(sensors, config.thermal)
        self.store = Store(config.data_dir / "ea.sqlite")

    def create_meeting(
        self, recording: str, title: str, meeting_date: date, topic: str | None = None
    ) -> Meeting:
        recording_path = Path(recording).expanduser().resolve()
        if not recording_path.is_file():
            raise FileNotFoundError(recording_path)
        duration = self.audio.probe(str(recording_path))
        meeting_id = uuid.uuid4().hex
        folder = self.config.data_dir / "meetings" / meeting_id
        folder.mkdir(parents=True)
        recording_copy = folder / recording_path.name
        shutil.copy2(recording_path, recording_copy)
        meeting = Meeting(meeting_id, title, meeting_date, folder, topic, duration)
        self.store.create_meeting(meeting)
        self.store.add_recording(meeting.id, recording_copy, 0, duration)
        return meeting

    def process(self, meeting: Meeting) -> MeetingResult:
        self.stt.validate_compute_type(self.config.stt)
        recordings = self.store.recordings(meeting.id)
        if not recordings:
            raise ValueError(f"Meeting {meeting.id} has no Recording")
        recording_path, duration = recordings[0]
        normalized_path = meeting.folder / "normalized.wav"
        self._ingest(meeting, recording_path, normalized_path)
        segments = self._transcribe(meeting, normalized_path)
        return self._run_stage_sequence(meeting, duration, segments)

    def process_pre_normalized(
        self, meeting: Meeting, normalized_path: str | Path
    ) -> MeetingResult:
        """Run all AI stages on a normalized Recording prepared by the caller."""
        self.stt.validate_compute_type(self.config.stt)
        recordings = self.store.recordings(meeting.id)
        if not recordings:
            raise ValueError(f"Meeting {meeting.id} has no Recording")
        _, duration = recordings[0]
        self.store.mark_stage_complete(meeting.id, StageName.INGEST)
        segments = self._transcribe(meeting, Path(normalized_path))
        return self._run_stage_sequence(meeting, duration, segments)

    def process_transcript(
        self, meeting: Meeting, segments: list[Segment], duration: float
    ) -> MeetingResult:
        """Generate a Summary and MoM from supplied mixed-language Transcript segments."""
        digest = (
            self.prompt_settings.input_digest
            or hashlib.sha256(
                "\n".join(item.text for item in segments).encode("utf-8")
            ).hexdigest()
        )
        self.store.save_segments(meeting.id, segments)
        self._publish(
            meeting,
            "transcript",
            meeting.folder / "transcript.md",
            transcript_markdown(segments),
            Provenance(
                prompt_version=PROMPT_VERSION,
                prompt_variant=self.prompt_settings.prompt_variant,
                vocabulary=self.prompt_settings.vocabulary_snapshot("transcript"),
            ),
        )
        self.store.mark_stage_complete(meeting.id, StageName.TRANSCRIBE)
        return self._run_stage_sequence(
            meeting, duration, segments, direct=True, input_digest=digest
        )

    def _run_stage_sequence(
        self,
        meeting: Meeting,
        duration: float,
        segments: list[Segment],
        direct: bool = False,
        input_digest: str | None = None,
    ) -> MeetingResult:
        english = (
            {item.id: item.text for item in segments}
            if direct
            else self._translate(meeting, segments)
        )
        facts = self._extract(meeting, segments, english)
        return self._render(
            meeting, duration, segments, facts, direct=direct, input_digest=input_digest
        )

    def read_meeting(self, meeting: Meeting) -> MeetingResult:
        def read(name: str) -> str:
            path = meeting.folder / name
            return path.read_text(encoding="utf-8") if path.exists() else ""

        return MeetingResult(
            meeting,
            read("transcript.md"),
            read("english-transcript.md"),
            read("summary.md"),
            Mom(read("mom.md")),
            self.store.provenance(meeting.id),
            self.effective_language,
            self.language_probability,
            self.effective_compute_type,
            self.store.segments(meeting.id),
            meeting.duration,
        )

    def _ingest(
        self, meeting: Meeting, recording_path: Path, normalized_path: Path
    ) -> None:
        if self.store.is_stage_complete(meeting.id, StageName.INGEST):
            return
        self.audio.probe(str(recording_path))
        self.audio.normalize(str(recording_path), str(normalized_path))
        self.store.mark_stage_complete(meeting.id, StageName.INGEST)

    def _transcribe(self, meeting: Meeting, normalized_path: Path) -> list[Segment]:
        if self.store.is_stage_complete(meeting.id, StageName.TRANSCRIBE):
            return self.store.segments(meeting.id)
        self.guard.check()
        self.llm.ensure_gpu_free()
        stt_values: dict[str, Any] = {
            key: value
            for key, value in self.prompt_settings.stt_prompt_values().items()
            if value is not None
        }
        stt_config = replace(
            self.config.stt,
            **stt_values,
            vocabulary=tuple(
                term.canonical for term in self.prompt_settings.stt_vocabulary
            ),
        )
        try:
            segments, stt_model, compute_type = self.stt.transcribe(
                str(normalized_path), stt_config
            )
            self.effective_language = (
                getattr(self.stt, "detected_language", None) or self.config.stt.language
            )
            self.language_probability = getattr(self.stt, "language_probability", None)
            self.effective_compute_type = compute_type
        finally:
            self.stt.release()
        for segment in segments:
            self._mark_segment_flags(segment)
        self.store.save_segments(meeting.id, segments)
        vocabulary_applied = getattr(self.stt, "vocabulary_applied", None)
        self._publish(
            meeting,
            "transcript",
            meeting.folder / "transcript.md",
            transcript_markdown(segments),
            Provenance(
                models={stt_model: {"digest": self.stt.model_identifier}},
                compute_type=compute_type,
                prompt_version=PROMPT_VERSION,
                decoding={
                    **self.config.stt.adapter_values(),
                    **(
                        {"vocabulary_applied": vocabulary_applied}
                        if isinstance(vocabulary_applied, bool)
                        else {}
                    ),
                },
                prompt_variant=self.prompt_settings.prompt_variant,
                vocabulary=self.prompt_settings.vocabulary_snapshot("transcript"),
            ),
        )
        self.store.mark_stage_complete(meeting.id, StageName.TRANSCRIBE)
        return segments

    def _mark_segment_flags(self, segment: Segment) -> None:
        if segment.compression_ratio > self.config.detection.compression_ratio:
            segment.flags.append("repetition loop")
        if (
            segment.no_speech_prob > self.config.detection.no_speech_prob
            and segment.avg_logprob < self.config.detection.avg_logprob
        ):
            segment.flags.append("likely text over silence")

    def _translate(self, meeting: Meeting, segments: list[Segment]) -> dict[str, str]:
        if self.store.is_stage_complete(meeting.id, StageName.TRANSLATE):
            return self.store.english(meeting.id)

        def run() -> dict[str, str]:
            english: dict[str, str] = {}
            instructions = self.prompt_settings.stage_prompt("translate")
            chunks = self._chunks(segments, instructions, StageName.TRANSLATE)
            for chunk in chunks:
                self.guard.check()
                ids = [segment.id for segment in chunk]
                content = {segment.id: segment.text for segment in chunk}
                schema = english_transcript_schema(ids)

                def valid_english_transcript(
                    candidate: Any,
                    expected_schema: dict[str, Any] = schema,
                    expected_ids: tuple[str, ...] = tuple(ids),
                ) -> bool:
                    return validate_object(candidate, expected_schema) and set(
                        candidate
                    ) == set(expected_ids)

                value = self._json_call(
                    instructions,
                    content,
                    StageName.TRANSLATE,
                    schema,
                    valid_english_transcript,
                )
                english.update({key: str(value[key]) for key in ids})
            return english

        english = self._run_llm_stage(StageName.TRANSLATE, run)
        self.store.save_english(meeting.id, english)
        provenance = self._llm_provenance(
            StageName.TRANSLATE,
            {"transcript": self.store.artefact_revision(meeting.id, "transcript")},
        )
        self._publish(
            meeting,
            "english_transcript",
            meeting.folder / "english-transcript.md",
            transcript_markdown(segments, english),
            provenance,
        )
        self.store.mark_stage_complete(meeting.id, StageName.TRANSLATE)
        return english

    def _extract(
        self, meeting: Meeting, segments: list[Segment], english: dict[str, str]
    ) -> list[dict[str, Any]]:
        if self.store.is_stage_complete(meeting.id, StageName.EXTRACT):
            return self.store.facts(meeting.id)

        def run() -> list[dict[str, Any]]:
            all_facts = []
            instructions = self.prompt_settings.stage_prompt("extract")
            chunks = self._chunks(
                [(segment.id, english[segment.id]) for segment in segments],
                instructions,
                StageName.EXTRACT,
            )
            for index, chunk in enumerate(chunks):
                self.guard.check()
                content = {segment_id: text for segment_id, text in chunk}
                value = self._json_call(
                    instructions,
                    content,
                    StageName.EXTRACT,
                    FACT_SCHEMA,
                    lambda candidate: validate_object(candidate, FACT_SCHEMA),
                )
                self.store.save_facts(meeting.id, index, value)
                all_facts.append(value)
            return all_facts

        facts = self._run_llm_stage(StageName.EXTRACT, run)
        self.store.mark_stage_complete(meeting.id, StageName.EXTRACT)
        return facts

    def _render(
        self,
        meeting: Meeting,
        duration: float,
        segments: list[Segment],
        facts: list[dict[str, Any]],
        direct: bool = False,
        input_digest: str | None = None,
    ) -> MeetingResult:
        facts_by_section = reconcile(facts, segments)
        english_revision = (
            None
            if direct
            else self.store.artefact_revision(meeting.id, "english_transcript")
        )
        input_refs: dict[str, int] = {}
        if not direct and english_revision is not None:
            input_refs["english_transcript"] = english_revision
        if self.store.is_stage_complete(meeting.id, StageName.SUMMARY):
            summary = self.store.artefact_content(meeting.id, "summary")
            if summary is None:
                raise RuntimeError("Summary artefact content is missing from SQLite")
        else:
            summary = (
                self._run_llm_stage(
                    StageName.SUMMARY, lambda: self._summarise(facts_by_section)
                ).strip()
                + "\n"
            )
            self._publish(
                meeting,
                "summary",
                meeting.folder / "summary.md",
                summary,
                self._llm_provenance(StageName.SUMMARY, input_refs, input_digest),
            )
            self.store.mark_stage_complete(meeting.id, StageName.SUMMARY)
        if not self.store.is_stage_complete(meeting.id, StageName.MOM):
            mom_text = mom_markdown(
                meeting, duration, summary.strip(), facts_by_section
            )
            provenance = self._mom_provenance(input_refs, input_digest)
            self._publish(
                meeting, "mom", meeting.folder / "mom.md", mom_text, provenance
            )
            self.store.mark_stage_complete(meeting.id, StageName.MOM)
        return self.read_meeting(meeting)

    def _summarise(self, facts_by_section: dict[str, list[dict[str, Any]]]) -> str:
        instructions = self.prompt_settings.stage_prompt("summary")
        content = json.dumps(facts_by_section, ensure_ascii=False)
        if len(content) <= self._max_content_chars(instructions, StageName.SUMMARY):
            return self._summary_call(content, instructions)

        fact_chunks = self._chunk_facts(facts_by_section, instructions)
        partials = [
            self._summary_call(json.dumps(chunk, ensure_ascii=False), instructions)
            for chunk in fact_chunks
        ]
        return self._combine_summaries(partials)

    def _chunk_facts(
        self, facts_by_section: dict[str, list[dict[str, Any]]], instructions: str
    ) -> list[dict[str, list[dict[str, Any]]]]:
        def fact_payload(
            facts: list[tuple[str, dict[str, Any]]],
        ) -> dict[str, list[dict[str, Any]]]:
            by_section: dict[str, list[dict[str, Any]]] = {}
            for section, entry in facts:
                by_section.setdefault(section, []).append(entry)
            return by_section

        entries = (
            (section, entry)
            for section, section_entries in facts_by_section.items()
            for entry in section_entries
        )
        chunks = self._pack_items(
            entries,
            self._max_content_chars(instructions, StageName.SUMMARY),
            fact_payload,
            lambda _entry: "Single fact entry exceeds configured LLM token budget",
        )
        return [fact_payload(chunk) for chunk in chunks]

    def _combine_summaries(self, partials: list[str]) -> str:
        instructions = self.prompt_settings.stage_prompt("summary_combine")
        max_chars = self._max_content_chars(instructions, StageName.SUMMARY)
        while len(partials) > 1:
            groups = self._pack_items(
                partials,
                max_chars,
                lambda summaries: summaries,
                lambda _summary: (
                    "Single partial Summary exceeds configured LLM token budget"
                ),
            )
            if len(groups) >= len(partials):
                raise ValueError(
                    "Partial Summaries cannot be combined within configured LLM token budget"
                )
            partials = [
                self._summary_call(
                    json.dumps(summaries, ensure_ascii=False), instructions
                )
                if len(summaries) > 1
                else summaries[0]
                for summaries in groups
            ]
        return partials[0]

    def _summary_call(self, content: str, instructions: str) -> str:
        self.guard.check()
        return self.llm.chat(
            instructions,
            content,
            self.config.llm.summary.adapter_values(self.config.llm.model),
        )

    def _mom_provenance(
        self, input_revisions: dict[str, int], input_digest: str | None
    ) -> Provenance:
        model = self.config.llm.model
        digest = self.llm.model_digest(model)
        return Provenance(
            models={model: {"digest": digest}},
            prompt_version=PROMPT_VERSION,
            prompt_variant=self.prompt_settings.prompt_variant,
            decoding={
                "extract": self.config.llm.extract.adapter_values(model),
                "summary": self.config.llm.summary.adapter_values(model),
            },
            vocabulary=self.prompt_settings.vocabulary_snapshot("summary"),
            input_revisions=input_revisions,
            input_digests={"reference_transcript": input_digest}
            if input_digest
            else {},
        )

    def _llm_provenance(
        self, stage: StageName, inputs: dict[str, int], input_digest: str | None = None
    ) -> Provenance:
        model = self.config.llm.model
        return Provenance(
            models={model: {"digest": self.llm.model_digest(model)}},
            prompt_version=PROMPT_VERSION,
            prompt_variant=self.prompt_settings.prompt_variant,
            decoding=self.config.llm.stage(stage).adapter_values(model).__dict__,
            vocabulary=self.prompt_settings.vocabulary_snapshot(stage.value),
            input_revisions=inputs,
            input_digests={"reference_transcript": input_digest}
            if input_digest
            else {},
        )

    def _run_llm_stage(self, stage: StageName, operation: Callable[[], T]) -> T:
        try:
            result = operation()
        except Exception as stage_error:
            timeout = self.config.llm.stage(stage).timeout
            self.last_failure_reason = (
                f"{stage.value}: TimeoutError after {timeout:g} s"
                if isinstance(stage_error, TimeoutError)
                else f"{stage.value}: {type(stage_error).__name__}"
            )
            try:
                self.llm.unload(self.config.llm.model)
            except Exception as unload_error:
                log.warning("LLM unload failed after %s stage error", stage.value)
                raise stage_error from unload_error
            raise
        self.llm.unload(self.config.llm.model)
        return result

    def _publish(
        self, meeting: Meeting, name: str, path: Path, text: str, provenance: Provenance
    ) -> None:
        previous = path.read_bytes() if path.exists() else None
        self._write_atomic(path, text)
        try:
            self.store.artefact(meeting.id, name, path, provenance, text)
        except BaseException:
            if previous is None:
                path.unlink(missing_ok=True)
            else:
                self._write_atomic(path, previous.decode("utf-8"))
            raise

    def _write_atomic(self, path: Path, text: str) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _json_call(
        self,
        instructions: str,
        content: dict[str, str],
        stage: StageName,
        schema: dict[str, Any],
        validator: Callable[[Any], bool],
    ) -> dict[str, Any]:
        config = self.config.llm.stage(stage).adapter_values(self.config.llm.model)
        serialized = json.dumps(content, ensure_ascii=False)
        for attempt in range(2):
            try:
                raw = self.llm.chat(instructions, serialized, config, schema)
            except RetryableLLMResponseError:
                if attempt == 1:
                    raise
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if validator(parsed):
                return cast(dict[str, Any], parsed)
        raise ValueError(
            f"{stage.value} response failed JSON/schema validation after one retry"
        )

    def _chunks(
        self, items: list[Any], instructions: str, stage: StageName
    ) -> list[list[Any]]:
        def segment_payload(chunk: list[Any]) -> dict[str, str]:
            return (
                {entry.id: entry.text for entry in chunk}
                if chunk and isinstance(chunk[0], Segment)
                else {entry[0]: entry[1] for entry in chunk}
            )

        return self._pack_items(
            items,
            self._max_content_chars(instructions, stage),
            segment_payload,
            lambda item: (
                f"Single segment {item.id if isinstance(item, Segment) else item[0]} "
                "exceeds configured LLM token budget"
            ),
        )

    def _pack_items(
        self,
        items: Iterable[T],
        max_chars: float,
        payload: Callable[[list[T]], Any],
        oversized_message: Callable[[T], str],
    ) -> list[list[T]]:
        chunks: list[list[T]] = []
        current: list[T] = []
        for item in items:
            current.append(item)
            if len(json.dumps(payload(current), ensure_ascii=False)) <= max_chars:
                continue
            current.pop()
            if current:
                chunks.append(current)
            current = [item]
            if len(json.dumps(payload(current), ensure_ascii=False)) > max_chars:
                raise ValueError(oversized_message(item))
        if current:
            chunks.append(current)
        return chunks

    def _max_content_chars(self, instructions: str, stage: StageName) -> float:
        config = self.config.llm.stage(stage)
        return (config.num_ctx - config.num_predict) * 2.5 - len(instructions) - 100
