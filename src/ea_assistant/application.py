from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, TypeVar, cast

from .adapters import LLM, Audio, Sensors, SpeechToText
from .config import AppConfig
from .domain import StageName
from .models import Meeting, MeetingResult, Mom, Provenance, Segment
from .prompts import EXTRACT, PROMPT_VERSION, SUMMARY, TRANSLATE
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
    ) -> None:
        self.audio, self.stt, self.llm = audio, stt, llm
        self.config = config
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
        meeting = Meeting(meeting_id, title, meeting_date, folder, topic)
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
        english = self._translate(meeting, segments)
        facts = self._extract(meeting, segments, english)
        return self._render(meeting, duration, segments, facts)

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
        try:
            segments, stt_model, compute_type = self.stt.transcribe(
                str(normalized_path), self.config.stt
            )
        finally:
            self.stt.release()
        for segment in segments:
            self._mark_segment_flags(segment)
        self.store.save_segments(meeting.id, segments)
        self._publish(
            meeting,
            "transcript",
            meeting.folder / "transcript.md",
            transcript_markdown(segments),
            Provenance(
                models={stt_model: {"digest": self.stt.model_identifier}},
                compute_type=compute_type,
                prompt_version=PROMPT_VERSION,
                decoding=self.config.stt.adapter_values(),
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
            chunks = self._chunks(segments, TRANSLATE, StageName.TRANSLATE)
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
                    TRANSLATE,
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
            chunks = self._chunks(
                [(segment.id, english[segment.id]) for segment in segments],
                EXTRACT,
                StageName.EXTRACT,
            )
            for index, chunk in enumerate(chunks):
                self.guard.check()
                content = {segment_id: text for segment_id, text in chunk}
                value = self._json_call(
                    EXTRACT,
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
    ) -> MeetingResult:
        facts_by_section = reconcile(facts, segments)
        english_revision = self.store.artefact_revision(
            meeting.id, "english_transcript"
        )
        if self.store.is_stage_complete(meeting.id, StageName.SUMMARY):
            summary = self.store.artefact_content(meeting.id, "summary")
            if summary is None:
                raise RuntimeError("Summary artefact content is missing from SQLite")
        else:
            content = json.dumps(facts_by_section, ensure_ascii=False)
            summary = (
                self._run_llm_stage(
                    StageName.SUMMARY, lambda: self._summary_call(content)
                ).strip()
                + "\n"
            )
            self._publish(
                meeting,
                "summary",
                meeting.folder / "summary.md",
                summary,
                self._llm_provenance(
                    StageName.SUMMARY, {"english_transcript": english_revision}
                ),
            )
            self.store.mark_stage_complete(meeting.id, StageName.SUMMARY)
        if not self.store.is_stage_complete(meeting.id, StageName.MOM):
            mom_text = mom_markdown(
                meeting, duration, summary.strip(), facts_by_section
            )
            provenance = self._mom_provenance(english_revision)
            self._publish(
                meeting, "mom", meeting.folder / "mom.md", mom_text, provenance
            )
            self.store.mark_stage_complete(meeting.id, StageName.MOM)
        return self.read_meeting(meeting)

    def _summary_call(self, content: str) -> str:
        self.guard.check()
        return self.llm.chat(
            SUMMARY,
            content,
            self.config.llm.summary.adapter_values(self.config.llm.model),
        )

    def _mom_provenance(self, english_revision: int) -> Provenance:
        model = self.config.llm.model
        digest = self.llm.model_digest(model)
        return Provenance(
            models={model: {"digest": digest}},
            prompt_version=PROMPT_VERSION,
            decoding={
                "extract": self.config.llm.extract.adapter_values(model),
                "summary": self.config.llm.summary.adapter_values(model),
            },
            input_revisions={"english_transcript": english_revision},
        )

    def _llm_provenance(self, stage: StageName, inputs: dict[str, int]) -> Provenance:
        model = self.config.llm.model
        return Provenance(
            models={model: {"digest": self.llm.model_digest(model)}},
            prompt_version=PROMPT_VERSION,
            decoding=self.config.llm.stage(stage).adapter_values(model).__dict__,
            input_revisions=inputs,
        )

    def _run_llm_stage(self, stage: StageName, operation: Callable[[], T]) -> T:
        try:
            result = operation()
        except Exception as stage_error:
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
        for _attempt in range(2):
            raw = self.llm.chat(instructions, serialized, config, schema)
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
        config = self.config.llm.stage(stage)
        max_chars = (
            (config.num_ctx - config.num_predict) * 2.5 - len(instructions) - 100
        )
        chunks: list[list[Any]] = []
        current: list[Any] = []
        for item in items:
            trial = current + [item]
            content = (
                {entry.id: entry.text for entry in trial}
                if trial and isinstance(trial[0], Segment)
                else {entry[0]: entry[1] for entry in trial}
            )
            if current and len(json.dumps(content, ensure_ascii=False)) > max_chars:
                chunks.append(current)
                current = [item]
            else:
                current = trial
            one = (
                {item.id: item.text}
                if isinstance(item, Segment)
                else {item[0]: item[1]}
            )
            if len(json.dumps(one, ensure_ascii=False)) > max_chars:
                raise ValueError(
                    f"Single segment {next(iter(one))} exceeds configured LLM token budget"
                )
        if current:
            chunks.append(current)
        return chunks
