from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from .domain import MomState


@dataclass
class Segment:
    id: str
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    flags: list[str] = field(default_factory=list)
    speaker: str = "Speaker 1"
    recording_index: int = 0
    recording_start: float | None = None
    recording_end: float | None = None


@dataclass
class Meeting:
    id: str
    title: str
    date: date
    folder: Path
    topic: str | None = None


@dataclass
class Mom:
    markdown: str
    state: MomState = MomState.DRAFT


@dataclass
class Provenance:
    models: dict[str, dict[str, str]] = field(default_factory=dict)
    compute_type: str | None = None
    prompt_version: str = "1"
    decoding: dict[str, Any] = field(default_factory=dict)
    vocabulary: list[str] = field(default_factory=list)
    input_revisions: dict[str, int] = field(default_factory=dict)


@dataclass
class MeetingResult:
    meeting: Meeting
    transcript: str
    english_transcript: str
    summary: str
    mom: Mom
    provenance: dict[str, Provenance]
