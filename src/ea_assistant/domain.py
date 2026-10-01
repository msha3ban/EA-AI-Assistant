from __future__ import annotations

from enum import StrEnum


class StageName(StrEnum):
    INGEST = "ingest"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    EXTRACT = "extract"
    SUMMARY = "summary"
    MOM = "mom"


class MomState(StrEnum):
    DRAFT = "Draft"


class FactStatus(StrEnum):
    AGREED = "agreed"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class FactSection(StrEnum):
    PURPOSE_AGENDA = "purpose_agenda"
    DISCUSSION_POINTS = "discussion_points"
    DECISIONS = "decisions"
    ACTION_ITEMS = "action_items"
    OPEN_QUESTIONS_RISKS = "open_questions_risks"
    NEXT_STEPS = "next_steps"
    TECHNICAL_DETAILS = "technical_details"


FACT_SECTIONS = tuple(FactSection)
