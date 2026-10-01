from __future__ import annotations

import logging
import math
from typing import Any

from .domain import FACT_SECTIONS, FactSection, FactStatus, MomState
from .models import Meeting, Segment

log = logging.getLogger(__name__)
SECTION_HEADINGS = {
    FactSection.PURPOSE_AGENDA: "## Purpose / Agenda",
    FactSection.DISCUSSION_POINTS: "## Discussion Points",
    FactSection.DECISIONS: "## Decisions",
    FactSection.ACTION_ITEMS: "## Action Items",
    FactSection.OPEN_QUESTIONS_RISKS: "## Open Questions / Risks",
    FactSection.NEXT_STEPS: "## Next Steps",
    FactSection.TECHNICAL_DETAILS: "## Technical Details",
}
SECTION_ORDER = ["## Summary"] + [
    SECTION_HEADINGS[section] for section in FACT_SECTIONS
]


def format_timestamp(seconds: float) -> str:
    value = max(0, int(seconds))
    return (
        f"{value // 3600}:{value % 3600 // 60:02d}:{value % 60:02d}"
        if value >= 3600
        else f"{value // 60:02d}:{value % 60:02d}"
    )


def duration_text(seconds: float) -> str:
    return format_timestamp(seconds)


def evidence_text(item: dict[str, Any]) -> str:
    refs = item.get("evidence", [])
    return f"[{', '.join(refs)}]" if refs else "Evidence: not stated"


def reconcile(
    facts: list[dict[str, Any]], segments: list[Segment]
) -> dict[str, list[dict[str, Any]]]:
    positions = {segment.id: segment.start for segment in segments}
    valid = set(positions)
    collected: dict[FactSection, list[tuple[float, dict[str, Any]]]] = {
        section: [] for section in FACT_SECTIONS
    }
    without_evidence = 0
    for chunk in facts:
        for section in FACT_SECTIONS:
            for raw in chunk.get(section.value, []):
                if not isinstance(raw, dict):
                    continue
                refs = list(
                    dict.fromkeys(
                        ref for ref in raw.get("evidence", []) if ref in valid
                    )
                )
                if not refs:
                    without_evidence += 1
                fact = {**raw, "evidence": refs}
                time = min((positions[ref] for ref in refs), default=math.inf)
                collected[section].append((time, fact))
    if without_evidence:
        log.warning(
            "Retained %d extracted facts without valid Evidence", without_evidence
        )
    result: dict[str, list[dict[str, Any]]] = {}
    for section, entries in collected.items():
        if section in (FactSection.DECISIONS, FactSection.TECHNICAL_DETAILS):
            key = "statement" if section is FactSection.DECISIONS else "subject"
            merged: dict[str, tuple[float, dict[str, Any]]] = {}
            for item_time, fact in sorted(entries, key=lambda pair: pair[0]):
                identity = str(fact.get(key, "")).casefold()
                if identity in merged:
                    prior_time, prior = merged[identity]
                    refs = list(dict.fromkeys(prior["evidence"] + fact["evidence"]))
                    merged[identity] = (prior_time, {**prior, **fact, "evidence": refs})
                else:
                    merged[identity] = (item_time, fact)
            entries = list(merged.values())
        result[section.value] = [
            fact for _, fact in sorted(entries, key=lambda pair: pair[0])
        ]
    return result


def mom_markdown(
    meeting: Meeting,
    duration: float,
    summary: str,
    facts: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        f"# {meeting.title}",
        f"Date: {meeting.date.isoformat()}",
        f"Duration: {duration_text(duration)}",
        f"Topic: {meeting.topic or 'none'}",
        f"State: {MomState.DRAFT.value}",
        "",
        "## Summary",
        summary,
        "",
        SECTION_HEADINGS[FactSection.PURPOSE_AGENDA],
    ]
    lines += [
        f"- {x.get('statement', 'not stated')} {evidence_text(x)}"
        for x in facts[FactSection.PURPOSE_AGENDA.value]
    ] or ["not stated"]
    lines += ["", SECTION_HEADINGS[FactSection.DISCUSSION_POINTS]] + [
        f"- {x.get('statement', '')} {evidence_text(x)}"
        for x in facts[FactSection.DISCUSSION_POINTS.value]
    ]
    lines += ["", SECTION_HEADINGS[FactSection.DECISIONS]]
    for index, item in enumerate(facts[FactSection.DECISIONS.value], 1):
        status = FactStatus(item.get("status", FactStatus.UNRESOLVED.value))
        lines.append(
            f"- D{index}: {item.get('statement', '')} (status: {status.value}; rationale: {item.get('rationale') or 'not stated'}) {evidence_text(item)}"
        )
    lines += [
        "",
        SECTION_HEADINGS[FactSection.ACTION_ITEMS],
        "| # | Item | Owner | Due date | Evidence |",
        "|---|---|---|---|---|",
    ]
    for index, item in enumerate(facts[FactSection.ACTION_ITEMS.value], 1):
        lines.append(
            f"| {index} | {item.get('item', '')} | {item.get('owner') or 'TBD'} | {item.get('due_date') or 'not stated'} | {', '.join(item.get('evidence', [])) or 'not stated'} |"
        )
    lines += ["", SECTION_HEADINGS[FactSection.OPEN_QUESTIONS_RISKS]] + [
        f"- {x.get('statement', '')} {evidence_text(x)}"
        for x in facts[FactSection.OPEN_QUESTIONS_RISKS.value]
    ]
    lines += ["", SECTION_HEADINGS[FactSection.NEXT_STEPS]] + [
        f"- {x.get('statement', '')} {evidence_text(x)}"
        for x in facts[FactSection.NEXT_STEPS.value]
    ]
    lines += ["", SECTION_HEADINGS[FactSection.TECHNICAL_DETAILS]]
    for item in facts[FactSection.TECHNICAL_DETAILS.value]:
        status = FactStatus(item.get("status", FactStatus.UNRESOLVED.value))
        lines.append(
            f"- **{item.get('subject', '')}**: {item.get('statement', '')} (status: {status.value}; units/conditions: {item.get('units') or 'not stated'}) {evidence_text(item)}"
        )
    return "\n".join(lines) + "\n"


def transcript_markdown(
    segments: list[Segment], english: dict[str, str] | None = None
) -> str:
    return (
        "\n".join(
            f"[{segment.id}] {format_timestamp(segment.start)}–{format_timestamp(segment.end)} {'⚑ ' + ', '.join(segment.flags) + ' ' if segment.flags else ''}**{segment.speaker}:** {(english or {}).get(segment.id, segment.text)}"
            for segment in segments
        )
        + "\n"
    )
