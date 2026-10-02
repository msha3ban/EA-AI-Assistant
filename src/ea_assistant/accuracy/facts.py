from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain import FactKind, FactSection
from ..render import SECTION_HEADINGS
from .metrics import normalize
from .toml_io import load_toml


@dataclass(frozen=True)
class CriticalFact:
    id: str
    kind: FactKind
    statement: str
    expect: tuple[str, ...]
    wrong_if: tuple[str, ...] = ()
    anchor: tuple[str, ...] = ()
    owner: str | None = None
    due: str | None = None

    def mapping(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "statement": self.statement,
            "expect": list(self.expect),
            "wrong_if": list(self.wrong_if),
            "anchor": list(self.anchor) if self.anchor else [self.expect[0]],
            "owner": self.owner,
            "due": self.due,
        }


def load_facts(path: str | Path) -> tuple[CriticalFact, ...]:
    raw = load_toml(path)
    if set(raw) != {"fact"} or not isinstance(raw["fact"], list):
        raise ValueError("Facts file must contain [[fact]] entries")
    allowed = {
        "id",
        "kind",
        "statement",
        "expect",
        "wrong_if",
        "anchor",
        "owner",
        "due",
    }
    parsed = []
    for fact in raw["fact"]:
        unknown = set(fact) - allowed
        if unknown:
            raise ValueError(
                f"Unknown critical fact keys: {', '.join(sorted(unknown))}"
            )
        if not {"id", "kind", "statement", "expect"} <= set(fact):
            raise ValueError(
                "Each critical fact requires id, kind, statement, and expect"
            )
        expect = fact["expect"]
        if (
            not isinstance(expect, list)
            or not expect
            or not all(isinstance(item, str) for item in expect)
        ):
            raise ValueError("Critical fact expect must be an array of strings")
        for field_name in ("wrong_if", "anchor"):
            value = fact.get(field_name, [])
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(
                    f"Critical fact {field_name} must be an array of strings"
                )
        parsed.append(
            CriticalFact(
                str(fact["id"]),
                FactKind(fact["kind"]),
                str(fact["statement"]),
                tuple(expect),
                tuple(fact.get("wrong_if", [])),
                tuple(fact.get("anchor", [])),
                str(fact["owner"]) if "owner" in fact else None,
                str(fact["due"]) if "due" in fact else None,
            )
        )
    return tuple(parsed)


def score_facts(
    markdown: str, facts: list[dict[str, Any]] | tuple[CriticalFact, ...]
) -> dict[str, Any]:
    facts = [
        fact.mapping() if isinstance(fact, CriticalFact) else fact for fact in facts
    ]
    sections: dict[str, list[dict[str, str]]] = {
        "decision": [],
        "action_item": [],
        "technical_detail": [],
    }
    current = ""
    current_label = ""
    in_table_body = False
    all_entries: list[dict[str, str]] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            in_table_body = False
            current_label = line[3:].strip()
            heading = current_label.lower()
            current = (
                "decision"
                if heading == SECTION_HEADINGS[FactSection.DECISIONS][3:].lower()
                else "action_item"
                if heading == SECTION_HEADINGS[FactSection.ACTION_ITEMS][3:].lower()
                else "technical_detail"
                if heading == SECTION_HEADINGS[FactSection.TECHNICAL_DETAILS][3:].lower()
                else ""
            )
        elif line.startswith("|"):
            if re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", line):
                in_table_body = True
                continue
            cols = [col.strip() for col in line.strip("|").split("|")]
            if in_table_body and len(cols) >= 4 and cols[0] != "#":
                entry = {"text": cols[1], "owner": cols[2], "due": cols[3], "section": current_label}
                if current == "action_item":
                    sections[current].append(entry)
                all_entries.append(entry)
        elif re.match(r"\s*(?:- |\* )\S", line):
            entry = {"text": line, "owner": "", "due": "", "section": current_label}
            if (current == "decision" and re.match(r"\s*-\s*D\d+:", line)) or current == "technical_detail":
                sections[current].append(entry)
            all_entries.append(entry)
        elif current_label.lower() == "summary" and line.strip():
            entry = {"text": line.strip(), "owner": "", "due": "", "section": current_label}
            all_entries.append(entry)
    found: list[str] = []
    wrong: list[str] = []
    missing: list[str] = []
    results: list[dict[str, Any]] = []
    matched: set[int] = set()
    for fact in facts:
        kind = str(fact["kind"])
        expect = [normalize(str(v)) for v in fact["expect"]]
        anchors = [normalize(str(v)) for v in fact.get("anchor", [fact["expect"][0]])]
        expected_candidates = sections.get(kind, [])
        candidates = expected_candidates
        if kind in {"date", "number"}:
            candidates = [
                {"text": sentence, "owner": "", "due": "", "section": ""}
                for line in markdown.splitlines()
                for sentence in re.split(r"(?<=[.!?؟؛])\s+", line)
                if sentence.strip()
            ]
            expected_candidates = candidates
        elif kind in sections:
            expected_ids = {id(entry) for entry in expected_candidates}
            fallback = [entry for entry in all_entries if id(entry) not in expected_ids]
            candidates = [*expected_candidates, *fallback]
        anchor_matches = [
            (i, entry)
            for i, entry in enumerate(candidates)
            if all(a in normalize(entry["text"]) for a in anchors)
        ]
        valid = [
            (i, entry)
            for i, entry in anchor_matches
            if all(value in normalize(entry["text"]) for value in expect)
        ]
        expected_ids = {id(entry) for entry in expected_candidates}
        def is_expected(entry: dict[str, str], expected_ids: set[int] = expected_ids) -> bool:
            return id(entry) in expected_ids
        def prioritize(matches: list[tuple[int, dict[str, str]]]) -> list[tuple[int, dict[str, str]]]:
            return sorted(matches, key=lambda item: not is_expected(item[1]))
        valid = prioritize(valid)
        anchor_matches = prioritize(anchor_matches)
        action_mismatches: list[tuple[int, dict[str, str], list[str]]] = []
        if kind == "action_item":
            for i, entry in valid:
                if not is_expected(entry):
                    continue
                reasons = []
                if fact.get("owner") and normalize(fact["owner"]) not in normalize(
                    entry["owner"]
                ):
                    reasons.append("owner")
                if fact.get("due") and normalize(fact["due"]) not in normalize(
                    entry["due"]
                ):
                    reasons.append("due")
                if reasons:
                    action_mismatches.append((i, entry, reasons))
            valid = [
                (i, entry)
                for i, entry in valid
                if not any(
                    mismatch_index == i for mismatch_index, _, _ in action_mismatches
                )
            ]
        wrong_matches = [
            (i, entry)
            for i, entry in anchor_matches
            if any(
                normalize(value) in normalize(entry["text"])
                for value in fact.get("wrong_if", [])
            )
        ]
        if wrong_matches:
            wrong.append(str(fact["id"]))
            matched_entry = wrong_matches[0][1]
            results.append(
                {
                    "id": str(fact["id"]),
                    "statement": str(fact["statement"]),
                    "status": "wrong",
                    "reason": "wrong_if",
                    "section": matched_entry["section"] or None,
                    "in_expected_section": is_expected(matched_entry),
                }
            )
            matched.add(id(wrong_matches[0][1]))
        elif action_mismatches:
            wrong.append(str(fact["id"]))
            _, mismatch_entry, reasons = action_mismatches[0]
            results.append(
                {
                    "id": str(fact["id"]),
                    "statement": str(fact["statement"]),
                    "status": "wrong",
                    "reason": ",".join(reasons),
                    "section": mismatch_entry["section"] or None,
                    "in_expected_section": is_expected(mismatch_entry),
                }
            )
            matched.add(id(mismatch_entry))
        elif valid:
            found.append(str(fact["id"]))
            results.append(
                {
                    "id": str(fact["id"]),
                    "statement": str(fact["statement"]),
                    "status": "found",
                    "section": valid[0][1]["section"] or None,
                    "in_expected_section": is_expected(valid[0][1]),
                }
            )
            matched.add(id(valid[0][1]))
        else:
            missing.append(str(fact["id"]))
            results.append(
                {
                    "id": str(fact["id"]),
                    "statement": str(fact["statement"]),
                    "status": "missing",
                    "section": None,
                    "in_expected_section": False,
                }
            )
    invented = [
        entry["text"]
        for kind in ("decision", "action_item")
        for i, entry in enumerate(sections[kind])
        if id(entry) not in matched
    ]
    return {
        "found": found,
        "wrong": wrong,
        "missing": missing,
        "results": results,
        "found_percent": len(found) * 100 / len(facts) if facts else None,
        "invented": invented,
    }
