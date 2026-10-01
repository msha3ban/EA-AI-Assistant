from __future__ import annotations

from typing import Any

from .domain import FACT_SECTIONS, FactSection, FactStatus

STRING = {"type": "string"}
EVIDENCE = {"type": "array", "items": STRING}


def _fact(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def english_transcript_schema(ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {segment_id: STRING for segment_id in ids},
        "required": ids,
        "additionalProperties": False,
    }


_STATUS = {"type": "string", "enum": [status.value for status in FactStatus]}
_SECTION_SCHEMAS: dict[FactSection, dict[str, Any]] = {
    FactSection.DECISIONS: _fact(
        {
            "statement": STRING,
            "rationale": {"type": ["string", "null"]},
            "status": _STATUS,
            "evidence": EVIDENCE,
        },
        ["statement", "rationale", "status", "evidence"],
    ),
    FactSection.ACTION_ITEMS: _fact(
        {
            "item": STRING,
            "owner": {"type": ["string", "null"]},
            "due_date": {"type": ["string", "null"]},
            "evidence": EVIDENCE,
        },
        ["item", "owner", "due_date", "evidence"],
    ),
    FactSection.TECHNICAL_DETAILS: _fact(
        {
            "subject": STRING,
            "statement": STRING,
            "status": _STATUS,
            "units": STRING,
            "evidence": EVIDENCE,
        },
        ["subject", "statement", "status", "units", "evidence"],
    ),
}
for _section in FACT_SECTIONS:
    if _section not in _SECTION_SCHEMAS:
        _SECTION_SCHEMAS[_section] = _fact(
            {"statement": STRING, "evidence": EVIDENCE}, ["statement", "evidence"]
        )
FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        section.value: {"type": "array", "items": _SECTION_SCHEMAS[section]}
        for section in FACT_SECTIONS
    },
    "required": [section.value for section in FACT_SECTIONS],
    "additionalProperties": False,
}


def validate_object(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    if schema.get("additionalProperties") is False and set(value) - set(
        schema.get("properties", {})
    ):
        return False
    if any(key not in value for key in schema.get("required", [])):
        return False
    return all(
        validate_value(item, schema.get("properties", {}).get(key, {}))
        for key, item in value.items()
    )


def validate_value(value: Any, schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if isinstance(kind, list):
        return (value is None and "null" in kind) or any(
            validate_value(value, {"type": option}) for option in kind
        )
    if kind == "string" and not isinstance(value, str):
        return False
    if kind == "array" and not isinstance(value, list):
        return False
    if kind == "object" and not isinstance(value, dict):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if kind == "array":
        return all(validate_value(item, schema.get("items", {})) for item in value)
    if kind == "object":
        return validate_object(value, schema)
    return True
