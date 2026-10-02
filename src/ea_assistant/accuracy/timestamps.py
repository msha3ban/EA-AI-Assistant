from __future__ import annotations

import re

_PREFIX = re.compile(r"^\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*")
_ALL_PREFIXES = re.compile(r"(?m)^\s*\[(?:\d{1,2}:)?\d{1,2}:\d{2}\]\s*")


def parse_timestamp_prefix(line: str) -> tuple[float, str] | None:
    match = _PREFIX.match(line)
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    total = int(seconds) + int(minutes) * 60 + (int(hours) * 3600 if hours else 0)
    return float(total), line[match.end() :]


def strip_line_timestamps(text: str) -> str:
    return _ALL_PREFIXES.sub("", text)
