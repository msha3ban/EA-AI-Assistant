from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

COLUMNS: list[tuple[str, str]] = [
    ("name", "Name"),
    ("pipeline", "Pipeline"),
    ("stt_model", "STT model"),
    ("compute_type", "Compute"),
    ("language", "Language"),
    ("vad", "VAD"),
    ("vocabulary_prompt", "Vocab prompt"),
    ("llm_model", "LLM model"),
    ("wer_raw", "WER raw"),
    ("wer_norm", "WER norm"),
    ("cer_raw", "CER raw"),
    ("cer_norm", "CER norm"),
    ("vocab_recognized_percent", "Vocab recognized %"),
    ("number_recall", "Number recall"),
    ("flag_recall", "Flag recall"),
    ("flag_precision", "Flag precision"),
    ("active_detectors", "Active detectors"),
    ("confidence_signals", "Confidence signals"),
    ("real_error_segments", "Real error segments"),
    ("flagged_segments", "Flagged segments"),
    ("mom_found", "MoM found"),
    ("mom_found_expected", "found in expected section"),
    ("mom_wrong", "wrong"),
    ("mom_missing", "missing"),
    ("mom_invented", "invented"),
    ("minutes_per_audio_hour", "min/audio hour"),
    ("peak_vram_mib", "Peak VRAM MiB"),
    ("peak_ram_mib", "Peak RAM MiB"),
    ("status", "Status"),
]
PERCENT_COLUMNS = {
    "wer_raw",
    "wer_norm",
    "cer_raw",
    "cer_norm",
    "vocab_recognized_percent",
    "number_recall",
    "flag_recall",
    "flag_precision",
}
INTEGER_COLUMNS = {"peak_vram_mib", "peak_ram_mib"}


def _cell(key: str, value: Any) -> str:
    if value is None:
        return "n/a" if key in PERCENT_COLUMNS else "—"
    if key in PERCENT_COLUMNS and isinstance(value, (int, float)):
        percent = value * 100 if key not in {"vocab_recognized_percent"} else value
        return f"{percent:.1f}%"
    if key in INTEGER_COLUMNS and isinstance(value, (int, float)):
        return f"{value:.0f}"
    if key == "minutes_per_audio_hour" and isinstance(value, (int, float)):
        return f"{value:.1f}"
    return str(value)


def table(rows: list[dict[str, Any]]) -> str:
    labels = [label for _, label in COLUMNS]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for row in rows:
        vals = [_cell(key, row.get(key)).replace("|", "\\|") for key, _ in COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_reports(result_dir: Path, payload: dict[str, Any], markdown: str) -> None:
    write_atomic(
        result_dir / "report.json", json.dumps(payload, ensure_ascii=False, indent=2)
    )
    write_atomic(result_dir / "report.md", markdown)
