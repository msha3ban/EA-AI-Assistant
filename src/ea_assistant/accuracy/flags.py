from __future__ import annotations

from bisect import bisect_left
from typing import Any

from ..config import DetectionConfig
from ..models import Segment
from .metrics import edit_score, normalize


def _alignment(
    reference: list[str], hypothesis: list[str]
) -> list[tuple[str | None, str | None]]:
    import numpy as np

    height, width = len(reference), len(hypothesis)
    ref_values = np.asarray(reference, dtype="U1")
    hyp_values = np.asarray(hypothesis, dtype="U1")
    directions = np.zeros((height + 1, width + 1), dtype=np.uint8)
    directions[1:, 0] = 1  # deletion
    directions[0, 1:] = 2  # insertion
    previous = np.arange(width + 1, dtype=np.int32)
    columns = np.arange(1, width + 1, dtype=np.int32)
    for row_index in range(1, height + 1):
        substitutions = (ref_values[row_index - 1] != hyp_values).astype(np.int32)
        diagonal = previous[:-1] + substitutions
        deletion = previous[1:] + 1
        base = np.minimum(deletion, diagonal)
        prefix = np.minimum.accumulate(np.minimum(base - columns, row_index))
        current = np.empty(width + 1, dtype=np.int32)
        current[0] = row_index
        current[1:] = columns + prefix
        insertion = current[:-1] + 1
        choose_diagonal = (diagonal <= deletion) & (diagonal <= insertion)
        choose_deletion = (~choose_diagonal) & (deletion <= insertion)
        directions[row_index, 1:] = np.where(
            choose_diagonal, 0, np.where(choose_deletion, 1, 2)
        )
        previous = current
    pairs: list[tuple[str | None, str | None]] = []
    i, j = height, width
    while i or j:
        direction = int(directions[i, j])
        if i and j and direction == 0:
            pairs.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i and direction == 1:
            pairs.append((reference[i - 1], None))
            i -= 1
        else:
            pairs.append((None, hypothesis[j - 1]))
            j -= 1
    return list(reversed(pairs))


def flag_recall(
    reference: str,
    segments: list[Segment],
    threshold: float = 0.25,
    logprob_threshold: float = -0.8,
    duration: float = 1.0,
    aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    ref_chars = list(normalize(reference, aliases).replace(" ", ""))
    segment_chars = [
        list(normalize(segment.text, aliases).replace(" ", "")) for segment in segments
    ]
    hypothesis = [char for chars in segment_chars for char in chars]
    boundaries: list[int] = []
    end = 0
    for chars in segment_chars:
        end += len(chars)
        boundaries.append(end)
    ref_by_segment: list[list[str]] = [[] for _ in segments]
    hyp_by_segment: list[list[str]] = [[] for _ in segments]
    hpos = 0
    for ref_char, hyp_char in _alignment(ref_chars, hypothesis):
        if hyp_char is not None:
            index = min(bisect_left(boundaries, hpos + 1), max(len(segments) - 1, 0))
            if segments:
                hyp_by_segment[index].append(hyp_char)
                if ref_char is not None:
                    ref_by_segment[index].append(ref_char)
            hpos += 1
        elif ref_char is not None and segments:
            index = min(bisect_left(boundaries, hpos + 1), len(segments) - 1)
            ref_by_segment[index].append(ref_char)
    rows = []
    for index, segment in enumerate(segments):
        reference_chars = ref_by_segment[index]
        hypothesis_chars = hyp_by_segment[index]
        cer = (
            1.0
            if hypothesis_chars and not reference_chars
            else edit_score(reference_chars, hypothesis_chars).rate
        )
        real_error = bool(cer is not None and cer > threshold)
        marker = bool(segment.flags)
        repetition = "repetition loop" in segment.flags
        over_silence = "likely text over silence" in segment.flags
        low = segment.confidence_signals and segment.avg_logprob < logprob_threshold
        low_words = "low-confidence words" in segment.flags
        low_average = "low average log probability" in segment.flags
        rows.append(
            {
                "segment": segment.id,
                "cer": cer,
                "real_error": real_error,
                "marker": marker,
                "repetition_loop": repetition,
                "likely_text_over_silence": over_silence,
                "low_logprob": low,
                "low_confidence_words": low_words,
                "low_average_log_probability": low_average,
                "combined": marker or low,
            }
        )

    def tally(selected: list[bool]) -> dict[str, Any]:
        tp = sum(flag and bool(row["real_error"]) for flag, row in zip(selected, rows))
        positives = sum(selected)
        errors = sum(bool(row["real_error"]) for row in rows)
        return {
            "recall": tp / errors if errors else None,
            "precision": tp / positives if positives else None,
            "true_positives": tp,
            "flagged_segments": positives,
            "real_error_segments": errors,
            "flags_per_audio_hour": positives * 3600 / duration if duration else None,
        }

    return {
        "segments": rows,
        "marker": tally([bool(row["marker"]) for row in rows]),
        "repetition_loop": tally([bool(row["repetition_loop"]) for row in rows]),
        "likely_text_over_silence": tally(
            [bool(row["likely_text_over_silence"]) for row in rows]
        ),
        "low_logprob": tally([bool(row["low_logprob"]) for row in rows]),
        "low_confidence_words": tally(
            [bool(row["low_confidence_words"]) for row in rows]
        ),
        "low_average_log_probability": tally(
            [bool(row["low_average_log_probability"]) for row in rows]
        ),
        "combined": tally([bool(row["combined"]) for row in rows]),
        "real_errors": sum(bool(row["real_error"]) for row in rows),
        "segment_count": len(rows),
    }


def detection_sweep(
    segments: list[Segment],
    labels: dict[str, Any],
    word_probabilities: list[float],
    minimum_words: list[int],
) -> list[dict[str, Any]]:
    errors = {
        str(row["segment"]): bool(row["real_error"])
        for row in labels["segments"]
    }
    segment_ids = {segment.id for segment in segments}
    if segment_ids != set(errors) or len(segments) != len(errors):
        raise ValueError("Detection sweep segment IDs do not match real-error labels")
    error_count = sum(errors.values())
    rows = []
    for probability in word_probabilities:
        for minimum in minimum_words:
            detection = DetectionConfig(
                word_probability=probability,
                min_low_confidence_words=minimum,
            )
            word_rule = dict(detection.detector_rules())["low-confidence words"]
            selected = [
                word_rule(segment)
                for segment in segments
            ]
            flagged = sum(selected)
            hits = sum(
                flag and errors[segment.id]
                for flag, segment in zip(selected, segments)
            )
            rows.append(
                {
                    "word_probability": probability,
                    "min_low_confidence_words": minimum,
                    "recall": hits / error_count if error_count else None,
                    "precision": hits / flagged if flagged else None,
                    "flagged_segments": flagged,
                }
            )
    return rows
