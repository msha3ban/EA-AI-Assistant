from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..models import VocabularyTerm
from .timestamps import strip_line_timestamps


@dataclass(frozen=True)
class EditScore:
    errors: int
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    rate: float | None


def edit_score(reference: list[str], hypothesis: list[str]) -> EditScore:
    try:
        import numpy as np
    except ImportError:
        errors, subs, dels, ins = _edit_score_rolling(reference, hypothesis)
    else:
        width = len(hypothesis) + 1
        columns = np.arange(1, width, dtype=np.int32)
        previous_cost = np.arange(width, dtype=np.int32)
        previous_subs = np.zeros(width, dtype=np.int32)
        previous_dels = np.zeros(width, dtype=np.int32)
        previous_ins = np.arange(width, dtype=np.int32)
        hypothesis_array = np.asarray(hypothesis)
        positions = np.arange(width, dtype=np.int32)
        for row_index, ref in enumerate(reference, 1):
            diagonal = previous_cost[:-1] + (hypothesis_array != ref)
            deletion = previous_cost[1:] + 1
            choose_diagonal = diagonal <= deletion
            seed_cost = np.where(choose_diagonal, diagonal, deletion)
            seed_subs = np.where(
                choose_diagonal,
                previous_subs[:-1] + (hypothesis_array != ref),
                previous_subs[1:],
            )
            seed_dels = np.where(
                choose_diagonal, previous_dels[:-1], previous_dels[1:] + 1
            )
            seed_ins = np.where(choose_diagonal, previous_ins[:-1], previous_ins[1:])
            keys = np.concatenate(([row_index], seed_cost - columns))
            minima = np.minimum.accumulate(keys)
            best_indices = np.maximum.accumulate(np.where(keys == minima, positions, 0))
            indices = best_indices[1:]
            current_cost = columns + minima[1:]
            current_subs = np.concatenate(([0], seed_subs))[indices]
            current_dels = np.concatenate(([row_index], seed_dels))[indices]
            current_ins = np.concatenate(([0], seed_ins))[indices] + columns - indices
            previous_cost = np.concatenate(([row_index], current_cost))
            previous_subs = np.concatenate(([0], current_subs))
            previous_dels = np.concatenate(([row_index], current_dels))
            previous_ins = np.concatenate(([0], current_ins))
        errors = int(previous_cost[-1])
        subs, dels, ins = (
            int(previous_subs[-1]),
            int(previous_dels[-1]),
            int(previous_ins[-1]),
        )
    return EditScore(
        errors,
        subs,
        dels,
        ins,
        len(reference),
        errors / len(reference) if reference else (0.0 if not hypothesis else None),
    )


def _edit_score_rolling(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int, int]:
    """Two-row pure-Python fallback used when the optional STT extra is absent."""
    width = len(hypothesis) + 1
    previous = [(j, 0, 0, j) for j in range(width)]
    for ref in reference:
        current = [(previous[0][0] + 1, 0, previous[0][2] + 1, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            prior = previous[j - 1]
            diagonal = (
                prior[0] + (ref != hyp),
                prior[1] + (ref != hyp),
                prior[2],
                prior[3],
            )
            deletion_prior = previous[j]
            deletion = (
                deletion_prior[0] + 1,
                deletion_prior[1],
                deletion_prior[2] + 1,
                deletion_prior[3],
            )
            insertion_prior = current[j - 1]
            insertion = (
                insertion_prior[0] + 1,
                insertion_prior[1],
                insertion_prior[2],
                insertion_prior[3] + 1,
            )
            current.append(
                min((diagonal, deletion, insertion), key=lambda item: item[0])
            )
        previous = current
    return previous[-1]


def _strip_timestamps(text: str) -> str:
    return strip_line_timestamps(text)


def normalize(text: str, aliases: dict[str, str] | None = None) -> str:
    text = _strip_timestamps(text)
    text = text.translate(str.maketrans("أإآٱىةؤئ", "اااا يهوي".replace(" ", "")))
    text = "".join(
        ch
        for ch in text
        if ch not in "ـ" and not ("\u064b" <= ch <= "\u0652") and ch != "\u0670"
    )
    digit_map = {ord(chr(0x0660 + i)): str(i) for i in range(10)} | {
        ord(chr(0x06F0 + i)): str(i) for i in range(10)
    }
    text = text.translate(digit_map).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    if aliases:
        for alias, canonical in sorted(
            aliases.items(), key=lambda pair: len(pair[0]), reverse=True
        ):
            alias_text = normalize(alias)
            target = normalize(canonical)
            if alias_text:
                text = re.sub(
                    rf"(?<!\w){re.escape(alias_text)}(?!\w)",
                    target,
                    text,
                    flags=re.IGNORECASE,
                )
    return " ".join(text.split())


def score_text(
    reference: str, hypothesis: str, aliases: dict[str, str] | None = None
) -> dict[str, EditScore]:
    raw_ref, raw_hyp = (
        " ".join(_strip_timestamps(reference).split()),
        " ".join(_strip_timestamps(hypothesis).split()),
    )
    norm_ref, norm_hyp = normalize(reference, aliases), normalize(hypothesis, aliases)
    return {
        "wer_raw": edit_score(raw_ref.split(), raw_hyp.split()),
        "wer_normalized": edit_score(norm_ref.split(), norm_hyp.split()),
        "cer_raw": edit_score(
            list("".join(raw_ref.split())), list("".join(raw_hyp.split()))
        ),
        "cer_normalized": edit_score(
            list(norm_ref.replace(" ", "")), list(norm_hyp.replace(" ", ""))
        ),
    }


def vocabulary_accuracy(
    reference: str,
    hypothesis: str,
    terms: list[dict[str, Any]] | tuple[VocabularyTerm, ...],
) -> dict[str, Any]:
    def occurrences(text: str, variants: list[str]) -> list[tuple[int, int, str]]:
        normalized = normalize(text)
        choices = sorted(
            {normalize(variant) for variant in variants if normalize(variant)},
            key=len,
            reverse=True,
        )
        if not choices:
            return []
        pattern = re.compile(
            r"(?<!\w)(?:"
            + "|".join(re.escape(choice) for choice in choices)
            + r")(?!\w)"
        )
        return [
            (match.start(), match.end(), match.group())
            for match in pattern.finditer(normalized)
        ]

    def exact_occurrences(text: str, canonical: str) -> int:
        pattern = re.compile(r"(?<!\w)" + re.escape(canonical) + r"(?!\w)")
        return sum(1 for _ in pattern.finditer(_strip_timestamps(text)))

    result: list[dict[str, str | int]] = []
    for term in terms:
        if isinstance(term, VocabularyTerm):
            canonical = term.canonical
            aliases = term.aliases
        else:
            canonical = str(term["canonical"])
            aliases = term.get("aliases", [])
        variants = [canonical, *aliases]
        ref_counts = len(occurrences(reference, variants))
        hyp_all = len(occurrences(hypothesis, variants))
        hyp_canonical = exact_occurrences(hypothesis, canonical)
        recognized = min(ref_counts, hyp_all)
        canonical_count = min(ref_counts, hyp_canonical)
        result.append(
            {
                "canonical": canonical,
                "reference": ref_counts,
                "recognized": recognized,
                "canonical_hits": canonical_count,
            }
        )
    total_ref = sum(int(row["reference"]) for row in result)
    recognized_total = sum(int(row["recognized"]) for row in result)
    return {
        "terms": result,
        "reference": total_ref,
        "recognized": recognized_total,
        "canonical": sum(int(row["canonical_hits"]) for row in result),
        "recognized_rate": recognized_total / total_ref if total_ref else None,
    }


_NUM_WORDS = {
    "صفر": 0,
    "واحد": 1,
    "واحده": 1,
    "واحدة": 1,
    "اتنين": 2,
    "اثنين": 2,
    "تنين": 2,
    "ثلاثه": 3,
    "ثلاثة": 3,
    "تلاته": 3,
    "تلات": 3,
    "تلاتة": 3,
    "ثلاث": 3,
    "اربعه": 4,
    "أربعة": 4,
    "اربعة": 4,
    "خمسه": 5,
    "خمسة": 5,
    "سته": 6,
    "ستة": 6,
    "سبعه": 7,
    "سبعة": 7,
    "تمانيه": 8,
    "تمانية": 8,
    "ثمانية": 8,
    "تسعه": 9,
    "تسعة": 9,
    "عشره": 10,
    "عشرة": 10,
    "حداشر": 11,
    "احدعشر": 11,
    "اتناشر": 12,
    "تلاتاشر": 13,
    "اربعتاشر": 14,
    "خمستاشر": 15,
    "ستاشر": 16,
    "سبعتاشر": 17,
    "تمنتاشر": 18,
    "تسعتاشر": 19,
    "عشرين": 20,
    "عشرون": 20,
    "ثلاثين": 30,
    "تلاتين": 30,
    "اربعين": 40,
    "أربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "تمانين": 80,
    "تسعين": 90,
    "مئة": 100,
    "مائه": 100,
    "مية": 100,
    "مائتان": 200,
    "مئات": 100,
    "ميتين": 200,
    "تلتمية": 300,
    "ربعمية": 400,
    "خمسمية": 500,
    "ستمية": 600,
    "سبعمية": 700,
    "تمنمية": 800,
    "تسعمية": 900,
    "الف": 1000,
    "ألف": 1000,
    "آلاف": 1000,
    "ألفين": 2000,
    "الفين": 2000,
}

_NORMAL_WORDS = {normalize(word): value for word, value in _NUM_WORDS.items()}


def extract_numbers(text: str) -> list[str]:
    digit_map = {ord(chr(0x0660 + i)): str(i) for i in range(10)} | {
        ord(chr(0x06F0 + i)): str(i) for i in range(10)
    }
    digit_text = _strip_timestamps(text).translate(digit_map)
    found = [
        number.replace(",", "")
        for number in re.findall(
            r"(?<!\w)\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<!\w)\d+(?:\.\d+)?", digit_text
        )
    ]
    words = re.findall(r"[\w\u0600-\u06ff]+", normalize(text))
    found.extend(str(_NORMAL_WORDS[word]) for word in words if word in _NORMAL_WORDS)
    return found


def number_accuracy(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = Counter(extract_numbers(reference)), Counter(extract_numbers(hypothesis))
    hits = sum((ref & hyp).values())
    return {
        "reference": dict(ref),
        "hypothesis": dict(hyp),
        "hits": hits,
        "recall": hits / sum(ref.values()) if ref else None,
        "precision": hits / sum(hyp.values()) if hyp else (1.0 if not ref else 0.0),
    }
