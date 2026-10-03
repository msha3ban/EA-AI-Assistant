from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..config import AppConfig, DetectionConfig, StageConfig, STTConfig
from ..domain import Pipeline, VocabularyPromptMode
from ..models import Vocabulary, VocabularyTerm
from .toml_io import load_toml


@dataclass(frozen=True)
class Excerpt:
    start: str
    end: str


@dataclass(frozen=True)
class SuiteRun:
    name: str
    pipeline: Pipeline
    vocabulary_prompt: VocabularyPromptMode
    stt: dict[str, Any]
    llm: dict[str, Any]
    detection: dict[str, Any] | None = None

    def mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pipeline": self.pipeline.value,
            "vocabulary_prompt": self.vocabulary_prompt.value,
            "stt": self.stt,
            "llm": self.llm,
            "detection": self.detection or {},
        }


@dataclass(frozen=True)
class Suite:
    name: str
    recording: Path
    reference_transcript: Path
    output_dir: Path
    suite_path: Path
    run: tuple[SuiteRun, ...]
    excerpt: Excerpt | None = None
    critical_facts: Path | None = None
    vocabulary: Path | None = None
    flag_cer_threshold: float = 0.25
    flag_logprob_threshold: float = -0.8
    vocabulary_fixture: Vocabulary | None = None
    sweep_detection: dict[str, list[float] | list[int]] | None = None

    def mapping(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "recording": str(self.recording),
            "reference_transcript": str(self.reference_transcript),
            "output_dir": str(self.output_dir),
            "suite_path": str(self.suite_path),
            "run": [run.mapping() for run in self.run],
            "flag_cer_threshold": self.flag_cer_threshold,
            "flag_logprob_threshold": self.flag_logprob_threshold,
            "sweep": {"detection": self.sweep_detection or default_detection_sweep()},
        }
        if self.excerpt:
            result["excerpt"] = {"start": self.excerpt.start, "end": self.excerpt.end}
        if self.critical_facts:
            result["critical_facts"] = str(self.critical_facts)
        if self.vocabulary:
            result["vocabulary"] = str(self.vocabulary)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.mapping()[key]


def load_suite(path: str | Path) -> Suite:
    suite_path = Path(path).expanduser().resolve()
    raw = load_toml(suite_path)
    allowed = {
        "name",
        "recording",
        "excerpt",
        "reference_transcript",
        "critical_facts",
        "vocabulary",
        "output_dir",
        "flag_cer_threshold",
        "flag_logprob_threshold",
        "run",
        "sweep",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown suite keys: {', '.join(sorted(unknown))}")
    for required in ("name", "recording", "reference_transcript", "run"):
        if required not in raw:
            raise ValueError(f"Missing suite key: {required}")

    def resolve(value: str) -> str:
        candidate = Path(value).expanduser()
        return str(
            (suite_path.parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )

    for field in ("recording", "reference_transcript", "critical_facts", "vocabulary"):
        if field in raw:
            raw[field] = resolve(str(raw[field]))
    output = raw.get("output_dir", "results")
    raw["output_dir"] = resolve(str(output))
    raw["suite_path"] = str(suite_path)
    if not isinstance(raw["run"], list) or not raw["run"]:
        raise ValueError("Suite must define at least one [[run]]")
    names: set[str] = set()
    for run in raw["run"]:
        unknown_run = set(run) - {"name", "pipeline", "vocabulary_prompt", "stt", "llm", "detection"}
        if unknown_run:
            raise ValueError(f"Unknown run keys: {', '.join(sorted(unknown_run))}")
        for key in ("name", "pipeline"):
            if key not in run:
                raise ValueError(f"Each run requires {key}")
        if run["pipeline"] not in {"two-step", "direct"}:
            raise ValueError(f"Unknown pipeline {run['pipeline']!r}")
        if run["name"] in names:
            raise ValueError(f"Duplicate run name {run['name']!r}")
        if Path(run["name"]).name != run["name"] or run["name"] in {".", ".."}:
            raise ValueError(
                f"Run name must be a single directory name: {run['name']!r}"
            )
        names.add(run["name"])
        if run.get("vocabulary_prompt", "off") not in {
            "off",
            "initial_prompt",
            "hotwords",
        }:
            raise ValueError(
                "vocabulary_prompt must be off, initial_prompt, or hotwords"
            )
    runs = tuple(
        SuiteRun(
            str(run["name"]),
            Pipeline(run["pipeline"]),
            VocabularyPromptMode(run.get("vocabulary_prompt", "off")),
            dict(run.get("stt", {})),
            dict(run.get("llm", {})),
            dict(run.get("detection", {})),
        )
        for run in raw["run"]
    )
    excerpt_data = raw.get("excerpt")
    excerpt = (
        Excerpt(str(excerpt_data["start"]), str(excerpt_data["end"]))
        if excerpt_data
        else None
    )
    vocab_fixture = (
        load_vocabulary(raw["vocabulary"]) if raw.get("vocabulary") else None
    )
    return Suite(
        str(raw["name"]),
        Path(raw["recording"]),
        Path(raw["reference_transcript"]),
        Path(raw["output_dir"]),
        suite_path,
        runs,
        excerpt,
        Path(raw["critical_facts"]) if "critical_facts" in raw else None,
        Path(raw["vocabulary"]) if "vocabulary" in raw else None,
        float(raw.get("flag_cer_threshold", 0.25)),
        float(raw.get("flag_logprob_threshold", -0.8)),
        vocab_fixture,
        parse_detection_sweep(raw.get("sweep", {}).get("detection", {})),
    )


def default_detection_sweep() -> dict[str, list[float] | list[int]]:
    return {"word_probability": [0.2, 0.35, 0.5, 0.65],
            "min_low_confidence_words": [1, 2]}


def parse_detection_sweep(raw: dict[str, Any]) -> dict[str, list[float] | list[int]]:
    unknown = set(raw) - {"word_probability", "min_low_confidence_words"}
    if unknown:
        raise ValueError(f"Unknown [sweep.detection] keys: {', '.join(sorted(unknown))}")
    values = {**default_detection_sweep(), **raw}
    probabilities = values["word_probability"]
    minimums = values["min_low_confidence_words"]
    if not isinstance(probabilities, list) or not probabilities or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1
        for value in probabilities
    ):
        raise ValueError("sweep word_probability must be a nonempty list of values between 0 and 1")
    if not isinstance(minimums, list) or not minimums or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in minimums
    ):
        raise ValueError("sweep min_low_confidence_words must be a nonempty list of positive integers")
    return {"word_probability": [float(value) for value in probabilities],
            "min_low_confidence_words": minimums}


def load_vocabulary(path: str | Path) -> Vocabulary:
    raw = load_toml(path)
    if set(raw) != {"term"} or not isinstance(raw["term"], list):
        raise ValueError("Vocabulary file must contain [[term]] entries")
    terms = []
    for item in raw["term"]:
        if set(item) - {"canonical", "aliases"} or not isinstance(
            item.get("canonical"), str
        ):
            raise ValueError("Vocabulary term requires canonical and optional aliases")
        aliases = item.get("aliases", [])
        if not isinstance(aliases, list) or not all(
            isinstance(alias, str) for alias in aliases
        ):
            raise ValueError("Vocabulary aliases must be an array of strings")
        terms.append(VocabularyTerm(item["canonical"], tuple(aliases)))
    return Vocabulary(tuple(terms))


def merge_config(base: AppConfig, run: dict[str, Any], data_dir: Path) -> AppConfig:
    detection_values = run.get("detection", {})
    unknown_detection = set(detection_values) - set(DetectionConfig.__dataclass_fields__)
    if unknown_detection:
        raise ValueError(f"Unknown [run.detection] keys: {', '.join(sorted(unknown_detection))}")
    detection = replace(base.detection, **detection_values)
    allowed_stt = set(STTConfig.__dataclass_fields__)
    stt_overrides = run.get("stt", {})
    unknown = set(stt_overrides) - allowed_stt
    if unknown:
        raise ValueError(f"Unknown [run.stt] keys: {', '.join(sorted(unknown))}")
    stt_values = dict(stt_overrides)
    if "vad_parameters" in stt_values:
        stt_values["vad_parameters"] = {
            **base.stt.vad_parameters,
            **stt_values["vad_parameters"],
        }
    stt = replace(base.stt, **stt_values)
    llmraw = run.get("llm", {})
    allowed_llm = {
        "model",
        "translate",
        "extract",
        "summary",
        "stages",
    }
    unknown = set(llmraw) - allowed_llm
    if unknown:
        raise ValueError(f"Unknown [run.llm] keys: {', '.join(sorted(unknown))}")
    stages = {}
    for name in ("translate", "extract", "summary"):
        nested_stages = llmraw.get("stages", {})
        unknown_nested = set(nested_stages) - {"translate", "extract", "summary"}
        if unknown_nested:
            raise ValueError(
                f"Unknown [run.llm.stages] keys: {', '.join(sorted(unknown_nested))}"
            )
        values = {**nested_stages.get(name, {}), **llmraw.get(name, {})}
        unknown_stage = set(values) - set(StageConfig.__dataclass_fields__)
        if unknown_stage:
            raise ValueError(
                f"Unknown [run.llm.{name}] keys: {', '.join(sorted(unknown_stage))}"
            )
        stages[name] = replace(getattr(base.llm, name), **values)
    llm = replace(base.llm, model=llmraw.get("model", base.llm.model), **stages)
    return replace(base, data_dir=data_dir, stt=stt, llm=llm, detection=detection)
