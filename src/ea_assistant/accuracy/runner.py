from __future__ import annotations

import hashlib
import logging
import multiprocessing
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..adapters import Audio
from ..application import Application
from ..config import AppConfig
from ..domain import Pipeline, VocabularyPromptMode
from ..models import Segment
from ..prompts import WHISPER_PROMPT_TOKEN_LIMIT, PromptSettings
from ..real_adapters import FfmpegAudio
from .facts import load_facts, score_facts
from .flags import flag_recall
from .metrics import number_accuracy, score_text, vocabulary_accuracy
from .report import _cell, table, write_reports
from .resources import ResourceSampler, SamplingWindow, SystemResourceSampler
from .suite import Suite, load_suite, load_vocabulary, merge_config
from .timestamps import parse_timestamp_prefix

log = logging.getLogger(__name__)
Factory = Callable[[AppConfig, PromptSettings], Application]


def _executor_entry(
    queue: Any, operation: Callable[..., Any], args: tuple[Any, ...]
) -> None:
    try:
        queue.put((True, operation(*args)))
    except Exception as exc:  # noqa: BLE001 - report worker failures across process boundary
        queue.put((False, (type(exc).__name__, str(exc))))


class InProcessExecutor:
    def run(self, operation: Callable[..., Any], *args: Any) -> Any:
        return operation(*args)


class SubprocessExecutor:
    """Spawn each run separately so process-lifetime RSS is isolated."""

    def run(self, operation: Callable[..., Any], *args: Any) -> Any:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        process = context.Process(target=_executor_entry, args=(queue, operation, args))
        process.start()
        succeeded, result = queue.get()
        process.join()
        queue.close()
        if process.exitcode != 0:
            raise RuntimeError(f"accuracy child exited with status {process.exitcode}")
        if not succeeded:
            name, message = result
            raise RuntimeError(f"accuracy child {name}: {message}")
        return result


class TwoStepPipeline:
    uses_stt = True

    def process(
        self,
        app: Application,
        meeting: Any,
        normalized_path: Path,
        reference_segments: list[Segment],
        duration: float,
    ) -> Any:
        return app.process_pre_normalized(meeting, normalized_path)

    def score(
        self,
        reference: str,
        hypothesis: str,
        segments: list[Segment],
        aliases: dict[str, str],
        terms: tuple[Any, ...],
        threshold: float,
        logprob_threshold: float,
        duration: float,
    ) -> dict[str, Any]:
        return {
            "transcript": _serialize(score_text(reference, hypothesis, aliases)),
            "flags": flag_recall(
                reference, segments, threshold, logprob_threshold, duration, aliases
            ),
            "vocabulary": vocabulary_accuracy(reference, hypothesis, terms),
            "numbers": number_accuracy(reference, hypothesis),
        }


class DirectPipeline:
    uses_stt = False

    def process(
        self,
        app: Application,
        meeting: Any,
        normalized_path: Path,
        reference_segments: list[Segment],
        duration: float,
    ) -> Any:
        return app.process_transcript(meeting, reference_segments, duration)

    def score(
        self,
        reference: str,
        hypothesis: str,
        segments: list[Segment],
        aliases: dict[str, str],
        terms: tuple[Any, ...],
        threshold: float,
        logprob_threshold: float,
        duration: float,
    ) -> dict[str, Any]:
        return {
            "transcript": None,
            "flags": None,
            "vocabulary": {"terms": [], "recognized_rate": None},
            "numbers": {"recall": None, "precision": None},
        }


PIPELINES: dict[str, TwoStepPipeline | DirectPipeline] = {
    "two-step": TwoStepPipeline(),
    "direct": DirectPipeline(),
}


def _repo_root() -> Path:
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / ".git").exists():
            return parent
    return source.parents[3]


def _segments(reference: str) -> list[Segment]:
    lines = [line.strip() for line in reference.splitlines() if line.strip()]
    segments = []
    seconds = 0.0
    for i, line in enumerate(lines, 1):
        timestamp = parse_timestamp_prefix(line)
        if timestamp:
            seconds, line = timestamp
        segments.append(Segment(f"s{i:04d}", seconds, seconds + 1.0, line))
        seconds += 1.0
    return segments


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: _serialize(getattr(value, key)) for key in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


_SAFE_FAILURE_MESSAGES = {
    "Ollama returned invalid JSON",
    "Ollama returned an incomplete or empty response",
    "Ollama response exhausted num_ctx and may be truncated",
    "Ollama GPU still has loaded models after cleanup timeout",
    "Unable to verify GPU is free: Ollama /api/ps failed",
}


_TOKEN_BUDGET_MESSAGE = re.compile(
    r"Ollama request exceeds token budget: estimated \d+ input \+ \d+ output exceeds num_ctx \d+"
)


def _safe_message(exc: Exception) -> str | None:
    message = str(exc)
    if message in _SAFE_FAILURE_MESSAGES or _TOKEN_BUDGET_MESSAGE.fullmatch(message):
        return message
    return None


def _safe_failure(exc: Exception, app: Application | None) -> str:
    message = _safe_message(exc)
    if app is not None and app.last_failure_reason:
        reason = app.last_failure_reason
        if message is not None and reason.endswith(": " + type(exc).__name__):
            return f"{reason}: {message}"
        return reason
    if message is not None:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: run failed"


def _table(rows: list[dict[str, Any]]) -> str:
    return table(rows)


def _markdown_detail(detail: dict[str, Any]) -> list[str]:
    if "error" in detail:
        lines = [f"Failed: {detail['error']}"]
        metrics = detail.get("metrics")
        if metrics:
            lines.append(f"Artifacts: `{detail.get('artifacts', '—')}`")
            english = metrics.get("english_transcript")
            if english:
                lines.append(f"English Transcript: `{english}`")
            lines.append("Transcript metrics are included in the run table.")
        return lines
    metrics = detail["metrics"]
    vocabulary = metrics["vocabulary"]
    lines = [f"Status: {detail['status']}", f"Artifacts: `{detail['artifacts']}`"]
    lines.extend(
        [
            "",
            "### Vocabulary terms",
            "",
            "| Canonical term | Reference occurrences | Recognised | Canonical spelling |",
            "|---|---:|---:|---:|",
        ]
    )
    for term in vocabulary.get("terms", []):
        lines.append(
            f"| {term['canonical']} | {term['reference']} | {term['recognized']} | {term['canonical_hits']} |"
        )
    lines.append(
        f"| **Total** | {vocabulary.get('reference', 0)} | {vocabulary.get('recognized', 0)} | {vocabulary.get('canonical', 0)} |"
    )
    numbers = metrics["numbers"]
    lines += [
        "",
        "### Numbers",
        "",
        f"Reference counts: `{numbers.get('reference', {})}`",
        f"Hypothesis counts: `{numbers.get('hypothesis', {})}`",
        f"Hits: {numbers.get('hits', '—')}; recall: {_cell('number_recall', numbers.get('recall'))}; precision: {_cell('number_recall', numbers.get('precision'))}.",
    ]
    flags = metrics.get("flags")
    lines += ["", "### Flagged Passage signals", ""]
    if flags:
        lines += [
            "| Signal | Recall | Precision | Flagged segments | Segments with real errors | Flags per audio hour |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for signal in (
            "repetition_loop",
            "likely_text_over_silence",
            "marker",
            "low_logprob",
            "combined",
        ):
            item = flags[signal]
            lines.append(
                f"| {signal} | {_cell('flag_recall', item['recall'])} | {_cell('flag_recall', item['precision'])} | {item['flagged_segments']} | {item['real_error_segments']} | {_cell('minutes_per_audio_hour', item['flags_per_audio_hour'])} |"
            )
    else:
        lines.append("Not applicable to direct runs.")
    facts = metrics.get("facts")
    lines += ["", "### Critical facts", ""]
    if facts:
        lines += ["| ID | Result | Section found | Expected fact |", "|---|---|---|---|"]
        for fact in facts["results"]:
            statement = fact["statement"].replace("|", "\\|")
            section = fact.get("section") or "—"
            if not fact.get("in_expected_section", False) and section != "—":
                section += " (other section)"
            lines.append(f"| {fact['id']} | {fact['status']} | {section} | {statement} |")
        lines.append(
            f"Found: {len(facts['found'])}/{len(facts['results'])} ({facts['found_percent']}%); wrong: {len(facts['wrong'])}; missing: {len(facts['missing'])}."
        )
        lines.append("Invented candidates for human review:")
        lines.extend(f"- {item}" for item in facts["invented"] or ["None"])
    else:
        lines.append("No critical-facts fixture was configured.")
    resource = detail["resource"]
    lines += [
        "",
        "### Runtime and resources",
        "",
        f"Elapsed: {detail['elapsed_seconds']:.3f} seconds; minutes per audio hour: {_cell('minutes_per_audio_hour', detail.get('minutes_per_audio_hour'))}.",
        f"Peak VRAM: {_cell('peak_vram_mib', resource.get('peak_vram_mib'))} MiB (delta {_cell('peak_vram_mib', resource.get('peak_vram_delta_mib'))} MiB); peak RAM: {_cell('peak_ram_mib', resource.get('peak_ram_mib'))} MiB.",
        f"Vocabulary initial prompt truncated to the Whisper prompt byte limit: {detail.get('vocabulary_prompt_truncated', False)}.",
    ]
    return lines


def _execute_run(
    data: dict[str, Any],
    run: dict[str, Any],
    terms: tuple[Any, ...],
    critical: tuple[Any, ...],
    reference: str,
    base: AppConfig,
    result_dir: Path,
    normalized_path: Path,
    factory: Factory,
    sampler: ResourceSampler | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    app: Application | None = None
    meeting: Any = None
    aliases: dict[str, str] = {}
    row: dict[str, Any] = {
        "name": run["name"],
        "pipeline": run["pipeline"],
        "vocabulary_prompt": run.get("vocabulary_prompt", "off"),
        "status": "ok",
    }
    run_dir = result_dir / run["name"]
    try:
        run_dir.mkdir()
        strategy = PIPELINES[run["pipeline"]]
        prompt_type = run.get("vocabulary_prompt", "off")
        canonical = [term.canonical for term in terms]
        aliases = {alias: term.canonical for term in terms for alias in term.aliases}
        run_with_stt = run
        config = merge_config(base, run_with_stt, run_dir / "data")
        typed_terms = tuple(term for term in terms)
        prompt_settings = PromptSettings(
            pipeline=Pipeline(run["pipeline"]),
            vocabulary=typed_terms,
            vocabulary_prompt=VocabularyPromptMode(prompt_type),
            input_digest=(
                hashlib.sha256(reference.encode("utf-8")).hexdigest()
                if run["pipeline"] == "direct"
                else None
            ),
        )
        prompt_truncated = bool(
            typed_terms
            and prompt_type == "initial_prompt"
            and len(("Vocabulary: " + ", ".join(canonical)).encode("utf-8"))
            > WHISPER_PROMPT_TOKEN_LIMIT
        )
        app = factory(config, prompt_settings)
        row.update(
            {
                "stt_model": config.stt.model if strategy.uses_stt else "—",
                "compute_type": config.stt.compute_type if strategy.uses_stt else "—",
                "language": config.stt.language if strategy.uses_stt else "—",
                "vad": ("on" if config.stt.vad_filter else "off")
                if strategy.uses_stt
                else "—",
                "llm_model": config.llm.model,
                "vocabulary_prompt_truncated": prompt_truncated,
            }
        )
        with SamplingWindow(sampler or SystemResourceSampler()) as window:
            started = time.perf_counter()
            meeting = app.create_meeting(
                str(normalized_path),
                data["name"],
                datetime.now().astimezone().date(),
            )
            duration = meeting.duration
            reference_segments = _segments(reference)
            result = strategy.process(
                app, meeting, normalized_path, reference_segments, duration
            )
            if strategy.uses_stt:
                effective = result.provenance.get("transcript")
                if effective and effective.compute_type:
                    row["compute_type"] = effective.compute_type
                if config.stt.language == "auto" and result.effective_language:
                    probability = result.language_probability
                    row["language"] = (
                        f"auto→{result.effective_language} ({probability:.2f})"
                        if probability is not None
                        else f"auto→{result.effective_language}"
                    )
                elif result.effective_language:
                    row["language"] = result.effective_language
            elapsed = time.perf_counter() - started
        resources = window.result()
        run_segments = result.segments
        transcript_hypothesis = "\n".join(segment.text for segment in run_segments)
        report_metrics = strategy.score(
            reference,
            transcript_hypothesis,
            run_segments,
            aliases,
            terms,
            float(data.get("flag_cer_threshold", 0.25)),
            float(data.get("flag_logprob_threshold", -0.8)),
            duration,
        )
        if report_metrics["transcript"] is not None:
            row.update(
                {
                    "wer_raw": report_metrics["transcript"]["wer_raw"]["rate"],
                    "wer_norm": report_metrics["transcript"]["wer_normalized"]["rate"],
                    "cer_raw": report_metrics["transcript"]["cer_raw"]["rate"],
                    "cer_norm": report_metrics["transcript"]["cer_normalized"]["rate"],
                    "flag_recall": report_metrics["flags"]["combined"]["recall"],
                    "real_error_segments": report_metrics["flags"]["combined"][
                        "real_error_segments"
                    ],
                    "flagged_segments": report_metrics["flags"]["combined"][
                        "flagged_segments"
                    ],
                }
            )
        else:
            row.update(
                {
                    key: "—"
                    for key in (
                        "wer_raw",
                        "wer_norm",
                        "cer_raw",
                        "cer_norm",
                        "flag_recall",
                        "real_error_segments",
                        "flagged_segments",
                    )
                }
            )
        report_metrics["facts"] = (
            score_facts(result.mom.markdown, critical) if critical else None
        )
        if report_metrics["facts"]:
            f = report_metrics["facts"]
            row.update(
                {
                    "mom_found": len(f["found"]),
                    "mom_found_expected": sum(1 for fact in f["results"] if fact["status"] == "found" and fact["in_expected_section"]),
                    "mom_wrong": len(f["wrong"]),
                    "mom_missing": len(f["missing"]),
                    "mom_invented": len(f["invented"]),
                }
            )
        else:
            row.update(
                {
                    key: "—"
                    for key in (
                        "mom_found",
                        "mom_found_expected",
                        "mom_wrong",
                        "mom_missing",
                        "mom_invented",
                    )
                }
            )
        row["vocab_recognized_percent"] = (
            report_metrics["vocabulary"]["recognized_rate"] * 100
            if report_metrics["vocabulary"].get("recognized_rate") is not None
            else None
        )
        row["number_recall"] = report_metrics["numbers"].get("recall")
        row["minutes_per_audio_hour"] = (
            elapsed / 60 / (duration / 3600) if duration else None
        )
        row.update(resources)
        detail = {
            "status": row["status"],
            "metrics": report_metrics,
            "artifacts": str(meeting.folder),
            "effective_config": _serialize(config),
            "resource": resources,
            "elapsed_seconds": elapsed,
            "minutes_per_audio_hour": row["minutes_per_audio_hour"],
            "vocabulary_prompt_truncated": prompt_truncated,
        }
    except Exception as exc:  # noqa: BLE001 - each model/config run must fail independently
        row["status"] = "failed"
        thermal_reason = (
            str(exc) if "Thermal guard stopped processing" in str(exc) else None
        )
        row["failure"] = thermal_reason or _safe_failure(exc, app)
        log.warning("Accuracy run %s failed (%s)", run["name"], type(exc).__name__)
        detail = {"error": row["failure"]}
        transcript_path = meeting.folder / "transcript.md" if meeting is not None else None
        if transcript_path is not None and transcript_path.exists():
            text = transcript_path.read_text(encoding="utf-8")
            hypothesis = "\n".join(
                match.group(1)
                for line in text.splitlines()
                if (match := re.search(r"\*\*[^*]+:\*\* (.*)$", line))
            )
            if hypothesis:
                metrics = _serialize(score_text(reference, hypothesis, aliases))
                segments = app.store.segments(meeting.id) if app is not None else []
                flags = flag_recall(
                    reference, segments,
                    float(data.get("flag_cer_threshold", 0.25)),
                    float(data.get("flag_logprob_threshold", -0.8)),
                    meeting.duration,
                    aliases,
                )
                vocabulary = vocabulary_accuracy(reference, hypothesis, terms)
                numbers = number_accuracy(reference, hypothesis)
                detail["metrics"] = {
                    "transcript": _serialize(metrics), "flags": flags,
                    "vocabulary": vocabulary, "numbers": numbers,
                    "english_transcript": str(meeting.folder / "english-transcript.md")
                    if (meeting.folder / "english-transcript.md").exists() else None,
                }
                for key, metric_key in (("wer_raw", "wer_raw"), ("wer_norm", "wer_normalized"),
                                        ("cer_raw", "cer_raw"), ("cer_norm", "cer_normalized")):
                    row[key] = metrics[metric_key]["rate"]
                row["vocab_recognized_percent"] = vocabulary["recognized_rate"] * 100 if vocabulary.get("recognized_rate") is not None else None
                row["number_recall"] = numbers.get("recall")
                combined = flags["combined"]
                row.update({"flag_recall": combined["recall"],
                            "real_error_segments": combined["real_error_segments"],
                            "flagged_segments": combined["flagged_segments"]})
        if meeting is not None:
            detail["artifacts"] = str(meeting.folder)
    return row, detail


def _validate_paths(data: dict[str, Any], allow_suite_in_repo: bool) -> Path:
    repo = _repo_root()
    suite_path = Path(data["suite_path"]).resolve() if data.get("suite_path") else None
    example_suite = (repo / "docs" / "accuracy-example" / "suite.toml").resolve()
    if (
        suite_path
        and suite_path.is_relative_to(repo)
        and (not allow_suite_in_repo or suite_path != example_suite)
    ):
        raise ValueError(
            "Suite file is inside the repository; --allow-suite-in-repo applies only to docs/accuracy-example/suite.toml"
        )
    output = Path(data["output_dir"]).resolve()
    if output.is_relative_to(repo):
        raise ValueError(
            "Accuracy output directory must resolve outside the repository"
        )
    return output


def _load_fixtures(
    data: dict[str, Any], parsed_suite: Suite | None
) -> tuple[tuple[Any, ...], tuple[Any, ...], str]:
    terms = (
        parsed_suite.vocabulary_fixture.terms
        if parsed_suite and parsed_suite.vocabulary_fixture
        else load_vocabulary(data["vocabulary"]).terms
        if data.get("vocabulary")
        else ()
    )
    facts = load_facts(data["critical_facts"]) if data.get("critical_facts") else ()
    reference = Path(data["reference_transcript"]).read_text(encoding="utf-8")
    return terms, facts, reference


def run_suite(
    suite: Suite | dict[str, Any] | str | Path,
    factory: Factory,
    base_config: AppConfig | None = None,
    sampler: ResourceSampler | None = None,
    only: list[str] | None = None,
    allow_suite_in_repo: bool = False,
    audio_preparer: Audio | None = None,
    executor: InProcessExecutor | SubprocessExecutor | None = None,
) -> dict[str, Any]:
    if isinstance(suite, (str, Path)):
        parsed = load_suite(suite)
        parsed_suite: Suite | None = parsed
        data = parsed.mapping()
    elif isinstance(suite, Suite):
        parsed_suite = suite
        data = parsed_suite.mapping()
    else:
        parsed_suite = None
        data = suite.copy()
    output = _validate_paths(data, allow_suite_in_repo)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result_dir = output / f"{data['name']}-{stamp}"
    work_dir = result_dir / "work"
    if work_dir.resolve().is_relative_to(_repo_root()):
        raise ValueError("Accuracy work directory must resolve outside the repository")
    result_dir.mkdir(parents=True, exist_ok=False)
    work_dir.mkdir(parents=True)
    terms, critical, reference = _load_fixtures(data, parsed_suite)
    run_defs = [run for run in data["run"] if only is None or run["name"] in only]
    if only and len(run_defs) != len(set(only)):
        raise ValueError("--only names must match suite run names")
    base = base_config or AppConfig()
    sampler = sampler or SystemResourceSampler()
    executor = executor or InProcessExecutor()
    normalized_path = work_dir / "normalized.wav"
    excerpt = data.get("excerpt") or {}
    if set(excerpt) - {"start", "end"} or (
        excerpt and set(excerpt) != {"start", "end"}
    ):
        raise ValueError("excerpt requires exactly start and end")
    (audio_preparer or FfmpegAudio()).normalize(
        str(data["recording"]),
        str(normalized_path),
        str(excerpt["start"]) if excerpt else None,
        str(excerpt["end"]) if excerpt else None,
    )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for run in run_defs:
        row, detail = executor.run(
            _execute_run,
            data,
            run,
            terms,
            critical,
            reference,
            base,
            result_dir,
            normalized_path,
            factory,
            sampler,
        )
        rows.append(row)
        details[run["name"]] = detail
    payload = {
        "suite": data["name"],
        "created_at": stamp,
        "report_dir": str(result_dir),
        "runs": rows,
        "details": details,
    }
    lines = [
        f"# Accuracy suite: {data['name']}",
        "",
        _table(rows),
        "",
        "'Invented' lists are candidates for human review, not an automatic judgement.",
    ]
    for name, detail in details.items():
        lines += ["", f"## {name}", "", *_markdown_detail(detail)]
    write_reports(result_dir, payload, "\n".join(lines) + "\n")
    print(_table(rows))
    return payload
