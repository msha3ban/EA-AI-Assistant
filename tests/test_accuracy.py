from __future__ import annotations

import json
import os
import random
import sqlite3
import time
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

from ea_assistant.accuracy.facts import CriticalFact, load_facts, score_facts
from ea_assistant.accuracy.flags import flag_recall
from ea_assistant.accuracy.metrics import (
    edit_score,
    normalize,
    number_accuracy,
    vocabulary_accuracy,
)
from ea_assistant.accuracy.report import table, write_reports
from ea_assistant.accuracy.resources import ResourceReading
from ea_assistant.accuracy.runner import (
    InProcessExecutor,
    SubprocessExecutor,
    run_suite,
)
from ea_assistant.accuracy.suite import Suite, SuiteRun, load_suite, merge_config
from ea_assistant.adapters import RetryableLLMResponseError
from ea_assistant.application import Application
from ea_assistant.config import AppConfig
from ea_assistant.domain import FACT_SECTIONS, VocabularyPromptMode
from ea_assistant.models import Segment
from ea_assistant.prompts import PromptSettings
from ea_assistant.testing import FakeAudio, FakeLLM, FakeSensors, FakeSpeechToText


def test_wer_cer_known_pairs_and_empty_reference() -> None:
    assert edit_score(["a", "b"], ["a", "b"]).rate == 0
    substitution = edit_score(["a", "b"], ["a", "c"])
    assert (
        substitution.substitutions,
        substitution.deletions,
        substitution.insertions,
    ) == (1, 0, 0)
    assert edit_score(["a"], ["a", "b"]).insertions == 1
    assert edit_score(["a", "b"], ["a"]).deletions == 1
    assert edit_score([], []).rate == 0
    assert edit_score([], ["x"]).rate is None


def _simple_edit_counts(
    reference: list[str], hypothesis: list[str]
) -> tuple[int, int, int, int]:
    table: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hypothesis) + 1)]
        for _ in range(len(reference) + 1)
    ]
    for j in range(1, len(hypothesis) + 1):
        table[0][j] = (j, 0, 0, j)
    for i in range(1, len(reference) + 1):
        table[i][0] = (i, 0, i, 0)
    for i, ref in enumerate(reference, 1):
        for j, hyp in enumerate(hypothesis, 1):
            prior = table[i - 1][j - 1]
            candidates = [
                (prior[0] + (ref != hyp), prior[1] + (ref != hyp), prior[2], prior[3]),
                (
                    table[i - 1][j][0] + 1,
                    table[i - 1][j][1],
                    table[i - 1][j][2] + 1,
                    table[i - 1][j][3],
                ),
                (
                    table[i][j - 1][0] + 1,
                    table[i][j - 1][1],
                    table[i][j - 1][2],
                    table[i][j - 1][3] + 1,
                ),
            ]
            table[i][j] = min(candidates, key=lambda value: value[0])
    return table[-1][-1]


def test_edit_score_matches_reference_and_scales_to_5k() -> None:
    for seed in range(5):
        generator = random.Random(seed)
        for _ in range(5):
            reference = [generator.choice("abc") for _ in range(16)]
            hypothesis = [generator.choice("abc") for _ in range(16)]
            expected = _simple_edit_counts(reference, hypothesis)
            actual = edit_score(reference, hypothesis)
            assert (
                actual.errors,
                actual.substitutions,
                actual.deletions,
                actual.insertions,
            ) == expected
    started = time.perf_counter()
    large = edit_score(["a"] * 5000, ["b"] * 5000)
    assert time.perf_counter() - started < 5
    assert (large.errors, large.substitutions, large.deletions, large.insertions) == (
        5000,
        5000,
        0,
        0,
    )


def test_normalization_rules_and_aliases() -> None:
    assert normalize("[00:12] أإآٱ ى ة ؤ ئ ـَ") == "اااا ي ه و ي"
    assert normalize("١٢۳٤") == "1234"
    assert normalize("HELLO، WORLD! (X)") == "hello world x"
    assert normalize("S4 HANA", {"S4 HANA": "SAP S/4HANA"}) == "sap s4hana"
    assert normalize("SAP S/4HANA") == "sap s4hana"


def test_vocabulary_recognized_and_canonical_counts() -> None:
    terms = [{"canonical": "SAP", "aliases": ["S A P"]}]
    result = vocabulary_accuracy("SAP and SAP", "S A P and SAP", terms)
    assert result["recognized"] == 2
    assert result["canonical"] == 1
    assert result["recognized_rate"] == 1
    lower = vocabulary_accuracy("SAP", "sap", [{"canonical": "SAP", "aliases": []}])
    assert lower["recognized"] == 1 and lower["canonical"] == 0


def test_vocabulary_occurrences_are_boundary_aware_and_non_overlapping() -> None:
    overlap = vocabulary_accuracy(
        "SAP S/4HANA S4HANA",
        "SAP S/4HANA S4HANA",
        [{"canonical": "SAP S/4HANA", "aliases": ["S4HANA"]}],
    )
    assert overlap["reference"] == overlap["recognized"] == 2
    short = vocabulary_accuracy(
        "SAPIENS SAP", "SAPIENS SAP", [{"canonical": "SAP", "aliases": []}]
    )
    assert short["reference"] == 1


def test_number_accuracy_digits_and_words() -> None:
    result = number_accuracy("١,٢٣٤ 2.5 اتنين تلاتين مية ألف", "1234 2.5 2 30 100 1000")
    assert result["recall"] == 1
    assert result["precision"] == 1
    dialect_numbers = number_accuracy(
        "ألفين ميتين تلتمية ربعمية خمسمية ستمية سبعمية تمنمية تسعمية تلات تمانية ميه",
        "2000 200 300 400 500 600 700 800 900 3 8 100",
    )
    assert dialect_numbers["recall"] == dialect_numbers["precision"] == 1


def test_flag_recall_per_signal_exact() -> None:
    segments = [
        Segment("a", 0, 1, "correct"),
        Segment("b", 1, 2, "wrong", -1, flags=["repetition loop"]),
        Segment("c", 2, 3, "also wrong", -1),
    ]
    result = flag_recall("correct right right", segments, threshold=0.25, duration=3600)
    assert result["marker"]["recall"] == 0.5
    assert result["low_logprob"]["recall"] == 1
    assert result["combined"]["precision"] == 1
    assert result["combined"]["flags_per_audio_hour"] == 2
    assert result["combined"]["real_error_segments"] == 2
    assert result["combined"]["flagged_segments"] == 2


def test_flag_recall_normalizes_vocabulary_aliases() -> None:
    result = flag_recall(
        "checkout",
        [Segment("s1", 0, 1, "تشيك اوت")],
        aliases={"تشيك اوت": "checkout"},
    )
    assert result["real_errors"] == 0
    assert result["segments"][0]["real_error"] is False


def test_report_uses_na_for_undefined_flag_metrics() -> None:
    result = flag_recall("same", [Segment("s1", 0, 1, "same")])
    assert result["combined"]["recall"] is None
    assert result["combined"]["precision"] is None
    rendered = table([{"name": "r", "flag_recall": result["combined"]["recall"]}])
    assert "n/a" in rendered and "None" not in rendered


def test_report_formats_accuracy_percentages_and_resource_units() -> None:
    rendered = table(
        [
            {
                "name": "r",
                "wer_raw": 0.01234,
                "vocab_recognized_percent": 91.26,
                "number_recall": 0.5,
                "minutes_per_audio_hour": 2.34,
                "peak_vram_mib": 100.7,
                "peak_ram_mib": 200.2,
            }
        ]
    )
    assert "1.2%" in rendered and "91.3%" in rendered and "50.0%" in rendered
    assert "2.3" in rendered and "101" in rendered and "200" in rendered


def test_report_files_are_written_atomically(tmp_path: Path) -> None:
    write_reports(tmp_path, {"suite": "demo"}, "# demo\n")
    assert json.loads((tmp_path / "report.json").read_text()) == {"suite": "demo"}
    assert (tmp_path / "report.md").read_text() == "# demo\n"
    assert not (tmp_path / "report.json.tmp").exists()
    assert not (tmp_path / "report.md.tmp").exists()


def test_critical_fact_results_and_invented_candidates() -> None:
    mom = "## Decisions\n- D1: Use RabbitMQ for integration\n- D2: Add an unrelated cache\n## Action Items\n| # | Item | Owner | Due date | Evidence |\n|---|---|---|---|---|\n| 1 | send file | Ahmed | 2026-10-15 | s1 |\n## Technical Details\n- **Kafka**: integration uses Kafka"
    facts = [
        {
            "id": "F1",
            "kind": "decision",
            "statement": "Use Kafka for integration",
            "expect": ["kafka", "integration"],
            "anchor": ["integration"],
            "wrong_if": ["rabbitmq"],
        },
        {
            "id": "F2",
            "kind": "action_item",
            "statement": "Send the file",
            "expect": ["file"],
            "anchor": ["file"],
            "owner": "Ahmed",
            "due": "2026-10-15",
        },
        {
            "id": "F3",
            "kind": "technical_detail",
            "statement": "Kafka integration",
            "expect": ["kafka", "integration"],
        },
        {"id": "F4", "kind": "date", "statement": "Date 2040", "expect": ["2040"]},
    ]
    result = score_facts(mom, facts)
    assert result["wrong"] == ["F1"]
    assert result["found"] == ["F2", "F3"]
    assert result["missing"] == ["F4"]
    assert "- D2: Add an unrelated cache" in result["invented"]


def test_critical_facts_match_other_mom_sections_and_report_location() -> None:
    result = score_facts(
        "## Purpose / Agenda\n- Use Kafka for integration\n## Decisions\n- D1: Add cache",
        [{"id": "F1", "kind": "decision", "statement": "Use Kafka",
          "expect": ["kafka", "integration"], "anchor": ["integration"]}],
    )
    assert result["found"] == ["F1"]
    assert result["results"][0]["section"] == "Purpose / Agenda"
    assert result["results"][0]["in_expected_section"] is False
    assert "- Use Kafka for integration" not in result["invented"]


def test_wrong_if_matches_summary_even_when_expected_section_has_correct_match() -> None:
    result = score_facts(
        "Date: 2026-10-01\nState: Draft\n## Summary\nRabbitMQ replaces Kafka.\n"
        "## Decisions\n| # | Item | Owner | Due | Evidence |\n|---|---|---|---|---|\n"
        "- D1: Use Kafka for integration",
        [{"id": "F1", "kind": "decision", "statement": "Use Kafka",
          "expect": ["kafka"], "anchor": ["kafka"], "wrong_if": ["rabbitmq"]}],
    )
    assert result["wrong"] == ["F1"]
    assert result["results"][0]["section"] == "Summary"


def test_critical_facts_load_as_frozen_typed_values(tmp_path: Path) -> None:
    path = tmp_path / "facts.toml"
    path.write_text(
        '[[fact]]\nid="F1"\nkind="decision"\nstatement="Use Kafka"\nexpect=["kafka"]\n',
        encoding="utf-8",
    )
    fact = load_facts(path)[0]
    assert isinstance(fact, CriticalFact) and fact.kind.value == "decision"


def test_action_owner_due_mismatch_is_wrong_and_number_wrong_if_is_local() -> None:
    mom = (
        "## Action Items\n"
        "| # | Item | Owner | Due date | Evidence |\n"
        "|---|---|---|---|---|\n"
        "| 1 | send report | Sara | 2026-10-10 | s1 |\n"
        "## Discussion Points\n"
        "The budget is 10, not 20. The network is Kafka."
    )
    facts = [
        {
            "id": "A1",
            "kind": "action_item",
            "statement": "send report",
            "expect": ["report"],
            "anchor": ["report"],
            "owner": "Ahmed",
            "due": "2026-10-15",
        },
        {
            "id": "N1",
            "kind": "number",
            "statement": "budget is 20",
            "expect": ["20"],
            "anchor": ["budget"],
            "wrong_if": ["10"],
        },
    ]
    result = score_facts(mom, facts)
    statuses = {fact["id"]: fact for fact in result["results"]}
    assert statuses["A1"]["status"] == "wrong"
    assert statuses["A1"]["reason"] == "owner,due"
    assert statuses["N1"]["status"] == "wrong"
    assert statuses["N1"]["reason"] == "wrong_if"


def test_number_wrong_if_outside_anchor_sentence_does_not_mark_fact_wrong() -> None:
    result = score_facts(
        "## Discussion Points\nBudget is 20. A separate proposal mentions 10.",
        [
            {
                "id": "N1",
                "kind": "number",
                "statement": "Budget is 20",
                "expect": ["20"],
                "anchor": ["budget"],
                "wrong_if": ["10"],
            }
        ],
    )
    assert result["found"] == ["N1"] and not result["wrong"]


def test_suite_relative_paths_merge_and_unknown_rejected(tmp_path: Path) -> None:
    (tmp_path / "recording.wav").write_bytes(b"wav")
    (tmp_path / "ref.txt").write_text("x", encoding="utf-8")
    path = tmp_path / "suite.toml"
    path.write_text(
        'name="s"\nrecording="recording.wav"\nreference_transcript="ref.txt"\n[[run]]\nname="r"\npipeline="two-step"\n[run.stt]\nlanguage="auto"\n[run.llm.extract]\nnum_predict=123\ntimeout=17\n',
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert isinstance(suite, Suite) and isinstance(suite.run[0], SuiteRun)
    assert suite.run[0].vocabulary_prompt is VocabularyPromptMode.OFF
    assert suite["recording"] == str((tmp_path / "recording.wav").resolve())
    cfg = merge_config(AppConfig(), suite["run"][0], tmp_path / "data")
    assert cfg.stt.language == "auto" and cfg.llm.extract.num_predict == 123
    assert cfg.llm.extract.timeout == 17
    suite_path = tmp_path / "bad.toml"
    suite_path.write_text(
        'name="x"\nrecording="a"\nreference_transcript="b"\nfoo=1\n[[run]]\nname="x"\npipeline="direct"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown suite keys"):
        load_suite(suite_path)
    with pytest.raises(ValueError, match=r"Unknown \[run.stt\]"):
        merge_config(AppConfig(), {"stt": {"bad": 1}}, tmp_path)


class FakeSampler:
    def __init__(self) -> None:
        self.i = 0

    def sample(self) -> ResourceReading:
        self.i += 1
        return ResourceReading(100 + self.i, 1000 - self.i, 200 + self.i)


def _process_id() -> int:
    return os.getpid()


def test_spawn_executor_isolates_a_runs_process_lifetime() -> None:
    assert SubprocessExecutor().run(_process_id) != os.getpid()
    assert isinstance(InProcessExecutor(), InProcessExecutor)


def test_cli_uses_spawn_executor_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ea_assistant import cli
    from ea_assistant.accuracy import runner

    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"wav")
    reference = tmp_path / "reference.txt"
    reference.write_text("example", encoding="utf-8")
    suite = tmp_path / "suite.toml"
    suite.write_text(
        f'name="smoke"\nrecording="{recording}"\nreference_transcript="{reference}"\n'
        f'output_dir="{tmp_path / "out"}"\n[[run]]\nname="r"\npipeline="direct"\n',
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    def fake_run_suite(*args: Any, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run_suite", fake_run_suite)
    assert cli.main(["accuracy", str(suite)]) == 0
    assert isinstance(seen["executor"], SubprocessExecutor)


def _make_app(
    config: AppConfig,
    prompt_settings: PromptSettings,
    stt_obj: FakeSpeechToText | None = None,
    llm_obj: FakeLLM | None = None,
) -> Application:
    return Application(
        FakeAudio(),
        stt_obj
        or FakeSpeechToText(
            [Segment("s0001", 0, 1, "SAP S/4HANA and 123", -1.0, 0.1, 0)]
        ),
        llm_obj or FakeLLM(),
        FakeSensors(),
        config,
        prompt_settings=prompt_settings,
    )


def test_two_step_run_all_metrics_vocab_prompts_and_provenance(tmp_path: Path) -> None:
    recording = tmp_path / "demo.wav"
    recording.write_bytes(b"fake")
    reference = tmp_path / "reference.txt"
    reference.write_text("[00:00] SAP S/4HANA and 123", encoding="utf-8")
    vocab = tmp_path / "vocab.toml"
    vocab.write_text(
        '[[term]]\ncanonical="SAP S/4HANA"\naliases=["S4 HANA"]\n', encoding="utf-8"
    )
    suite = {
        "name": "two",
        "recording": str(recording),
        "excerpt": {"start": "00:01", "end": "00:02"},
        "reference_transcript": str(reference),
        "vocabulary": str(vocab),
        "output_dir": str(tmp_path / "out"),
        "run": [
            {
                "name": "r",
                "pipeline": "two-step",
                "vocabulary_prompt": "initial_prompt",
                "stt": {"language": "auto"},
            },
            {"name": "hot", "pipeline": "two-step", "vocabulary_prompt": "hotwords"},
        ],
    }
    apps: list[Application] = []
    stt = FakeSpeechToText([Segment("s0001", 0, 1, "SAP S/4HANA and 123", -1, 0, 0)])
    llm = FakeLLM()

    def factory(config: AppConfig, prompt_settings: PromptSettings) -> Application:
        app = _make_app(config, prompt_settings, stt_obj=stt, llm_obj=llm)
        apps.append(app)
        return app

    preparer = FakeAudio()
    report = run_suite(suite, factory, sampler=FakeSampler(), audio_preparer=preparer)
    row = report["runs"][0]
    assert row["status"] == "ok" and row["wer_raw"] == 0
    assert row["language"] == "auto→ar (0.97)"
    assert (
        stt.configs[0].initial_prompt and "SAP S/4HANA" in stt.configs[0].initial_prompt
    )
    assert stt.configs[1].hotwords == "SAP S/4HANA"
    assert len(preparer.normalized) == 1
    assert preparer.normalized[0][2:] == ("00:01", "00:02")
    assert any(
        "Spell these Vocabulary terms exactly" in prompt for prompt in llm.instructions
    )
    assert (
        row["minutes_per_audio_hour"] is not None
        and row["peak_vram_delta_mib"] is not None
        and row["peak_ram_mib"] is not None
    )
    assert "numbers" in report["details"]["r"]["metrics"]
    assert "Canonical term" in Path(report["report_dir"], "report.md").read_text()
    assert {
        "wer_raw",
        "wer_norm",
        "cer_raw",
        "cer_norm",
        "minutes_per_audio_hour",
        "peak_vram_mib",
        "peak_ram_mib",
        "status",
    } <= row.keys()
    db = Path(report["details"]["r"]["artifacts"]).parents[1] / "ea.sqlite"
    with sqlite3.connect(db) as connection:
        provenance = json.loads(
            connection.execute(
                "SELECT provenance FROM artefacts WHERE name='transcript'"
            ).fetchone()[0]
        )
    assert provenance["vocabulary"]


def test_direct_skips_stt_and_wer_flags_are_not_applicable(tmp_path: Path) -> None:
    recording = tmp_path / "demo.wav"
    recording.write_bytes(b"fake")
    reference = tmp_path / "reference.txt"
    reference.write_text("هنستخدم Kafka", encoding="utf-8")
    vocab = tmp_path / "vocab.toml"
    vocab.write_text(
        '[[term]]\ncanonical="Kafka"\naliases=["كافكا"]\n', encoding="utf-8"
    )
    stt = FakeSpeechToText()
    llm = FakeLLM()
    suite = {
        "name": "direct",
        "recording": str(recording),
        "reference_transcript": str(reference),
        "vocabulary": str(vocab),
        "output_dir": str(tmp_path / "out"),
        "run": [{"name": "d", "pipeline": "direct"}],
    }
    report = run_suite(
        suite,
        lambda cfg, prompt_settings: _make_app(cfg, prompt_settings, stt, llm),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    row = report["runs"][0]
    assert stt.calls == 0 and row["wer_raw"] == "—" and row["flag_recall"] == "—"
    assert any("Kafka" in prompt for prompt in llm.instructions)
    assert Path(report["details"]["d"]["artifacts"], "mom.md").exists()


def test_safety_refuses_repository_paths(tmp_path: Path) -> None:
    suite = {
        "name": "x",
        "recording": "/tmp/a",
        "reference_transcript": "/tmp/b",
        "output_dir": str(Path(__file__).parents[1]),
        "run": [{"name": "r", "pipeline": "direct"}],
    }
    with pytest.raises(ValueError, match="outside the repository"):
        run_suite(suite, lambda *_: cast(Application, None))
    inside = Path(__file__).parents[1] / "example-suite.toml"
    inside.write_text(
        'name="x"\nrecording="/tmp/a"\nreference_transcript="/tmp/b"\noutput_dir="/tmp/out"\n[[run]]\nname="r"\npipeline="direct"\n',
        encoding="utf-8",
    )
    try:
        with pytest.raises(ValueError, match="Suite file is inside"):
            run_suite(inside, lambda *_: cast(Application, None))
    finally:
        inside.unlink()


def test_default_pytest_excludes_accuracy_marker() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert any(
        marker.startswith("accuracy:")
        for marker in config["tool"]["pytest"]["ini_options"]["markers"]
    )
    assert "not accuracy" in config["tool"]["pytest"]["ini_options"]["addopts"]


def test_two_runs_in_table_failure_does_not_stop_later_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recording = tmp_path / "a.wav"
    recording.write_bytes(b"a")
    ref = tmp_path / "r.txt"
    ref.write_text("hi", encoding="utf-8")
    suite = {
        "name": "multi",
        "recording": str(recording),
        "reference_transcript": str(ref),
        "output_dir": str(tmp_path / "out"),
        "run": [
            {"name": "bad", "pipeline": "two-step", "stt": {"compute_type": "invalid"}},
            {"name": "good", "pipeline": "direct"},
        ],
    }
    report = run_suite(
        suite,
        lambda cfg, prompt_settings: _make_app(cfg, prompt_settings),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    assert [r["status"] for r in report["runs"]] == ["failed", "ok"]
    table = capsys.readouterr().out
    assert "| bad |" in table and "| good |" in table
    assert (Path(report["report_dir"]) / "report.json").exists()


def test_thermal_guard_failure_is_reported_and_next_run_executes(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "a.wav"
    recording.write_bytes(b"a")
    ref = tmp_path / "r.txt"
    ref.write_text("hi", encoding="utf-8")
    suite = {
        "name": "thermal",
        "recording": str(recording),
        "reference_transcript": str(ref),
        "output_dir": str(tmp_path / "out"),
        "run": [
            {"name": "hot", "pipeline": "two-step"},
            {"name": "cool", "pipeline": "direct"},
        ],
    }
    count = 0

    def factory(config: AppConfig, prompt_settings: PromptSettings) -> Application:
        nonlocal count
        count += 1
        sensors = FakeSensors(temp=100 if count == 1 else 40)
        return Application(
            FakeAudio(),
            FakeSpeechToText(),
            FakeLLM(),
            sensors,
            config,
            prompt_settings=prompt_settings,
        )

    report = run_suite(
        suite, factory, sampler=FakeSampler(), audio_preparer=FakeAudio()
    )
    assert report["runs"][0]["status"] == "failed"
    assert "Thermal guard stopped processing" in report["runs"][0]["failure"]
    assert report["runs"][1]["status"] == "ok"


def test_logs_never_include_reference_or_fact_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "SENSITIVE-DO-NOT-LOG-7931"
    recording = tmp_path / "a.wav"
    recording.write_bytes(b"a")
    ref = tmp_path / "r.txt"
    ref.write_text(marker, encoding="utf-8")
    facts = tmp_path / "facts.toml"
    facts.write_text(
        f'[[fact]]\nid="F1"\nkind="decision"\nstatement="{marker}"\nexpect=["{marker}"]\n',
        encoding="utf-8",
    )
    suite = {
        "name": "private",
        "recording": str(recording),
        "reference_transcript": str(ref),
        "critical_facts": str(facts),
        "output_dir": str(tmp_path / "out"),
        "run": [
            {"name": "bad", "pipeline": "two-step", "stt": {"compute_type": "invalid"}}
        ],
    }
    run_suite(
        suite,
        lambda cfg, prompt_settings: _make_app(cfg, prompt_settings),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    assert marker not in caplog.text


def test_initial_prompt_is_byte_bounded_and_truncation_is_reported(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "a.wav"
    recording.write_bytes(b"a")
    reference = tmp_path / "r.txt"
    reference.write_text("short reference", encoding="utf-8")
    vocabulary = tmp_path / "vocab.toml"
    vocabulary.write_text(
        '[[term]]\ncanonical="' + ("SAP" * 100) + '"\naliases=[]\n',
        encoding="utf-8",
    )
    suite = {
        "name": "bounded",
        "recording": str(recording),
        "reference_transcript": str(reference),
        "vocabulary": str(vocabulary),
        "output_dir": str(tmp_path / "out"),
        "run": [
            {
                "name": "r",
                "pipeline": "two-step",
                "vocabulary_prompt": "initial_prompt",
            }
        ],
    }
    stt = FakeSpeechToText()
    report = run_suite(
        suite,
        lambda config, prompt_settings: _make_app(config, prompt_settings, stt),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    assert stt.configs[0].initial_prompt is not None
    assert len(stt.configs[0].initial_prompt.encode("utf-8")) <= 224
    assert report["runs"][0]["vocabulary_prompt_truncated"] is True


def test_actual_vocabulary_prompt_budget_splits_long_extract_into_chunks(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = tmp_path / "reference.txt"
    reference.write_text("reference", encoding="utf-8")
    vocabulary = tmp_path / "vocabulary.toml"
    vocabulary.write_text(
        '[[term]]\ncanonical="' + ("LongVocabulary" * 240) + '"\naliases=[]\n',
        encoding="utf-8",
    )
    segments = [Segment(f"s{i:04d}", i, i + 1, "x" * 1400) for i in range(5)]

    def respond(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Translate"):
            return json.dumps(content)
        if instructions.startswith("Extract"):
            return json.dumps({section.value: [] for section in FACT_SECTIONS})
        return "summary"

    llm = FakeLLM(responder=respond)
    stt = FakeSpeechToText(segments)
    suite = {
        "name": "budget",
        "recording": str(recording),
        "reference_transcript": str(reference),
        "vocabulary": str(vocabulary),
        "output_dir": str(tmp_path / "out"),
        "run": [{"name": "r", "pipeline": "two-step", "vocabulary_prompt": "hotwords"}],
    }
    result = run_suite(
        suite,
        lambda config, settings: _make_app(config, settings, stt_obj=stt, llm_obj=llm),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    assert result["runs"][0]["status"] == "ok"
    assert llm.extract_calls > 1


def test_timeout_failure_names_stage_type_and_configured_duration(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = tmp_path / "reference.txt"
    reference.write_text("reference", encoding="utf-8")

    def timeout(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Extract"):
            raise TimeoutError("private transport detail")
        return "{}"

    report = run_suite(
        {
            "name": "timeouts",
            "recording": str(recording),
            "reference_transcript": str(reference),
            "output_dir": str(tmp_path / "out"),
            "run": [{"name": "r", "pipeline": "direct"}],
        },
        lambda config, settings: _make_app(
            config, settings, llm_obj=FakeLLM(responder=timeout)
        ),
        sampler=FakeSampler(),
        audio_preparer=FakeAudio(),
    )
    assert report["runs"][0]["failure"] == "extract: TimeoutError after 600 s"
    assert report["runs"][0]["wer_raw"] == "—"
    assert "private transport detail" not in str(report)


def test_failed_extract_keeps_transcript_metrics_and_safe_ollama_message(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = tmp_path / "reference.txt"
    reference.write_text("SAP S/4HANA and 123", encoding="utf-8")

    def fail_extract(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Extract"):
            raise RetryableLLMResponseError("Ollama returned invalid JSON")
        if instructions.startswith("Translate"):
            return json.dumps(content)
        return "{}"

    report = run_suite(
        {"name": "failed", "recording": str(recording),
         "reference_transcript": str(reference), "output_dir": str(tmp_path / "out"),
         "run": [{"name": "r", "pipeline": "two-step"}]},
        lambda config, settings: _make_app(
            config, settings,
            stt_obj=FakeSpeechToText([Segment("s1", 0, 1, "SAP S/4HANA and 123")]),
            llm_obj=FakeLLM(responder=fail_extract),
        ), sampler=FakeSampler(), audio_preparer=FakeAudio(),
    )
    row = report["runs"][0]
    assert row["status"] == "failed"
    assert row["wer_raw"] == 0
    assert row["cer_raw"] == 0
    assert row["failure"] == "extract: RetryableLLMResponseError: Ollama returned invalid JSON"
    assert report["details"]["r"]["metrics"]["english_transcript"] is not None
    markdown = (Path(report["report_dir"]) / "report.md").read_text(encoding="utf-8")
    assert "### Vocabulary terms" in markdown
    assert "### Numbers" in markdown
    assert "### Flagged Passage signals" in markdown
    assert "SAP S/4HANA and 123" not in str(report)


def test_summary_token_budget_failure_keeps_numeric_diagnostic(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = tmp_path / "reference.txt"
    reference.write_text("SAP S/4HANA and 123", encoding="utf-8")
    budget = (
        "Ollama request exceeds token budget: "
        "estimated 9000 input + 2048 output exceeds num_ctx 8192"
    )

    def fail_summary(instructions: str, content: dict[str, str]) -> str:
        if instructions.startswith("Write"):
            raise ValueError(budget)
        if instructions.startswith("Extract"):
            return json.dumps({section.value: [] for section in FACT_SECTIONS})
        return json.dumps(content)

    report = run_suite(
        {"name": "failed", "recording": str(recording),
         "reference_transcript": str(reference), "output_dir": str(tmp_path / "out"),
         "run": [{"name": "r", "pipeline": "two-step"}]},
        lambda config, settings: _make_app(
            config, settings,
            stt_obj=FakeSpeechToText([Segment("s1", 0, 1, "SAP S/4HANA and 123")]),
            llm_obj=FakeLLM(responder=fail_summary),
        ), sampler=FakeSampler(), audio_preparer=FakeAudio(),
    )
    assert report["runs"][0]["failure"] == f"summary: ValueError: {budget}"


def test_failed_speech_to_text_reports_no_transcript_metrics(tmp_path: Path) -> None:
    recording = tmp_path / "recording.wav"
    recording.write_bytes(b"audio")
    reference = tmp_path / "reference.txt"
    reference.write_text("SAP S/4HANA and 123", encoding="utf-8")

    class FailingSpeechToText(FakeSpeechToText):
        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("model failed")

    report = run_suite(
        {"name": "failed", "recording": str(recording),
         "reference_transcript": str(reference), "output_dir": str(tmp_path / "out"),
         "run": [{"name": "r", "pipeline": "two-step"}]},
        lambda config, settings: _make_app(
            config, settings, stt_obj=FailingSpeechToText([]),
        ), sampler=FakeSampler(), audio_preparer=FakeAudio(),
    )
    row = report["runs"][0]
    assert row["status"] == "failed"
    assert row.get("wer_raw", "—") == "—"
    assert "metrics" not in report["details"]["r"]
