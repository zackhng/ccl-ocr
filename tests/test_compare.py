"""Comparison report and saved-run round trips.

Runs are built by hand so every expected verdict follows from the numbers given.
"""

from __future__ import annotations

import json

import pytest

from ocrbench import cli, compare
from ocrbench.metrics import CharMetrics, RegionMetrics, SampleMetrics, TextMetrics, WordMetrics
from ocrbench.runner import RunResult, aggregate_regions, aggregate_words, load, save


def sm(sid: str, total: float, capture: str = "scan", found: int = 9, n_gt: int = 10,
       text: TextMetrics | None = None, words: WordMetrics | None = None) -> SampleMetrics:
    return SampleMetrics(
        sample_id=sid, source="synth", capture=capture, script="latin", dpi=300,
        regions=RegionMetrics(n_gt=n_gt, n_pred=n_gt, found=found, coverage_sum=found * 0.9),
        text=text, words=words, timings_ms={"total": total},
    )


def run(engine: str, samples: list[SampleMetrics], **env) -> RunResult:
    return RunResult(
        run_id=f"run-{engine}", engine=engine, samples=samples,
        environment={"platform": "test", "processor": "cpu", "cv2_threads": 8, **env},
        run_config={"repeats": 5, "warmup": 1, "note": ""},
    )


def by_name(checks):
    return {(c.slice, c.name.split(" (")[0]): c for c in checks}


class TestRoundTrip:
    def test_saved_run_reloads_with_identical_aggregates(self, tmp_path):
        original = run("ccl", [
            sm("a", 10.0, words=WordMetrics(n_gt=4, n_surviving=5, hit=3, coverage_sum=2.7)),
            sm("b", 20.0, capture="photo", found=3,
               text=TextMetrics(n_gt_chars=50, edits=4, inserted_chars=1, n_gt_words=10,
                                n_pred_words=9, matched_words=8)),
        ])
        original.samples[0].chars = CharMetrics(n_gt=10, n_surviving=11, isolated=7, merged=2, missed=1)
        loaded = load(save(original, tmp_path, subdir="ccl"))

        assert [s.sample_id for s in loaded.samples] == ["a", "b"]
        assert loaded.samples[0].chars.isolation_recall == pytest.approx(0.7)
        assert loaded.samples[1].text.cer == pytest.approx(5 / 50)
        assert aggregate_regions(s.regions for s in loaded.samples).line_recall == \
            pytest.approx(aggregate_regions(s.regions for s in original.samples).line_recall)
        # coverage_sum is persisted, so mean coverage survives exactly.
        assert aggregate_words(s.words for s in loaded.samples).mean_coverage == pytest.approx(2.7 / 4)

    def test_load_accepts_file_or_folder(self, tmp_path):
        folder = save(run("paddle", [sm("a", 5.0)]), tmp_path, subdir="paddle")
        assert load(folder).engine == load(folder / "results.json").engine == "paddle"

    def test_old_results_without_coverage_sum_still_load(self, tmp_path):
        d = run("ccl", [sm("a", 1.0, words=WordMetrics(n_gt=4, hit=2, coverage_sum=2.0))]).as_dict()
        del d["samples"][0]["words"]["coverage_sum"]
        path = tmp_path / "results.json"
        path.write_text(json.dumps(d), encoding="utf-8")
        assert load(path).samples[0].words.mean_coverage == pytest.approx(0.5)


class TestChecks:
    def test_fast_and_accurate_candidate_passes(self):
        results = {
            "ccl": run("ccl", [sm(f"s{i}", 10.0, capture="photo" if i % 2 else "scan") for i in range(10)]),
            "paddle": run("paddle", [sm(f"s{i}", 100.0, capture="photo" if i % 2 else "scan") for i in range(10)]),
        }
        cs = compare.checks(results)
        assert cs and all(c.passed for c in cs)

    def test_slow_candidate_fails_latency(self):
        results = {"ccl": run("ccl", [sm("a", 60.0)]), "paddle": run("paddle", [sm("a", 100.0)])}
        c = by_name(compare.checks(results))[("overall", "latency P50 ratio")]
        assert c.value == pytest.approx(0.6)
        assert c.passed is False

    def test_recall_gap_uses_detection_baseline(self):
        results = {
            "ccl": run("ccl", [sm("a", 1.0, found=7)]),
            "paddle": run("paddle", [sm("a", 100.0, found=10)]),
            "paddle-det": run("paddle-det", [sm("a", 20.0, found=8)]),
        }
        c = by_name(compare.checks(results))[("overall", "line recall gap")]
        assert c.value == pytest.approx(0.1)  # against paddle-det's 80%, not paddle's 100%

    def test_candidate_is_configurable(self):
        """Phase 3+: a new pipeline of ours, measured against a saved Paddle baseline."""
        results = {"ccl-cnn": run("ccl-cnn", [sm("a", 30.0)]), "paddle": run("paddle", [sm("a", 100.0)])}
        cs = compare.checks(results, compare.Roles(candidate="ccl-cnn"))
        assert any("ccl-cnn" in c.name for c in cs)
        assert compare.checks(results) == []  # no 'ccl' run, so nothing to decide

    def test_only_common_documents_are_compared(self):
        results = {
            "ccl": run("ccl", [sm("a", 10.0), sm("b", 10.0)]),
            "paddle": run("paddle", [sm("a", 100.0), sm("c", 1.0)]),
        }
        c = by_name(compare.checks(results))[("overall", "latency P50 ratio")]
        assert "over 1 docs" in c.detail
        assert c.value == pytest.approx(0.1)


class TestReport:
    def test_warns_when_runs_were_measured_differently(self):
        a = run("ccl", [sm("a", 1.0)])
        b = run("paddle", [sm("a", 9.0)], cv2_threads=12)
        b.run_config["note"] = "machine busy: GPU training"
        warnings = compare.comparability_warnings({"ccl": a, "paddle": b})
        assert any("cv2 threads" in w for w in warnings)
        assert any("machine busy" in w for w in warnings)
        assert "Latency comparability warnings" in compare.build({"ccl": a, "paddle": b}, "x")

    def test_no_warnings_for_matching_runs(self):
        assert compare.comparability_warnings({
            "ccl": run("ccl", [sm("a", 1.0)]), "paddle": run("paddle", [sm("a", 9.0)]),
        }) == []

    def test_report_command_rebuilds_comparison_from_saved_runs(self, tmp_path):
        save(run("paddle", [sm("a", 100.0), sm("b", 100.0)]), tmp_path, subdir="old-paddle")
        save(run("ccl", [sm("a", 10.0), sm("b", 10.0)]), tmp_path, subdir="new-ccl")
        out = tmp_path / "cmp"
        rc = cli.main(["report", "--out", str(out), "--candidate", "mine",
                       "--runs", str(tmp_path / "old-paddle"), f"mine={tmp_path / 'new-ccl'}"])
        assert rc == 0
        text = (out / "comparison.md").read_text(encoding="utf-8")
        assert "**GO**" in text and "`mine`" in text

    def test_report_rejects_duplicate_names(self, tmp_path):
        save(run("ccl", [sm("a", 1.0)]), tmp_path, subdir="one")
        save(run("ccl", [sm("a", 1.0)]), tmp_path, subdir="two")
        rc = cli.main(["report", "--out", str(tmp_path / "x"),
                       "--runs", str(tmp_path / "one"), str(tmp_path / "two")])
        assert rc == 2
