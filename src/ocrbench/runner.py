"""Benchmark runner: latency distribution + isolation metrics over the whole set.

Two decisions worth stating, because both change the numbers:

**Percentiles are taken across documents, not across repeats.** Each sample is run
``warmup + repeats`` times and its *median* per-stage timing becomes that document's
cost; the distribution we report is then over documents. The question the product cares
about is "how slow is the 95th-percentile *document*", not "how slow was the 95th
percentile of a timing loop on one easy page".

**Rates are aggregated from counts, not averaged over samples.** A per-sample mean of
isolation recall weights a 12-character ID card the same as a 900-character form, which
quietly inflates the headline whenever the sparse pages are the easy ones. Summing
numerators and denominators and dividing once is the honest aggregation.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

from ocr.config import DEFAULT_CONFIG, PipelineConfig
from ocr.engine import CCLEngine, Engine, load_image

from .gt import BenchmarkStore, Sample
from .metrics import (
    CharMetrics,
    GroupingMetrics,
    RegionMetrics,
    SampleMetrics,
    TextMetrics,
    WordMetrics,
    evaluate,
)

PERCENTILES = (50, 95, 99)


@dataclass(slots=True)
class RunConfig:
    repeats: int = 3
    warmup: int = 1
    limit: int | None = None
    sources: tuple[str, ...] = ()
    captures: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    note: str = ""
    """Free text stored with the results, e.g. "machine busy: GPU training running".
    Latency is only as good as the machine was idle, and that is not recoverable from
    the numbers afterwards."""

    def matches(self, s: Sample) -> bool:
        if self.sources and s.source not in self.sources:
            return False
        if self.captures and s.capture not in self.captures:
            return False
        if self.scripts and s.script not in self.scripts:
            return False
        return True


@dataclass(slots=True)
class RunResult:
    run_id: str
    engine: str
    samples: list[SampleMetrics] = field(default_factory=list)
    latency: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    pipeline_config: dict = field(default_factory=dict)
    run_config: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "engine": self.engine,
            "environment": self.environment,
            "pipeline_config": self.pipeline_config,
            "run_config": self.run_config,
            "latency": self.latency,
            "skipped": self.skipped,
            "samples": [s.as_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(
            run_id=d["run_id"],
            engine=d["engine"],
            samples=[SampleMetrics.from_dict(x) for x in d.get("samples", [])],
            latency=d.get("latency", {}),
            environment=d.get("environment", {}),
            pipeline_config=d.get("pipeline_config", {}),
            run_config=d.get("run_config", {}),
            skipped=d.get("skipped", []),
        )


def load(path: str | Path) -> RunResult:
    """Load a saved run: a ``results.json`` file, or the folder holding one."""
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    return RunResult.from_dict(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- aggregate


def aggregate_chars(metrics: Iterable[CharMetrics]) -> CharMetrics:
    out = CharMetrics()
    any_seen = False
    for m in metrics:
        any_seen = True
        out.n_gt += m.n_gt
        out.n_surviving += m.n_surviving
        out.isolated += m.isolated
        out.merged += m.merged
        out.over_segmented += m.over_segmented
        out.missed += m.missed
        out.junk += m.junk
        out.junk_in_nontext += m.junk_in_nontext
        out.small_gt += m.small_gt
        out.small_retained += m.small_retained
        out.gt_complete = out.gt_complete and m.gt_complete
    if not any_seen:
        out.gt_complete = False
    return out


def aggregate_words(metrics: Iterable[WordMetrics | None]) -> WordMetrics:
    out = WordMetrics()
    any_seen = False
    granularities: set[str] = set()
    for m in metrics:
        if m is None:
            continue
        any_seen = True
        if m.n_gt:
            granularities.add(m.granularity)
        out.n_gt += m.n_gt
        out.n_surviving += m.n_surviving
        out.hit += m.hit
        out.crossing += m.crossing
        out.junk += m.junk
        out.coverage_sum += m.coverage_sum
        out.gt_complete = out.gt_complete and m.gt_complete
    if not any_seen:
        out.gt_complete = False
    # "mixed" is a warning, not a label: word- and line-level coverage are not
    # comparable, so an aggregate spanning both should be read per-slice instead.
    out.granularity = granularities.pop() if len(granularities) == 1 else (
        "mixed" if granularities else "none"
    )
    return out


def aggregate_regions(metrics: Iterable[RegionMetrics | None]) -> RegionMetrics:
    out = RegionMetrics()
    any_seen = False
    for m in metrics:
        if m is None or m.n_gt == 0:
            continue
        any_seen = True
        out.n_gt += m.n_gt
        out.n_pred += m.n_pred
        out.found += m.found
        out.coverage_sum += m.coverage_sum
        out.spurious += m.spurious
        out.gt_complete = out.gt_complete and m.gt_complete
    if not any_seen:
        out.gt_complete = False
    return out


def aggregate_text(metrics: Iterable[TextMetrics | None]) -> TextMetrics:
    out = TextMetrics()
    any_seen = False
    for m in metrics:
        if m is None:
            continue
        any_seen = True
        out.n_gt_chars += m.n_gt_chars
        out.edits += m.edits
        out.inserted_chars += m.inserted_chars
        out.n_gt_words += m.n_gt_words
        out.n_pred_words += m.n_pred_words
        out.matched_words += m.matched_words
        out.gt_complete = out.gt_complete and m.gt_complete
    if not any_seen:
        out.gt_complete = False
    return out


def aggregate_grouping(metrics: Iterable[GroupingMetrics | None]) -> GroupingMetrics:
    out = GroupingMetrics()
    any_seen = False
    for m in metrics:
        if m is None:
            continue
        any_seen = True
        out.n_gt_lines += m.n_gt_lines
        out.n_pred_lines += m.n_pred_lines
        out.tp_lines += m.tp_lines
        out.n_gt_words += m.n_gt_words
        out.n_pred_words += m.n_pred_words
        out.tp_words += m.tp_words
        out.word_splits += m.word_splits
        out.word_merges += m.word_merges
        out.p_tp_lines += m.p_tp_lines
        out.p_pred_lines += m.p_pred_lines
        out.p_tp_words += m.p_tp_words
        out.p_pred_words += m.p_pred_words
        out.tp_lines_any += m.tp_lines_any
        out.tp_words_any += m.tp_words_any
        out.gt_complete = out.gt_complete and m.gt_complete
    if not any_seen:
        out.gt_complete = False
    return out


def slice_by(
    samples: Sequence[SampleMetrics], key: Callable[[SampleMetrics], str]
) -> dict[str, list[SampleMetrics]]:
    out: dict[str, list[SampleMetrics]] = {}
    for s in samples:
        out.setdefault(key(s), []).append(s)
    return dict(sorted(out.items()))


def dpi_bucket(s: SampleMetrics) -> str:
    d = s.dpi
    if d < 100:
        return "<100"
    if d < 150:
        return "100-149"
    if d < 250:
        return "150-249"
    return ">=250"


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    out = {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in PERCENTILES}
    out["mean"] = round(float(arr.mean()), 3)
    out["max"] = round(float(arr.max()), 3)
    out["n"] = len(values)
    return out


def latency_summary(samples: Sequence[SampleMetrics]) -> dict[str, dict[str, float]]:
    stages: dict[str, list[float]] = {}
    for s in samples:
        for stage, ms in s.timings_ms.items():
            stages.setdefault(stage, []).append(ms)
    # 'total' last so it reads as the bottom line of the table.
    ordered = [k for k in stages if k != "total"] + (["total"] if "total" in stages else [])
    return {stage: percentiles(stages[stage]) for stage in ordered}


# ------------------------------------------------------------------------------- run


def run(
    store: BenchmarkStore,
    engine: Engine | None = None,
    run_config: RunConfig | None = None,
    pipeline_config: PipelineConfig | None = None,
    progress: bool = True,
) -> RunResult:
    engine = engine or CCLEngine(pipeline_config or DEFAULT_CONFIG)
    rc = run_config or RunConfig()

    sample_ids = store.sample_ids()
    if not sample_ids:
        raise RuntimeError(
            f"no ground truth under {store.gt_dir}. "
            "Generate some first: python -m ocrbench.cli synth --count 150"
        )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = RunResult(
        run_id=run_id,
        engine=getattr(engine, "name", type(engine).__name__),
        environment={
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            # Thread counts move latency more than anything else in this comparison.
            "cv2_threads": cv2.getNumThreads(),
            "engine_versions": getattr(engine, "versions", None),
        },
        pipeline_config=(pipeline_config or getattr(engine, "config", DEFAULT_CONFIG)).to_dict(),
        run_config={
            "repeats": rc.repeats, "warmup": rc.warmup, "limit": rc.limit,
            "sources": list(rc.sources), "captures": list(rc.captures), "scripts": list(rc.scripts),
            "note": rc.note,
            "started_at": run_id,
        },
    )

    processed = 0
    started = time.perf_counter()
    for sid in sample_ids:
        sample = store.read(sid)
        if not rc.matches(sample):
            continue
        image_path = store.image_path(sid)
        if not image_path.exists():
            # Real-dataset ground truth can be committed while its pixels are not
            # redistributable. Skipping loudly beats silently shrinking the benchmark.
            result.skipped.append(sid)
            continue

        image = load_image(str(image_path))

        for _ in range(rc.warmup):
            engine.run(image)

        runs = [engine.run(image) for _ in range(max(1, rc.repeats))]
        page = runs[-1]
        # Median per stage across repeats: robust to a single GC pause or a scheduler
        # hiccup, which a mean would carry straight into the P99.
        stage_names = {k for r in runs for k in r.timings_ms}
        page.timings_ms = {
            stage: round(float(np.median([r.timings_ms.get(stage, 0.0) for r in runs])), 3)
            for stage in stage_names
        }

        result.samples.append(evaluate(sample, page))
        processed += 1
        if progress and processed % 25 == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{result.engine}] {processed} samples, {elapsed:.0f}s elapsed", flush=True)
        if rc.limit and processed >= rc.limit:
            break

    if not result.samples:
        raise RuntimeError("no samples matched the run filters")

    result.latency = latency_summary(result.samples)
    result.run_config["wall_s"] = round(time.perf_counter() - started, 1)
    return result


def save(result: RunResult, root: str | Path = "bench/results", subdir: str | None = None) -> Path:
    """Write ``results.json`` under ``<root>/<subdir or run_id>``."""
    out_dir = Path(root) / (subdir or result.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(result.as_dict(), indent=1), encoding="utf-8"
    )
    return out_dir
