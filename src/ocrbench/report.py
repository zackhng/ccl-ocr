"""Markdown report.

Organised so the uncomfortable numbers are impossible to miss: the aggregate first,
then every slice that could be hiding inside it. A single isolation-recall figure over
a mixed benchmark is close to useless — photographed ID cards and clean scanned forms
fail in different ways and at different rates, and the fix differs accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from .metrics import SampleMetrics
from .runner import (
    RunResult,
    aggregate_chars,
    aggregate_words,
    dpi_bucket,
    latency_summary,
    slice_by,
)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def latency_table(samples: Sequence[SampleMetrics]) -> str:
    summary = latency_summary(samples)
    rows = []
    for stage, stats in summary.items():
        rows.append([
            f"**{stage}**" if stage == "total" else stage,
            f"{stats['p50']:.1f}", f"{stats['p95']:.1f}", f"{stats['p99']:.1f}",
            f"{stats['max']:.1f}", f"{stats['mean']:.1f}",
        ])
    return _table(["stage", "P50 ms", "P95 ms", "P99 ms", "max ms", "mean ms"], rows)


def char_rows(groups: dict[str, list[SampleMetrics]]) -> list[list[object]]:
    rows = []
    for name, group in groups.items():
        chars = [s.chars for s in group if s.chars is not None]
        if not chars:
            continue
        agg = aggregate_chars(chars)
        if agg.n_gt == 0:
            continue
        rows.append([
            name, len(chars), agg.n_gt,
            _pct(agg.isolation_recall), _pct(agg.over_segmentation_rate),
            _pct(agg.merge_rate), _pct(agg.miss_rate),
            _pct(agg.junk_rate), _pct(agg.small_retention),
        ])
    return rows


CHAR_HEADERS = [
    "slice", "samples", "GT chars", "isolated", "over-seg", "merged", "missed",
    "junk", "small kept",
]


def word_rows(groups: dict[str, list[SampleMetrics]]) -> list[list[object]]:
    rows = []
    for name, group in groups.items():
        agg = aggregate_words([s.words for s in group])
        if agg.n_gt == 0:
            continue
        rows.append([
            name, len(group), agg.granularity, agg.n_gt, _pct(agg.hit_rate),
            _pct(agg.mean_coverage), _pct(agg.crossing_rate), _pct(agg.junk_rate),
        ])
    return rows


WORD_HEADERS = [
    "slice", "samples", "unit", "GT regions", "hit", "coverage", "crossing", "junk",
]


def _slice_section(
    title: str,
    samples: Sequence[SampleMetrics],
    key: Callable[[SampleMetrics], str],
) -> str:
    groups = slice_by(samples, key)
    parts = [f"### {title}"]
    rows = char_rows(groups)
    if rows:
        parts.append(_table(CHAR_HEADERS, rows))
    else:
        parts.append("_No character-level ground truth in this slice._")
    parts.append("")
    parts.append(_table(WORD_HEADERS, word_rows(groups)))
    return "\n\n".join(parts)


def build(result: RunResult) -> str:
    samples = result.samples
    all_chars = aggregate_chars([s.chars for s in samples if s.chars is not None])
    all_words = aggregate_words([s.words for s in samples])
    n_char_samples = sum(1 for s in samples if s.chars is not None)

    lat = result.latency.get("total", {})
    parts: list[str] = [
        f"# Benchmark run `{result.run_id}`",
        "",
        f"- engine: **{result.engine}**",
        f"- samples: **{len(samples)}** ({n_char_samples} with character-level ground truth)",
        f"- repeats: {result.run_config.get('repeats')} "
        f"(+{result.run_config.get('warmup')} warm-up), percentiles taken across documents",
        f"- platform: {result.environment.get('platform')} / Python {result.environment.get('python')}",
    ]
    if result.skipped:
        parts.append(
            f"- **skipped {len(result.skipped)} sample(s)** with ground truth but no image "
            f"(e.g. `{result.skipped[0]}`) — run the matching fetch script"
        )
    parts.append("")

    parts += [
        "## Headline",
        "",
        _table(
            ["metric", "value"],
            [
                ["character isolation recall", _pct(all_chars.isolation_recall)],
                ["over-segmentation rate", _pct(all_chars.over_segmentation_rate)],
                ["merge rate", _pct(all_chars.merge_rate)],
                ["miss rate", _pct(all_chars.miss_rate)],
                ["junk rate (of surviving components)", _pct(all_chars.junk_rate)],
                ["small-mark retention", _pct(all_chars.small_retention)],
                [f"region hit rate ({all_words.granularity})", _pct(all_words.hit_rate)],
                [f"region coverage ({all_words.granularity})", _pct(all_words.mean_coverage)],
                ["end-to-end P50 / P95 / P99 ms",
                 f"{lat.get('p50', 0):.1f} / {lat.get('p95', 0):.1f} / {lat.get('p99', 0):.1f}"],
            ],
        ),
        "",
        "Character metrics are computed only over samples that carry character-level "
        "ground truth — today that is the synthetic set. Word metrics cover everything. "
        "Rates are aggregated from raw counts, so a long document weighs more than a "
        "sparse one.",
        "",
        "## Latency by stage",
        "",
        latency_table(samples),
        "",
        "## Slices",
        "",
        "An aggregate hides the failure mode. These are the cuts that matter.",
        "",
        _slice_section("By capture mode", samples, lambda s: s.capture),
        "",
        _slice_section("By source", samples, lambda s: s.source),
        "",
        _slice_section("By script", samples, lambda s: s.script),
        "",
        _slice_section("By effective DPI", samples, dpi_bucket),
        "",
        _slice_section("By template", samples, lambda s: s.template or "-"),
        "",
        "## Latency by capture mode",
        "",
    ]

    for name, group in slice_by(samples, lambda s: s.capture).items():
        parts += [f"### {name} ({len(group)} samples)", "", latency_table(group), ""]

    parts += [
        "## Component routing",
        "",
        _table(
            ["slice", "text", "diacritic", "rule", "blob", "noise"],
            [
                [name] + [
                    sum(s.kind_counts.get(k, 0) for s in group)
                    for k in ("text", "diacritic", "rule", "blob", "noise")
                ]
                for name, group in slice_by(samples, lambda s: s.capture).items()
            ],
        ),
        "",
        "## How to read this",
        "",
        "- **Isolated** is the only outcome Phase 3 can consume as designed. It is the "
        "ceiling on end-to-end accuracy, before a single weight is trained.",
        "- **Over-segmentation** points at preprocessing — thresholds too aggressive, "
        "resolution too low, strokes breaking.",
        "- **Merging** points at resolution and at glyphs touching rules; it is the "
        "failure mode a bigger classifier cannot fix.",
        "- **Junk** costs latency, not accuracy: each one is a wasted inference. Compare "
        "it against the filter's routing counts above.",
        "- **Small-mark retention** below ~95% means the size gates are eating "
        "punctuation. On a cheque that turns `1,234.56` into `123456`.",
        "",
    ]
    return "\n".join(parts)


def write(result: RunResult, out_dir: str | Path) -> Path:
    path = Path(out_dir) / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(result), encoding="utf-8")
    return path
