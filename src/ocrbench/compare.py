"""Phase 6 bake-off report: several engines, the same documents, one verdict.

The verdict is computed against criteria fixed *before* the run (``CRITERIA``), and
is printed first. Criteria chosen after seeing the numbers are not criteria.

What this comparison can and cannot say:

- **Latency is not like-for-like yet.** CCL stops at filtered components; ``paddle``
  returns text. What the gap measures is the *budget* left for Phases 3-5 (classifier
  and reconstruction) if CCL is to stay ahead. ``paddle-det`` is the like-for-like
  number: both sides stop at "where is the text".
- **Localisation is like-for-like**, via :func:`ocrbench.metrics.region_metrics`,
  which scores line boxes and character boxes on the same terms.
- **Recognition accuracy is one-sided.** Only Paddle produces text, so its CER is a
  target for Phase 3+, not a comparison.

Latency ratios are computed **per document, paired**: the same image through both
engines. That removes the between-document variance two independent distributions
would carry, which on this benchmark (40 ms ID cards to 2 s noisy photos) dwarfs the
difference being measured.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .metrics import SampleMetrics
from .report import _pct, _table
from .runner import RunResult, aggregate_regions, aggregate_text, percentiles, slice_by

CRITERIA = {
    "latency_ratio_max": 0.50,
    "line_recall_gap_max": 0.10,
}
"""CCL proceeds to Phase 3 only if, overall *and* on photographs:

- its P50 and P95 are each at most ``latency_ratio_max`` of full Paddle's — leaving at
  least half of Paddle's time for the classifier and reconstruction still to be built;
- its line recall is within ``line_recall_gap_max`` of Paddle's detector.
"""

VERDICT_SLICES: dict[str, Callable[[SampleMetrics], bool]] = {
    "overall": lambda s: True,
    "photo": lambda s: s.capture == "photo",
}


@dataclass(slots=True)
class Check:
    slice: str
    name: str
    value: float | None
    limit: float
    passed: bool | None
    detail: str


def _total(s: SampleMetrics) -> float:
    return float(s.timings_ms.get("total", 0.0))


def _pctl(values: Sequence[float], p: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), p)) if values else None


def _ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def _paired(a: RunResult, b: RunResult, keep: Callable[[SampleMetrics], bool] = lambda s: True
            ) -> list[tuple[SampleMetrics, SampleMetrics]]:
    by_id = {s.sample_id: s for s in b.samples}
    return [(s, by_id[s.sample_id]) for s in a.samples if s.sample_id in by_id and keep(s)]


@dataclass(frozen=True, slots=True)
class Roles:
    """Which saved run plays which part in the verdict.

    The candidate is whatever pipeline of ours is under test — ``ccl`` today, a
    ``ccl-cnn`` run once Phase 3 exists. The baselines are the off-the-shelf engines it
    has to beat, typically measured once and reloaded from disk."""

    candidate: str = "ccl"
    baseline: str = "paddle"
    baseline_det: str = "paddle-det"


DEFAULT_ROLES = Roles()


def checks(results: dict[str, RunResult], roles: Roles = DEFAULT_ROLES) -> list[Check]:
    results = common_samples(results)
    cand, full = results.get(roles.candidate), results.get(roles.baseline)
    det = results.get(roles.baseline_det) or full
    cn = roles.candidate
    out: list[Check] = []
    for slice_name, keep in VERDICT_SLICES.items():
        if cand and full:
            pairs = _paired(cand, full, keep)
            for p in (50, 95):
                c = _pctl([_total(x) for x, _ in pairs], p)
                f = _pctl([_total(y) for _, y in pairs], p)
                ratio = None if not c or not f else c / f
                out.append(Check(
                    slice_name, f"latency P{p} ratio ({cn} / {full.engine})", ratio,
                    CRITERIA["latency_ratio_max"],
                    None if ratio is None else ratio <= CRITERIA["latency_ratio_max"],
                    f"{cn} {_ms(c)} ms vs {full.engine} {_ms(f)} ms over {len(pairs)} docs",
                ))
        if cand and det:
            rc = aggregate_regions(s.regions for s in cand.samples if keep(s))
            rd = aggregate_regions(s.regions for s in det.samples if keep(s))
            if rc.n_gt and rd.n_gt:
                gap = rd.line_recall - rc.line_recall
                out.append(Check(
                    slice_name, f"line recall gap ({det.engine} - {cn})", gap,
                    CRITERIA["line_recall_gap_max"], gap <= CRITERIA["line_recall_gap_max"],
                    f"{cn} {_pct(rc.line_recall)} vs {det.engine} {_pct(rd.line_recall)} "
                    f"over {rc.n_gt} GT lines",
                ))
    return out


def verdict_section(results: dict[str, RunResult], roles: Roles = DEFAULT_ROLES) -> list[str]:
    cs = checks(results, roles)
    decided = [c for c in cs if c.passed is not None]
    if not decided:
        return ["## Verdict", "",
                f"_Not computable: needs both `{roles.candidate}` and `{roles.baseline}` runs._", ""]
    failed = [c for c in decided if not c.passed]
    head = (
        f"**GO** — `{roles.candidate}` meets every criterion."
        if not failed else
        f"**NO-GO** — {len(failed)} of {len(decided)} criteria failed for `{roles.candidate}`."
    )
    rows = [
        [c.slice, c.name, "n/a" if c.value is None else f"{c.value:.2f}",
         f"<= {c.limit:.2f}", "n/a" if c.passed is None else ("pass" if c.passed else "**FAIL**"),
         c.detail]
        for c in cs
    ]
    return [
        "## Verdict", "", head, "",
        _table(["slice", "criterion", "value", "limit", "result", "detail"], rows), "",
        f"Criteria fixed before the run: `{roles.candidate}` P50 and P95 each <= "
        f"{CRITERIA['latency_ratio_max']:.0%} of `{roles.baseline}`'s (paired per document), "
        f"and line recall within {CRITERIA['line_recall_gap_max'] * 100:.0f} points of "
        f"`{roles.baseline_det}` — overall and on photographs.", "",
    ]


def latency_section(results: dict[str, RunResult], roles: Roles = DEFAULT_ROLES) -> list[str]:
    parts = ["## Latency", "", "End-to-end ms per document (median of repeats), percentiles "
             "across documents.", ""]
    for title, key in (("By capture mode", lambda s: s.capture), ("By source", lambda s: s.source)):
        rows = []
        for name, res in results.items():
            groups = {"all": list(res.samples), **slice_by(res.samples, key)}
            for g, group in groups.items():
                st = percentiles([_total(s) for s in group])
                rows.append([name, g, st["n"], _ms(st["p50"]), _ms(st["p95"]),
                             _ms(st["p99"]), _ms(st["max"])])
        parts += [f"### {title}", "",
                  _table(["engine", "slice", "docs", "P50", "P95", "P99", "max"], rows), ""]

    cand, full = results.get(roles.candidate), results.get(roles.baseline)
    det = results.get(roles.baseline_det)
    cn = roles.candidate
    if cand and full:
        rows = []
        for g, keep in (("all", lambda s: True), ("scan", lambda s: s.capture == "scan"),
                        ("photo", lambda s: s.capture == "photo")):
            pairs = _paired(cand, full, keep)
            if not pairs:
                continue
            ratio = [_total(c) / _total(f) for c, f in pairs if _total(f) > 0]
            budget = [_total(f) - _total(c) for c, f in pairs]
            rows.append([g, len(pairs), f"{_pctl(ratio, 50):.2f}", f"{_pctl(ratio, 95):.2f}",
                         _ms(_pctl(budget, 50)), _ms(_pctl(budget, 5))])
        parts += [
            f"### Paired: `{cn}` against `{full.engine}` on the same document", "",
            _table(["slice", "docs", "ratio P50", "ratio P95", "budget P50 ms",
                    "budget P5 ms"], rows), "",
            f"**Budget** is `{full.engine}`'s time minus `{cn}`'s on the same document: what "
            f"remains for the stages `{cn}` has not built yet before it stops being faster. "
            "P5 is the tight end — the documents with least room.", "",
        ]
    if full and det:
        pairs = _paired(full, det)
        rec = [_total(f) - _total(d) for f, d in pairs]
        share = [(_total(f) - _total(d)) / _total(f) for f, d in pairs if _total(f) > 0]
        parts += [
            f"`{full.engine}` recognition cost (full minus detection-only, paired over "
            f"{len(pairs)} docs): P50 {_ms(_pctl(rec, 50))} ms, P95 {_ms(_pctl(rec, 95))} ms — "
            f"{_pctl(share, 50):.0%} of its time at the median.", "",
        ]
    return parts


def slowest_section(results: dict[str, RunResult], n: int = 5) -> list[str]:
    parts = ["## Slowest documents", "",
             "The tail is where a latency claim breaks. Per-stage ms for each engine's "
             f"{n} slowest documents.", ""]
    for name, res in results.items():
        worst = sorted(res.samples, key=_total, reverse=True)[:n]
        rows = []
        for s in worst:
            stages = ", ".join(f"{k} {v:.0f}" for k, v in sorted(
                s.timings_ms.items(), key=lambda kv: -kv[1]) if k != "total")
            rows.append([s.sample_id, s.capture, f"{_total(s):.0f}", stages])
        parts += [f"### {name}", "", _table(["sample", "capture", "total ms", "stages (ms)"], rows), ""]
    return parts


def localisation_section(results: dict[str, RunResult]) -> list[str]:
    parts = [
        "## Localisation", "",
        "Per ground-truth line: the share of its horizontal span covered by the engine's "
        "text boxes (gaps under one line-height closed, so character boxes and line boxes "
        "score alike). **Found** = span coverage >= 50%. **Spurious** = predicted boxes "
        "on no GT line; n/a where GT is incomplete. Cheques carry no line GT and are absent.",
        "",
    ]
    for title, key in (("By capture mode", lambda s: s.capture),
                       ("By source", lambda s: s.source),
                       ("By script", lambda s: s.script)):
        rows = []
        for name, res in results.items():
            groups = {"all": list(res.samples), **slice_by(res.samples, key)}
            for g, group in groups.items():
                agg = aggregate_regions(s.regions for s in group)
                if agg.n_gt == 0:
                    continue
                rows.append([name, g, agg.n_gt, _pct(agg.line_recall),
                             _pct(agg.mean_coverage), _pct(agg.spurious_rate)])
        parts += [f"### {title}", "",
                  _table(["engine", "slice", "GT lines", "found", "coverage", "spurious"], rows), ""]
    return parts


def text_section(results: dict[str, RunResult]) -> list[str]:
    parts = ["## Recognition accuracy (target for Phase 3+)", ""]
    any_rows = False
    for title, key in (("By capture mode", lambda s: s.capture),
                       ("By source", lambda s: s.source),
                       ("By script", lambda s: s.script),
                       ("By template", lambda s: s.template or "-")):
        rows = []
        for name, res in results.items():
            groups = {"all": list(res.samples), **slice_by(res.samples, key)}
            for g, group in groups.items():
                agg = aggregate_text(s.text for s in group)
                if agg.n_gt_chars == 0:
                    continue
                rows.append([name, g, agg.n_gt_chars, _pct(agg.cer), _pct(agg.word_f1)])
        if rows:
            any_rows = True
            parts += [f"### {title}", "",
                      _table(["engine", "slice", "GT chars", "CER", "word F1"], rows), ""]
    if not any_rows:
        parts += ["_No engine in this run returned text._", ""]
    else:
        parts += [
            "CER = (edits on GT lines + characters predicted on no GT line) / GT characters, "
            "with whitespace removed, and case folded for SROIE, whose transcripts are "
            "all upper case. Word F1 is bag-of-words and space-sensitive, so it is where "
            "dropped spaces (\"ROCNO:538358-H\") are charged; Han is tokenised by "
            "character. SROIE transcripts also omit some printed text on annotated lines, "
            "which inflates CER for every engine alike.", "",
        ]
    return parts


def common_samples(results: dict[str, RunResult]) -> dict[str, RunResult]:
    """Restrict every run to the documents all of them measured.

    Runs saved at different times (a Paddle baseline measured once, a CCL run from
    today) need not cover the same set — a ``--limit``, a filter, or a sample added
    since. Aggregates over different documents are not comparable, so the comparison
    is made over the intersection, and the report says how many were dropped."""
    ids = set.intersection(*({s.sample_id for s in r.samples} for r in results.values()))
    return {
        name: replace(r, samples=[s for s in r.samples if s.sample_id in ids])
        for name, r in results.items()
    }


def comparability_warnings(results: dict[str, RunResult]) -> list[str]:
    """Differences between runs that make their latencies not directly comparable."""
    warnings: list[str] = []

    def differs(label: str, get: Callable[[RunResult], object]) -> None:
        values = {name: get(r) for name, r in results.items()}
        if len({repr(v) for v in values.values()}) > 1:
            warnings.append(f"{label} differs: " + ", ".join(f"`{n}`={v}" for n, v in values.items()))

    differs("machine", lambda r: (r.environment.get("platform"), r.environment.get("processor")))
    differs("repeats", lambda r: (r.run_config.get("repeats"), r.run_config.get("warmup")))
    differs("cv2 threads", lambda r: r.environment.get("cv2_threads"))
    for name, r in results.items():
        if r.run_config.get("note"):
            warnings.append(f"`{name}` note: {r.run_config['note']}")
    return warnings


def build(results: dict[str, RunResult], run_id: str, roles: Roles = DEFAULT_ROLES) -> str:
    sizes = {name: len(r.samples) for name, r in results.items()}
    results = common_samples(results)
    first = next(iter(results.values()))
    env = first.environment
    n_common = len(first.samples)
    parts = [
        f"# Phase 6 bake-off `{run_id}`", "",
        f"- engines: {', '.join(f'**{n}**' for n in results)}",
        f"- documents: {n_common} common to every run"
        + ("" if len(set(sizes.values())) == 1 and n_common == next(iter(sizes.values()))
           else f" (runs held {', '.join(f'{n}: {k}' for n, k in sizes.items())}; "
                "the rest are excluded)"),
        f"- repeats: {first.run_config.get('repeats')} (+{first.run_config.get('warmup')} "
        "warm-up) per document; percentiles across documents",
        f"- platform: {env.get('platform')} / {env.get('processor')} / Python {env.get('python')}",
    ]
    for name, res in results.items():
        parts.append(f"- `{name}` run `{res.run_id}`"
                     + (f", {res.run_config['wall_s']:.0f}s wall" if res.run_config.get("wall_s") else "")
                     + f", config: `{res.pipeline_config}`"
                     + (f" versions: `{res.environment.get('engine_versions')}`"
                        if res.environment.get("engine_versions") else "")
                     + f" cv2 threads: {res.environment.get('cv2_threads')}")
    warnings = comparability_warnings(results)
    if warnings:
        parts += ["", "**Latency comparability warnings** — read the latency sections with "
                  "these in mind:", ""] + [f"- {w}" for w in warnings]
    parts.append("")
    parts += verdict_section(results, roles)
    parts += latency_section(results, roles)
    parts += localisation_section(results)
    parts += text_section(results)
    parts += slowest_section(results)
    parts += [
        "## Caveats", "",
        "- CCL's latency covers preprocess → binarise → CCL → filter only. It has no "
        "classifier and no reconstruction yet; full Paddle includes recognition. The "
        "budget columns are the honest reading, not the raw ratio.",
        "- Paddle runs PP-OCRv5 *mobile* on CPU with oneDNN, orientation and unwarping "
        "off, and its own resize defaults (no long-side cap; CCL caps at 1600 px).",
        "- Character-level and word-level isolation metrics are CCL-only and live in "
        "each engine's own `summary.md`.",
        "",
    ]
    return "\n".join(parts)


def write(results: dict[str, RunResult], out_dir: str | Path, run_id: str,
          roles: Roles = DEFAULT_ROLES) -> Path:
    path = Path(out_dir) / "comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(results, run_id, roles), encoding="utf-8")
    return path
