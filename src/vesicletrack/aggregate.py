"""Roll per-vesicle results up to every level, and write a CSV for each.

The data are nested: vesicle -> video (one cell) -> whatever grouping the experiment
has (genotype, batch, treatment). This module produces one table per level plus a tidy
long-format file for plotting.

    vesicles_all.csv      one row per VESICLE, nothing removed, every metric
    vesicles_filtered.csv the subset passing filters
    per_video.csv         one row per VIDEO
    per_<key>.csv         one row per group, batch, genotype ... whatever you name
    long.csv              tidy: level, unit, group, metric, value

WHY VIDEO IS THE UNIT THAT MATTERS
----------------------------------
Vesicles within one cell are not independent - they share a cell, a transfection, a
field of view and a focal plane. Treating each vesicle as a replicate inflates n by a
factor of hundreds and produces significance that will not survive a nested analysis.

So group-level statistics here are computed ACROSS VIDEOS, from video means: the
`n` reported at group level is the number of videos, never the number of vesicles, and
`sem` is the standard error of the video means. The per-vesicle count is still
reported, as `n_vesicles`, but it is never used as the sample size.

If you want per-vesicle statistics anyway - for a distribution shape, say - use
`vesicles_all.csv` directly and say plainly in the methods that vesicles were pooled.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Metrics rolled up per video. Median per vesicle-level metric: these distributions are
# heavily right-skewed (a few large excursions), so a mean per video is dragged by
# outliers that a median ignores.
DEFAULT_METRICS = ["net", "gross", "directed", "net_rate", "gross_rate",
                   "directed_rate", "observed_frames", "span_frames", "observed_s",
                   "span_s", "frac_observed", "n_gaps", "longest_gap",
                   "sigma_px", "sigma_deconv_px", "fwhm_px"]


def summarise_video(vesicles: pd.DataFrame, name: str | None = None,
                    metrics: list | None = None, meta: dict | None = None) -> dict:
    """One row for one video: counts, class fractions, and the median of each metric."""
    metrics = metrics or DEFAULT_METRICS
    rec: dict = {"video": name}
    rec.update(meta or {})
    rec["n_vesicles"] = int(len(vesicles))
    if not len(vesicles):
        return rec

    if "passes_filter" in vesicles:
        rec["n_pass"] = int(vesicles.passes_filter.sum())
        rec["n_fail"] = int((~vesicles.passes_filter).sum())
    if "klass" in vesicles:
        counts = vesicles.klass.value_counts()
        for k in ("mover", "confined", "excluded"):
            rec[f"n_{k}"] = int(counts.get(k, 0))
        scored = int(counts.get("mover", 0) + counts.get("confined", 0))
        rec["frac_mover"] = (counts.get("mover", 0) / scored) if scored else np.nan
    if "is_censored" in vesicles:
        rec["n_censored"] = int(vesicles.is_censored.sum())
        rec["frac_censored"] = float(vesicles.is_censored.mean())
    if "roi" in vesicles:
        for r, n in vesicles.roi.value_counts().items():
            rec[f"n_roi_{r}"] = int(n)

    for m in metrics:
        if m in vesicles.columns:
            v = pd.to_numeric(vesicles[m], errors="coerce").dropna()
            rec[f"{m}_median"] = float(v.median()) if len(v) else np.nan
    return rec


def per_video(results_or_frames, metrics: list | None = None,
              use_filtered: bool = True) -> pd.DataFrame:
    """One row per video, from Result objects or (name, DataFrame) pairs.

    use_filtered: summarise only vesicles passing the filters (the usual choice). The
    unfiltered counts are still reported as n_vesicles / n_fail either way, so how much
    the filter removed is always visible.
    """
    rows = []
    for item in results_or_frames:
        if hasattr(item, "vesicles"):
            allv, name = item.vesicles, item.name
            meta = {k: item.summary.get(k) for k in ("duration_s", "frames")
                    if k in item.summary}
        else:
            name, allv = item
            meta = {}
        use = allv[allv.passes_filter] if (use_filtered and
                                           "passes_filter" in allv) else allv
        rec = summarise_video(use, name=name, metrics=metrics)
        rec["n_vesicles_all"] = int(len(allv))
        for c in ("batch", "group", "genotype", "condition", "treatment", "animal"):
            if c in allv.columns and len(allv):
                rec[c] = allv[c].iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def per_group(video_table: pd.DataFrame, by: str | list, metrics: list | None = None
              ) -> pd.DataFrame:
    """Aggregate VIDEO rows to a grouping. n is the number of VIDEOS, not vesicles.

    Each metric gets mean, sd, sem and n across videos. `sem` here is the quantity to
    put on a bar chart; a sem computed across vesicles would be roughly sqrt(n_vesicles)
    times too small and is not offered.
    """
    by = [by] if isinstance(by, str) else list(by)
    missing = [b for b in by if b not in video_table.columns]
    if missing:
        raise KeyError(f"grouping column(s) {missing} not in the video table. "
                       f"Available: {sorted(video_table.columns)}. Pass them to "
                       "analyse(..., batch=..., group=...) or via a sample sheet.")
    cols = metrics or [c for c in video_table.columns
                       if c.endswith("_median") or c in
                       ("frac_mover", "n_vesicles", "frac_censored")]
    out = []
    for keys, d in video_table.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(by, keys))
        rec["n_videos"] = int(len(d))
        rec["n_vesicles_total"] = int(d.n_vesicles.sum()) if "n_vesicles" in d else 0
        for c in cols:
            if c not in d.columns:
                continue
            v = pd.to_numeric(d[c], errors="coerce").dropna()
            rec[f"{c}_mean"] = float(v.mean()) if len(v) else np.nan
            rec[f"{c}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            rec[f"{c}_sem"] = (float(v.std(ddof=1) / np.sqrt(len(v)))
                               if len(v) > 1 else np.nan)
            rec[f"{c}_n"] = int(len(v))
        out.append(rec)
    return pd.DataFrame(out)


def to_long(video_table: pd.DataFrame, id_cols: list | None = None) -> pd.DataFrame:
    """Tidy long format for plotting: one row per (video, metric)."""
    id_cols = id_cols or [c for c in
                          ("video", "batch", "group", "genotype", "condition",
                           "treatment", "animal")
                          if c in video_table.columns]
    val_cols = [c for c in video_table.columns
                if c not in id_cols and pd.api.types.is_numeric_dtype(video_table[c])]
    long = video_table.melt(id_vars=id_cols, value_vars=val_cols,
                            var_name="metric", value_name="value")
    long.insert(0, "level", "video")
    return long


def write_all(results, output_dir, by=None, metrics: list | None = None,
              use_filtered: bool = True) -> dict:
    """Write every level to CSV. Returns {name: path}.

    `by` is the grouping column(s) - e.g. "genotype" or ["batch", "genotype"]. Omit it
    and only the vesicle and video levels are written.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict = {}

    frames = []
    for r in results:
        v = r.vesicles.copy() if hasattr(r, "vesicles") else r[1].copy()
        v.insert(0, "video", r.name if hasattr(r, "name") else r[0])
        frames.append(v)
    allv = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    allv.to_csv(out / "vesicles_all.csv", index=False)
    written["vesicles_all"] = out / "vesicles_all.csv"
    if "passes_filter" in allv:
        allv[allv.passes_filter].to_csv(out / "vesicles_filtered.csv", index=False)
        written["vesicles_filtered"] = out / "vesicles_filtered.csv"

    vid = per_video(results, metrics=metrics, use_filtered=use_filtered)
    vid.to_csv(out / "per_video.csv", index=False)
    written["per_video"] = out / "per_video.csv"

    if by:
        for key in ([by] if isinstance(by, str) else by):
            if key in vid.columns:
                g = per_group(vid, key, metrics=metrics)
                g.to_csv(out / f"per_{key}.csv", index=False)
                written[f"per_{key}"] = out / f"per_{key}.csv"

    to_long(vid).to_csv(out / "long.csv", index=False)
    written["long"] = out / "long.csv"
    return written


def load_results(output_dir, pattern: str = "*/vesicles_all.csv") -> pd.DataFrame:
    """Re-pool per-movie CSVs written earlier, without re-running the pipeline."""
    frames = []
    for p in sorted(Path(output_dir).glob(pattern)):
        d = pd.read_csv(p)
        d.insert(0, "video", p.parent.name)
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
