"""Filters that label vesicles rather than deleting them.

Every tracked vesicle appears in the output with a `passes_filter` boolean and a
`filter_reason` naming the first rule it failed. The subset that passes is also written
separately, so a video yields two lists:

    vesicles_all.csv        every vesicle, every number, nothing removed
    vesicles_filtered.csv   the subset passing the configured filters

Rows are not deleted upstream because once a vesicle is gone there is no way to ask
how many were excluded, whether the excluded ones differ systematically, or what a
different threshold would have given. Keeping everything makes the filter a
reversible, auditable choice.

The one exception is upstream of this module: tracks below `link.min_length_frames`
(default 3) never reach scoring, because a two-frame track has a single step and no
metric can be computed from it. That floor is far below any scientific cut, and the
count discarded by it is reported in the summary.

Warning: an upper bound on lifetime (`max_observed_frames` / `max_lifetime_s`)
preferentially keeps tracks the tracker lost early, so filtering on it selects for
tracking failure rather than for short-lived biology. The censored count in
`summary.json` is reported so the effect stays visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rule_list(cfg):
    """(name, predicate) pairs. Predicate returns True for vesicles that pass."""
    f = cfg.filters
    rules = []
    if f.min_observed_frames is not None:
        rules.append((f"observed_frames<{f.min_observed_frames}",
                      lambda d: d.observed_frames >= f.min_observed_frames))
    if f.min_span_frames is not None:
        rules.append((f"span_frames<{f.min_span_frames}",
                      lambda d: d.span_frames >= f.min_span_frames))
    if f.max_observed_frames is not None:
        rules.append((f"observed_frames>{f.max_observed_frames}",
                      lambda d: d.observed_frames <= f.max_observed_frames))
    if f.min_lifetime_s is not None:
        rules.append((f"span_s<{f.min_lifetime_s}",
                      lambda d: d.span_s >= f.min_lifetime_s))
    if f.max_lifetime_s is not None:
        rules.append((f"span_s>{f.max_lifetime_s}",
                      lambda d: d.span_s <= f.max_lifetime_s))
    if f.min_frac_observed is not None and f.min_frac_observed > 0:
        rules.append((f"frac_observed<{f.min_frac_observed}",
                      lambda d: d.frac_observed >= f.min_frac_observed))
    if f.max_longest_gap is not None:
        rules.append((f"longest_gap>{f.max_longest_gap}",
                      lambda d: d.longest_gap <= f.max_longest_gap))
    if f.require_directed_measurable:
        # This checks the computed flag rather than a proxy for it. Whether `directed`
        # exists depends on how many observed frames land in each tau-window, which no
        # threshold on span or on total observed frames can express: a track can pass
        # span>=180 and observed>=40 and still be unmeasurable if its frames cluster
        # into too few windows. Filtering on the flag makes "filtered" mean usable by
        # construction, whatever the other parameters are.
        rules.append(("directed not measurable",
                      lambda d: d.directed_measurable.astype(bool)
                      if "directed_measurable" in d
                      else pd.Series(True, index=d.index)))
    if f.exclude_censored:
        rules.append(("censored", lambda d: ~d.is_censored.astype(bool)))
    if f.exclude_classes:
        bad = set(f.exclude_classes)
        rules.append((f"klass in {sorted(bad)}", lambda d: ~d.klass.isin(bad)))
    if f.rois:
        keep = set(f.rois)
        rules.append((f"roi not in {sorted(keep)}",
                      lambda d: d.roi.isin(keep) if "roi" in d else
                      pd.Series(True, index=d.index)))
    if f.min_sigma_px is not None:
        rules.append((f"sigma_px<{f.min_sigma_px}",
                      lambda d: d.sigma_px >= f.min_sigma_px
                      if "sigma_px" in d else pd.Series(True, index=d.index)))
    if f.max_sigma_px is not None:
        rules.append((f"sigma_px>{f.max_sigma_px}",
                      lambda d: d.sigma_px <= f.max_sigma_px
                      if "sigma_px" in d else pd.Series(True, index=d.index)))
    return rules


def apply_filters(vesicles: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add passes_filter and filter_reason. Returns a copy; drops nothing."""
    if not len(vesicles):
        out = vesicles.copy()
        out["passes_filter"] = pd.Series(dtype=bool)
        out["filter_reason"] = pd.Series(dtype=object)
        return out

    d = vesicles.copy()
    passes = pd.Series(True, index=d.index)
    reason = pd.Series("", index=d.index, dtype=object)
    for name, pred in _rule_list(cfg):
        ok = pred(d).fillna(False).astype(bool)
        newly_failed = passes & ~ok
        reason[newly_failed] = name          # only the first failing rule is kept
        passes &= ok
    d["passes_filter"] = passes
    d["filter_reason"] = reason.where(~passes, "")
    return d


def filter_summary(vesicles: pd.DataFrame) -> dict:
    """Counts by outcome: how many passed, how many failed, and the reasons."""
    if not len(vesicles) or "passes_filter" not in vesicles:
        return {"n_all": int(len(vesicles)), "n_pass": 0, "n_fail": 0, "reasons": {}}
    fail = vesicles[~vesicles.passes_filter]
    return {
        "n_all": int(len(vesicles)),
        "n_pass": int(vesicles.passes_filter.sum()),
        "n_fail": int(len(fail)),
        "reasons": fail.filter_reason.value_counts().to_dict(),
    }
