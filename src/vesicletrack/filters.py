"""Filters that LABEL vesicles rather than deleting them.

Every tracked vesicle appears in the output with a `passes_filter` boolean and a
`filter_reason` naming the first rule it failed. The subset that passes is also written
separately, so a video yields two lists:

    vesicles_all.csv        every vesicle, every number, nothing removed
    vesicles_filtered.csv   the subset passing the configured filters

Deleting rows upstream is convenient and wrong. Once a vesicle is gone you cannot ask
how many were excluded, whether the excluded ones differ systematically, or what a
different threshold would have given - and those are exactly the questions a reviewer
asks. Keeping everything makes the filter a reversible, auditable choice.

The one exception is upstream of this module: tracks below `link.min_length_frames`
(default 3) never reach scoring, because a two-frame track has a single step and no
metric can be computed from it. That floor is deliberately far below any scientific
cut, and the count discarded by it is reported in the summary.

SELECTION BIAS WARNING, applied by `max_observed_frames` / `max_lifetime_s`: an upper
bound on lifetime preferentially keeps tracks the tracker LOST early. Filtering on it
selects for tracking failure, not for short-lived biology. `report_bias` puts the
resulting censored fraction in the summary so the effect is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rule_list(cfg):
    """(name, predicate) pairs. Predicate returns True for vesicles that PASS."""
    f = cfg.filters
    rules = []
    if f.min_observed_frames is not None:
        rules.append((f"observed_frames<{f.min_observed_frames}",
                      lambda d: d.observed_frames >= f.min_observed_frames))
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
        reason[newly_failed] = name          # first failing rule, not a list
        passes &= ok
    d["passes_filter"] = passes
    d["filter_reason"] = reason.where(~passes, "")
    return d


def filter_summary(vesicles: pd.DataFrame) -> dict:
    """Counts by outcome, so what a filter removed is always visible."""
    if not len(vesicles) or "passes_filter" not in vesicles:
        return {"n_all": int(len(vesicles)), "n_pass": 0, "n_fail": 0, "reasons": {}}
    fail = vesicles[~vesicles.passes_filter]
    return {
        "n_all": int(len(vesicles)),
        "n_pass": int(vesicles.passes_filter.sum()),
        "n_fail": int(len(fail)),
        "reasons": fail.filter_reason.value_counts().to_dict(),
    }
