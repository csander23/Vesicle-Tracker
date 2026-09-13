"""Linking detections into tracks, then cleaning the result.

trackpy does the assignment. Two clean-up steps follow, both of which change counts
materially on real data:

  min_length   short tracks are dominated by detection noise and have no usable
               metrics - a 3-frame "vesicle" has no measurable direction
  merge        two tracks occupying the same place are one vesicle that the linker
               split at a dropout; without merging, the same vesicle is counted twice
               and its metrics are computed on two half-length pieces
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def link(spots: pd.DataFrame, cfg) -> pd.DataFrame:
    """Returns columns particle, frame, x, y."""
    import trackpy as tp
    tp.quiet()
    if not len(spots):
        return pd.DataFrame(columns=["particle", "frame", "x", "y"])
    tr = tp.link(spots, search_range=cfg.link.search_range_px,
                 memory=cfg.link.memory_frames)
    # MERGE FIRST, then filter. Filtering first deletes exactly the short fragments
    # that merging exists to rejoin: a vesicle broken into three 20-frame pieces was
    # discarded entirely at a 40-frame floor, when the merged track would have been 60.
    tr = merge_colocated(tr, cfg.link.merge_radius_px,
                         max_gap=cfg.link.merge_max_gap_frames,
                         overlap_tol=cfg.link.merge_overlap_tolerance)
    tr = filter_short(tr, cfg.link.min_length_frames)
    return tr.sort_values(["particle", "frame"]).reset_index(drop=True)[
        ["particle", "frame", "x", "y"]]


def filter_short(tr: pd.DataFrame, min_len: int) -> pd.DataFrame:
    if not len(tr):
        return tr
    n = tr.groupby("particle").size()
    return tr[tr.particle.isin(n[n >= min_len].index)].copy()


def merge_colocated(tr: pd.DataFrame, radius: float, max_gap: int = 15,
                    overlap_tol: int = 2) -> pd.DataFrame:
    """Rejoin fragments of one vesicle that the linker split at a long dropout.

    Fragment B is merged into fragment A only if ALL of:
      1. they are essentially disjoint in time - B starts no more than `overlap_tol`
         frames before A ends. Two fragments that coexist are two vesicles.
      2. the dropout is short: B starts within `max_gap` frames of A ending. This is
         the "look back" window, and it is what stops a vesicle being joined to an
         unrelated one that arrived at the same spot much later.
      3. they are close AT THE JUNCTION: A's last position is within `radius` of B's
         first. Mean position is the wrong test - a vesicle that moved has a mean
         nowhere near either endpoint.

    Union-find, so a chain of three fragments becomes one vesicle rather than two pairs.
    """
    if not len(tr) or radius <= 0:
        return tr

    ends = (tr.sort_values("frame").groupby("particle")
              .agg(f0=("frame", "first"), f1=("frame", "last"),
                   x0=("x", "first"), y0=("y", "first"),
                   x1=("x", "last"), y1=("y", "last")))
    ends = ends.sort_values("f0")
    ids = ends.index.to_numpy()
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rec = ends.to_dict("index")
    order = list(ids)
    for ai, a in enumerate(order):
        A = rec[a]
        for b in order[ai + 1:]:
            B = rec[b]
            gap = B["f0"] - A["f1"]
            if gap > max_gap:
                break                       # sorted by start: no later one is closer
            if gap < -overlap_tol:
                continue                    # they coexist -> different vesicles
            if np.hypot(A["x1"] - B["x0"], A["y1"] - B["y0"]) <= radius:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    out = tr.copy()
    out["particle"] = [find(p) for p in out.particle]
    # a merge across a small overlap can leave two rows for one frame; average them
    out = out.groupby(["particle", "frame"], as_index=False)[["x", "y"]].mean()
    codes = {p: k for k, p in enumerate(sorted(out.particle.unique()))}
    out["particle"] = out.particle.map(codes)
    return out
