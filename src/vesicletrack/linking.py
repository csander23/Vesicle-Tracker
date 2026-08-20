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
    tr = filter_short(tr, cfg.link.min_length_frames)
    tr = merge_colocated(tr, cfg.link.merge_radius_px)
    return tr.sort_values(["particle", "frame"]).reset_index(drop=True)[
        ["particle", "frame", "x", "y"]]


def filter_short(tr: pd.DataFrame, min_len: int) -> pd.DataFrame:
    if not len(tr):
        return tr
    n = tr.groupby("particle").size()
    return tr[tr.particle.isin(n[n >= min_len].index)].copy()


def merge_colocated(tr: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Merge tracks whose mean positions sit within `radius` of each other.

    Uses union-find so a chain of three fragments becomes one vesicle, not two pairs.
    """
    if not len(tr) or radius <= 0:
        return tr
    cen = tr.groupby("particle")[["x", "y"]].mean()
    ids = cen.index.to_numpy()
    xy = cen.to_numpy()
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    from scipy.spatial import cKDTree
    for i, j in cKDTree(xy).query_pairs(radius):
        union(ids[i], ids[j])

    out = tr.copy()
    out["particle"] = [find(p) for p in out.particle]
    # a merged vesicle can now hold two rows for one frame; average them
    out = out.groupby(["particle", "frame"], as_index=False)[["x", "y"]].mean()
    codes = {p: k for k, p in enumerate(sorted(out.particle.unique()))}
    out["particle"] = out.particle.map(codes)
    return out
