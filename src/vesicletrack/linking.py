"""Linking detections into tracks, then repairing what the linker got wrong.

The order is the one validated against hand-checked vesicles on a crowded cell:

  1. trackpy links each detection to the nearest one in the next frame, within
     `link.search_range_px`, and may bridge a dropout of up to `link.memory_frames`.
  2. Tracks with fewer than `link.min_length_frames` detected frames are dropped. This
     is the one place the pipeline deletes rather than labels: a two-frame fragment
     has no measurable motion, and the merge step below compares every pair of
     tracks, which is not feasible over thousands of fragments. The number dropped is
     reported in the run summary.
  3. Gap recovery (recover.py) fills gaps and extends ends from the probability maps.
  4. Tracks whose mean positions lie within `link.merge_colocated_px` of each other
     are one vesicle that the linker split, typically a bright vesicle that produced
     two detections at once.
  5. Trajectory merge joins fragments of one vesicle in two ways: fragment B starts
     within `merge_gap_frames` of fragment A ending and within `merge_gap_radius_px`
     of A's last position (or of where A's velocity would have taken it), or the two
     coexist for at least `merge_overlap_frames` with a median separation under
     `merge_overlap_radius_px`. The overlap radius is deliberately tight: real
     identity swaps sit about 1 px apart, while distinct neighbouring vesicles in a
     dense field sit about 6 px apart, and a looser radius fused them.

Merged fragments that share a frame are averaged to one position per frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .recover import recover

COLUMNS = ["particle", "frame", "x", "y", "recovered"]


def link(spots: pd.DataFrame, maps: np.ndarray, cfg, n_frames: int,
         mask: np.ndarray | None = None) -> tuple[pd.DataFrame, int]:
    """Returns (tracks, n_dropped_short). The `recovered` column of the tracks marks
    positions that came from gap recovery rather than from a detection."""
    import trackpy as tp
    tp.quiet()
    empty = pd.DataFrame(columns=COLUMNS)
    if not len(spots):
        return empty, 0
    tr = tp.link(spots[["frame", "x", "y"]], search_range=cfg.link.search_range_px,
                 memory=cfg.link.memory_frames)
    tr, n_dropped = filter_short(tr, cfg.link.min_length_frames)
    if not len(tr):
        return empty, n_dropped
    tr = tr.assign(recovered=False)
    if cfg.recover.enabled:
        tr, _ = recover(tr, maps, cfg, n_frames, mask=mask)
    tr = merge_colocated(tr, cfg.link.merge_colocated_px)
    tr = merge_trajectories(tr, cfg.link)
    return relabel(tr), n_dropped


def filter_short(tr: pd.DataFrame, min_len: int) -> tuple[pd.DataFrame, int]:
    n = tr.groupby("particle").size()
    keep = n[n >= min_len].index
    return tr[tr.particle.isin(keep)].copy(), int((n < min_len).sum())


def relabel(tr: pd.DataFrame) -> pd.DataFrame:
    """One position per (particle, frame), particles numbered 0..n-1 by first frame."""
    if not len(tr):
        return pd.DataFrame(columns=COLUMNS)
    out = (tr.groupby(["particle", "frame"], as_index=False)
             .agg(x=("x", "mean"), y=("y", "mean"), recovered=("recovered", "all")))
    first = out.groupby("particle").frame.min().sort_values(kind="stable")
    codes = {p: k for k, p in enumerate(first.index)}
    out["particle"] = out.particle.map(codes)
    return out.sort_values(["particle", "frame"]).reset_index(drop=True)[COLUMNS]


def _union_find(ids):
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
    return find, union


def merge_colocated(tr: pd.DataFrame, radius: float) -> pd.DataFrame:
    """Union tracks whose mean positions are within `radius` px. 0 disables."""
    if not len(tr) or radius <= 0:
        return tr
    means = tr.groupby("particle")[["x", "y"]].mean()
    find, union = _union_find(list(means.index))
    ids = means.index.to_numpy()
    for i, j in cKDTree(means.to_numpy()).query_pairs(radius):
        union(ids[i], ids[j])
    out = tr.copy()
    out["particle"] = [find(p) for p in out.particle]
    return out


def merge_trajectories(tr: pd.DataFrame, link_cfg) -> pd.DataFrame:
    """Join fragments of one vesicle across a gap or across an overlap (see module doc)."""
    if not len(tr):
        return tr
    seg = {}
    for p, g in tr.sort_values("frame").groupby("particle"):
        f = g.frame.to_numpy(np.int64)
        x = g.x.to_numpy(float)
        y = g.y.to_numpy(float)
        dur = max(int(f[-1] - f[0]), 1)
        seg[p] = dict(f=f, x=x, y=y, f0=int(f[0]), f1=int(f[-1]),
                      sx=x[0], sy=y[0], ex=x[-1], ey=y[-1],
                      vx=(x[-1] - x[0]) / dur, vy=(y[-1] - y[0]) / dur)
    order = sorted(seg, key=lambda p: seg[p]["f0"])
    find, union = _union_find(order)
    gap_max, gap_r = link_cfg.merge_gap_frames, link_cfg.merge_gap_radius_px
    ovl_min, ovl_r = link_cfg.merge_overlap_frames, link_cfg.merge_overlap_radius_px
    for i, a in enumerate(order):
        A = seg[a]
        for b in order[i + 1:]:
            B = seg[b]
            gap = B["f0"] - A["f1"]
            if gap > gap_max:
                break                               # sorted by start: nothing later can qualify
            span_ovl = min(A["f1"], B["f1"]) - max(A["f0"], B["f0"])
            merged = False
            if span_ovl >= ovl_min:
                md = _shared_median_distance(A, B)
                merged = md is not None and md <= ovl_r
            if not merged and -ovl_min <= gap:
                d = np.hypot(A["ex"] - B["sx"], A["ey"] - B["sy"])
                dp = np.hypot(A["ex"] + A["vx"] * max(gap, 0) - B["sx"],
                              A["ey"] + A["vy"] * max(gap, 0) - B["sy"])
                merged = min(d, dp) <= gap_r
            if merged:
                union(a, b)
    out = tr.copy()
    out["particle"] = [find(p) for p in out.particle]
    return out


def _shared_median_distance(A: dict, B: dict):
    common = np.intersect1d(A["f"], B["f"])
    if not len(common):
        return None
    ia = np.searchsorted(A["f"], common)
    ib = np.searchsorted(B["f"], common)
    return float(np.median(np.hypot(A["x"][ia] - B["x"][ib], A["y"][ia] - B["y"][ib])))
