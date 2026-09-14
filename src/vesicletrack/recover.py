"""Gap recovery: look again, more leniently, where a track says a vesicle should be.

deepBLINK scores each frame on its own. When a vesicle dims for a few frames its
probability drops below the detection threshold and the track has a gap, or ends
early, although the vesicle is still there. Lowering the threshold everywhere would
mostly add noise. Lowering it only where a track predicts a position uses the frames
before and after as evidence, which is what a human does when following a spot that
fades.

Two cases, both read from the probability maps kept by detect.py:

  internal gaps   the position is interpolated linearly between the frames on either
                  side of the gap, and the best peak near it is accepted if its
                  probability exceeds `recover.threshold` and it lies within
                  `recover.radius_px` of the prediction
  track ends      the last known position is held and the search continues frame by
                  frame, moving the anchor to each hit, for up to
                  `recover.max_extend_frames` frames or until `recover.max_misses`
                  consecutive frames give nothing

The closest peak within the radius is taken, not the strongest, so a brighter
neighbour cannot capture the track. On a crowded real cell this raised the fraction
of frames in which tracked vesicles were seen from 68% to 95% without introducing
identity swaps at radius 2 px; at 4 px it did.

Recovered points snap to the network's 4 px grid cells, so a stationary vesicle whose
recovered frames alternate between two adjacent cells shows a 4 px back-and-forth
that is not motion. Net and directed distances are unaffected by it; gross path and
per-frame speeds are, which is one reason they are not the primary readouts. The
`recovered` column on every track row says which positions came from this step.
"""
from __future__ import annotations

from math import hypot

import numpy as np
import pandas as pd

from .detect import CELL


def redetect(maps: np.ndarray, t: int, px: float, py: float, threshold: float,
             radius_px: float, search_cells: int):
    """The closest peak to (px, py) in frame t, within radius_px, above threshold."""
    G = maps.shape[1]
    gc, gr = int(round(px / CELL)), int(round(py / CELL))
    best, best_d = None, radius_px
    for r in range(max(gr - search_cells, 0), min(gr + search_cells + 1, G)):
        for c in range(max(gc - search_cells, 0), min(gc + search_cells + 1, G)):
            p = float(maps[t, r, c, 0])
            if p <= threshold:
                continue
            y = (r + float(maps[t, r, c, 1])) * CELL
            x = (c + float(maps[t, r, c, 2])) * CELL
            d = hypot(x - px, y - py)
            if d <= best_d:
                best_d, best = d, (x, y, p)
    return best


def recover(tracks: pd.DataFrame, maps: np.ndarray, cfg, n_frames: int,
            mask: np.ndarray | None = None) -> tuple[pd.DataFrame, int]:
    """Fill internal gaps and extend the ends of every track. Returns (tracks, n_added).

    The returned frame has columns particle, frame, x, y, recovered.
    """
    r = cfg.recover
    rows = []
    n_added = 0

    def inside(x, y):
        if mask is None:
            return True
        yi, xi = int(round(y)), int(round(x))
        return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])

    for p, g in tracks.sort_values("frame").groupby("particle", sort=False):
        f = g.frame.to_numpy(np.int64)
        x = g.x.to_numpy(float)
        y = g.y.to_numpy(float)
        pts = {int(fr): (xx, yy, False) for fr, xx, yy in zip(f, x, y)}
        for a, b in zip(f[:-1], f[1:]):
            if b - a <= 1:
                continue
            xa, ya, _ = pts[int(a)]
            xb, yb, _ = pts[int(b)]
            for t in range(int(a) + 1, int(b)):
                w = (t - a) / (b - a)
                hit = redetect(maps, t, xa + w * (xb - xa), ya + w * (yb - ya),
                               r.threshold, r.radius_px, r.search_cells)
                if hit and inside(hit[0], hit[1]):
                    pts[t] = (hit[0], hit[1], True)
                    n_added += 1
        for direction, anchor in ((-1, int(f[0])), (1, int(f[-1]))):
            ax, ay, _ = pts[anchor]
            misses = 0
            for k in range(1, r.max_extend_frames + 1):
                t = anchor + direction * k
                if t < 0 or t >= n_frames or t in pts:
                    break
                hit = redetect(maps, t, ax, ay, r.threshold, r.radius_px, r.search_cells)
                if hit and inside(hit[0], hit[1]):
                    pts[t] = (hit[0], hit[1], True)
                    ax, ay = hit[0], hit[1]
                    n_added += 1
                    misses = 0
                else:
                    misses += 1
                    if misses >= r.max_misses:
                        break
        rows.extend((p, t, xx, yy, rec) for t, (xx, yy, rec) in pts.items())
    out = pd.DataFrame(rows, columns=["particle", "frame", "x", "y", "recovered"])
    return out.sort_values(["particle", "frame"]).reset_index(drop=True), n_added
