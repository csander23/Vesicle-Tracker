"""Per-frame spot detection.

DAOStarFinder is used rather than a learned detector so the toolkit has no model
weights to ship and no training data to match: threshold and PSF width are the only
things that change between microscopes, and both are in the config.

Two steps beyond a plain call:

  roundness cut   rejects streaks, edges and cosmic rays, which otherwise enter the
                  linker as spurious one-frame spots
  separation NMS  DAOStarFinder will return several peaks on one bright vesicle; left
                  in, the linker splits that vesicle into parallel tracks and the
                  per-cell count inflates
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter


def _nms(x, y, flux, min_sep):
    """Greedy non-maximum suppression: keep the brightest of any cluster."""
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(flux)[::-1]
    keep, taken = [], np.zeros(len(x), bool)
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        d = np.hypot(x - x[i], y - y[i])
        taken |= d < min_sep
        taken[i] = True
    return np.array(sorted(keep), dtype=int)


def detect_frame(img: np.ndarray, cfg) -> pd.DataFrame:
    d = cfg.detect
    a = img.astype(np.float32)
    if d.background_sigma and d.background_sigma > 0:
        a = a - gaussian_filter(a, d.background_sigma)
    mean, median, std = sigma_clipped_stats(a, sigma=3.0)
    if std <= 0 or not np.isfinite(std):
        return pd.DataFrame(columns=["x", "y", "flux"])
    finder = DAOStarFinder(fwhm=d.psf_sigma_px * gaussian_sigma_to_fwhm,
                          threshold=d.threshold_sigma * std,
                          roundlo=-d.roundness, roundhi=d.roundness)
    tbl = finder(a - median)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(columns=["x", "y", "flux"])
    x = np.asarray(tbl["xcentroid"], float)
    y = np.asarray(tbl["ycentroid"], float)
    flux = np.asarray(tbl["flux"], float)
    keep = _nms(x, y, flux, d.min_separation_px)
    return pd.DataFrame({"x": x[keep], "y": y[keep], "flux": flux[keep]})


def detect_stack(stack: np.ndarray, cfg, mask: np.ndarray | None = None,
                 progress=None) -> pd.DataFrame:
    """Detect in every frame. Returns columns frame, x, y, flux.

    mask, if given, is a boolean (Y, X) array; detections outside it are dropped, which
    is how a cell outline or a soma exclusion is applied.
    """
    rows = []
    it = range(len(stack))
    if progress is not None:
        it = progress(it)
    for t in it:
        df = detect_frame(stack[t], cfg)
        if mask is not None and len(df):
            yi = np.clip(df.y.round().astype(int), 0, mask.shape[0] - 1)
            xi = np.clip(df.x.round().astype(int), 0, mask.shape[1] - 1)
            df = df[mask[yi, xi]]
        if len(df):
            df = df.assign(frame=t)
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["frame", "x", "y", "flux"])
    return pd.concat(rows, ignore_index=True)[["frame", "x", "y", "flux"]]
