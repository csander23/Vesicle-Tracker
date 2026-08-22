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

import inspect
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter


@lru_cache(maxsize=1)
def _uses_roundness_range() -> bool:
    """photutils >= 3.0 replaced roundlo/roundhi with roundness_range=(lo, hi).

    Detected from the signature rather than from a version string, so this keeps
    working whenever the change actually lands rather than when we guess it did.
    """
    return "roundness_range" in inspect.signature(DAOStarFinder).parameters


def _roundness_kwargs(r: float) -> dict:
    return ({"roundness_range": (-r, r)} if _uses_roundness_range()
            else {"roundlo": -r, "roundhi": r})


def _col(tbl, *names):
    """First column name that exists.

    photutils 3.0 renamed xcentroid -> x_centroid and will drop the old names in 4.0.
    Accepting either keeps this working across that break instead of emitting a
    deprecation warning per detection now and failing outright later.
    """
    for n in names:
        if n in tbl.colnames:
            return n
    raise KeyError(f"none of {names} in detection table; got {tbl.colnames}")


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
    # Estimate the noise on real pixels only. On a movie that has been ROI-masked to
    # zero outside the cell, the zeros dominate the sigma-clipped statistics, std
    # collapses toward 0 and the threshold with it - or the median shifts and most
    # detections vanish. Either way the count depends on how much of the frame was
    # blanked, which is not a property of the sample.
    finite = np.isfinite(a)
    zero_frac = float((a == 0).mean())
    sample = a[finite & (a != 0)] if zero_frac > 0.10 else a[finite]
    if sample.size < 64:
        return pd.DataFrame(columns=["x", "y", "flux"])
    mean, median, std = sigma_clipped_stats(sample, sigma=3.0)
    if std <= 0 or not np.isfinite(std):
        return pd.DataFrame(columns=["x", "y", "flux"])
    finder = DAOStarFinder(fwhm=d.psf_sigma_px * gaussian_sigma_to_fwhm,
                           threshold=d.threshold_sigma * std,
                           **_roundness_kwargs(d.roundness))
    # photutils warns when a frame yields nothing, or nothing passes the shape cuts.
    # Both are normal (a blank frame, a frame between blinks) and are handled below,
    # so the warning is noise that would read as an error to a user.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = finder(a - median)
    if tbl is None or len(tbl) == 0:
        return pd.DataFrame(columns=["x", "y", "flux"])
    x = np.asarray(tbl[_col(tbl, "x_centroid", "xcentroid")], float)
    y = np.asarray(tbl[_col(tbl, "y_centroid", "ycentroid")], float)
    flux = np.asarray(tbl[_col(tbl, "flux")], float)
    keep = _nms(x, y, flux, d.min_separation_px)
    return pd.DataFrame({"x": x[keep], "y": y[keep], "flux": flux[keep]})


def detect_stack(stack: np.ndarray, cfg, mask: np.ndarray | None = None,
                 progress=None) -> pd.DataFrame:
    """Detect in every frame. Returns columns frame, x, y, flux.

    mask, if given, is a boolean (Y, X) array; detections outside it are dropped, which
    is how a cell outline or a soma exclusion is applied.

    THE MASK MUST BE IN DRIFT-CORRECTED COORDINATES. This function runs after
    preprocess.correct_drift, so the stack has been aligned to the median of the first
    `drift.reference_frames` frames. A mask drawn on the RAW movie is offset by the
    drift and will clip the wrong pixels - silently, because a slightly wrong mask
    still returns plausible detections. Draw it on `Result.stack[0]`, or on a
    projection of the corrected stack, not on the original file.
    """
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != stack.shape[1:]:
            raise ValueError(
                f"mask shape {mask.shape} does not match the image {stack.shape[1:]}. "
                "A mismatched mask silently labels the wrong pixels, so this is "
                "refused rather than clipped into range.")
        if mask.dtype != bool:
            mask = mask != 0
    rows = []
    it = range(len(stack))
    if progress is not None:
        it = progress(it)
    for t in it:
        df = detect_frame(stack[t], cfg)
        if mask is not None and len(df):
            yi = np.clip(df.y.round().astype(int), 0, mask.shape[0] - 1)
            xi = np.clip(df.x.round().astype(int), 0, mask.shape[1] - 1)
            df = df[np.asarray(mask[yi, xi], dtype=bool)]
        if len(df):
            df = df.assign(frame=t)
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["frame", "x", "y", "flux"])
    return pd.concat(rows, ignore_index=True)[["frame", "x", "y", "flux"]]
