r"""How big is each vesicle?

Measured as the intensity-weighted second moment of a small window around the spot:

    sigma_meas = sqrt( 0.5 * (Sxx + Syy) ),    Sxx = sum w dx^2 / sum w

with w the background-subtracted intensity, negatives clipped to zero. This is chosen
over per-spot Gaussian fitting because it is a closed-form sum: fitting hundreds of
spots across a thousand frames with scipy.curve_fit takes minutes per movie, while the
moment is vectorised over all spots in a frame at once and is within a few percent of
the fitted sigma for well-separated spots.

READ THIS BEFORE USING THE NUMBERS
----------------------------------
A vesicle is smaller than the diffraction limit. What the microscope records is the
PSF, essentially regardless of the vesicle's true size, so `sigma_px` is mostly an
instrument property. Reporting it as "vesicle size" without qualification is wrong.

The honest treatment, and what this module does:

  sigma_px          what was measured. Includes the PSF.
  sigma_deconv_px   sqrt(sigma_meas^2 - sigma_psf^2), the width in excess of the PSF.
                    This is the closest thing to a true size, and it is only meaningful
                    when the object is genuinely resolvable.
  at_diffraction_limit  True when sigma_meas <= sigma_psf, i.e. the subtraction under
                    the square root went negative. sigma_deconv_px is then 0, NOT NaN
                    and NOT a small positive number: the object is unresolved and its
                    size is unknown, bounded above by roughly the PSF width.

If most vesicles come back with at_diffraction_limit True, the correct conclusion is
that this data cannot size them - not that they are all the same size. Use the flag,
do not average over it.

CALIBRATE THE PSF FROM THE DATA, NOT FROM THEORY. The second moment is biased upward
by noise: on synthetic spots generated with sigma exactly 1.30 px it measures 1.394,
and deconvolving against the theoretical 1.30 then reports 0.50 px of "size" for
objects that have none. Estimating sigma_psf as a low percentile of the measured
distribution in the same movie (config: psf_from_percentile) absorbs that bias,
because the smallest objects in the field ARE point sources measured the same way.
Deconvolved size then means "larger than the smallest thing here", which is a claim
the data can actually support.

THERE IS STILL A NOISE FLOOR, measured on this pipeline:

    synthetic point sources, true size ZERO   -> sigma 1.394 px, deconvolved 0.41 px
    real Rab5 endosomes, same settings        -> sigma 2.010 px, deconvolved 0.99 px

A homogeneous population of point sources cannot be distinguished from a population of
slightly-larger-than-point objects, because the low percentile and the median differ by
noise alone. Treat ~0.4 px of deconvolved width as indistinguishable from zero here, and
recalibrate that floor for your own optics by running the pipeline on sub-resolution
beads or on examples/make_synthetic.py.

PREFER sigma_px FOR COMPARISONS. When two conditions are imaged identically the PSF
contribution is common to both, so a difference in raw sigma_px is real and needs no
deconvolution. Deconvolved values are for the harder question of absolute size, and
they carry the floor above.

Comparisons ACROSS conditions imaged identically are still informative even when every
object is unresolved, because the PSF contribution is common: a shift in sigma_px
between genotypes is real even if no single value is a true diameter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def measure_tracks(stack: np.ndarray, tracks: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-track size, measured on a sample of each track's own frames.

    Frames are sampled evenly within each track (cap: size.max_frames) and then grouped
    BY FRAME, so every spot in a frame is measured in one vectorised call rather than
    one call per spot. Sizing all 1350 frames of a long track buys no precision that
    200 evenly spaced frames do not already give.

    The PSF width used for deconvolution is size.psf_sigma_px when set, otherwise the
    `psf_from_percentile` percentile of every width measured in this movie (see the
    module docstring for why the data beat theory here). The value used is written to
    every row as psf_sigma_used_px.
    """
    if not cfg.size.enabled or not len(tracks):
        return pd.DataFrame(columns=["particle"])
    cap = max(1, int(cfg.size.max_frames))
    picks = []
    for p, d in tracks.groupby("particle", sort=False):
        d = d.sort_values("frame")
        idx = (np.linspace(0, len(d) - 1, min(cap, len(d))).round().astype(int)
               if len(d) > cap else np.arange(len(d)))
        picks.append(d.iloc[np.unique(idx)])
    sel = pd.concat(picks, ignore_index=True)

    per_frame = []
    for fr, d in sel.groupby("frame", sort=True):
        sz = measure_frame(stack[int(fr)], d.x.values, d.y.values, cfg)
        sz["particle"] = d.particle.values
        per_frame.append(sz)
    allsz = pd.concat(per_frame, ignore_index=True)

    psf = cfg.size.psf_sigma_px
    if psf is None:
        psf = estimate_psf_sigma(allsz.sigma_px.values, cfg.size.psf_from_percentile)
    rows = []
    for p, d in allsz.groupby("particle", sort=True):
        rec = summarise_track(d, cfg)
        rec["particle"] = int(p)
        rows.append(rec)
    out = pd.DataFrame(rows)
    dec, lim = deconvolve(out.sigma_px.values, psf)
    out["sigma_deconv_px"] = dec
    out["at_diffraction_limit"] = lim
    out["psf_sigma_used_px"] = psf
    out["fwhm_px"] = out.sigma_px * 2.3548200450309493
    return out


def measure_frame(img: np.ndarray, x: np.ndarray, y: np.ndarray, cfg) -> pd.DataFrame:
    """Second-moment width for every spot in one frame. Vectorised over spots.

    Returns columns: sigma_px, flux_window, peak, bg, n_valid_px, edge (bool).
    """
    s = cfg.size
    r = int(s.window_px)
    H, W = img.shape
    n = len(x)
    if n == 0:
        return pd.DataFrame(columns=["sigma_px", "flux_window", "peak", "bg",
                                     "n_valid_px", "edge"])

    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    edge = (xi - r < 0) | (yi - r < 0) | (xi + r >= W) | (yi + r >= H)
    xic = np.clip(xi, r, W - r - 1)
    yic = np.clip(yi, r, H - r - 1)

    d = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")                 # (K, K)
    rows = yic[:, None, None] + dy[None]                      # (n, K, K)
    cols = xic[:, None, None] + dx[None]
    win = img[rows, cols].astype(np.float64)                  # (n, K, K)

    # Background from the outer ring of the window: local, and immune to a neighbouring
    # spot sitting in one corner because it is a median, not a mean.
    ring = (np.abs(dy) == r) | (np.abs(dx) == r)
    bg = np.median(win[:, ring], axis=1)
    peak = win.max(axis=(1, 2))
    w = win - bg[:, None, None]
    np.clip(w, 0, None, out=w)

    # offsets measured from the true (sub-pixel) centroid, not the rounded pixel
    ddx = dx[None] + (xic - x)[:, None, None]
    ddy = dy[None] + (yic - y)[:, None, None]

    m = w.sum(axis=(1, 2))
    ok = m > 0
    sxx = np.full(n, np.nan)
    syy = np.full(n, np.nan)
    sxx[ok] = (w[ok] * ddx[ok] ** 2).sum(axis=(1, 2)) / m[ok]
    syy[ok] = (w[ok] * ddy[ok] ** 2).sum(axis=(1, 2)) / m[ok]
    sigma = np.sqrt(0.5 * (sxx + syy))

    return pd.DataFrame({
        "sigma_px": sigma, "flux_window": m, "peak": peak, "bg": bg,
        "n_valid_px": (w > 0).sum(axis=(1, 2)), "edge": edge,
    })


def estimate_psf_sigma(sigma_meas: np.ndarray, percentile: float = 5.0) -> float:
    """PSF width estimated from the data: a low percentile of measured widths.

    The smallest objects in a field are point sources, and measuring them with the same
    biased estimator as everything else means the bias cancels in the subtraction. This
    is preferable to a theoretical PSF width for exactly that reason.
    """
    v = np.asarray(sigma_meas, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return float("nan")
    return float(np.percentile(v, percentile))


def deconvolve(sigma_meas: np.ndarray, sigma_psf: float):
    """Width in excess of the PSF, and a flag for objects that are unresolved.

    Returns (sigma_deconv, at_limit). Where the object is unresolved the deconvolved
    width is 0 and at_limit is True - it is not a small number and not NaN, because
    "unresolved" is a real, reportable state, not a missing value.
    """
    sigma_meas = np.asarray(sigma_meas, float)
    var = sigma_meas ** 2 - float(sigma_psf) ** 2
    at_limit = ~(var > 0)
    out = np.sqrt(np.where(var > 0, var, 0.0))
    out[np.isnan(sigma_meas)] = np.nan
    at_limit[np.isnan(sigma_meas)] = False
    return out, at_limit


def summarise_track(sizes: pd.DataFrame, cfg) -> dict:
    """Per-track size from its per-frame measurements.

    Median, not mean: a single frame where a neighbouring vesicle drifts into the
    window inflates the mean and leaves the median alone. Frames measured at the image
    edge are excluded - their window is truncated and the width is biased.
    """
    s = sizes[~sizes.edge] if "edge" in sizes else sizes
    v = s.sigma_px.to_numpy(float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(sigma_px=np.nan, sigma_px_sd=np.nan, n_size_frames=0,
                    flux_median=np.nan, peak_median=np.nan)
    return dict(
        sigma_px=float(np.median(v)),
        sigma_px_sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        n_size_frames=int(len(v)),
        flux_median=float(np.median(s.flux_window)),
        peak_median=float(np.median(s.peak)),
    )
