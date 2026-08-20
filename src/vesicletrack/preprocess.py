"""Stage drift correction.

Sample drift over a minute-long recording moves every vesicle together. Uncorrected it
adds a common directed component to every track, which is exactly what the directed
metric is designed to detect - so drift reads as transport, in every vesicle at once.
Correcting it is not optional for these data.

Registration is done on heavily blurred frames so it locks onto static cell structure
rather than onto the moving vesicles themselves, and the shift series is median
filtered so a single bad frame cannot inject a jump.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


def correct_drift(stack: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (corrected_stack, shifts, span_px).

    shifts is (T, 2) in (dy, dx). span_px is the total excursion - report it, because a
    large value on a dataset you believed was stable means the stage moved and the
    uncorrected metrics would have been wrong.
    """
    if not cfg.drift.enabled:
        return stack, np.zeros((len(stack), 2), np.float32), 0.0

    sig = cfg.drift.smoothing_sigma
    nref = min(cfg.drift.reference_frames, len(stack))
    ref = gaussian_filter(np.median(stack[:nref], axis=0).astype(np.float32), sig)

    shifts = np.zeros((len(stack), 2), np.float32)
    for t, frame in enumerate(stack):
        s, _, _ = phase_cross_correlation(
            ref, gaussian_filter(frame.astype(np.float32), sig),
            upsample_factor=10, normalization=None)
        shifts[t] = s

    k = cfg.drift.median_filter
    if k and k > 1:
        shifts = np.column_stack([median_filter(shifts[:, 0], k),
                                  median_filter(shifts[:, 1], k)])

    span = float(np.hypot(np.ptp(shifts[:, 1]), np.ptp(shifts[:, 0])))
    if span < cfg.drift.min_span_px:
        return stack, shifts, span

    out = np.empty_like(stack)
    for t, frame in enumerate(stack):
        out[t] = nd_shift(frame.astype(np.float32), shifts[t], order=1,
                          mode="nearest").round().astype(stack.dtype)
    return out, shifts, span
