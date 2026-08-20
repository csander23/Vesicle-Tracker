#!/usr/bin/env python
"""Generate a synthetic movie with KNOWN ground truth, for testing and for the demo.

Contains, on a noisy background:
  N_STATIC   vesicles that do not move (they only jitter by localisation noise)
  N_MOVER    vesicles moving at a constant velocity
  optional global drift, so drift correction can be checked

Positions are rejection-sampled so that no two vesicles come within MIN_SEP of each
other at ANY frame - a mover's whole path is checked, not just its start. Without that
guarantee a mover passing close to a static vesicle causes an identity swap in the
linker, and the fixture stops being a fixture: the "correct" answer changes with the
field size.

Because the truth is known, this is what to run after changing a parameter: the movers
should come back as movers and the static ones should not.

  python examples/make_synthetic.py [out.tif]
"""
import sys

import numpy as np
import tifffile

T, H, W = 400, 160, 160
N_STATIC, N_MOVER = 24, 6
MIN_SEP = 14.0           # px between any two vesicles, over the WHOLE movie
PSF, AMP, BG, NOISE = 1.3, 900.0, 300.0, 25.0
JITTER = 0.25            # px, localisation-scale wobble on every spot
MOVER_SPEED = 0.035      # px/frame -> 14 px over the movie
DRIFT = 3.0              # px total, linear, applied to everything


def render(shape, xs, ys, amp, psf):
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    img = np.zeros(shape, np.float32)
    for cx, cy in zip(xs, ys):
        img += amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * psf ** 2))
    return img


def _paths(x, y, vx=0.0, vy=0.0):
    """Sample a vesicle's position at a few times spanning the movie."""
    ts = np.linspace(0, T - 1, 9)
    return np.array([x + vx * ts, y + vy * ts])          # (2, 9)


def _place(rng, tries=20000):
    """Rejection-sample positions so no two vesicles approach within MIN_SEP."""
    placed = []            # list of (2, 9) sampled paths
    sx, sy, mx, my, vx, vy = [], [], [], [], [], []

    def clear(cand):
        for p in placed:
            d = np.hypot(cand[0][:, None] - p[0][None, :],
                         cand[1][:, None] - p[1][None, :])
            if d.min() < MIN_SEP:
                return False
        return True

    for _ in range(tries):
        if len(mx) < N_MOVER:
            a = rng.uniform(0, 2 * np.pi)
            u, w = MOVER_SPEED * np.cos(a), MOVER_SPEED * np.sin(a)
            span = MOVER_SPEED * T
            x = rng.uniform(16 + abs(u) * T * (u < 0) * -1 + max(0, -u * T),
                            W - 16 - max(0, u * T))
            y = rng.uniform(16 + max(0, -w * T), H - 16 - max(0, w * T))
            cand = _paths(x, y, u, w)
            if cand[0].min() < 12 or cand[0].max() > W - 12: continue
            if cand[1].min() < 12 or cand[1].max() > H - 12: continue
            if clear(cand):
                placed.append(cand); mx.append(x); my.append(y)
                vx.append(u); vy.append(w)
        elif len(sx) < N_STATIC:
            x, y = rng.uniform(12, W - 12), rng.uniform(12, H - 12)
            cand = _paths(x, y)
            if clear(cand):
                placed.append(cand); sx.append(x); sy.append(y)
        else:
            break
    if len(mx) < N_MOVER or len(sx) < N_STATIC:
        raise RuntimeError(f"could not place {N_STATIC}+{N_MOVER} vesicles at "
                           f"MIN_SEP={MIN_SEP} in {W}x{H}; lower the counts or "
                           "enlarge the field")
    return (np.array(sx), np.array(sy), np.array(mx), np.array(my),
            np.array(vx), np.array(vy))


def main(out="examples/synthetic.tif", seed=0):
    rng = np.random.default_rng(seed)
    sx, sy, mx, my, vx, vy = _place(rng)

    stack = np.empty((T, H, W), np.uint16)
    truth = []
    for t in range(T):
        dx = DRIFT * t / T
        X = np.concatenate([sx, mx + vx * t]) + dx + rng.normal(0, JITTER, N_STATIC + N_MOVER)
        Y = np.concatenate([sy, my + vy * t]) + rng.normal(0, JITTER, N_STATIC + N_MOVER)
        img = BG + render((H, W), X, Y, AMP, PSF)
        img = rng.normal(img, NOISE)
        stack[t] = np.clip(img, 0, 65535).astype(np.uint16)
        if t == 0:
            truth = ([("static", a, b) for a, b in zip(sx, sy)] +
                     [("mover", a, b) for a, b in zip(mx, my)])
    tifffile.imwrite(out, stack)
    print(f"wrote {out}  {stack.shape}  "
          f"{N_STATIC} static + {N_MOVER} movers "
          f"({MOVER_SPEED * T:.1f} px total travel), drift {DRIFT} px, "
          f"min separation {MIN_SEP} px")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/synthetic.tif")
