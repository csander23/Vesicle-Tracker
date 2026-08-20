#!/usr/bin/env python
"""Generate a synthetic movie with KNOWN ground truth, for testing and for the demo.

Contains, on a noisy background:
  N_STATIC   vesicles that do not move (they only jitter by localisation noise)
  N_MOVER    vesicles moving at a constant velocity
  optional global drift, so drift correction can be checked

Because the truth is known, this is what to run after changing a parameter: the movers
should come back as movers and the static ones should not.

  python examples/make_synthetic.py [out.tif]
"""
import sys

import numpy as np
import tifffile

T, H, W = 400, 256, 256
N_STATIC, N_MOVER = 40, 6
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


def main(out="examples/synthetic.tif", seed=0):
    rng = np.random.default_rng(seed)
    sx = rng.uniform(20, W - 20, N_STATIC); sy = rng.uniform(20, H - 20, N_STATIC)
    mx = rng.uniform(40, W - 40, N_MOVER);  my = rng.uniform(40, H - 40, N_MOVER)
    ang = rng.uniform(0, 2 * np.pi, N_MOVER)
    vx, vy = MOVER_SPEED * np.cos(ang), MOVER_SPEED * np.sin(ang)

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
          f"({MOVER_SPEED * T:.1f} px total travel), drift {DRIFT} px")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/synthetic.tif")
