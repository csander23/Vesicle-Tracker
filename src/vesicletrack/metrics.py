r"""Per-vesicle movement metrics: net, gross, and directed.

The two obvious readouts both fail on high-frame-rate data, in opposite directions.

  GROSS  = sum of per-frame step lengths. A stationary vesicle still accumulates
           localisation noise every frame, so over a thousand frames it racks up
           hundreds of pixels of "movement". Gross path is dominated by noise.

  NET    = |end - start|. Blind to a vesicle that runs out and comes back, or makes
           several directed runs in different directions. A genuinely motile but
           complex trajectory scores near zero.

DIRECTED sits between them. Positions are averaged in windows of tau frames before
the path is measured:

    c_k = mean of r_i over window k,     L(tau) = sum_k |c_{k+1} - c_k|

Uncorrelated localisation error averages down as 1/sqrt(tau) while real transport is
untouched, and reversals faster than tau cancel inside a window. Because coarsening
can only shorten a path (triangle inequality), the ordering

    gross = L(dt)  >=  L(tau)  >=  L(T) = net

holds for every vesicle at every tau - a useful invariant to assert on new data. If it
fails, something upstream is wrong.

`directed_runs` additionally splits the coarse steps into directionally persistent runs
and sums those that clear a minimum, which credits a vesicle making three runs in three
directions and gives nothing to back-and-forth jiggle.

Significance comes from a PER-VESICLE null: each coarse step keeps its magnitude but is
given a random heading, i.e. an isotropic walk with that vesicle's exact step-size
distribution. Whatever directed distance that still accumulates is what run-detection
finds by chance. Note the null randomises DIRECTION, not order - permuting order leaves
a smooth run's step vectors all pointing the same way, so the null reproduces the
observation and the clearest movers score p ~ 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coarse(x, y, f, tau: int):
    """Average positions within tau-frame windows. Returns (cx, cy, ct_frames)."""
    f = np.asarray(f, dtype=np.int64)
    b = (f - f[0]) // tau
    keep = np.flatnonzero(np.diff(b, prepend=b[0] - 1))
    edges = np.append(keep, len(f))
    cx = np.array([x[a:z].mean() for a, z in zip(edges[:-1], edges[1:])])
    cy = np.array([y[a:z].mean() for a, z in zip(edges[:-1], edges[1:])])
    ct = np.array([f[a:z].mean() for a, z in zip(edges[:-1], edges[1:])])
    return cx, cy, ct


def path_at_tau(x, y, f, tau: int) -> float:
    """L(tau): path length measured at timescale tau. The directed distance.

    Returns NaN - not 0.0 - when the track spans fewer than two tau-windows.

    That distinction is the whole point. A track shorter than tau yields a single
    centroid, so no path can be formed. Returning 0.0 there is indistinguishable from
    a vesicle that genuinely did not move, and it is silently wrong: a vesicle
    travelling 20 px in a straight line over 60 frames reported directed = 0.00 at the
    shipped tau of 90, while net said 20.00. Worse, net_coarse was 0.0 for the same
    reason, so check_ordering saw 20 >= 0 >= 0 and passed. NaN makes the gap visible
    and keeps it out of any median.
    """
    cx, cy, _ = coarse(x, y, f, tau)
    if len(cx) < 2:
        return float("nan")
    return float(np.hypot(np.diff(cx), np.diff(cy)).sum())


def runs(steps: np.ndarray, max_turn_deg: float, min_disp: float, min_steps: int):
    """Split step vectors into directionally persistent runs.

    A run continues while the next step turns less than max_turn from the run's current
    heading. Returns (total displacement, n runs kept, longest run, steps inside runs).

    min_steps >= 2 matters: allowing one-step runs makes the total converge on plain
    path length for real and random tracks alike, and the metric loses its contrast
    against the null.
    """
    n = len(steps)
    if n == 0:
        return 0.0, 0, 0.0, 0
    cosmax = np.cos(np.deg2rad(max_turn_deg))
    total = longest = 0.0
    nkept = nsteps = 0
    i = 0
    while i < n:
        acc = steps[i].copy()
        j = i + 1
        while j < n:
            h, s = np.hypot(*acc), np.hypot(*steps[j])
            if h < 1e-9 or s < 1e-9:
                break
            if float(acc @ steps[j]) / (h * s) < cosmax:
                break
            acc = acc + steps[j]
            j += 1
        d = float(np.hypot(*acc))
        if d >= min_disp and (j - i) >= min_steps:
            total += d
            nkept += 1
            nsteps += j - i
            longest = max(longest, d)
        i = j
    return total, nkept, longest, nsteps


def _null_directed(steps, rng, nperm, max_turn_deg, min_disp, min_steps):
    mags = np.hypot(steps[:, 0], steps[:, 1])
    out = np.empty(nperm)
    for k in range(nperm):
        th = rng.uniform(0, 2 * np.pi, len(mags))
        rnd = np.stack([mags * np.cos(th), mags * np.sin(th)], axis=1)
        out[k] = runs(rnd, max_turn_deg, min_disp, min_steps)[0]
    return out


def gyration(x, y):
    cx, cy = x - x.mean(), y - y.mean()
    t = np.array([[(cx * cx).mean(), (cx * cy).mean()],
                  [(cx * cy).mean(), (cy * cy).mean()]])
    ev = np.sort(np.linalg.eigvalsh(t))[::-1]
    rg = float(np.sqrt(ev.sum()))
    aniso = float((ev[0] - ev[1]) / ev.sum()) if ev.sum() > 0 else 0.0
    return rg, aniso


def metrics_for_track(x, y, f, cfg, rng, n_movie_frames: int | None = None) -> dict:
    m = cfg.metrics
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = np.asarray(f, np.int64)
    dur = float((f[-1] - f[0]) * cfg.dt_seconds) or cfg.dt_seconds

    # LIFETIME. span and observed are different numbers whenever the linker bridged a
    # dropout, and on real data they differ by more than half: median span 398 frames
    # vs 167 observed. Reporting only one of them describes a vesicle as present in
    # frames where nothing was detected, so both are carried through.
    span = int(f[-1] - f[0] + 1)
    gaps = np.diff(f) - 1
    gaps = gaps[gaps > 0]

    out = {
        "n_frames": int(len(f)), "frame_start": int(f[0]), "frame_end": int(f[-1]),
        "duration_s": dur,
        "span_frames": span,
        "observed_frames": int(len(f)),
        "missing_frames": int(span - len(f)),
        "frac_observed": float(len(f) / span),
        "n_gaps": int(len(gaps)),
        "longest_gap": int(gaps.max()) if len(gaps) else 0,
        "span_s": float(span * cfg.dt_seconds),
        "observed_s": float(len(f) * cfg.dt_seconds),
        "net": float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
        "gross": float(np.hypot(np.diff(x), np.diff(y)).sum()),
        "directed": path_at_tau(x, y, f, m.tau_directed_frames),
    }
    dcx, dcy, _ = coarse(x, y, f, m.tau_directed_frames)
    out["net_coarse"] = (float(np.hypot(dcx[-1] - dcx[0], dcy[-1] - dcy[0]))
                         if len(dcx) >= 2 else float("nan"))
    # Explicit flag so "not measurable at this tau" is a filterable state rather than
    # something a reader has to infer from a NaN.
    out["directed_measurable"] = bool(len(dcx) >= 2)
    out["n_tau_windows"] = int(len(dcx))

    cx, cy, _ = coarse(x, y, f, m.tau_frames)
    nan_keys = ["coarse_path", "directed_runs", "directed_null", "directed_excess",
                "directed_z", "directed_p", "n_runs", "longest_run", "frac_in_runs",
                "persistence", "max_excursion", "rg", "aniso"]
    if len(cx) < 3:
        out.update({k: np.nan for k in nan_keys})
        out["n_runs"] = 0
    else:
        steps = np.stack([np.diff(cx), np.diff(cy)], axis=1)
        mags = np.hypot(steps[:, 0], steps[:, 1])
        out["coarse_path"] = float(mags.sum())

        total, nkept, longest, nsteps = runs(
            steps, m.max_turn_deg, m.min_run_disp_px, m.min_run_steps)
        out.update(directed_runs=total, n_runs=nkept, longest_run=longest,
                   frac_in_runs=nsteps / len(steps))

        if m.n_permutations > 0:
            null = _null_directed(steps, rng, m.n_permutations, m.max_turn_deg,
                                  m.min_run_disp_px, m.min_run_steps)
            sd = null.std(ddof=1)
            out["directed_null"] = float(null.mean())
            out["directed_excess"] = float(total - null.mean())
            out["directed_z"] = float((total - null.mean()) / sd) if sd > 1e-9 else 0.0
            out["directed_p"] = float((np.sum(null >= total) + 1) /
                                      (m.n_permutations + 1))
        else:
            out.update(directed_null=np.nan, directed_excess=np.nan,
                       directed_z=np.nan, directed_p=np.nan)

        ok = mags > 1e-9
        if ok.sum() >= 2:
            u = steps[ok] / mags[ok, None]
            out["persistence"] = float(np.mean(np.sum(u[:-1] * u[1:], axis=1)))
        else:
            out["persistence"] = np.nan

        d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
        out["max_excursion"] = float(np.sqrt(d2.max()))
        out["rg"], out["aniso"] = gyration(cx, cy)

    # CENSORING. A vesicle already present in frame 0, or still present in the last
    # frame, has a lifetime we did not observe the end of. Averaging those in with
    # complete observations biases mean lifetime DOWNWARD, because the truncated value
    # is always shorter than the truth. Flag them; do not silently pool them.
    if n_movie_frames is not None:
        out["censored_start"] = bool(f[0] <= 0)
        out["censored_end"] = bool(f[-1] >= n_movie_frames - 1)
        out["is_censored"] = bool(out["censored_start"] or out["censored_end"])
    else:
        out["censored_start"] = out["censored_end"] = out["is_censored"] = False

    st = np.hypot(np.diff(x), np.diff(y))
    out["max_step"] = float(st.max()) if len(st) else 0.0
    out["n_big_steps"] = int((st > cfg.classify.big_step_px).sum())
    for k in ("net", "gross", "directed"):
        out[f"{k}_rate"] = out[k] / dur
    return out


def score_tracks(tracks: pd.DataFrame, cfg,
                 n_movie_frames: int | None = None) -> pd.DataFrame:
    """One row per vesicle, with metrics, unit conversions and a class label.

    EVERY tracked vesicle gets a row. Nothing is dropped here - short, gappy and
    suspect tracks are all scored and labelled so they can be filtered downstream on
    evidence rather than disappearing upstream on a threshold.
    """
    rng = np.random.default_rng(cfg.metrics.random_seed)
    rows = []
    for p, d in tracks.groupby("particle", sort=True):
        d = d.sort_values("frame")
        rec = metrics_for_track(d.x.values, d.y.values, d.frame.values, cfg, rng,
                                n_movie_frames=n_movie_frames)
        rec["particle"] = int(p)
        rec["x0"], rec["y0"] = float(d.x.iloc[0]), float(d.y.iloc[0])
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = classify(df, cfg)
    if cfg.um_per_px:
        for c in ("net", "net_coarse", "gross", "directed", "coarse_path",
                  "max_excursion", "rg"):
            df[f"{c}_um"] = df[c] * cfg.um_per_px
        for c in ("sigma_px", "sigma_deconv_px"):
            if c in df.columns:
                df[c.replace("_px", "_um")] = df[c] * cfg.um_per_px
        for c in ("net_rate", "gross_rate", "directed_rate"):
            df[f"{c}_um_s"] = df[c] * cfg.um_per_px
    front = ["particle", "klass", "observed_frames", "span_frames", "observed_s",
             "span_s", "frac_observed", "is_censored",
             "net", "gross", "directed", "directed_p"]
    return df[front + [c for c in df.columns if c not in front]]


def classify(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Label each vesicle: excluded / mover / confined.

    `excluded` is checked first and is a tracking-quality judgement, not biology: a
    single large jump, or repeated large steps, is the signature of an identity swap
    between two nearby vesicles. Leaving those in inflates the mover count.
    """
    c = cfg.classify
    df = df.copy()
    excluded = (df.max_step > c.exclude_max_step_px) | \
               (df.n_big_steps >= c.exclude_n_big_steps)
    if c.use_permutation:
        moving = (df.directed_p <= c.p_threshold) & (df.net >= c.mover_min_net_px)
        moving = moving.fillna(False)
    else:
        moving = (df.net_rate >= c.mover_rate_px_per_s) & (df.net >= c.mover_min_net_px)
    df["klass"] = np.where(excluded, "excluded",
                           np.where(moving.fillna(False), "mover", "confined"))
    return df


def check_ordering(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Rows violating gross >= directed >= net_coarse. Should be empty.

    Compared against net_coarse, not net - see the module docstring. A non-empty result
    means coarse-graining lengthened a path, which is geometrically impossible, so the
    cause is upstream: duplicated frames, unsorted tracks, or NaNs.
    """
    d = df.dropna(subset=["gross", "directed", "net_coarse"])
    bad = d[(d.gross + tol < d.directed) | (d.directed + tol < d.net_coarse)]
    return bad[["particle", "gross", "directed", "net_coarse", "net"]]
