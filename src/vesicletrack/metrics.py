r"""Per-vesicle movement metrics: net, gross, and directed.

The two simple readouts both fail on high-frame-rate data.

  gross  = sum of per-frame step lengths. A stationary vesicle still accumulates
           localisation noise every frame, so over a thousand frames it reports
           hundreds of pixels of "movement". Gross path is dominated by noise.

  net    = |end - start|. A vesicle that runs out and comes back, or makes several
           runs in different directions, scores near zero however far it travelled.

The directed metric averages positions in windows of tau frames before measuring the
path:

    c_k = mean of r_i over window k,     L(tau) = sum_k |c_{k+1} - c_k|

Uncorrelated localisation error averages down as 1/sqrt(tau) while real transport is
unchanged, and reversals faster than tau cancel inside a window. Coarsening can only
shorten a path (triangle inequality), so

    gross = L(dt)  >=  L(tau)  >=  L(T) = net_coarse

holds for every vesicle at every tau. check_ordering() tests it; if it fails,
something upstream is wrong.

Two families of columns, at two timescales:

    directed        = L(tau_directed_frames), the path at the coarse timescale,
                      with directed_measurable and n_tau_windows.
    runs_total      = the run-detection metric, computed on coarse steps at the
                      shorter tau_frames. runs_p, runs_z, runs_excess and runs_null
                      all describe this quantity.

runs_p is the p-value of runs_total, not of directed. Both tau values are written as
columns so the file records which window each metric used.

runs_total splits the coarse steps into directionally persistent runs and sums those
that clear a minimum length. A vesicle making three runs in three directions is
credited for all three; back-and-forth jitter gets nothing.

Significance comes from a per-vesicle null: each coarse step keeps its magnitude and
is given a random heading, which gives an isotropic walk with that vesicle's own
step-size distribution. The directed distance that still accumulates is what run
detection finds by chance. The null randomises direction and leaves step order alone.
Permuting order would leave a smooth run's step vectors all pointing the same way, so
the null would reproduce the observation and the clearest movers would score p ~ 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def coarse(x, y, f, tau: int, occupancy_frac: float):
    """Average positions within tau-frame windows. Returns (cx, cy, ct, n_per_window).

    Windows are `f // tau`, an absolute grid shared by every vesicle in the movie,
    rather than `(f - f[0]) // tau`. Binning relative to each track's own first frame
    makes the result depend on where that track happens to start against the window
    grid: the same simulated trajectory gave different directed values when started
    at frame 0, 1, 2, 3 or 4.

    Windows holding fewer than `round(tau * occupancy_frac)` observed frames (at
    least 1) are dropped. A window with one observed frame contributes a raw position
    with full localisation noise, while a full window contributes a mean with noise
    reduced by 1/sqrt(tau), so keeping sparse windows makes the metric follow
    detection dropout rather than motion: on stationary simulated vesicles, directed
    rose from 0.117 to 0.164 as the observed fraction fell from 1.00 to 0.42.
    `n_per_window` is returned so callers can see what was kept.

    `occupancy_frac` is `metrics.min_window_occupancy` and has no default here, so
    that scoring, rendering and tests all coarse-grain with the same floor. The
    shipped value is 0.25. A floor of 0.5 was too strict for real data: vesicles here
    are only about 42% observed, so a 90-frame window holds roughly 38 detections,
    and a floor of 45 dropped windows from good tracks and took the fraction of
    tracks with a measurable directed value from 18% to 8%. 0.25 is below the real
    observation rate but still excludes one- and two-frame windows.
    """
    f = np.asarray(f, dtype=np.int64)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    min_occupancy = max(1, int(round(tau * occupancy_frac)))

    b = f // tau                                    # absolute grid
    uniq, inv = np.unique(b, return_inverse=True)
    counts = np.bincount(inv)
    cx = np.bincount(inv, weights=x) / counts
    cy = np.bincount(inv, weights=y) / counts
    ct = np.bincount(inv, weights=f.astype(float)) / counts

    keep = counts >= min_occupancy
    return cx[keep], cy[keep], ct[keep], counts[keep]


def path_at_tau(x, y, f, tau: int, occupancy_frac: float) -> float:
    """L(tau): path length measured at timescale tau. The directed distance.

    Returns NaN, not 0.0, when the track spans fewer than two tau-windows. A track
    shorter than tau gives a single centroid, so no path can be formed. Returning 0.0
    would be indistinguishable from a vesicle that did not move: a vesicle travelling
    20 px in a straight line over 60 frames would report directed = 0.00 at tau = 90
    while net says 20.00, and net_coarse would be 0.0 too, so check_ordering would
    pass. NaN keeps such tracks out of every median and marks them as unmeasured.
    """
    cx, cy, _, _ = coarse(x, y, f, tau, occupancy_frac)
    return _path(cx, cy)


def _path(cx, cy) -> float:
    return float(np.hypot(np.diff(cx), np.diff(cy)).sum()) if len(cx) >= 2 else float("nan")


def runs(steps: np.ndarray, max_turn_deg: float, min_disp: float, min_steps: int):
    """Split step vectors into directionally persistent runs.

    A run continues while the next step turns less than max_turn from the run's current
    heading. Returns (total displacement, n runs kept, longest run, steps inside runs).

    `steps` is (n, 2) for one track, or (P, n, 2) for P tracks with the same number
    of steps. The permutation null has that shape, so it is scored in one pass down
    the step axis instead of P separate Python loops. With a batch input each return
    value is an array of length P. The two forms give identical answers, since the
    per-track arithmetic is the same additions in the same order.

    min_steps must be >= 2. Allowing one-step runs makes the total converge on plain
    path length for real and random tracks alike, and the metric loses its contrast
    against the null.
    """
    S = np.asarray(steps, float)
    single = S.ndim == 2
    if single:
        S = S[None]
    P, n, _ = S.shape
    total = np.zeros(P); longest = np.zeros(P)
    nkept = np.zeros(P, int); nsteps = np.zeros(P, int)
    if n == 0:
        return ((total[0], int(nkept[0]), longest[0], int(nsteps[0])) if single
                else (total, nkept, longest, nsteps))
    cosmax = np.cos(np.deg2rad(max_turn_deg))
    acc = S[:, 0].copy()                    # the run being accumulated, per track
    start = np.zeros(P, int)                # index of the step that opened it

    def close(j, which):
        """End the current run at step j for the tracks in `which`; credit if it qualifies."""
        d = np.hypot(acc[:, 0], acc[:, 1])
        keep = which & (d >= min_disp) & ((j - start) >= min_steps)
        total[keep] += d[keep]
        nkept[keep] += 1
        nsteps[keep] += (j - start)[keep]
        longest[keep] = np.maximum(longest[keep], d[keep])

    for j in range(1, n):
        s = S[:, j]
        h = np.hypot(acc[:, 0], acc[:, 1]); m = np.hypot(s[:, 0], s[:, 1])
        ok = (h >= 1e-9) & (m >= 1e-9)
        cos = np.full(P, -np.inf)
        np.divide(acc[:, 0] * s[:, 0] + acc[:, 1] * s[:, 1], h * m, out=cos, where=ok)
        cont = ok & (cos >= cosmax)
        close(j, ~cont)
        acc[~cont] = s[~cont]; start[~cont] = j
        acc[cont] += s[cont]
    close(n, np.ones(P, bool))
    if single:
        return float(total[0]), int(nkept[0]), float(longest[0]), int(nsteps[0])
    return total, nkept, longest, nsteps


def _null_runs(steps, rng, nperm, max_turn_deg, min_disp, min_steps):
    """Direction-randomised null: every step keeps its length, gets a random heading."""
    mags = np.hypot(steps[:, 0], steps[:, 1])
    th = rng.uniform(0, 2 * np.pi, (nperm, len(mags)))
    rnd = np.stack([mags * np.cos(th), mags * np.sin(th)], axis=2)      # (P, n, 2)
    return runs(rnd, max_turn_deg, min_disp, min_steps)[0]


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

    # A non-finite coordinate marks the track invalid rather than scoring it. Scored,
    # it would give gross and directed NaN but runs_total 0.0 and p 1.0, so the
    # vesicle would be labelled `confined` on the basis of a corrupt track, and
    # check_ordering would not see it.
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.all():
        return dict(frame_start=int(f[0]), frame_end=int(f[-1]),
                    n_nonfinite=int((~finite).sum()), invalid=True,
                    duration_s=float((f[-1] - f[0]) * cfg.dt_seconds) or cfg.dt_seconds,
                    span_frames=int(f[-1] - f[0] + 1), observed_frames=int(len(f)),
                    directed_measurable=False)
    dur = float((f[-1] - f[0]) * cfg.dt_seconds) or cfg.dt_seconds

    # Lifetime. Span and observed frames differ whenever the linker bridged a
    # dropout, and on real data by a lot (median span 398 frames vs 167 observed).
    # Both are reported, since span alone describes a vesicle as present in frames
    # where nothing was detected.
    span = int(f[-1] - f[0] + 1)
    gaps = np.diff(f) - 1
    gaps = gaps[gaps > 0]

    out = {
        "frame_start": int(f[0]), "frame_end": int(f[-1]),
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
    }
    # The coarse timescale, once: `directed` and `net_coarse` are two readings of the
    # same coarse-grained path, so they must come from the same centroids.
    dcx, dcy, _, _ = coarse(x, y, f, m.tau_directed_frames, m.min_window_occupancy)
    out["directed"] = _path(dcx, dcy)
    out["net_coarse"] = (float(np.hypot(dcx[-1] - dcx[0], dcy[-1] - dcy[0]))
                         if len(dcx) >= 2 else float("nan"))
    # Explicit flag so "not measurable at this tau" is a filterable state rather than
    # something a reader has to infer from a NaN.
    out["directed_measurable"] = bool(len(dcx) >= 2)
    out["n_tau_windows"] = int(len(dcx))
    out["tau_frames"] = int(m.tau_frames)
    out["tau_directed_frames"] = int(m.tau_directed_frames)

    cx, cy, _, _ = coarse(x, y, f, m.tau_frames, m.min_window_occupancy)
    nan_keys = ["coarse_path", "runs_total", "runs_null", "runs_excess",
                "runs_z", "runs_p", "n_runs", "longest_run", "frac_in_runs",
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
        out.update(runs_total=total, n_runs=nkept, longest_run=longest,
                   frac_in_runs=nsteps / len(steps))

        if m.n_permutations > 0:
            null = _null_runs(steps, rng, m.n_permutations, m.max_turn_deg,
                              m.min_run_disp_px, m.min_run_steps)
            sd = null.std(ddof=1)
            out["runs_null"] = float(null.mean())
            out["runs_excess"] = float(total - null.mean())
            # A null with no spread means every permutation scored the same and the
            # observation beat all of them, which is the strongest possible evidence.
            # z = 0.0 there would read as no evidence (a steady drift gets
            # runs_p = 0.005 and would get z = 0.0), so +inf is reported instead.
            if not np.isfinite(sd):
                out["runs_z"] = float("nan")            # fewer than 2 permutations
            elif sd > 1e-9:
                out["runs_z"] = float((total - null.mean()) / sd)
            elif total > null.mean():
                out["runs_z"] = float("inf")
            else:
                out["runs_z"] = 0.0
            out["runs_p"] = float((np.sum(null >= total) + 1) /
                                  (m.n_permutations + 1))
        else:
            out.update(runs_null=np.nan, runs_excess=np.nan,
                       runs_z=np.nan, runs_p=np.nan)

        ok = mags > 1e-9
        if ok.sum() >= 2:
            u = steps[ok] / mags[ok, None]
            out["persistence"] = float(np.mean(np.sum(u[:-1] * u[1:], axis=1)))
        else:
            out["persistence"] = np.nan

        d2 = (cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2
        out["max_excursion"] = float(np.sqrt(d2.max()))
        out["rg"], out["aniso"] = gyration(cx, cy)

    # Censoring. A vesicle present in frame 0 or in the last frame has a lifetime
    # whose start or end was not observed. Pooling those with complete observations
    # biases mean lifetime downward, because the truncated value is always shorter
    # than the truth, so they are flagged.
    if n_movie_frames is not None:
        out["censored_start"] = bool(f[0] <= 0)
        out["censored_end"] = bool(f[-1] >= n_movie_frames - 1)
        out["is_censored"] = bool(out["censored_start"] or out["censored_end"])
    else:
        out["censored_start"] = out["censored_end"] = out["is_censored"] = False

    out["invalid"] = False
    out["n_nonfinite"] = 0
    st = np.hypot(np.diff(x), np.diff(y))
    out["max_step"] = float(st.max()) if len(st) else 0.0
    out["n_big_steps"] = int((st > cfg.classify.big_step_px).sum())
    for k in ("net", "gross", "directed"):
        out[f"{k}_rate"] = out[k] / dur
    return out


EMPTY_SCHEMA = [
    "particle", "klass", "observed_frames", "span_frames", "observed_s", "span_s",
    "frac_observed", "is_censored", "net", "gross", "directed",
    "directed_measurable", "runs_total", "runs_p", "frame_start",
    "frame_end", "duration_s", "missing_frames", "n_gaps", "longest_gap",
    "net_coarse", "coarse_path", "runs_null", "runs_excess", "runs_z", "n_runs",
    "longest_run", "frac_in_runs", "persistence", "max_excursion", "rg", "aniso",
    "max_step", "n_big_steps", "net_rate", "gross_rate", "directed_rate",
    "censored_start", "censored_end", "invalid", "n_nonfinite",
    "n_tau_windows", "tau_frames", "tau_directed_frames", "x0", "y0",
]


def score_tracks(tracks: pd.DataFrame, cfg,
                 n_movie_frames: int | None = None) -> pd.DataFrame:
    """One row per vesicle, with metrics, unit conversions and a class label.

    Every tracked vesicle gets a row. Short, gappy and suspect tracks are all scored
    and labelled so they can be filtered downstream rather than dropped here.
    """
    rows = []
    dup = tracks.duplicated(subset=["particle", "frame"]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate (particle, frame) row(s) in the track table. Duplicates "
            "inflate gross path and max_step and flip vesicles to `excluded`; they "
            "indicate a merge that did not average coincident frames.")
    for p, d in tracks.groupby("particle", sort=True):
        d = d.sort_values("frame")
        # Seed per track. With one shared generator a vesicle's null, and so its
        # p-value and class, would depend on how many vesicles were scored before it,
        # so adding a track to the movie could change another track's class.
        rng = np.random.default_rng([cfg.metrics.random_seed, int(p)])
        rec = metrics_for_track(d.x.values, d.y.values, d.frame.values, cfg, rng,
                                n_movie_frames=n_movie_frames)
        rec["particle"] = int(p)
        rec["x0"], rec["y0"] = float(d.x.iloc[0]), float(d.y.iloc[0])
        rows.append(rec)
    if not rows:
        # An empty frame with the full schema. A bare DataFrame() has no columns, so
        # check_ordering would fail on a movie with no vesicles and pd.concat of
        # per-movie tables would produce all-NaN columns.
        return pd.DataFrame(columns=EMPTY_SCHEMA)
    df = pd.DataFrame(rows)
    # Invalid tracks return a short record, so fill in the full schema before
    # anything downstream indexes into it. Missing entries become NaN.
    for c in EMPTY_SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
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
             "net", "gross", "directed", "directed_measurable",
             "runs_total", "runs_p"]
    return df[front + [c for c in df.columns if c not in front]]


def classify(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Label each vesicle: excluded / mover / confined.

    `excluded` is checked first. It is a tracking-quality label rather than a
    biological one: a single large jump, or repeated large steps, is the signature of
    an identity swap between two nearby vesicles, and leaving those in inflates the
    mover count.
    """
    c = cfg.classify
    df = df.copy()
    def col(name, default):
        """Invalid tracks return a short record, so not every column is guaranteed."""
        return (df[name] if name in df.columns
                else pd.Series(default, index=df.index))

    invalid = col("invalid", False).fillna(False).astype(bool)
    excluded = ((col("max_step", 0.0) > c.exclude_max_step_px) |
                (col("n_big_steps", 0) >= c.exclude_n_big_steps)).fillna(False)
    if c.use_permutation:
        moving = ((col("runs_p", np.nan) <= c.p_threshold)
                  & (col("net", np.nan) >= c.mover_min_net_px)).fillna(False)
    else:
        moving = ((col("net_rate", np.nan) >= c.mover_rate_px_per_s)
                  & (col("net", np.nan) >= c.mover_min_net_px)).fillna(False)
    df["klass"] = np.where(invalid, "invalid",
                           np.where(excluded, "excluded",
                                    np.where(moving, "mover", "confined")))
    return df


def check_ordering(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Rows violating gross >= directed >= net_coarse. Should be empty.

    Compared against net_coarse rather than net; see the module docstring. A non-empty
    result means coarse-graining lengthened a path, which is geometrically impossible,
    so the cause is upstream: duplicated frames, unsorted tracks, or NaNs.
    """
    if not len(df) or "gross" not in df.columns:
        return pd.DataFrame(columns=["particle", "gross", "directed",
                                     "net_coarse", "net"])
    d = df.dropna(subset=["gross", "directed", "net_coarse"])
    bad = d[(d.gross + tol < d.directed) | (d.directed + tol < d.net_coarse)]
    return bad[["particle", "gross", "directed", "net_coarse", "net"]]
