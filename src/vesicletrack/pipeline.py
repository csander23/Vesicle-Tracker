"""The whole analysis as one call.

    result = analyse("cell.tif", Config.load("config/default.yaml"))

Everything is decided by the config; the only other input is the path. `analyse`
returns a Result holding the tracks, the per-vesicle table and the run summary, and
knows how to write its own outputs.

Order of operations, and why:

  load -> drift-correct -> detect -> link -> score -> classify -> render

Drift correction comes before detection because stage drift moves every vesicle
together and would otherwise be measured as transport in all of them at once.
Classification comes after scoring because the mover test is a comparison against each
vesicle's own permutation null, which needs the metrics first.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from . import detect as _detect
from . import filters as _filters
from . import io as _io
from . import linking as _link
from . import metrics as _metrics
from . import preprocess as _pre
from . import render as _render
from . import roi as _roi
from . import size as _size


def measure_track_sizes(stack: np.ndarray, tracks: pd.DataFrame, cfg) -> pd.DataFrame:
    """Per-track size, measured on a sample of each track's own frames.

    Frames are sampled evenly within each track (cap: size.max_frames) and then grouped
    BY FRAME, so every spot in a frame is measured in one vectorised call rather than
    one call per spot. Sizing all 1350 frames of a long track buys no precision that
    200 evenly spaced frames do not already give.
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
        sz = _size.measure_frame(stack[int(fr)], d.x.values, d.y.values, cfg)
        sz["particle"] = d.particle.values
        per_frame.append(sz)
    allsz = pd.concat(per_frame, ignore_index=True)

    psf = cfg.effective_psf_sigma
    if psf is None:
        psf = _size.estimate_psf_sigma(allsz.sigma_px.values,
                                       cfg.size.psf_from_percentile)
    rows = []
    for p, d in allsz.groupby("particle", sort=True):
        rec = _size.summarise_track(d, cfg)
        rec["particle"] = int(p)
        rows.append(rec)
    out = pd.DataFrame(rows)
    dec, lim = _size.deconvolve(out.sigma_px.values, psf)
    out["sigma_deconv_px"] = dec
    out["at_diffraction_limit"] = lim
    out["psf_sigma_used_px"] = psf
    out["fwhm_px"] = out.sigma_px * 2.3548200450309493
    return out


@dataclass
class Result:
    name: str
    config: Config
    tracks: pd.DataFrame
    vesicles: pd.DataFrame
    summary: dict
    stack: np.ndarray | None = field(default=None, repr=False)
    roi_labels: np.ndarray | None = field(default=None, repr=False)
    roi_names: list = field(default_factory=list)

    # ------------------------------------------------------------- summaries
    def counts(self) -> pd.Series:
        if not len(self.vesicles):
            return pd.Series(dtype=int)
        return self.vesicles.klass.value_counts()

    def movers(self) -> pd.DataFrame:
        return self.vesicles[self.vesicles.klass == "mover"]

    @property
    def filtered(self) -> pd.DataFrame:
        """The subset passing the configured filters. `.vesicles` keeps everything."""
        if "passes_filter" not in self.vesicles:
            return self.vesicles
        return self.vesicles[self.vesicles.passes_filter]

    def check(self) -> pd.DataFrame:
        """Vesicles violating gross >= directed >= net. Should be empty."""
        return _metrics.check_ordering(self.vesicles)

    # --------------------------------------------------------------- writing
    def save(self, output_dir=None, render: bool | None = None) -> dict:
        cfg = self.config
        out = Path(output_dir or cfg.output_dir) / self.name
        # Two movies with the same stem in different folders would otherwise write to
        # the same directory and the second would overwrite the first, silently.
        if out.exists() and (out / "summary.json").exists():
            try:
                prev = json.loads((out / "summary.json").read_text()).get("source")
            except Exception:
                prev = None
            if prev and prev != self.summary.get("source"):
                raise FileExistsError(
                    f"{out} already holds results for a DIFFERENT movie ({prev}). "
                    f"Two inputs share the stem {self.name!r}. Pass name=... to "
                    "analyse(), or use distinct output directories.")
        out.mkdir(parents=True, exist_ok=True)
        # Stale per-vesicle figures from a previous parameter set would otherwise sit
        # beside the new tables and be read as belonging to them.
        for sub in ("vesicles", "vesicle_videos"):
            d = out / sub
            if d.exists():
                for f in d.glob("*"):
                    f.unlink()
        written: dict = {}

        cfg.save(out / "config_used.yaml")
        written["tracks"] = _io.save_table(self.tracks, out / "tracks.parquet")
        # Two lists per video, as asked: everything, and the usable subset. The `all`
        # file is the one to keep - any filter can be re-derived from it later.
        written["vesicles_all"] = _io.save_table(self.vesicles,
                                                 out / "vesicles_all.csv")
        written["vesicles_filtered"] = _io.save_table(self.filtered,
                                                      out / "vesicles_filtered.csv")
        (out / "summary.json").write_text(json.dumps(self.summary, indent=2,
                                                     default=str))
        written["summary"] = out / "summary.json"

        # render=False means NO figures at all. Previously it suppressed only the
        # three-panel and still wrote distances.png plus up to 25 per-vesicle figures.
        any_render = True if render is None else bool(render)
        do_panel = (cfg.render.three_panel if render is None else bool(render))
        if do_panel and self.stack is not None and len(self.tracks):
            written["three_panel"] = _render.three_panel(
                self.stack, self.tracks, self.vesicles, cfg,
                out / "three_panel.png", title=self.name)
        if any_render and len(self.vesicles):
            written["distances"] = _render.distance_summary(
                self.vesicles, cfg, out / "distances.png")

        if any_render and self.stack is not None and len(self.vesicles):
            sel = self._selection()
            if cfg.render.per_vesicle_images:
                d = out / "vesicles"
                for _, row in sel.iterrows():
                    tr = self.tracks[self.tracks.particle == row.particle]
                    _render.vesicle_image(self.stack, tr, row, cfg,
                                          d / f"vesicle_{int(row.particle):04d}.png")
                written["vesicle_images"] = d
            if cfg.render.per_vesicle_videos:
                d = out / "vesicle_videos"
                made = [_render.vesicle_video(
                    self.stack, self.tracks[self.tracks.particle == row.particle],
                    row, cfg, d / f"vesicle_{int(row.particle):04d}.mp4")
                    for _, row in sel.iterrows()]
                if any(m is not None for m in made):
                    written["vesicle_videos"] = d
            if cfg.render.overview_video:
                v = _render.overview_video(self.stack, self.tracks, self.vesicles,
                                           cfg, out / "overview.mp4")
                if v:
                    written["overview_video"] = v
        return written

    def _selection(self) -> pd.DataFrame:
        """Which vesicles get per-vesicle output: movers first, then the longest.

        Capped by render.max_vesicle_outputs so a dense field cannot emit thousands of
        files. The cap is reported in the summary rather than applied silently.
        """
        v = self.vesicles
        order = v.assign(_m=(v.klass == "mover").astype(int)).sort_values(
            ["_m", "directed"], ascending=[False, False])
        return order.head(self.config.render.max_vesicle_outputs)


def analyse(path, config: Config | None = None, *, name: str | None = None,
            rois=None, mask: np.ndarray | None = None, channel: int | None = None,
            z_project: str | None = None, keep_stack: bool = True,
            progress=None, verbose: bool = True, **meta) -> Result:
    """Run the full pipeline on one movie.

    rois    ROI source (see roi.load_rois): ImageJ .roi/.zip, label or mask image,
            array, or {name: polygon}. Vesicles are LABELLED by region; those outside
            every region are kept and labelled "outside", not discarded.
    mask    legacy: a boolean array restricting DETECTION. Prefer `rois`, which keeps
            the outside population instead of deleting it.
    **meta  extra columns to stamp on every row (batch=, group=, genotype=, ...) so
            results from many movies can be pooled without re-parsing filenames.
    """
    cfg = config or Config()
    cfg.validate()
    path = Path(path)
    name = name or path.stem
    t0 = time.time()

    def say(m):
        if verbose:
            print(m, flush=True)

    stack = _io.load_stack(path, channel=channel, z_project=z_project)
    say(f"{name}: {stack.shape[0]} frames, {stack.shape[2]}x{stack.shape[1]} px")

    roi_labels, roi_names = _roi.load_rois(rois, stack.shape[1:])
    if mask is None and roi_names and getattr(cfg, "restrict_detection_to_rois", False):
        mask = roi_labels > 0

    stack_c, shifts, span = _pre.correct_drift(stack, cfg)
    if cfg.drift.enabled:
        say(f"  drift {span:.2f} px" + ("  (below threshold, not applied)"
                                        if span < cfg.drift.min_span_px else ""))

    spots = _detect.detect_stack(stack_c, cfg, mask=mask, progress=progress)
    say(f"  {len(spots)} detections "
        f"({len(spots) / max(1, len(stack_c)):.1f} per frame)")

    tracks = _link.link(spots, cfg)
    n_ves = tracks.particle.nunique() if len(tracks) else 0
    say(f"  {n_ves} vesicles after linking "
        f"(hard floor {cfg.link.min_length_frames} frames)")

    vesicles = _metrics.score_tracks(tracks, cfg, n_movie_frames=len(stack_c))

    if len(vesicles) and cfg.size.enabled:
        sz = measure_track_sizes(stack_c, tracks, cfg)
        if len(sz):
            vesicles = vesicles.merge(sz, on="particle", how="left")
            say(f"  size: median sigma {vesicles.sigma_px.median():.2f} px "
                f"(PSF used {sz.psf_sigma_used_px.iloc[0]:.2f}, "
                f"{int(vesicles.at_diffraction_limit.sum())} unresolved)")

    if len(vesicles):
        ass = _roi.assign_tracks(tracks, roi_labels, roi_names)
        vesicles = vesicles.merge(ass, on="particle", how="left")
        if roi_names:
            say(f"  roi: " + ", ".join(
                f"{k}={v}" for k, v in vesicles.roi.value_counts().items()))

    for k, v in meta.items():
        vesicles[k] = v
    vesicles = _filters.apply_filters(vesicles, cfg)
    fsum = _filters.filter_summary(vesicles)
    say(f"  filters: {fsum['n_pass']}/{fsum['n_all']} pass"
        + (f"  (failed: {fsum['reasons']})" if fsum["reasons"] else ""))
    counts = vesicles.klass.value_counts().to_dict() if len(vesicles) else {}
    say(f"  {counts}")

    bad = _metrics.check_ordering(vesicles) if len(vesicles) else pd.DataFrame()
    if len(bad):
        say(f"  WARNING: {len(bad)} vesicles violate gross >= directed >= net")

    summary = dict(
        name=name, source=str(path), frames=int(stack.shape[0]),
        height=int(stack.shape[1]), width=int(stack.shape[2]),
        dt_seconds=cfg.dt_seconds, um_per_px=cfg.um_per_px,
        duration_s=float(stack.shape[0] * cfg.dt_seconds),
        drift_span_px=span, n_detections=int(len(spots)), n_vesicles=int(n_ves),
        counts=counts, ordering_violations=int(len(bad)),
        filters=fsum if len(vesicles) else {},
        roi_names=roi_names, roi_coverage=_roi.coverage(roi_labels, roi_names),
        censored=int(vesicles.is_censored.sum()) if len(vesicles) else 0,
        vesicle_outputs_capped_at=cfg.render.max_vesicle_outputs,
        runtime_s=round(time.time() - t0, 1),
    )
    for c in ("net", "directed", "gross"):
        if len(vesicles):
            summary[f"median_{c}"] = float(vesicles[c].median())
    say(f"  done in {summary['runtime_s']} s")

    return Result(name=name, config=cfg, tracks=tracks, vesicles=vesicles,
                  summary=summary, stack=stack_c if keep_stack else None,
                  roi_labels=roi_labels, roi_names=roi_names)


def analyse_many(paths, config: Config | None = None, output_dir=None,
                 save: bool = True, verbose: bool = True, **kw) -> pd.DataFrame:
    """Run over several movies; returns one summary row per movie.

    A failure on one movie is recorded and the batch continues - one unreadable file
    should not cost the whole run.
    """
    rows = []
    for p in paths:
        try:
            r = analyse(p, config, verbose=verbose, **kw)
            if save:
                r.save(output_dir)
            rows.append(r.summary)
        except Exception as e:                                  # noqa: BLE001
            if verbose:
                print(f"FAILED {p}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(name=Path(p).stem, source=str(p), error=repr(e)))
    return pd.DataFrame(rows)
