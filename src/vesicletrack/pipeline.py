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
from . import io as _io
from . import linking as _link
from . import metrics as _metrics
from . import preprocess as _pre
from . import render as _render


@dataclass
class Result:
    name: str
    config: Config
    tracks: pd.DataFrame
    vesicles: pd.DataFrame
    summary: dict
    stack: np.ndarray | None = field(default=None, repr=False)

    # ------------------------------------------------------------- summaries
    def counts(self) -> pd.Series:
        if not len(self.vesicles):
            return pd.Series(dtype=int)
        return self.vesicles.klass.value_counts()

    def movers(self) -> pd.DataFrame:
        return self.vesicles[self.vesicles.klass == "mover"]

    def check(self) -> pd.DataFrame:
        """Vesicles violating gross >= directed >= net. Should be empty."""
        return _metrics.check_ordering(self.vesicles)

    # --------------------------------------------------------------- writing
    def save(self, output_dir=None, render: bool | None = None) -> dict:
        cfg = self.config
        out = Path(output_dir or cfg.output_dir) / self.name
        out.mkdir(parents=True, exist_ok=True)
        written: dict = {}

        cfg.save(out / "config_used.yaml")
        written["tracks"] = _io.save_table(self.tracks, out / "tracks.parquet")
        written["vesicles"] = _io.save_table(self.vesicles, out / "vesicles.csv")
        (out / "summary.json").write_text(json.dumps(self.summary, indent=2,
                                                     default=str))
        written["summary"] = out / "summary.json"

        do_render = cfg.render.three_panel if render is None else render
        if do_render and self.stack is not None and len(self.tracks):
            written["three_panel"] = _render.three_panel(
                self.stack, self.tracks, self.vesicles, cfg,
                out / "three_panel.png", title=self.name)
        if len(self.vesicles):
            written["distances"] = _render.distance_summary(
                self.vesicles, cfg, out / "distances.png")

        if self.stack is not None and len(self.vesicles):
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
            mask: np.ndarray | None = None, channel: int | None = None,
            z_project: str | None = None, keep_stack: bool = True,
            progress=None, verbose: bool = True) -> Result:
    """Run the full pipeline on one movie."""
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
        f"(min {cfg.link.min_length_frames} frames)")

    vesicles = _metrics.score_tracks(tracks, cfg)
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
        vesicle_outputs_capped_at=cfg.render.max_vesicle_outputs,
        runtime_s=round(time.time() - t0, 1),
    )
    for c in ("net", "directed", "gross"):
        if len(vesicles):
            summary[f"median_{c}"] = float(vesicles[c].median())
    say(f"  done in {summary['runtime_s']} s")

    return Result(name=name, config=cfg, tracks=tracks, vesicles=vesicles,
                  summary=summary, stack=stack_c if keep_stack else None)


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
