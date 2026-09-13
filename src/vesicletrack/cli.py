"""Command line entry point:  vesicletrack "data/*.tif" -c config/default.yaml -o output"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import yaml

from .config import Config
from .io import read_sample_sheet
from .pipeline import analyse_many


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vesicletrack",
                                 description="Detect, track and score vesicles.")
    ap.add_argument("inputs", nargs="+", help="movie file(s) or glob(s)")
    ap.add_argument("-c", "--config", default=None, help="YAML (or JSON) config")
    ap.add_argument("-o", "--output", default=None, help="output directory")
    ap.add_argument("--dt", type=float, default=None, help="frame interval, seconds")
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--mask", default=None,
                    help="where to DETECT, applied to every movie: a binary image "
                         "(.tif/.png/.npy) or ImageJ .roi/.zip. Use it to restrict "
                         "the analysis to one traced cell.")
    ap.add_argument("--rois", default=None,
                    help="regions to LABEL, applied to every movie: ImageJ .roi/.zip, "
                         "a mask or label image, or .npy. Vesicles outside every "
                         "region are KEPT and labelled 'outside', not discarded.")
    ap.add_argument("--channel", type=int, default=None,
                    help="channel index, for multi-channel files")
    ap.add_argument("--z-project", choices=["max", "mean"], default=None,
                    help="flatten a z axis")
    ap.add_argument("--sheet", default=None,
                    help="CSV sample sheet: a `file` column plus any metadata columns "
                         "(batch, group, genotype, ...) stamped onto every vesicle row")
    ap.add_argument("--by", default=None,
                    help="comma-separated metadata column(s) to aggregate by, e.g. "
                         "genotype or batch,genotype. Writes per_<key>.csv")
    ap.add_argument("--videos", action="store_true",
                    help="also write per-vesicle videos")
    ap.add_argument("--overview-video", action="store_true",
                    help="also write a whole-field video, tracks coloured by class")
    ap.add_argument("--no-figures", action="store_true",
                    help="tables only; skip every image and video")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    over = {}
    if a.dt is not None:
        over["dt_seconds"] = a.dt
    if a.um_per_px is not None:
        over["um_per_px"] = a.um_per_px
    if a.videos:
        over["render.per_vesicle_videos"] = True
    if a.overview_video:
        over["render.overview_video"] = True
    if a.no_figures:
        over.update({"render.three_panel": False,
                     "render.per_vesicle_images": False,
                     "render.per_vesicle_videos": False,
                     "render.overview_video": False})
    try:
        cfg = Config.load(a.config, **over)
        sheet = read_sample_sheet(a.sheet) if a.sheet else None
    except (ValueError, OSError, yaml.YAMLError) as e:
        # A bad parameter is user error, not a bug: say what is wrong and stop,
        # rather than printing a traceback the user has to read backwards.
        print(f"vesicletrack: configuration error: {e}", file=sys.stderr)
        return 2

    files: list[str] = []
    missing: list[str] = []
    for pat in a.inputs:
        hits = sorted(glob.glob(pat))
        if hits:
            files.extend(hits)
        elif Path(pat).exists():
            files.append(pat)
        else:
            missing.append(pat)
    # A mistyped path must not be silently dropped: reporting success on a subset of
    # what was asked for is how a batch quietly analyses the wrong set of movies.
    if missing:
        print("vesicletrack: no such file(s): " + ", ".join(missing), file=sys.stderr)
        if not files:
            return 2

    by = [b.strip() for b in a.by.split(",")] if a.by else None
    df = analyse_many(files, cfg, a.output, sheet=sheet, by=by, verbose=not a.quiet,
                      rois=a.rois, mask=a.mask, channel=a.channel,
                      z_project=a.z_project)
    n_fail = int(df["error"].notna().sum()) if "error" in df else 0
    if not a.quiet:
        out = Path(a.output or cfg.output_dir) / "batch_summary.csv"
        print(f"\n{len(df)} movie(s) -> {out}"
              + (f"   ({n_fail} FAILED)" if n_fail else ""))
    # Exit non-zero if anything failed, so a CI job or a shell loop notices.
    if n_fail == len(df) and len(df):
        return 1
    return 3 if (n_fail or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
