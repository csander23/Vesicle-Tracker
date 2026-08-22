"""Command line entry point:  vesicletrack run movie.tif -c config/default.yaml"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import Config
from .pipeline import analyse


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vesicletrack",
                                 description="Detect, track and score vesicles.")
    ap.add_argument("inputs", nargs="+", help="movie file(s) or glob(s)")
    ap.add_argument("-c", "--config", default=None, help="YAML config")
    ap.add_argument("-o", "--output", default=None, help="output directory")
    ap.add_argument("--dt", type=float, default=None, help="frame interval, seconds")
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--rois", default=None,
                    help="ROI source applied to every movie: ImageJ .roi/.zip, a mask "
                         "or label image, or .npy. Vesicles outside every region are "
                         "KEPT and labelled 'outside', not discarded.")
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
    if a.no_figures:
        over.update({"render.three_panel": False,
                     "render.per_vesicle_images": False,
                     "render.per_vesicle_videos": False,
                     "render.overview_video": False})
    try:
        cfg = Config.load(a.config, **over)
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

    # A sample sheet keeps this lab's filename conventions OUT of the package: the
    # user supplies the mapping from file to metadata, rather than the package
    # guessing it from a naming scheme that is only true here.
    sheet = {}
    if a.sheet:
        sh = pd.read_csv(a.sheet)
        if "file" not in sh.columns:
            ap.error(f"{a.sheet} needs a `file` column")
        for row in sh.to_dict("records"):
            key = Path(row.pop("file")).name
            sheet[key] = {k: v for k, v in row.items() if pd.notna(v)}

    results, rows = [], []
    for f in files:
        meta = sheet.get(Path(f).name, {})
        try:
            r = analyse(f, cfg, rois=a.rois, channel=a.channel,
                        z_project=a.z_project, verbose=not a.quiet, **meta)
            r.save(a.output)
            results.append(r)
            rows.append(r.summary)
        except Exception as e:                                  # noqa: BLE001
            if not a.quiet:
                print(f"FAILED {f}: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(name=Path(f).stem, source=str(f), error=repr(e)))
    df = pd.DataFrame(rows)

    # Every level, in one place, so the CSVs the user actually analyses are produced
    # by the same command that produced the per-movie output.
    if results:
        from . import aggregate
        by = [b.strip() for b in a.by.split(",")] if a.by else None
        w = aggregate.write_all(results, Path(a.output or cfg.output_dir), by=by)
        if not a.quiet:
            print("\naggregated:")
            for k, v in w.items():
                print(f"  {k:20s} {v}")
    out = Path(a.output or cfg.output_dir) / "batch_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_fail = int(df["error"].notna().sum()) if "error" in df else 0
    if not a.quiet:
        print(f"\n{len(df)} movie(s) -> {out}"
              + (f"   ({n_fail} FAILED)" if n_fail else ""))
    # Exit non-zero if anything failed, so a CI job or a shell loop notices.
    if n_fail == len(df) and len(df):
        return 1
    return 3 if (n_fail or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
