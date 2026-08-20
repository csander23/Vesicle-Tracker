"""Command line entry point:  vesicletrack run movie.tif -c config/default.yaml"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .config import Config
from .pipeline import analyse, analyse_many


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vesicletrack",
                                 description="Detect, track and score vesicles.")
    ap.add_argument("inputs", nargs="+", help="movie file(s) or glob(s)")
    ap.add_argument("-c", "--config", default=None, help="YAML config")
    ap.add_argument("-o", "--output", default=None, help="output directory")
    ap.add_argument("--dt", type=float, default=None, help="frame interval, seconds")
    ap.add_argument("--um-per-px", type=float, default=None)
    ap.add_argument("--videos", action="store_true",
                    help="also write per-vesicle videos")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    over = {}
    if a.dt is not None:
        over["dt_seconds"] = a.dt
    if a.um_per_px is not None:
        over["um_per_px"] = a.um_per_px
    if a.videos:
        over["render.per_vesicle_videos"] = True
    cfg = Config.load(a.config, **over)

    files: list[str] = []
    for pat in a.inputs:
        hits = sorted(glob.glob(pat))
        files.extend(hits if hits else [pat])
    files = [f for f in files if Path(f).exists()]
    if not files:
        ap.error("no input files matched")

    df = analyse_many(files, cfg, output_dir=a.output, verbose=not a.quiet)
    out = Path(a.output or cfg.output_dir) / "batch_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if not a.quiet:
        print(f"\n{len(df)} movie(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
