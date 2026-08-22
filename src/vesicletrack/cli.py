"""Command line entry point:  vesicletrack run movie.tif -c config/default.yaml"""
from __future__ import annotations

import argparse
import glob
import sys
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
    try:
        cfg = Config.load(a.config, **over)
    except (ValueError, OSError) as e:
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

    df = analyse_many(files, cfg, output_dir=a.output, verbose=not a.quiet)
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
