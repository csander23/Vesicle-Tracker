"""Shared fixtures: the synthetic movie with known truth, and a config scaled to it.

The synthetic movie is 400 frames at 20 fps, shorter than the real 60 s recordings, so
the coarse-graining windows are scaled down to match (tau 10 / 40 frames instead of
22 / 90). Everything else is the shipped default.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesicletrack import Config  # noqa: E402

MOVIE = ROOT / "examples" / "synthetic.tif"
N_STATIC, N_MOVER = 24, 6          # examples/make_synthetic.py plants exactly these
N_TOTAL = N_STATIC + N_MOVER


def base_cfg(**kw) -> Config:
    """The shipped defaults, scaled to the synthetic movie, with rendering off."""
    d = {"dt_seconds": 0.05, "metrics.tau_frames": 10,
         "metrics.tau_directed_frames": 40, "link.min_length_frames": 30,
         "metrics.n_permutations": 50,
         "render.three_panel": False, "render.per_vesicle_images": False}
    d.update(kw)
    return Config.load(ROOT / "config" / "default.yaml", **d)


@pytest.fixture(scope="session")
def movie() -> Path:
    if not MOVIE.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic.py"),
                        str(MOVIE)], check=True)
    return MOVIE
