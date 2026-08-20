"""vesicletrack - parameterised single-vesicle detection, tracking and movement scoring.

    from vesicletrack import Config, analyse
    cfg = Config.load("config/default.yaml", dt_seconds=0.0446, um_per_px=0.107)
    res = analyse("cell.tif", cfg)
    res.vesicles.head()
    res.save()

See README.md for what the three distance metrics mean and why gross and net alone are
not enough.
"""
from .config import Config
from .pipeline import Result, analyse, analyse_many
from . import detect, io, linking, metrics, preprocess, render

__version__ = "0.1.0"
__all__ = ["Config", "Result", "analyse", "analyse_many",
           "detect", "io", "linking", "metrics", "preprocess", "render"]
