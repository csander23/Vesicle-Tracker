"""Per-frame spot detection with deepBLINK.

deepBLINK (Eichenberger et al., 2021) is a convolutional network trained to find
diffraction-limited spots. This package ships its pretrained `vesicle` model. For each
frame the network returns a grid with one cell per 4x4 pixels; each cell holds the
probability that a spot centre lies in it and the centre's sub-pixel offset within
the cell. A cell above `detect.prob_threshold` is a detection.

The probability maps of the whole movie are kept (float16, about 130 MB for 1350
frames of 512x512) because gap recovery (recover.py) reads them again at a lower
threshold wherever a track has a gap. The network has no memory between frames, so a
vesicle that dims for a few frames simply drops out; knowing it was there just before
and just after is reason enough to accept a weaker response at the predicted position.

Coordinates follow deepblink.data.get_coordinate_list: with cell size s, a cell (r, c)
with offsets (dr, dc) is the point y = (r + dr) s, x = (c + dc) s.

Frames are normalised to zero mean and unit standard deviation, which is deepBLINK's
own normalisation, and padded by reflection to a square whose side is a power of two,
which is what the network expects. Detections that fall in the padding are discarded.

TensorFlow is imported only when a model is loaded, so importing the package does not
load it. The Keras 2 loader is selected before that import because the model file is
a Keras 2 HDF5 model.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CELL = 4                    # image pixels per cell of deepBLINK's output grid
_MODELS = Path(__file__).parent / "models"


def model_path(name_or_path: str) -> Path:
    """A bundled model name ("vesicle") or a path to any deepBLINK .h5 model."""
    p = Path(name_or_path).expanduser()
    if p.suffix == ".h5" and p.exists():
        return p
    bundled = _MODELS / f"{name_or_path}.h5"
    if bundled.exists():
        return bundled
    have = sorted(q.stem for q in _MODELS.glob("*.h5"))
    raise FileNotFoundError(
        f"detect.model={name_or_path!r} is neither a bundled model ({have}) nor an "
        "existing .h5 file")


@lru_cache(maxsize=2)
def load_model(path: str):
    """The Keras model, loaded once per path and kept for the process."""
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from deepblink.io import load_model as _load
    return _load(path)


def padded_size(height: int, width: int) -> int:
    """Side of the square the network sees: the next power of two of the larger side."""
    return 1 << max(height - 1, width - 1).bit_length()


def normalise(frame: np.ndarray) -> np.ndarray:
    f = frame.astype(np.float32)
    sd = float(f.std())
    if sd < 1e-12:                       # a flat frame has no spots; avoid 0/0
        return np.zeros_like(f)
    return (f - f.mean()) / sd


def probability_maps(stack: np.ndarray, cfg, progress=None) -> np.ndarray:
    """deepBLINK's raw output for every frame: (T, G, G, 3) float16.

    Channel 0 is the probability, channels 1 and 2 the row and column offsets.
    """
    model = load_model(str(model_path(cfg.detect.model)))
    T, H, W = stack.shape
    S = padded_size(H, W)
    G = S // CELL
    pad = ((0, S - H), (0, S - W))
    mode = "reflect" if (S - H < H and S - W < W) else "edge"
    batch = max(1, int(cfg.detect.batch_frames))
    maps = np.empty((T, G, G, 3), np.float16)
    starts = range(0, T, batch)
    if progress is not None:
        starts = progress(starts)
    for start in starts:
        idx = slice(start, min(start + batch, T))
        frames = [normalise(f) for f in stack[idx]]
        if S != H or S != W:
            frames = [np.pad(f, pad, mode) for f in frames]
        pred = model.predict(np.stack(frames)[..., None], verbose=0)
        maps[idx] = pred.astype(np.float16)
    return maps


def maps_for(stack: np.ndarray, cfg, name: str, progress=None) -> np.ndarray:
    """Probability maps, read from detect.cache_dir when this exact stack was seen before.

    The network is the slow step and the maps depend only on the corrected stack and
    the model, so a sweep over linking or metric parameters does not need to run it
    again. The cache key is a hash of the stack contents, so a different drift
    correction or a different movie with the same name gets its own file.
    """
    if not cfg.detect.cache_dir:
        return probability_maps(stack, cfg, progress)
    digest = hashlib.blake2b(np.ascontiguousarray(stack), digest_size=12).hexdigest()
    model = model_path(cfg.detect.model).stem
    path = Path(cfg.detect.cache_dir).expanduser() / f"{name}_{model}_{digest}.npy"
    if path.exists():
        return np.load(path)
    maps = probability_maps(stack, cfg, progress)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, maps)
    return maps


def detections_from_maps(maps: np.ndarray, prob_threshold: float, height: int,
                         width: int, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Every grid cell above the threshold, as a spot. Columns frame, x, y, prob.

    mask, if given, is a boolean (Y, X) array; detections outside it are dropped.
    It is applied in the same coordinates as the stack the maps were computed from,
    i.e. after drift correction.
    """
    p = maps[..., 0].astype(np.float32)
    t, r, c = np.nonzero(p > prob_threshold)
    y = (r + maps[t, r, c, 1].astype(np.float32)) * CELL
    x = (c + maps[t, r, c, 2].astype(np.float32)) * CELL
    prob = p[t, r, c]
    keep = (y < height) & (x < width)            # drop spots in the padding
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (height, width):
            raise ValueError(
                f"mask shape {mask.shape} does not match the image {(height, width)}. "
                "A mismatched mask would label the wrong pixels, so it is refused.")
        yi = np.clip(np.rint(y).astype(int), 0, height - 1)
        xi = np.clip(np.rint(x).astype(int), 0, width - 1)
        keep &= (mask != 0)[yi, xi]
    df = pd.DataFrame({"frame": t[keep].astype(np.int64), "x": x[keep].astype(float),
                       "y": y[keep].astype(float), "prob": prob[keep].astype(float)})
    return df.sort_values(["frame", "y", "x"]).reset_index(drop=True)
