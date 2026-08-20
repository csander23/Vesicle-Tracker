"""Reading image stacks and writing results.

Input is a 2D time series: shape (T, Y, X). TIFF is read with tifffile; ND2 is read
with nd2 or aicsimageio if either is installed. A stack with a channel or z axis must
be reduced before it gets here - `load_stack` will say so rather than guess which axis
is time, because guessing wrong silently produces tracks of nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_stack(path: str | Path, channel: int | None = None,
               z_project: str | None = None) -> np.ndarray:
    """Return a (T, Y, X) array.

    channel    index to take when the file has a channel axis
    z_project  "max" | "mean" to flatten a z axis; None requires there to be none
    """
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(str(path))
    elif suf == ".nd2":
        arr = _load_nd2(path)
    elif suf in (".npy",):
        arr = np.load(path)
    else:
        raise ValueError(f"unsupported file type {suf!r}; expected .tif/.tiff/.nd2/.npy")

    arr = np.asarray(arr)
    if arr.ndim == 5:                                  # T,C,Z,Y,X
        arr = arr[:, channel if channel is not None else 0]
    if arr.ndim == 4:                                  # T,C,Y,X or T,Z,Y,X
        if channel is not None:
            arr = arr[:, channel]
        elif z_project == "max":
            arr = arr.max(axis=1)
        elif z_project == "mean":
            arr = arr.mean(axis=1)
        else:
            raise ValueError(
                f"{path.name} has shape {arr.shape}: 4 axes. Pass channel=... to pick "
                "a channel, or z_project='max'/'mean' to flatten z. Refusing to guess "
                "which axis is time.")
    if arr.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"stack has only {arr.shape[0]} frames; nothing to track")
    return arr


def _load_nd2(path: Path) -> np.ndarray:
    try:
        import nd2
        return nd2.imread(str(path))
    except ImportError:
        pass
    try:
        from aicsimageio import AICSImage
    except ImportError as e:                                # pragma: no cover
        raise ImportError("reading .nd2 needs either `nd2` or `aicsimageio` "
                          "installed") from e
    img = AICSImage(str(path))
    return img.get_image_data("TYX")


def frame_interval_from_nd2(path: str | Path) -> float | None:
    """Median frame interval in seconds, if the file records timestamps.

    Worth calling once on a new dataset: dt_seconds set wrongly rescales every rate in
    the output without changing anything visible in the images.
    """
    try:
        import nd2
    except ImportError:
        return None
    with nd2.ND2File(str(path)) as f:
        ev = getattr(f, "events", None)
        if not ev:
            return None
        try:
            t = pd.DataFrame(ev())["Time [s]"].to_numpy(float)
        except Exception:
            return None
    return float(np.median(np.diff(t))) if len(t) > 2 else None


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Parquet when pyarrow is available (tracks get long), CSV otherwise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path
