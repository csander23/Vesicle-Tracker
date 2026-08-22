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
        arr = _load_nd2(path, channel=channel)
        channel = None if arr.ndim == 3 else channel
    elif suf in (".npy",):
        arr = np.load(path)
    else:
        raise ValueError(f"unsupported file type {suf!r}; expected .tif/.tiff/.nd2/.npy")

    arr = np.asarray(arr)
    orig_shape = arr.shape
    if arr.ndim == 5:                                  # T,C,Z,Y,X
        # Handled fully here and RETURNED. Previously this fell through into the 4-D
        # branch below, where `channel` was still set and got applied a second time -
        # as a z index - so the loader silently returned a single z-plane whose index
        # happened to equal the channel number, discarding z_project entirely.
        if channel is None or z_project is None:
            raise ValueError(
                f"{path.name} has shape {orig_shape} (T, C, Z, Y, X): pass both "
                "channel=... and z_project='max'/'mean'. Refusing to guess which "
                "axis is time.")
        arr = arr[:, channel]
        arr = arr.max(axis=1) if z_project == "max" else arr.mean(axis=1)
        return _check_3d(arr, path, orig_shape)
    if arr.ndim == 4:                                  # T,C,Y,X or T,Z,Y,X
        if channel is not None:
            arr = arr[:, channel]
        elif z_project == "max":
            arr = arr.max(axis=1)
        elif z_project == "mean":
            arr = arr.mean(axis=1)
        else:
            raise ValueError(
                f"{path.name} has shape {orig_shape}: 4 axes. Pass channel=... to pick "
                "a channel, or z_project='max'/'mean' to flatten z. Refusing to guess "
                "which axis is time.")
    return _check_3d(arr, path, orig_shape)


def _check_3d(arr, path, orig_shape):
    if arr.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {arr.shape} "
                         f"(file was {orig_shape})")
    if arr.shape[0] < 3:
        raise ValueError(f"{path.name} has only {arr.shape[0]} frames "
                         f"(shape {orig_shape}); nothing to track")
    return arr


def _load_nd2(path: Path, channel: int | None = None) -> np.ndarray:
    """Read an ND2, preserving whatever axes it has for the caller to reduce.

    The aicsimageio fallback previously requested "TYX" unconditionally, which silently
    ignored `channel` and returned channel 0 - so the same file gave different answers
    depending on which reader happened to be installed. It now asks for the channel
    axis too and lets load_stack apply the caller's choice.
    """
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
    if len(img.channel_names) > 1:
        if channel is None:
            raise ValueError(
                f"{path.name} has {len(img.channel_names)} channels "
                f"({list(img.channel_names)}): pass channel=... . Refusing to "
                "default to channel 0.")
        return img.get_image_data("TYX", C=channel)
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
