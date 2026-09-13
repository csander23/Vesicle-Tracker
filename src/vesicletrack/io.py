"""Reading image stacks and writing results.

Input is a 2D time series: shape (T, Y, X). TIFF is read with tifffile; ND2 is read
with nd2 or aicsimageio if either is installed. A stack with a channel or z axis must
be reduced before it gets here. `load_stack` raises an error rather than guessing
which axis is time, because a wrong guess produces tracks of nothing with no
indication that anything went wrong.
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
    axes = None
    if suf in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            axes = getattr(tf.series[0], "axes", None)
            arr = tf.asarray()
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
        # Handled fully here and returned. If this fell through to the 4-D branch
        # below, `channel` would be applied a second time as a z index and the loader
        # would return a single z-plane, ignoring z_project.
        if channel is None or z_project is None:
            raise ValueError(
                f"{path.name} has shape {orig_shape} (T, C, Z, Y, X): pass both "
                "channel=... and z_project='max'/'mean'. Refusing to guess which "
                "axis is time.")
        arr = arr[:, channel]
        arr = arr.max(axis=1) if z_project == "max" else arr.mean(axis=1)
        return _check_3d(arr, path, orig_shape)
    # The file usually records its axes: tifffile exposes series[0].axes as e.g.
    # 'ZYX' or 'TYX'. A z-stack has the same 3-D shape as a time series and would
    # otherwise be tracked, producing tracks of nothing. Use the metadata when present.
    if arr.ndim == 3 and axes and len(axes) == 3 and axes[0] not in ("T", "I", "Q"):
        raise ValueError(
            f"{path.name} has axes {axes!r}: the first axis is {axes[0]!r}, not time. "
            f"This looks like a {'z-stack' if axes[0] == 'Z' else 'non-time series'}, "
            "not a time-lapse. Reduce it to (T, Y, X) first, or pass z_project=.")
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
    return _check_3d(arr, path, orig_shape, axes)


def _check_3d(arr, path, orig_shape, axes=None):
    if arr.ndim != 3:
        raise ValueError(f"expected a (T, Y, X) stack, got shape {arr.shape} "
                         f"(file was {orig_shape})")
    if arr.shape[0] < 3:
        raise ValueError(f"{path.name} has only {arr.shape[0]} frames "
                         f"(shape {orig_shape}); nothing to track")
    # No metadata (.npy): a stack with fewer "frames" than pixels on a side is far
    # more likely a z-stack or a transposed array than a real recording.
    if axes is None and arr.shape[0] <= min(arr.shape[1], arr.shape[2]):
        import warnings
        warnings.warn(
            f"{path.name} has shape {arr.shape}: only {arr.shape[0]} frames for a "
            f"{arr.shape[1]}x{arr.shape[2]} image. If this is a z-stack or a "
            "transposed array, every result will be meaningless. Check the axis "
            "order.", stacklevel=3)
    return arr


def _load_nd2(path: Path, channel: int | None = None) -> np.ndarray:
    """Read an ND2, preserving whatever axes it has for the caller to reduce.

    Both readers honour `channel`, so the same file gives the same result whichever
    reader is installed.
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

    Call it once on a new dataset: a wrong dt_seconds rescales every rate in the
    output and nothing in the images shows it.
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


def read_sample_sheet(path: str | Path) -> dict:
    """{file name: {column: value}} from a CSV with a `file` column.

    The sheet is how metadata (batch, genotype, ...) reaches the output: the user
    supplies the mapping from file to metadata, and nothing is parsed from filenames,
    since naming conventions differ between labs. Files are matched on their base
    name, so the sheet can list bare names or full paths. Empty cells are left out
    rather than stamped as NaN.
    """
    sh = pd.read_csv(path)
    if "file" not in sh.columns:
        raise ValueError(f"{path} needs a `file` column; got {list(sh.columns)}")
    out = {}
    for row in sh.to_dict("records"):
        key = Path(row.pop("file")).name
        out[key] = {k: v for k, v in row.items() if pd.notna(v)}
    return out


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Parquet when pyarrow is available (tracks get long), CSV otherwise."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path
