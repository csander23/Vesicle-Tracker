"""Regions of interest: which region of the image each vesicle is in.

Accepts ROIs from any of:

    ImageJ / Fiji     .roi, or a RoiSet .zip of several   (needs `roifile`)
    label image       .tif / .png where 0 = background and 1..N are regions
    binary mask       .tif / .png / .npy, one region
    numpy array       bool (one region) or int (label image)
    polygons          {"soma": [(x, y), ...], "processes": [...]}

Everything is converted to one int label array, 0 meaning "outside every ROI", plus a
list of names.

Nothing is discarded. Vesicles outside every ROI are tracked, measured and reported
like the rest, labelled `outside`. "Outside" is usually a real comparison group
(cytoplasm vs soma, cell vs background), and dropping those vesicles would make that
comparison impossible. To restrict the analysis, filter on the `roi` column
afterwards. The filtered output does this when `filters.rois` is set.

A vesicle that moves between regions is assigned by majority of its observed frames,
and flagged with `roi_changed` and `roi_frac` so the ambiguous ones can be found.
Assigning by first frame instead would mislabel the motile vesicles, which are the
ones a transport study is about.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTSIDE = "outside"


def load_rois(source, shape: tuple[int, int]) -> tuple[np.ndarray, list[str]]:
    """Return (labels, names). labels is int (H, W); 0 = outside. names[i] is label i+1."""
    H, W = shape
    if source is None:
        return np.zeros((H, W), np.int32), []

    if isinstance(source, dict) and source and all(
            isinstance(v, (list, tuple, np.ndarray)) for v in source.values()):
        return _from_polygons(source, shape)

    if isinstance(source, np.ndarray):
        return _from_array(source, shape)

    if isinstance(source, (list, tuple)):
        labels = np.zeros((H, W), np.int32)
        names: list[str] = []
        for item in source:
            lab, nm = load_rois(item, shape)
            for k, n in enumerate(nm, start=1):
                names.append(n)
                labels[(lab == k) & (labels == 0)] = len(names)
        return labels, names

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"ROI source not found: {p}")
    suf = p.suffix.lower()
    if suf in (".roi", ".zip"):
        return _from_imagej(p, shape)
    if suf == ".npy":
        return _from_array(np.load(p), shape)
    if suf in (".tif", ".tiff", ".png"):
        import tifffile
        if suf == ".png":
            from matplotlib.image import imread
            a = imread(str(p))
            a = a[..., 0] if a.ndim == 3 else a
            a = (a * 255).astype(np.int32) if a.dtype.kind == "f" else a.astype(np.int32)
        else:
            a = tifffile.imread(str(p))
        return _from_array(a, shape)
    raise ValueError(f"unsupported ROI file type {suf!r}")


def _from_array(a: np.ndarray, shape) -> tuple[np.ndarray, list[str]]:
    a = np.asarray(a)
    if a.shape[:2] != tuple(shape):
        raise ValueError(f"ROI array shape {a.shape[:2]} does not match image {shape}. "
                         "An ROI drawn on a cropped or resized image will silently "
                         "label the wrong pixels, so this is refused.")
    if a.dtype == bool:
        return a.astype(np.int32), ["roi_1"]
    vals = [v for v in np.unique(a) if v != 0]
    if len(vals) == 1:                              # binary mask stored as 0/255
        return (a != 0).astype(np.int32), ["roi_1"]
    lab = np.zeros(a.shape[:2], np.int32)
    names = []
    for k, v in enumerate(vals, start=1):
        lab[a == v] = k
        names.append(f"roi_{int(v)}")
    return lab, names


def _from_polygons(polys: dict, shape) -> tuple[np.ndarray, list[str]]:
    from matplotlib.path import Path as MplPath
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    lab = np.zeros((H, W), np.int32)
    names = []
    for k, (name, poly) in enumerate(polys.items(), start=1):
        inside = MplPath(np.asarray(poly, float)).contains_points(pts).reshape(H, W)
        lab[inside & (lab == 0)] = k
        names.append(str(name))
    return lab, names


def _from_imagej(path: Path, shape) -> tuple[np.ndarray, list[str]]:
    try:
        import roifile
    except ImportError as e:                                    # pragma: no cover
        raise ImportError("reading ImageJ ROIs needs `pip install roifile`") from e
    rois = roifile.roiread(str(path))
    rois = rois if isinstance(rois, list) else [rois]
    polys = {}
    for i, r in enumerate(rois, start=1):
        c = np.asarray(r.coordinates(), float)
        polys[getattr(r, "name", None) or f"roi_{i}"] = c
    return _from_polygons(polys, shape)


def label_at(labels: np.ndarray, x, y) -> np.ndarray:
    """ROI index (0 = outside) at each position, clipped to the image."""
    if labels is None or not labels.size:
        return np.zeros(len(x), np.int32)
    H, W = labels.shape
    yi = np.clip(np.rint(np.asarray(y)).astype(int), 0, H - 1)
    xi = np.clip(np.rint(np.asarray(x)).astype(int), 0, W - 1)
    return labels[yi, xi]


def assign_tracks(tracks: pd.DataFrame, labels: np.ndarray,
                  names: list[str]) -> pd.DataFrame:
    """One row per particle: roi, roi_frac, roi_changed, n_rois_visited.

    Assignment is by majority of observed frames. `roi_frac` is the fraction of frames
    in the assigned region, so a value near 0.5 marks a vesicle that straddles a
    boundary and probably should not be counted as either.
    """
    lut = [OUTSIDE] + list(names)
    if not len(tracks):
        return pd.DataFrame(columns=["particle", "roi", "roi_frac", "roi_changed",
                                     "n_rois_visited"])
    idx = label_at(labels, tracks.x.values, tracks.y.values)
    df = pd.DataFrame({"particle": tracks.particle.values, "idx": idx})
    rows = []
    for p, d in df.groupby("particle"):
        counts = d.idx.value_counts()
        top = int(counts.index[0])
        rows.append(dict(
            particle=int(p),
            roi=lut[top] if top < len(lut) else f"roi_{top}",
            roi_frac=float(counts.iloc[0] / len(d)),
            roi_changed=bool(len(counts) > 1),
            n_rois_visited=int(len(counts)),
        ))
    return pd.DataFrame(rows)


def coverage(labels: np.ndarray, names: list[str]) -> dict:
    """Fraction of the frame occupied by each region (a sanity check on the ROI)."""
    if labels is None or not labels.size:
        return {}
    tot = labels.size
    out = {OUTSIDE: float((labels == 0).sum() / tot)}
    for k, n in enumerate(names, start=1):
        out[n] = float((labels == k).sum() / tot)
    return out
