"""Figures and videos.

Three kinds of output, each switchable in `render:` config:

  three_panel          RAW | ALL VESICLES | CLASSIFIED, side by side on one image.
                       The middle panel answers "did detection find the vesicles",
                       the right one answers "did it classify them sensibly" - the
                       two failure modes worth checking before trusting any number.

  per-vesicle images   one trajectory per image, annotated with its net, gross and
                       directed distances and its permutation p. The three distances
                       are drawn as well as printed: net as a straight arrow
                       start->end, directed as the coarse-grained polyline, gross as
                       the raw trace. Seeing all three on the same axes is the fastest
                       way to understand why they differ.

  videos               per-vesicle crops and/or a whole-field overview, written via
                       ffmpeg with a moving trail.

Display uses a percentile stretch shared across every frame of a stack, so brightness
changes in a video are real and not the renderer re-levelling each frame.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KLASS_COLOR = {"mover": "#2ecc40", "confined": "#00c8ff", "excluded": "#ff4136"}


# ------------------------------------------------------------------ helpers
def stretch(stack: np.ndarray, pct=(1.0, 99.7)) -> tuple[float, float]:
    sub = stack[:: max(1, len(stack) // 20)]
    lo, hi = np.percentile(sub, pct)
    return float(lo), float(max(hi, lo + 1e-6))


def to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _projection(stack: np.ndarray) -> np.ndarray:
    return stack.max(axis=0)


# -------------------------------------------------------------- three panel
def three_panel(stack, tracks, vesicles, cfg, out_path, title=""):
    """RAW | ALL VESICLES | CLASSIFIED. Returns the written path."""
    lo, hi = stretch(stack, cfg.render.percentiles)
    proj = to_uint8(_projection(stack), lo, hi)
    kl = dict(zip(vesicles.particle, vesicles.klass)) if len(vesicles) else {}

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), facecolor="white")
    for a in ax:
        a.imshow(proj, cmap="gray", interpolation="nearest")
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)

    ax[0].set_title("Raw (max projection)", fontsize=12, weight="bold")

    for p, d in tracks.groupby("particle"):
        ax[1].plot(d.x, d.y, lw=0.7, color="#ffdc00", alpha=0.9)
    n_ves = tracks.particle.nunique() if len(tracks) else 0
    ax[1].set_title(f"All vesicles tracked  (n={n_ves})", fontsize=12, weight="bold")

    counts = {}
    for p, d in tracks.groupby("particle"):
        k = kl.get(p, "confined")
        counts[k] = counts.get(k, 0) + 1
        ax[2].plot(d.x, d.y, lw=0.9, color=KLASS_COLOR.get(k, "#aaaaaa"), alpha=0.95)
    lab = "   ".join(f"{k} {counts.get(k, 0)}"
                     for k in ("mover", "confined", "excluded"))
    ax[2].set_title(f"Classified\n{lab}", fontsize=12, weight="bold")
    handles = [plt.Line2D([], [], color=c, lw=2, label=k)
               for k, c in KLASS_COLOR.items()]
    ax[2].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.7)

    if title:
        fig.suptitle(title, fontsize=13, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97 if title else 1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.render.dpi, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------- per-vesicle trajectory
def vesicle_image(stack, track, row, cfg, out_path, pad: int = 24):
    """One vesicle: crop, trajectory, and the three distances drawn and printed."""
    x, y, f = track.x.values, track.y.values, track.frame.values
    lo, hi = stretch(stack, cfg.render.percentiles)
    H, W = stack.shape[1:]
    x0 = int(max(0, np.floor(x.min()) - pad)); x1 = int(min(W, np.ceil(x.max()) + pad))
    y0 = int(max(0, np.floor(y.min()) - pad)); y1 = int(min(H, np.ceil(y.max()) + pad))
    crop = to_uint8(stack[f[0]:f[-1] + 1, y0:y1, x0:x1].max(axis=0), lo, hi)

    from .metrics import coarse
    cx, cy, _ = coarse(x, y, f, cfg.metrics.tau_directed_frames)

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2), facecolor="white",
                           gridspec_kw={"width_ratios": [1, 1]})
    ax[0].imshow(crop, cmap="gray", interpolation="nearest",
                 extent=[x0, x1, y1, y0])
    ax[0].plot(x, y, lw=0.8, color="#ffdc00", alpha=0.8, label="gross (raw path)")
    ax[0].plot(cx, cy, lw=2.0, color="#2ecc40", label="directed (coarse path)")
    ax[0].annotate("", xy=(x[-1], y[-1]), xytext=(x[0], y[0]),
                   arrowprops=dict(arrowstyle="->", color="#ff4136", lw=2))
    ax[0].plot([], [], color="#ff4136", lw=2, label="net (end - start)")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax[0].set_title(f"vesicle {int(row.particle)}   [{row.klass}]",
                    fontsize=12, weight="bold")

    unit, k = ("µm", cfg.um_per_px) if cfg.um_per_px else ("px", 1.0)
    lines = [
        f"frames {int(row.frame_start)}–{int(row.frame_end)}  "
        f"({row.duration_s:.1f} s)",
        "",
        f"gross     {row.gross * k:8.2f} {unit}",
        f"directed  {row.directed * k:8.2f} {unit}",
        f"net       {row.net * k:8.2f} {unit}",
        "",
        f"gross rate     {row.gross_rate * k:7.3f} {unit}/s",
        f"directed rate  {row.directed_rate * k:7.3f} {unit}/s",
        f"net rate       {row.net_rate * k:7.3f} {unit}/s",
    ]
    if not pd.isna(row.get("directed_p", np.nan)):
        lines += ["", f"permutation p  {row.directed_p:.3f}",
                  f"directed excess {row.directed_excess * k:6.2f} {unit}"]
    lines += ["", f"tau = {cfg.tau_directed_seconds:.2f} s "
                  f"({cfg.metrics.tau_directed_frames} frames)"]
    ax[1].axis("off")
    ax[1].text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
               family="monospace", fontsize=10.5, transform=ax[1].transAxes)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.render.dpi, facecolor="white")
    plt.close(fig)
    return out_path


def distance_summary(vesicles, cfg, out_path):
    """Net / directed / gross across all vesicles - the ordering, made visible."""
    unit, k = ("µm", cfg.um_per_px) if cfg.um_per_px else ("px", 1.0)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    for col, c in [("gross", "#ffb700"), ("directed", "#2ecc40"), ("net", "#ff4136")]:
        v = np.sort(vesicles[col].dropna().to_numpy()) * k
        if len(v):
            ax[0].plot(v, np.linspace(0, 1, len(v)), lw=2, color=c, label=col)
    ax[0].set_xscale("symlog", linthresh=1e-2)
    ax[0].set_xlabel(f"distance ({unit})"); ax[0].set_ylabel("cumulative fraction")
    ax[0].legend(fontsize=9); ax[0].set_title("Distance distributions",
                                              fontsize=12, weight="bold")
    for kl, sub in vesicles.groupby("klass"):
        ax[1].scatter(sub.net * k, sub.directed * k, s=14, alpha=0.75,
                      color=KLASS_COLOR.get(kl, "#888"), label=kl)
    lim = max(1e-3, float((vesicles.directed * k).max()))
    ax[1].plot([0, lim], [0, lim], ls="--", lw=1, color="#888")
    ax[1].set_xlabel(f"net ({unit})"); ax[1].set_ylabel(f"directed ({unit})")
    ax[1].legend(fontsize=9); ax[1].set_title("directed vs net  (y=x dashed)",
                                              fontsize=12, weight="bold")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.render.dpi, facecolor="white")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------- video out
_FFMPEG_WARNED = False


def resolve_ffmpeg(exe: str) -> str | None:
    """Find ffmpeg: an explicit path, then PATH, then next to the running python.

    The last case matters for conda envs, where ffmpeg is installed inside the env and
    is NOT on PATH unless the env is activated - the usual situation in a notebook.
    """
    p = Path(exe)
    if p.is_absolute() and p.exists():
        return str(p)
    found = shutil.which(exe)
    if found:
        return found
    import sys
    cand = Path(sys.executable).parent / exe
    return str(cand) if cand.exists() else None


def _have_ffmpeg(exe: str) -> bool:
    global _FFMPEG_WARNED
    if resolve_ffmpeg(exe) is not None:
        return True
    if not _FFMPEG_WARNED:
        _FFMPEG_WARNED = True
        print(f"vesicletrack: ffmpeg not found (looked for {exe!r} on PATH and next "
              "to the running python). Video output is being SKIPPED; images are "
              "unaffected. Install ffmpeg, or set render.ffmpeg to its full path.",
              flush=True)
    return False


def _write_video(frames, out_path, fps, exe):
    """frames: iterable of uint8 (H, W) or (H, W, 3). Piped straight to ffmpeg."""
    frames = list(frames)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    if h % 2:
        frames = [f[:-1] for f in frames]; h -= 1
    if w % 2:
        frames = [f[:, :-1] for f in frames]; w -= 1
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exe = resolve_ffmpeg(exe) or exe
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", str(out_path)]
    # A present-but-broken ffmpeg raises BrokenPipeError mid-write. That must not kill
    # save() after the whole analysis has already run - the tables are the valuable
    # output and they are already on disk.
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            if f.ndim == 2:
                f = np.repeat(f[:, :, None], 3, axis=2)
            p.stdin.write(f.astype(np.uint8).tobytes())
        p.stdin.close()
        p.wait()
    except (BrokenPipeError, OSError) as e:
        print(f"vesicletrack: ffmpeg failed ({type(e).__name__}: {e}); "
              f"skipping {out_path.name}. Tables and images are unaffected.",
              flush=True)
        return None
    return out_path if out_path.exists() else None


def vesicle_video(stack, track, row, cfg, out_path, pad: int = 24):
    """Cropped movie of one vesicle with a fading trail."""
    if not _have_ffmpeg(cfg.render.ffmpeg):
        return None
    import cv2
    x, y, f = track.x.values, track.y.values, track.frame.values
    lo, hi = stretch(stack, cfg.render.percentiles)
    H, W = stack.shape[1:]
    x0 = int(max(0, np.floor(x.min()) - pad)); x1 = int(min(W, np.ceil(x.max()) + pad))
    y0 = int(max(0, np.floor(y.min()) - pad)); y1 = int(min(H, np.ceil(y.max()) + pad))
    pos = {int(fr): (xx, yy) for fr, xx, yy in zip(f, x, y)}
    col = tuple(int(c * 255) for c in
                matplotlib.colors.to_rgb(KLASS_COLOR.get(row.klass, "#ffffff")))
    scale = max(1, int(round(160 / max(1, x1 - x0))))
    frames = []
    for t in range(int(f[0]), int(f[-1]) + 1, cfg.render.frame_step):
        img = to_uint8(stack[t, y0:y1, x0:x1], lo, hi)
        rgb = np.repeat(img[:, :, None], 3, axis=2).copy()
        trail = [pos[s] for s in range(max(int(f[0]), t - cfg.render.trail_frames),
                                       t + 1) if s in pos]
        for (ax_, ay_), (bx_, by_) in zip(trail[:-1], trail[1:]):
            cv2.line(rgb, (int(ax_ - x0), int(ay_ - y0)),
                     (int(bx_ - x0), int(by_ - y0)), col, 1, cv2.LINE_AA)
        if t in pos:
            cv2.circle(rgb, (int(pos[t][0] - x0), int(pos[t][1] - y0)), 3, col, 1,
                       cv2.LINE_AA)
        if scale > 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
        frames.append(rgb)
    return _write_video(frames, out_path, cfg.render.fps, cfg.render.ffmpeg)


def overview_video(stack, tracks, vesicles, cfg, out_path):
    """Whole field with every track drawn, coloured by class."""
    if not _have_ffmpeg(cfg.render.ffmpeg):
        return None
    import cv2
    lo, hi = stretch(stack, cfg.render.percentiles)
    kl = dict(zip(vesicles.particle, vesicles.klass)) if len(vesicles) else {}
    by_frame: dict[int, list] = {}
    for p, d in tracks.groupby("particle"):
        c = tuple(int(v * 255) for v in
                  matplotlib.colors.to_rgb(KLASS_COLOR.get(kl.get(p, "confined"))))
        for fr, xx, yy in zip(d.frame.values, d.x.values, d.y.values):
            by_frame.setdefault(int(fr), []).append((xx, yy, c, p))
    hist: dict[int, list] = {}
    frames = []
    for t in range(0, len(stack), cfg.render.frame_step):
        rgb = np.repeat(to_uint8(stack[t], lo, hi)[:, :, None], 3, axis=2).copy()
        for xx, yy, c, p in by_frame.get(t, []):
            h = hist.setdefault(p, [])
            h.append((xx, yy))
            if cfg.render.trail_frames > 0:
                del h[:-cfg.render.trail_frames]
            else:
                del h[:-1]          # 0 means no trail, not an infinite one
            for (ax_, ay_), (bx_, by_) in zip(h[:-1], h[1:]):
                cv2.line(rgb, (int(ax_), int(ay_)), (int(bx_), int(by_)), c, 1,
                         cv2.LINE_AA)
            cv2.circle(rgb, (int(xx), int(yy)), 2, c, -1, cv2.LINE_AA)
        frames.append(rgb)
    return _write_video(frames, out_path, cfg.render.fps, cfg.render.ffmpeg)
