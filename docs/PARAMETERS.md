# Parameter reference

Every parameter, what it does, and which direction to move it. The shipped
[`config/default.yaml`](../config/default.yaml) carries the same notes inline.

**Tune in this order.** Later sections rarely need touching if the earlier ones are right.

1. `dt_seconds`, `um_per_px` — describe the microscope
2. `detect.threshold_sigma`, `detect.psf_sigma_px` — find the vesicles
3. `link.search_range_px`, `link.min_length_frames` — connect them
4. `metrics.tau_directed_frames` — define "consistent direction"
5. everything else

---

## Acquisition

| parameter | default | unit | what it does |
|---|---|---|---|
| `dt_seconds` | 0.0446 | s | Frame interval. **Required.** Every `*_rate` column and both τ values convert through it. |
| `um_per_px` | `null` | µm/px | Pixel size. `null` reports pixels only; setting it adds `*_um` columns. |
| `output_dir` | `output` | path | Results go to `<output_dir>/<movie name>/`. |

> Set `dt_seconds` wrong and **every rate is wrong while nothing looks wrong** — the
> images are unchanged. `io.frame_interval_from_nd2()` reads it from an ND2 that
> records timestamps.

## `drift` — stage drift correction

Drift moves every vesicle *together*, so uncorrected it appears as directed transport
in all of them at once — precisely what the directed metric detects. **Leave enabled**
unless you know the stage was fixed. Measured span appears in `summary.json`.

| parameter | default | unit | what it does |
|---|---|---|---|
| `enabled` | `true` | | Turn correction off entirely. |
| `smoothing_sigma` | 6.0 | px | Blur applied before registering. Large values lock onto static cell structure instead of the moving vesicles. |
| `reference_frames` | 20 | frames | Median-projected to form the registration reference. |
| `median_filter` | 5 | frames | Temporal smoothing of the shift series; stops one bad frame injecting a jump. `0`/`1` disables. |
| `min_span_px` | 0.5 | px | Below this total drift, the shift is measured but not applied. |

## `detect` — per-frame spot detection

Tune `threshold_sigma` first, then check the **middle panel** of `three_panel.png`:
were the vesicles found, and was anything invented?

| parameter | default | unit | raise it | lower it |
|---|---|---|---|---|
| `threshold_sigma` | 3.0 | σ | too many spurious spots | vesicles being missed |
| `psf_sigma_px` | 1.3 | px | spots are larger than default | spots are tighter |
| `roundness` | 0.7 | 0–1 | to accept irregular shapes | to reject streaks/edges harder |
| `min_separation_px` | 4.0 | px | one vesicle gives several peaks | genuinely adjacent vesicles merge |
| `background_sigma` | 0.0 | px | uneven illumination (try ~PSF×10) | — (`0` = off) |
| `method` | `dao` | | Only `dao` (photutils `DAOStarFinder`) is built in. |

> `min_separation_px` matters more than it looks. Without NMS, a bright vesicle yields
> several peaks, the linker splits it into parallel tracks, and the per-cell count
> inflates.

## `link` — detections into tracks

| parameter | default | unit | what it does |
|---|---|---|---|
| `search_range_px` | 3.0 | px | Max frame-to-frame displacement. **Too large is the dangerous direction** — it invites identity swaps between neighbours. |
| `memory_frames` | 15 | frames | How long a vesicle may vanish (blink, defocus) and still be relinked to the same identity. |
| `min_length_frames` | 40 | frames | Shorter tracks are dropped: too few frames to measure direction. Should exceed `metrics.tau_frames`. |
| `merge_radius_px` | 4.0 | px | Tracks whose mean positions sit this close are treated as one vesicle the linker split at a dropout. `0` disables. |

## `metrics` — movement

`net` = |end − start| · `gross` = summed steps · `directed` = path length at timescale τ.
See the [README](../README.md) for why the first two are not enough.

| parameter | default | unit | what it does |
|---|---|---|---|
| `tau_frames` | 22 (~1 s) | frames | Coarse-graining window for **run detection**. Larger averages away more localisation noise. |
| `tau_directed_frames` | 90 (~4 s) | frames | The timescale defining "consistent direction" — this sets the `directed` column. Larger = only slower, more persistent motion counts. Must be ≥ `tau_frames`. |
| `max_turn_deg` | 60 | ° | A step turning more than this ends a run. |
| `min_run_disp_px` | 1.0 | px | Runs shorter than this are not counted as transport. |
| `min_run_steps` | 2 | steps | A run must persist ≥ 2 coarse steps. **Must be ≥ 2** — `1` collapses `directed` onto plain path length and the metric loses all contrast against the null. Validation rejects it. |
| `n_permutations` | 200 | | Per-vesicle null. `p` floors at `1/(n+1)`, so 200 → smallest possible p is 0.005. `0` disables the p-value. |
| `random_seed` | 0 | | Affects **only** the null, never the measurement. |

> **Choosing τ.** τ = 4 s was chosen against simulated ground truth: it recovers 86% of
> a known 5 px directed run against a pure-noise floor of 2.6 px. Scale both τ values
> by your frame rate — ~1 s and ~4 s in *frames* — then check that mover/confined
> separation rises with τ and flattens (the notebook does this sweep).

## `classify` — labels

| parameter | default | what it does |
|---|---|---|
| `use_permutation` | `true` | `true`: a mover **beats its own null**. `false`: fixed rate threshold. |
| `p_threshold` | 0.05 | Used when `use_permutation` is true. |
| `mover_rate_px_per_s` | 0.15 | Used **only** when `use_permutation` is false. |
| `mover_min_net_px` | 4.0 | Guard so a tiny but significant displacement cannot score as a mover. |
| `exclude_max_step_px` | 5.0 | One jump this large ⇒ suspected mislink. |
| `exclude_n_big_steps` | 5 | This many big steps ⇒ suspected repeated identity swap. |
| `big_step_px` | 3.0 | What counts as "big" for the rule above. |

> `excluded` is a **tracking-quality** judgement, not biology. These are the signatures
> of identity swaps between nearby vesicles; leaving them in inflates the mover count.
> A rising `excluded` count usually means `link.search_range_px` is too large or the
> field is too dense.

## `render` — output

| parameter | default | what it does |
|---|---|---|
| `three_panel` | `true` | `RAW │ ALL VESICLES │ CLASSIFIED` — the QC image. |
| `per_vesicle_images` | `true` | One image per vesicle: trajectory with all three distances drawn and printed. |
| `per_vesicle_videos` | `false` | One **file per vesicle**; off by default. |
| `overview_video` | `false` | Whole field, tracks coloured by class. |
| `max_vesicle_outputs` | 25 | Cap on per-vesicle files (movers first) so a dense field cannot emit thousands. Recorded in `summary.json` rather than applied silently. |
| `trail_frames` | 40 | Length of the fading trail in videos. |
| `frame_step` | 4 | Temporal downsample for videos (4 = every 4th frame). |
| `fps` | 20 | Playback rate of written videos. |
| `dpi` | 150 | Raster resolution of PNG figures. |
| `percentiles` | `[1.0, 99.7]` | Display stretch. **Looks only** — never affects a measurement. |
| `ffmpeg` | `ffmpeg` | Full path if not auto-found. Missing ffmpeg skips videos with a warning; images are unaffected. |

---

## Overriding without editing the file

Dotted paths, and `copy()` never mutates the original — which makes a sweep a loop:

```python
cfg = Config.load("config/default.yaml", dt_seconds=0.05, um_per_px=0.107)
cfg.set("detect.threshold_sigma", 2.5)

for thr in [2.5, 3.0, 3.5]:
    analyse(movie, cfg.copy(**{"detect.threshold_sigma": thr}))
```

Unknown keys are rejected at load, so a typo like `thresold_sigma` fails loudly instead
of being silently ignored while you wonder why the parameter did nothing.

Every run writes `config_used.yaml` beside its outputs, so any figure traces back to
the exact parameters that made it.
