# Parameter reference

Every parameter, what it does, and which direction to move it. The shipped
[`config/default.yaml`](../config/default.yaml) carries the same notes inline.

Tune in this order. Later sections rarely need changing if the earlier ones are right.

1. `dt_seconds`, `um_per_px`: describe the microscope
2. `detect.threshold_sigma`, `detect.psf_sigma_px`: find the vesicles
3. `link.search_range_px`, `link.min_length_frames`: connect them
4. `metrics.tau_directed_frames`: define "consistent direction"
5. everything else

---

## Acquisition

| parameter | default | unit | what it does |
|---|---|---|---|
| `dt_seconds` | 0.0446 | s | Frame interval. Required. Every `*_rate` column and both τ values convert through it. |
| `um_per_px` | `null` | µm/px | Pixel size. `null` reports pixels only; setting it adds `*_um` columns. |
| `output_dir` | `output` | path | Results go to `<output_dir>/<movie name>/`. |

> If `dt_seconds` is wrong, every rate is wrong and the images give no hint of it.
> `io.frame_interval_from_nd2()` reads it from an ND2 that records timestamps.

## `drift`: stage drift correction

Drift moves every vesicle together, so uncorrected it appears as directed transport
in all of them at once, which is what the directed metric measures. Leave it enabled
unless the stage is known to have been fixed. The measured span is written to
`summary.json`.

| parameter | default | unit | what it does |
|---|---|---|---|
| `enabled` | `true` | | Turn correction off entirely. |
| `smoothing_sigma` | 6.0 | px | Blur applied before registering. Large values lock onto static cell structure instead of the moving vesicles. |
| `reference_frames` | 20 | frames | Median-projected to form the registration reference. |
| `median_filter` | 5 | frames | Temporal smoothing of the shift series; stops one bad frame injecting a jump. `0`/`1` disables. |
| `min_span_px` | 0.5 | px | Below this total drift, the shift is measured but not applied. |

## `detect`: per-frame spot detection

Tune `threshold_sigma` first, then check the middle panel of `three_panel.png`: were
the vesicles found, and were any spurious spots detected?

| parameter | default | unit | raise it | lower it |
|---|---|---|---|---|
| `threshold_sigma` | 3.0 | σ | too many spurious spots | vesicles being missed |
| `psf_sigma_px` | 1.3 | px | spots are larger than default | spots are tighter |
| `roundness` | 0.7 | 0–1 | to accept irregular shapes | to reject streaks/edges harder |
| `min_separation_px` | 4.0 | px | one vesicle gives several peaks | genuinely adjacent vesicles merge |
| `background_sigma` | 0.0 | px | uneven illumination (try ~PSF×10) | (`0` = off) |
| `method` | `dao` | | Only `dao` (photutils `DAOStarFinder`) is built in. |

> `min_separation_px` matters more than it looks. Without non-maximum suppression, a
> bright vesicle yields several peaks, the linker splits it into parallel tracks, and
> the per-cell count inflates.

## `link`: detections into tracks

| parameter | default | unit | what it does |
|---|---|---|---|
| `search_range_px` | 3.0 | px | Max frame-to-frame displacement. Too large is the risky direction: it allows identity swaps between neighbours. |
| `memory_frames` | 15 | frames | How long a vesicle may vanish (blink, defocus) and still be relinked to the same identity. |
| `min_length_frames` | 3 | frames | Hard floor only. Below this no metric exists at all (a 2-frame track has one step). The scientific length cut is `filters.min_observed_frames`, which labels instead of deleting. |
| `merge_radius_px` | 4.0 | px | End-to-start distance allowed when rejoining a fragment. `0` disables merging. |
| `merge_max_gap_frames` | 15 | frames | The "look back" window: how long after losing a vesicle the tracker may still rejoin a new fragment to it. |
| `merge_overlap_tolerance` | 2 | frames | Overlap still treated as one vesicle. Fragments that genuinely coexist are different vesicles and are never merged. |

> Merging is deliberately conservative. Joining tracks whose mean positions are close
> fuses two different vesicles that occupied the same spot at different times; on real
> data that produced tracks with 120-frame gaps under a `memory` of 15. A merge
> requires temporal disjointness, a short gap, and proximity at the junction rather
> than on average.

## `size`: how big each vesicle is

Intensity-weighted second moment of a window around each spot: fast (vectorised per
frame) and within a few percent of a Gaussian fit for well-separated spots.

| parameter | default | unit | what it does |
|---|---|---|---|
| `enabled` | `true` | | Turn sizing off to save time. |
| `window_px` | 4 | px | Half-width of the measurement window (4 → 9×9). Too large pulls in neighbours; too small truncates the spot. |
| `psf_sigma_px` | `null` | px | PSF width to deconvolve. `null` calibrates it from this movie, which is the recommended setting. This is a different quantity from `detect.psf_sigma_px`, which only sets the detection kernel. |
| `psf_from_percentile` | 5.0 | % | When calibrating, which percentile of measured widths counts as "a point source". |
| `max_frames` | 200 | frames | Frames sampled per track. More adds no precision. |

> A vesicle is below the diffraction limit, so what the microscope records is mostly
> the PSF. Three columns come out: `sigma_px` (measured, includes PSF),
> `sigma_deconv_px` (excess over the PSF), and `at_diffraction_limit` (True when the
> object is unresolved; deconvolved size is then `0`, not NaN, because "unresolved" is
> a real state rather than missing data).
>
> There is a noise floor. The second moment is biased upward by noise. Measured on
> this pipeline: synthetic point sources with a true size of zero report σ = 1.394 px
> and 0.41 px of deconvolved "size"; real Rab5 endosomes at the same settings give
> σ = 2.010 px and 0.99 px. Treat ~0.4 px of deconvolved width as indistinguishable
> from zero, and recalibrate that floor for your own optics with sub-resolution beads
> or `examples/make_synthetic.py`.
>
> Prefer `sigma_px` for comparisons. Between conditions imaged identically the PSF
> contribution is common, so a difference in raw σ is real and needs no deconvolution.

## `filters`: which vesicles count

Filters add a label. No vesicle is removed from the output. Every vesicle is written
to `vesicles_all.csv` with `passes_filter` and `filter_reason`; the passing subset also
goes to `vesicles_filtered.csv`. Any threshold can be re-derived later from the `all`
file, so a filter can be changed afterwards without losing data. `null` = rule off.

| parameter | default | what it does |
|---|---|---|
| `min_observed_frames` | 40 | Frames actually detected. A tracking-quality cut, not the `directed` requirement. |
| `min_span_frames` | `null` | First-to-last extent. `null` = off. |
| `require_directed_measurable` | `true` | Keep only vesicles for which `directed` exists. This is the exact condition, so every row of `vesicles_filtered.csv` has a `directed` value. |
| `max_observed_frames` | `null` | Upper bound on detected frames. |
| `min_lifetime_s` | `null` | On span, first to last sighting. |
| `max_lifetime_s` | `null` | Upper bound on span. See the bias warning below. |
| `min_frac_observed` | 0.0 | Rejects tracks that are mostly gap (`0.5` = at least half the span seen). |
| `max_longest_gap` | `null` | Longest single dropout allowed, in frames. |
| `exclude_censored` | `false` | Drop tracks touching the first or last frame. |
| `exclude_classes` | `[excluded]` | `excluded` = suspected identity swap; a tracking judgement, not biology. |
| `rois` | `[]` | Keep only these regions, e.g. `[soma]`. Empty keeps all, including `outside`. Regions come from `analyse(..., rois=)` / `--rois`; to restrict detection to one cell use `mask=` / `--mask` instead, which is not a filter. |
| `min_sigma_px` / `max_sigma_px` | `null` | Size bounds. |

> `directed` needs two τ-windows to exist, which is a requirement on span rather than
> on observed frames. A track spanning less than `2 × tau_directed_frames` cannot form
> a coarse path, so `directed` is NaN and `directed_measurable` is False. Real
> vesicles are only ~40% observed, so one spanning 400 frames holds ~163 detections; a
> gate of 180 observed frames would reject the genuine vesicles. Use
> `min_span_frames` for a span requirement, and `require_directed_measurable` for the
> exact condition, since whether `directed` exists also depends on how the detections
> fall into windows. Config validation warns when `min_span_frames` admits tracks that
> cannot be scored.

> `max_lifetime_s` selects for tracking failure. An upper bound on lifetime
> preferentially keeps tracks the tracker lost early; it does not select for
> short-lived biology. Use it only with a reason, and check the censored fraction.

> Censoring. A vesicle already present in frame 0, or still present in the last
> frame, has a lifetime whose start or end was not observed. Pooling those with
> complete observations biases mean lifetime downward. They are flagged
> `is_censored`, `censored_start`, `censored_end`. On a 400-frame crop of real data
> nearly every long-lived vesicle is censored, so check this before quoting a mean
> lifetime.

## `metrics`: movement

`net` = |end − start| · `gross` = summed steps · `directed` = path length at timescale τ.
See the [README](../README.md) for why the first two are not enough.

| parameter | default | unit | what it does |
|---|---|---|---|
| `tau_frames` | 22 (~1 s) | frames | Coarse-graining window for run detection. Larger averages away more localisation noise. |
| `tau_directed_frames` | 90 (~4 s) | frames | The timescale defining "consistent direction"; this sets the `directed` column. Larger = only slower, more persistent motion counts. Must be ≥ `tau_frames`. |
| `max_turn_deg` | 60 | ° | A step turning more than this ends a run. |
| `min_run_disp_px` | 1.0 | px | Runs shorter than this are not counted as transport. |
| `min_window_occupancy` | 0.25 | fraction | Share of τ a window must actually contain to be used. Below this it is dropped. |
| `min_run_steps` | 2 | steps | A run must persist ≥ 2 coarse steps. Must be ≥ 2: `1` collapses `directed` onto plain path length and the metric loses its contrast against the null. Validation rejects it. |
| `n_permutations` | 200 | | Per-vesicle null. `p` floors at `1/(n+1)`, so 200 → smallest possible p is 0.005. `0` disables the p-value. Validation rejects a `p_threshold` below that floor, which would make `mover` unreachable. |
| `random_seed` | 0 | | Affects only the null, not the measurement. |

> Two metric families, two timescales. `directed` = L(`tau_directed_frames`), the
> coarse path. `runs_total` = the run-detection metric on coarse steps at the shorter
> `tau_frames`, and `runs_p` / `runs_z` / `runs_excess` belong to `runs_total`, not to
> `directed`. Do not report `runs_p` as the significance of `directed`. Both τ values
> are written as columns so the file records which window each metric used.

> Choosing τ. τ = 4 s was chosen against simulated ground truth: it recovers 86% of a
> known 5 px directed run against a pure-noise floor of 2.6 px. Scale both τ values by
> your frame rate (~1 s and ~4 s in frames), then check that mover/confined separation
> rises with τ and flattens (the notebook does this sweep).

## `classify`: labels

| parameter | default | what it does |
|---|---|---|
| `use_permutation` | `true` | `true`: a mover must beat its own null. `false`: fixed rate threshold. |
| `p_threshold` | 0.05 | Used when `use_permutation` is true. |
| `mover_rate_px_per_s` | 0.15 | Used only when `use_permutation` is false. |
| `mover_min_net_px` | 4.0 | Guard so a tiny but significant displacement cannot score as a mover. |
| `exclude_max_step_px` | 5.0 | One jump this large ⇒ suspected mislink. |
| `exclude_n_big_steps` | 5 | This many big steps ⇒ suspected repeated identity swap. |
| `big_step_px` | 3.0 | What counts as "big" for the rule above. |

> `excluded` is a tracking-quality judgement, not biology. These are the signatures of
> identity swaps between nearby vesicles; leaving them in inflates the mover count. A
> rising `excluded` count usually means `link.search_range_px` is too large or the
> field is too dense.

## `render`: output

| parameter | default | what it does |
|---|---|---|
| `three_panel` | `true` | `raw │ all vesicles │ classified`, the QC image. |
| `per_vesicle_images` | `true` | One image per vesicle: trajectory with all three distances drawn and printed. |
| `per_vesicle_videos` | `false` | One file per vesicle; off by default. |
| `overview_video` | `false` | Whole field, tracks coloured by class. |
| `max_vesicle_outputs` | 25 | Cap on per-vesicle files (movers first) so a dense field cannot emit thousands. Recorded in `summary.json`. |
| `trail_frames` | 40 | Trail length in acquisition frames (not sampled frames), identical in both video kinds. `0` = no trail. The trail is solid, not fading. |
| `frame_step` | 4 | Temporal downsample for videos (4 = every 4th frame). |
| `fps` | 20 | Playback rate of written videos. |
| `dpi` | 150 | Raster resolution of PNG figures. |
| `percentiles` | `[1.0, 99.7]` | Display stretch. Affects the display only, never a measurement. |
| `ffmpeg` | `ffmpeg` | Full path if not auto-found. Missing ffmpeg skips videos with a warning; images are unaffected. |

---

## Overriding without editing the file

Dotted paths, and `copy()` does not change the original, so a sweep is a loop:

```python
cfg = Config.load("config/default.yaml", dt_seconds=0.05, um_per_px=0.107)
cfg.set("detect.threshold_sigma", 2.5)

for thr in [2.5, 3.0, 3.5]:
    analyse(movie, cfg.copy(**{"detect.threshold_sigma": thr}))
```

Unknown keys are rejected at load, so a typo like `thresold_sigma` raises an error
instead of being ignored.

Every run writes `config_used.yaml` beside its outputs, so any figure can be traced
back to the exact parameters that made it.
