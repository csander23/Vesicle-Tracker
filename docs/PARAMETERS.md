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
| `min_length_frames` | 3 | frames | **Hard floor only** — below this no metric exists at all (a 2-frame track has one step). The scientific length cut is `filters.min_observed_frames`, which labels instead of deleting. |
| `merge_radius_px` | 4.0 | px | End-to-start distance allowed when rejoining a fragment. `0` disables merging. |
| `merge_max_gap_frames` | 15 | frames | **The "look back" window.** How long after losing a vesicle the tracker may still rejoin a new fragment to it. |
| `merge_overlap_tolerance` | 2 | frames | Overlap still treated as one vesicle. Fragments that genuinely coexist are different vesicles and are never merged. |

> **Why merging is conservative.** The obvious implementation joins tracks whose *mean*
> positions are close. That fuses two **different** vesicles that occupied the same spot
> at different times — on real data it produced tracks with 120-frame gaps under a
> `memory` of 15, silently inflating lifetimes. A merge now requires temporal
> disjointness, a short gap, and proximity **at the junction**, not on average.

## `size` — how big each vesicle is

Intensity-weighted second moment of a window around each spot: fast (vectorised per
frame) and within a few percent of a Gaussian fit for well-separated spots.

| parameter | default | unit | what it does |
|---|---|---|---|
| `enabled` | `true` | | Turn sizing off to save time. |
| `window_px` | 4 | px | Half-width of the measurement window (4 → 9×9). Too large pulls in neighbours; too small truncates the spot. |
| `psf_sigma_px` | `null` | px | PSF width to deconvolve. **`null` calibrates it from this movie** — recommended. |
| `psf_from_percentile` | 5.0 | % | When calibrating, which percentile of measured widths counts as "a point source". |
| `max_frames` | 200 | frames | Frames sampled per track. More buys no precision. |

> **Read this before quoting a size.** A vesicle is below the diffraction limit, so what
> the microscope records is mostly the PSF. Three columns come out:
> `sigma_px` (measured, includes PSF), `sigma_deconv_px` (excess over the PSF), and
> `at_diffraction_limit` (True when the object is unresolved — deconvolved size is then
> `0`, not NaN, because "unresolved" is a real state, not missing data).
>
> **There is a noise floor.** The second moment is biased upward by noise. Measured on
> this pipeline: synthetic point sources with a true size of **zero** report
> σ = 1.394 px and 0.41 px of deconvolved "size"; real Rab5 endosomes at the same
> settings give σ = 2.010 px and 0.99 px. Treat ~0.4 px of deconvolved width as
> indistinguishable from zero, and recalibrate that floor for your own optics with
> sub-resolution beads or `examples/make_synthetic.py`.
>
> **Prefer `sigma_px` for comparisons.** Between conditions imaged identically the PSF
> contribution is common, so a difference in raw σ is real and needs no deconvolution.

## `filters` — which vesicles count

**Filters label; they never delete.** Every vesicle is written to `vesicles_all.csv`
with `passes_filter` and `filter_reason`; the passing subset also goes to
`vesicles_filtered.csv`. Any threshold can be re-derived later from the `all` file, so
a filter is a reversible, auditable choice rather than lost data. `null` = rule off.

| parameter | default | what it does |
|---|---|---|
| `min_observed_frames` | 180 | Frames actually **detected** (not span). Must be ≥ `2 × tau_directed_frames`. |
| `max_observed_frames` | `null` | Upper bound on detected frames. |
| `min_lifetime_s` | `null` | On **span** — first to last sighting. |
| `max_lifetime_s` | `null` | Upper bound on span. **See the bias warning below.** |
| `min_frac_observed` | 0.0 | Rejects tracks that are mostly gap (`0.5` = at least half the span seen). |
| `max_longest_gap` | `null` | Longest single dropout allowed, in frames. |
| `exclude_censored` | `false` | Drop tracks touching the first or last frame. |
| `exclude_classes` | `[excluded]` | `excluded` = suspected identity swap; a tracking judgement, not biology. |
| `rois` | `[]` | Keep only these regions, e.g. `[soma]`. Empty keeps all, including `outside`. |
| `min_sigma_px` / `max_sigma_px` | `null` | Size bounds. |

> **`directed` needs two τ-windows to exist.** A track shorter than
> `2 × tau_directed_frames` spans under two coarse windows, so no coarse path can be
> formed and `directed` comes back **NaN**, flagged `directed_measurable = False`.
> Earlier this returned `0.0`, which was silently wrong — a vesicle travelling 20 px in
> a straight line over 60 frames reported `directed = 0.00` while `net` said `20.00`,
> and because `net_coarse` was 0 too the ordering check passed. Config validation now
> warns when the length filter admits tracks that cannot be scored.

> **`max_lifetime_s` selects for tracking failure.** An upper bound on lifetime
> preferentially keeps tracks the tracker *lost* early — it does not select for
> short-lived biology. Use it only with a reason, and check the censored fraction.

> **Censoring.** A vesicle already present in frame 0, or still present in the last
> frame, has a lifetime whose end was never observed. Pooling those with complete
> observations biases mean lifetime **downward**. They are flagged `is_censored`,
> `censored_start`, `censored_end` — on a 400-frame crop of real data nearly every
> long-lived vesicle is censored, so check this before quoting a mean lifetime.

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
| `n_permutations` | 200 | | Per-vesicle null. `p` floors at `1/(n+1)`, so 200 → smallest possible p is 0.005. `0` disables the p-value. Validation rejects a `p_threshold` below that floor, which would make `mover` unreachable. |
| `random_seed` | 0 | | Affects **only** the null, never the measurement. |

> **Two metric families, two timescales — do not mix them up.**
> `directed` = L(`tau_directed_frames`), the coarse path. `runs_total` = the
> run-detection metric on coarse steps at the shorter `tau_frames`, and
> **`runs_p` / `runs_z` / `runs_excess` belong to `runs_total`, not to `directed`.**
> These were originally named `directed_p` / `directed_runs` and sat beside `directed`
> in the output, which invited "directed displacement was significant (p < 0.05)"
> written about a different number. Both τ values are emitted as columns so the file
> is self-describing.

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
| `trail_frames` | 40 | Trail length in **acquisition frames** (not sampled frames), identical in both video kinds. `0` = no trail. The trail is solid, not fading. |
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
