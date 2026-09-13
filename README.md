# vesicletrack

Detect, track and score intracellular vesicles in time-lapse microscopy. All parameters
live in one config file; the only other input is the movie.

```python
from vesicletrack import Config, analyse

cfg = Config.load("config/default.yaml", dt_seconds=0.0446, um_per_px=0.107)
res = analyse("cell.tif", cfg)

res.counts()        # mover / confined / excluded
res.vesicles        # one row per vesicle, all metrics
res.save()          # tables + figures + videos
```

From the shell, one command takes movies to every level of CSV:

```bash
vesicletrack "data/*.tif" -c config/default.yaml -o output \
    --sheet samples.csv --by genotype --mask cell_outline.tif --rois soma_and_processes.zip
```

`--mask` restricts where detection runs (one traced cell); `--rois` labels what was
detected by region (soma vs process). Both are optional.

`samples.csv` supplies the metadata. Nothing is parsed from filenames:

```csv
file,genotype,batch
cellA.tif,WT,B1
cellB.tif,ApoE4,B1
```

---

## The problem this solves

The two simple measures of vesicle movement both fail on high-frame-rate data, and
they fail differently:

| metric | definition | failure |
|---|---|---|
| **gross** | Σ per-frame step lengths | a stationary vesicle accumulates localisation noise every frame. Over 1000 frames it "travels" hundreds of pixels. Gross path is mostly noise. |
| **net** | \|end − start\| | blind to a vesicle that runs out and comes back, or makes several runs in different directions. A genuinely motile trajectory scores ≈ 0. |

In the demo below, a vesicle that moved 14 px reports 179 px of gross path, and the
stationary ones report 181 px. Gross does not separate them.

The directed metric averages positions in windows of τ frames before measuring the
path:

```
c_k = mean of r_i over window k          L(τ) = Σ_k |c_{k+1} − c_k|
```

Uncorrelated localisation error averages down as 1/√τ while real transport is
unchanged, and reversals faster than τ cancel inside a window. Coarse-graining can
only shorten a path, so the ordering

```
gross = L(Δt)  ≥  L(τ)  ≥  L(T) = net_coarse
```

holds for every vesicle at every τ. `Result.check()` asserts it; a non-empty result
means something upstream is wrong rather than a problem with the metric.

### Significance, not a threshold

A vesicle is a mover if it beats its own null rather than a fixed number. Each coarse
step keeps its magnitude and is given a random heading, which gives an isotropic walk
with that vesicle's own step-size distribution. The directed distance that still
accumulates is what run-detection finds by chance, and the per-vesicle permutation
p-value follows from it.

> The null randomises step direction and leaves step order alone. Permuting step order
> would leave a smooth run's vectors all pointing the same way, so the null would
> reproduce the observation and the clearest movers would score p ≈ 1.

---

## Install

```bash
conda env create -f environment.yml && conda activate vesicletrack
pip install -e .
pytest -q                      # 66 tests, ~1.5 min
```

or

```bash
pip install -e ".[all]"      # opencv + nd2 + pyarrow
```

`ffmpeg` is needed only for video output. It is looked for on `PATH` and next to the
running Python (where conda installs it). If it is missing, videos are skipped with a
printed warning and images are unaffected.

---

## Try it without data

```bash
python examples/make_synthetic.py           # 24 static + 6 movers, known truth, 3 px drift
python -m vesicletrack.cli examples/synthetic.tif --dt 0.05 -o examples/output
```

Recovers all 30 vesicles and exactly the 6 planted movers, p = 0.005 for the movers
and p = 1.00 for the static ones. That is `vesicles_filtered.csv`; `vesicles_all.csv`
also holds the 670 short noise fragments (under 40 observed frames) that the filters
labelled and kept. None of them scores as a mover.

| class | net | directed | gross | p |
|---|---|---|---|---|
| mover (n=6) | 13.96 px | 11.68 px | 179.1 px | 0.005 |
| confined (n=24) | 0.44 px | 0.55 px | 180.9 px | 1.000 |

Net separates the classes 32-fold. Gross does not separate them at all; the confined
vesicles report more gross path than the movers. This is the reason for the directed
metric.

`p = 0.005` is the floor, `1/(n_permutations + 1)`: with 200 permutations the null
never beat the observation. It cannot go lower without raising
`metrics.n_permutations`.

Planted positions are rejection-sampled with a 14 px minimum separation held over the
whole movie, so no mover passes close enough to a static vesicle to cause an identity
swap. Without that the expected answer would depend on field size.

Run this after changing any parameter. It is a quick check that detection still works.

---

## Config files: YAML or JSON

Use YAML. `config/default.yaml` is the single source of truth and the only config file
shipped, because it carries comments, and the comments document the parameters.

```python
Config.load("config/default.yaml")     # the normal path
Config.load(None, dt_seconds=0.05)     # all defaults, overridden inline
Config.load("my_run.json")             # JSON also loads, if you have one
```

JSON is supported for configs generated by another program (a sweep script, a LIMS, a
web front end) rather than maintained by hand. It is not shipped as a second copy of
the defaults, because two files holding the same values diverge as soon as one is
edited.

`save()` writes whichever format the extension asks for, so a config generated in a
notebook can be passed to another program:

```python
cfg.save("runs/exp1.json")     # JSON
cfg.save("runs/exp1.yaml")     # YAML
cfg.to_json()                  # or just the string
```

Unknown keys are rejected in either format, so a typo such as `thresold_sigma` raises
an error at load instead of being ignored.

Overrides use dotted paths and do not mutate the original, so a sweep is a loop:

```python
for thr in [2.5, 3.0, 3.5]:
    analyse(movie, cfg.copy(**{"detect.threshold_sigma": thr}))
```

## Setting parameters for a new dataset

Two fields are required and cannot be read from the images:

```yaml
dt_seconds: 0.0446    # every rate scales through this
um_per_px: 0.107      # null keeps everything in pixels
```

If `dt_seconds` is wrong, every rate is wrong and the images give no hint of it.
`io.frame_interval_from_nd2()` reads it from an ND2 if the file records timestamps.

Then, in rough order of importance:

| parameter | raise it if | lower it if |
|---|---|---|
| `detect.threshold_sigma` | too many spurious spots | vesicles are being missed |
| `detect.psf_sigma_px` | spots are larger than the default 1.3 px σ | |
| `link.search_range_px` | vesicles move far between frames | you see identity swaps |
| `link.min_length_frames` | tracks are too short to score | you lose real short-lived vesicles |
| `metrics.tau_directed_frames` | noise still dominates | genuine fast reversals are being erased |

`metrics.tau_frames` ≈ 1 s and `tau_directed_frames` ≈ 4 s at your frame rate is a
reasonable starting point. τ = 4 s was chosen against simulated ground truth: it
recovers 86% of a known 5 px directed run against a pure-noise floor of 2.6 px.

Drift correction is on by default and should stay on. Stage drift moves every vesicle
together, so it reads as directed transport in all of them at once and would be scored
as such. The measured drift span is reported in `summary.json`; a large value on data
you believed was stable means the uncorrected numbers would have been wrong.

---

## Output: every level, nothing dropped

```
output/
  vesicles_all.csv      one row per vesicle, pooled across movies, nothing removed
  vesicles_filtered.csv the subset passing the filters
  per_video.csv         one row per video
  per_<key>.csv         one row per genotype / batch / whatever you passed to --by
  long.csv              tidy: level, unit, group, metric, value
  batch_summary.csv     one row per movie: counts, drift, runtime, failures

  <movie>/
    config_used.yaml     exact parameters, so a figure always traces to its run
    tracks.parquet       particle, frame, x, y  (every vesicle, every frame)
    vesicles_all.csv     this movie's vesicles, with passes_filter + filter_reason
    vesicles_filtered.csv
    summary.json         counts, drift span, filter breakdown, ROI coverage
    three_panel.png      raw | all vesicles | classified
    distances.png        net / directed / gross distributions
    vesicles/*.png       per vesicle: trajectory + all three distances
    vesicle_videos/*.mp4 per-vesicle crops                  (render.per_vesicle_videos)
    overview.mp4         whole field, coloured by class     (render.overview_video)
```

Filters add a label. No vesicle is removed from the output. Every tracked vesicle is in
`vesicles_all.csv` with `passes_filter` and `filter_reason` naming the rule it failed.
Any threshold can be re-derived from that file later, so a filter is a reversible,
auditable choice, and the question "how many did we exclude, and were they different?"
can always be answered.

Group-level `n` is the number of videos, not the number of vesicles. Vesicles within
one cell share a cell, a transfection and a field of view; treating each as a
replicate inflates `n` by hundreds and produces significance that will not survive a
nested analysis. `per_<key>.csv` reports `n_videos` and a `sem` across video means.
There is no option to compute group statistics per vesicle.

The three panels answer two questions to check before trusting any number: did
detection find the vesicles (middle), and did classification label them sensibly
(right).

Per-vesicle output is capped by `render.max_vesicle_outputs` (default 25, movers
first) so a dense field cannot emit thousands of files. The cap is recorded in
`summary.json`.

### Mask and ROI

Mask and ROI answer different questions and are separate arguments:

```python
res = analyse("cell.tif", cfg, mask="cell_outline.tif")     # where to detect
res = analyse("cell.tif", cfg, rois="soma_and_processes.zip") # how to label what was found
```

`mask` restricts detection: nothing outside it is tracked. Use it to analyse one cell
when a field holds several. `rois` labels each vesicle by region and keeps the rest.
Both take the same inputs (ImageJ `.roi`/`.zip`, mask or label images, arrays,
polygons), and both are applied in drift-corrected coordinates. When drift is not
negligible, draw them on a projection of `Result.stack` rather than the raw file.

#### ROI: in and out, never dropped

```python
res = analyse("cell.tif", cfg, rois="soma_and_processes.zip")   # ImageJ RoiSet
res = analyse("cell.tif", cfg, rois={"soma": [(x, y), ...],     # or polygons
                                     "processes": [...]})
res.vesicles.roi.value_counts()      # soma / processes / outside
```

Accepts ImageJ `.roi` and `.zip`, label or mask images, arrays, and named polygons.
Vesicles outside every region are kept and labelled `outside`, because outside is
usually the comparison group and dropping them would leave nothing to compare against.
Vesicles that move between regions are assigned by majority and flagged
(`roi_changed`, `roi_frac`).

### Lifetime, gaps and censoring

`span_frames` (first to last sighting) and `observed_frames` (frames actually detected)
differ whenever the linker bridged a dropout; on real data the medians were 398 vs 167.
Reporting only span describes a vesicle as present in frames where nothing was
detected, so both are carried, with `n_gaps`, `longest_gap` and `frac_observed`.

`is_censored` marks vesicles present in the first or last frame: their true lifetime
was not observed, and pooling them with complete observations biases mean lifetime
downward.

### Size

`sigma_px` (measured), `sigma_deconv_px` (excess over the PSF), `at_diffraction_limit`.
See [`size.py`](src/vesicletrack/size.py) for the caveats: a vesicle is below the
diffraction limit, so σ is mostly the PSF, and the estimator has a measured noise floor
of ~0.4 px (synthetic point sources of zero true size deconvolve to 0.41 px). Prefer
raw `sigma_px` when comparing conditions imaged identically; the PSF contribution is
common to both and cancels.

### Key columns in `vesicles_all.csv`

| column | meaning |
|---|---|
| `net`, `gross`, `directed` | the three distances, px (`*_um` when `um_per_px` set) |
| `net_coarse` | \|c_K − c_1\|; the quantity the ordering theorem bounds |
| `*_rate` | per second |
| `directed_measurable` | False when the track spans under two τ-windows; `directed` is then NaN rather than 0 |
| `runs_total` | run-detection metric at the shorter `tau_frames` |
| `runs_p`, `runs_z`, `runs_excess` | the permutation test of `runs_total`, not of `directed` |
| `span_frames`, `observed_frames` | track span vs frames actually detected; they differ whenever a dropout was bridged |
| `n_gaps`, `longest_gap`, `frac_observed` | how much of the span was really seen |
| `is_censored` | present in the first or last frame, so its true lifetime is unknown |
| `sigma_px`, `sigma_deconv_px`, `at_diffraction_limit` | size (see the caveats above) |
| `roi`, `roi_frac`, `roi_changed` | which region, and whether it moved between regions |
| `passes_filter`, `filter_reason` | filter label and the rule failed; nothing is removed |
| `klass` | `mover` / `confined` / `excluded` / `invalid` |
| `persistence` | lag-1 direction cosine; ≈ −0.5 for pure localisation noise |
| `rg`, `aniso` | radius of gyration, anisotropy |

`invalid` means a non-finite coordinate; no biological claim is made about it.
`excluded` is a tracking-quality judgement rather than biology: a single large jump or
repeated large steps is the signature of an identity swap between nearby vesicles.
Leaving those in inflates the mover count.

---

## Caveats

- Detection is DAOStarFinder, chosen so the toolkit ships no model weights. A learned
  detector will do better on dim or dense fields.
- Tracking is `trackpy` nearest-neighbour linking. It has no explicit model of
  merging/splitting vesicles; fusion events appear as one track ending and another
  beginning.
- The `mover` / `confined` split is a statement about directional persistence, not
  about mechanism. A vesicle can be motor-driven and still score confined if it never
  holds a direction for longer than τ.
- Everything is 2D. Movement through the focal plane is invisible and reads as a
  vesicle disappearing.

---

## Layout

```
vesicletrack/
├── README.md                  what it does and why
├── config/
│   └── default.yaml           every parameter, commented. Copy and edit this
├── docs/
│   └── PARAMETERS.md          full reference: units, defaults, which way to tune
├── notebooks/
│   └── vesicle_analysis.ipynb walkthrough: run → inspect → tune → batch
├── examples/
│   └── make_synthetic.py      ground-truth movie (24 static + 6 movers)
├── tests/                     pytest -q
│   ├── conftest.py            the synthetic movie and a config scaled to it
│   ├── test_smoke.py          does it get the known answer right?
│   ├── test_robustness.py     edge cases, odd inputs, every switch
│   └── test_docs.py           docs cannot drift from the code
└── src/vesicletrack/
    ├── config.py              parameters, validation, YAML/JSON round-trip
    ├── io.py                  load stacks (tif/nd2/npy), sample sheets, write tables
    ├── preprocess.py          stage drift correction
    ├── detect.py              per-frame spot detection + NMS
    ├── linking.py             trackpy linking, fragment merge, short-track floor
    ├── metrics.py             net / gross / directed, permutation null, classification
    ├── size.py                vesicle width, PSF calibration, deconvolution
    ├── roi.py                 masks and regions: load, label, assign
    ├── filters.py             label (never delete) by the `filters:` rules
    ├── aggregate.py           vesicle -> video -> group tables, n = videos
    ├── render.py              three-panel, per-vesicle images, videos
    ├── pipeline.py            analyse() / analyse_many() / Result
    └── cli.py                 command line
```

Suggested reading order: `config/default.yaml` (what is adjustable), then `metrics.py`
(what is measured), then `pipeline.py` (the order things happen in).

Parameter reference: [docs/PARAMETERS.md](docs/PARAMETERS.md) lists every parameter
with its units, default, and which direction to move it. The same notes are inline in
`config/default.yaml`, and a test checks that both stay consistent with the code.

## Licence

MIT.
