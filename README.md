# vesicletrack

Detect, track and score intracellular vesicles in live imaging microscopy. 

```python
from vesicletrack import Config, analyse

cfg = Config.load("config/default.yaml", dt_seconds=0.0446, um_per_px=0.107)
res = analyse("cell.tif", cfg)

res.counts()        # mover / confined / excluded
res.vesicles        # one row per vesicle, all metrics
res.save()          # tables + figures + videos
```

Or from the shell — one command from movies to every level of CSV:

```bash
vesicletrack "data/*.tif" -c config/default.yaml -o output \
    --sheet samples.csv --by genotype --mask cell_outline.tif --rois soma_and_processes.zip
```

`--mask` says where to *detect* (one traced cell); `--rois` says how to *label* what
was detected (soma vs process). Both are optional.

`samples.csv` supplies the metadata (the package never guesses it from filenames):

```csv
file,genotype,batch
cellA.tif,WT,B1
cellB.tif,ApoE4,B1
```

---

## The problem this solves

The two obvious ways to measure vesicle movement both fail on high-frame-rate data,
in opposite directions:

| metric | definition | failure |
|---|---|---|
| **gross** | Σ per-frame step lengths | a *stationary* vesicle accumulates localisation noise every frame. Over 1000 frames it "travels" hundreds of pixels. Gross path is mostly noise. |
| **net** | \|end − start\| | blind to a vesicle that runs out and comes back, or makes several runs in different directions. A genuinely motile trajectory scores ≈ 0. |

In the demo below, a vesicle that moved **14 px** reports **179 px** of gross path — and
the *stationary* ones report **181 px**. Gross cannot tell them apart at all.

**directed** sits between them. Positions are averaged in windows of τ frames before
the path is measured:

```
c_k = mean of r_i over window k          L(τ) = Σ_k |c_{k+1} − c_k|
```

Uncorrelated localisation error averages down as 1/√τ while real transport is
untouched, and reversals faster than τ cancel inside a window. Because coarse-graining
can only shorten a path, the ordering

```
gross = L(Δt)  ≥  L(τ)  ≥  L(T) = net_coarse
```

holds for **every** vesicle at **every** τ. `Result.check()` asserts it; a non-empty
result means something upstream is wrong, not that the metric misbehaved.

### Significance, not a threshold

A vesicle is a *mover* if it beats **its own** null, not a fixed number. Each coarse
step keeps its magnitude but is given a random heading — an isotropic walk with that
vesicle's exact step-size distribution. Whatever directed distance that still
accumulates is what run-detection finds by chance, and the per-vesicle permutation
p-value follows.

> The null randomises **direction**, not order. Permuting step *order* leaves a smooth
> run's vectors all pointing the same way, so the null reproduces the observation and
> the clearest movers score p ≈ 1. This is a trap worth knowing about.

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

`ffmpeg` is needed only for video output. It is auto-detected on `PATH` and next to
the running Python (which is where conda puts it); if it is missing, videos are
skipped with a printed warning and images are unaffected.

---

## Try it without data

```bash
python examples/make_synthetic.py           # 24 static + 6 movers, known truth, 3 px drift
python -m vesicletrack.cli examples/synthetic.tif --dt 0.05 -o examples/output
```

Recovers all 30 vesicles and exactly the 6 planted movers, p = 0.005 against
p = 1.00 for the static ones. This is `vesicles_filtered.csv`; `vesicles_all.csv` also
holds the 670 short noise fragments (under 40 observed frames) that the filters
labelled rather than deleted - none of them scores as a mover.

| class | net | directed | gross | p |
|---|---|---|---|---|
| mover (n=6) | 13.96 px | 11.68 px | 179.1 px | 0.005 |
| confined (n=24) | 0.44 px | 0.55 px | 180.9 px | 1.000 |

Net separates the classes 32-fold; **gross separates them not at all** — the confined
vesicles report *more* gross path than the movers. That is the whole argument for the
directed metric, in one table.

`p = 0.005` is the floor, `1/(n_permutations + 1)`: with 200 permutations the null
never once beat the observation. It is not a p-value that can go lower without raising
`metrics.n_permutations`.

Planted positions are rejection-sampled with a 14 px minimum separation held over the
*whole* movie, so no mover ever passes close enough to a static vesicle to cause an
identity swap. Without that the "correct" answer changes with field size.

Run this after changing any parameter — it is the fastest check that a change did not
break detection.

---

## Config files: YAML or JSON

**YAML is the format to use.** `config/default.yaml` is the single source of truth and
the only config file shipped, because it carries comments — and the comments are half
of what makes the parameters usable.

```python
Config.load("config/default.yaml")     # the normal path
Config.load(None, dt_seconds=0.05)     # all defaults, overridden inline
Config.load("my_run.json")             # JSON also loads, if you have one
```

JSON is supported for the case where a config is *generated* by another program — a
sweep script, a LIMS, a web front end — rather than maintained by a person. It is not
shipped as a second copy of the defaults: two files holding the same values drift the
moment one is edited.

`save()` writes whichever format the extension asks for, so a config generated in a
notebook can be handed straight to another program:

```python
cfg.save("runs/exp1.json")     # JSON
cfg.save("runs/exp1.yaml")     # YAML
cfg.to_json()                  # or just the string
```

Unknown keys are rejected in either format, so `thresold_sigma` fails loudly at load
instead of being silently ignored and leaving you wondering why the parameter did
nothing.

Overrides use dotted paths and never mutate the original, which makes a sweep a loop:

```python
for thr in [2.5, 3.0, 3.5]:
    analyse(movie, cfg.copy(**{"detect.threshold_sigma": thr}))
```

## Setting parameters for a new dataset

Two fields are **required** and are not guessable from the images:

```yaml
dt_seconds: 0.0446    # every rate scales through this
um_per_px: 0.107      # null keeps everything in pixels
```

Set `dt_seconds` wrong and every rate is wrong while nothing looks wrong.
`io.frame_interval_from_nd2()` reads it from an ND2 if the file records timestamps.

Then, in rough order of how much they matter:

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

**Drift correction is on by default and should stay on.** Stage drift moves every
vesicle together, so it reads as directed transport in all of them at once —
precisely what the directed metric is built to detect. The measured drift span is
reported in `summary.json`; a large value on data you believed was stable means the
uncorrected numbers would have been wrong.

---

## Output — every level, nothing dropped

```
output/
  vesicles_all.csv      one row per VESICLE, pooled across movies, nothing removed
  vesicles_filtered.csv the subset passing the filters
  per_video.csv         one row per VIDEO
  per_<key>.csv         one row per genotype / batch / whatever you passed to --by
  long.csv              tidy: level, unit, group, metric, value
  batch_summary.csv     one row per movie: counts, drift, runtime, failures

  <movie>/
    config_used.yaml     exact parameters — a figure always traces to its run
    tracks.parquet       particle, frame, x, y  (every vesicle, every frame)
    vesicles_all.csv     this movie's vesicles, with passes_filter + filter_reason
    vesicles_filtered.csv
    summary.json         counts, drift span, filter breakdown, ROI coverage
    three_panel.png      RAW | ALL VESICLES | CLASSIFIED
    distances.png        net / directed / gross distributions
    vesicles/*.png       per vesicle: trajectory + all three distances
    vesicle_videos/*.mp4 per-vesicle crops                  (render.per_vesicle_videos)
    overview.mp4         whole field, coloured by class     (render.overview_video)
```

**Filters label; they never delete.** Every tracked vesicle is in `vesicles_all.csv`
with `passes_filter` and `filter_reason` naming the rule it failed. Any threshold can
be re-derived from that file later, so a filter is a reversible, auditable choice
rather than lost data — and you can always answer "how many did we exclude, and were
they different?".

**Group-level `n` counts videos, not vesicles.** Vesicles within one cell share a cell,
a transfection and a field of view; treating each as a replicate inflates `n` by
hundreds and produces significance that will not survive a nested analysis.
`per_<key>.csv` reports `n_videos` and a `sem` across video means. The per-vesicle
route to pseudoreplication is not offered.

The three panels answer the two questions worth asking before trusting any number:
did detection *find* the vesicles (middle), and did classification *label* them
sensibly (right).

Per-vesicle output is capped by `render.max_vesicle_outputs` (default 25, movers
first) so a dense field cannot emit thousands of files. The cap is recorded in
`summary.json` rather than applied silently.

### Mask and ROI

Two different questions, two different arguments:

```python
res = analyse("cell.tif", cfg, mask="cell_outline.tif")     # WHERE to detect
res = analyse("cell.tif", cfg, rois="soma_and_processes.zip") # how to LABEL what was found
```

`mask` restricts detection: nothing outside it is ever tracked. Use it for "this cell
only" when a field holds several. `rois` labels each vesicle by region and keeps the
rest. Both take the same inputs (ImageJ `.roi`/`.zip`, mask or label images, arrays,
polygons), and both are applied in **drift-corrected** coordinates - draw them on a
projection of `Result.stack`, not the raw file, when drift is not negligible.

#### ROI: in and out, never dropped

```python
res = analyse("cell.tif", cfg, rois="soma_and_processes.zip")   # ImageJ RoiSet
res = analyse("cell.tif", cfg, rois={"soma": [(x, y), ...],     # or polygons
                                     "processes": [...]})
res.vesicles.roi.value_counts()      # soma / processes / outside
```

Accepts ImageJ `.roi` and `.zip`, label or mask images, arrays, and named polygons.
Vesicles **outside every region are kept and labelled `outside`**, because outside is
usually the comparison group — a pipeline that silently drops them cannot answer the
question the ROI was drawn to ask. Vesicles that move between regions are assigned by
majority and flagged (`roi_changed`, `roi_frac`).

### Lifetime, gaps and censoring

`span_frames` (first to last sighting) and `observed_frames` (frames actually detected)
are different numbers whenever the linker bridged a dropout — on real data the medians
were **398 vs 167**. Reporting only span describes a vesicle as present in frames where
nothing was detected, so both are carried, with `n_gaps`, `longest_gap` and
`frac_observed`.

`is_censored` marks vesicles present in the first or last frame: their true lifetime was
never observed, and pooling them with complete observations biases mean lifetime
**downward**.

### Size

`sigma_px` (measured), `sigma_deconv_px` (excess over the PSF), `at_diffraction_limit`.
Read [`size.py`](src/vesicletrack/size.py) before quoting a number: a vesicle is below
the diffraction limit, so σ is mostly the PSF, and the estimator has a measured noise
floor of **~0.4 px** — synthetic point sources of *zero* true size deconvolve to
0.41 px. Prefer raw `sigma_px` when comparing conditions imaged identically; the PSF
contribution is common to both and cancels.

### Key columns in `vesicles_all.csv`

| column | meaning |
|---|---|
| `net`, `gross`, `directed` | the three distances, px (`*_um` when `um_per_px` set) |
| `net_coarse` | \|c_K − c_1\|; the quantity the ordering theorem bounds |
| `*_rate` | per second |
| `directed_measurable` | False when the track spans under two τ-windows; `directed` is then NaN, never 0 |
| `runs_total` | run-detection metric at the **shorter** `tau_frames` |
| `runs_p`, `runs_z`, `runs_excess` | the permutation test **of `runs_total`** — not of `directed` |
| `span_frames`, `observed_frames` | track span vs frames actually detected — different whenever a dropout was bridged |
| `n_gaps`, `longest_gap`, `frac_observed` | how much of the span was really seen |
| `is_censored` | present in the first or last frame, so its true lifetime is unknown |
| `sigma_px`, `sigma_deconv_px`, `at_diffraction_limit` | size (read the caveats) |
| `roi`, `roi_frac`, `roi_changed` | which region, and whether it moved between regions |
| `passes_filter`, `filter_reason` | filters label, they never delete |
| `klass` | `mover` / `confined` / `excluded` / `invalid` |
| `persistence` | lag-1 direction cosine; ≈ −0.5 for pure localisation noise |
| `rg`, `aniso` | radius of gyration, anisotropy |

`invalid` means a non-finite coordinate: no biological claim is made about it.
`excluded` is a **tracking-quality** judgement, not biology: a single large jump or
repeated large steps is the signature of an identity swap between nearby vesicles.
Leaving those in inflates the mover count.

---

## Caveats

- Detection is DAOStarFinder, chosen so the toolkit ships no model weights. A learned
  detector will do better on dim or dense fields.
- Tracking is `trackpy` nearest-neighbour linking. It has no explicit model of
  merging/splitting vesicles; fusion events appear as one track ending and another
  beginning.
- The `mover` / `confined` split is a statement about **directional persistence**, not
  about mechanism. A vesicle can be motor-driven and still score confined if it never
  holds a direction for longer than τ.
- Everything is 2D. Movement through the focal plane is invisible and will read as a
  vesicle disappearing.

---

## Layout

```
vesicletrack/
├── README.md                  you are here — what it does and why
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

Read in this order to understand it: `config/default.yaml` (what is adjustable) →
`metrics.py` (what is measured) → `pipeline.py` (the order things happen in).

**Parameter reference: [docs/PARAMETERS.md](docs/PARAMETERS.md)** — every parameter
with its units, default, and which direction to move it. The same notes are inline in
`config/default.yaml`, and a test asserts the two never drift apart from the code.

## License

MIT.
