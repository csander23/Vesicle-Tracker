# vesicletrack

Detect, track and score intracellular vesicles in live imaging microscopy.

The goal of this software is to extract data on vesicle movement from live-imaging
videos of fluorescently labelled vesicles. Each frame is motion corrected, vesicles are
found with a pretrained neural network (deepBLINK, Eichenberger et al. 2021), the
detections are joined into tracks with the gaps the network leaves repaired, and every
vesicle gets three movement distances: gross, the path it actually travelled; net, start
to end; and directed, the path with the frame-by-frame jitter smoothed away. A
significance test then says whether a vesicle moved more than chance. The aim is to let
new questions be asked about the quantity and quality of intracellular vesicle movement.

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

## What it measures

The two simple measures of vesicle movement both fail on high-frame-rate data, and
they fail differently:

| metric | definition | failure |
|---|---|---|
| **gross** | Σ per-frame step lengths | a stationary vesicle accumulates localisation noise every frame. Over 1000 frames it "travels" hundreds of pixels. Gross path is mostly noise. |
| **net** | \|end − start\| | blind to a vesicle that runs out and comes back, or makes several runs in different directions. A genuinely motile trajectory scores ≈ 0. |

In the demo below, a vesicle that moved 14 px reports 178 px of gross path, and the
stationary ones report 186 px. Gross does not separate them.

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

## How detection and tracking work

**Detection** is deepBLINK (Eichenberger et al., 2021, https://github.com/BBQuercus/deepBlink),
a convolutional network trained to find diffraction-limited spots, with its
pretrained `vesicle` model bundled in the package (9.5 MB). The network scores each
frame on its own: for every 4×4-pixel cell it returns the probability that a spot
centre lies there and the centre's sub-pixel offset. A cell above
`detect.prob_threshold` (default 0.5) is a detection. On a laptop CPU this takes
about 45 ms per frame, so a 60 s movie at 22 frames per second needs about a minute.

**Linking** joins detections in consecutive frames with trackpy (nearest neighbour
within `link.search_range_px`, a memory of `link.memory_frames` for short dropouts).

**Gap recovery** is what makes the network usable for tracking. It scores frames
independently, so a vesicle that dims for a few frames drops out and its track breaks.
Wherever a track has a gap, or ends, the probability map is read again at the
predicted position with a lower threshold (`recover.threshold`, default 0.1), and the
closest peak within `recover.radius_px` is accepted. Knowing the vesicle was there
just before and just after is the evidence that justifies the lower threshold. On a
crowded real cell this raised the fraction of frames in which tracked vesicles were
seen from 68% to 95% without introducing identity swaps. Every recovered position is
flagged in the `recovered` column of `tracks.parquet`, and every vesicle carries a
`frac_recovered`.

**Merging** rejoins fragments of one vesicle: fragments that follow each other
closely in time and space, and fragments that coexist within a tight radius (an
identity swap or a double detection). The rules were checked against vesicles
followed by eye on a crowded cell.

The probability maps are the slow part and depend only on the drift-corrected movie
and the model. Set `detect.cache_dir` and every re-run with different linking or
metric settings skips the network.

---

## Install

```bash
git clone https://github.com/csander23/Vesicle-Tracker.git
cd Vesicle-Tracker
conda env create -f environment.yml && conda activate vesicletrack
pytest -q
```

`environment.yml` creates a Python 3.11 environment and installs the package and
every dependency with pip, including TensorFlow (CPU is enough) and deepBLINK. It
installs the Python packages through pip rather than conda so that TensorFlow, deepBLINK
and the compiled scientific libraries share one NumPy build; mixing the two produced a
binary mismatch on macOS.

Without conda:

```bash
pip install -e ".[all]"      # opencv + nd2 + pyarrow + jupyter + pytest
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
and p = 1.00 for the static ones:

| class | net | directed | gross | p |
|---|---|---|---|---|
| mover (n=6) | 14.08 px | 11.65 px | 178.4 px | 0.005 |
| confined (n=24) | 0.40 px | 0.57 px | 185.6 px | 1.000 |

Net separates the classes 35-fold. Gross does not separate them at all; the confined
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

Unknown keys are rejected in either format, so a typo such as `prob_treshold` raises
an error at load instead of being ignored.

Overrides use dotted paths and do not mutate the original, so a sweep is a loop:

```python
cfg.set("detect.cache_dir", "probmaps")            # network runs once
for tau in [45, 90, 180]:
    analyse(movie, cfg.copy(**{"metrics.tau_directed_frames": tau}))
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
| `detect.prob_threshold` | spurious spots in the middle panel of `three_panel.png` | vesicles are being missed |
| `link.search_range_px` | vesicles move far between frames | you see identity swaps (rising `excluded` count) |
| `link.min_length_frames` | short fragments clutter the output | you lose real short-lived vesicles |
| `metrics.tau_directed_frames` | noise still dominates | genuine fast reversals are being erased |
| `recover.radius_px` | recovery misses a vesicle that wandered | recovered points jump to neighbours |

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
    tracks.parquet       particle, frame, x, y, recovered  (every vesicle, every frame)
    vesicles_all.csv     this movie's vesicles, with passes_filter + filter_reason
    vesicles_filtered.csv
    summary.json         counts, drift span, frames recovered, filter breakdown
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
can always be answered. The one deletion in the pipeline happens earlier: tracks with
fewer than `link.min_length_frames` detected frames never reach scoring, and the count
is in `summary.json`.

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

`mask` restricts detection and gap recovery: nothing outside it is tracked. Use it to
analyse one cell when a field holds several. `rois` labels each vesicle by region and
keeps the rest. Both take the same inputs (ImageJ `.roi`/`.zip`, mask or label images,
arrays, polygons), and both are applied in drift-corrected coordinates. When drift is
not negligible, draw them on a projection of `Result.stack` rather than the raw file.

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

`span_frames` (first to last sighting) and `observed_frames` (frames in which the
track has a position) differ whenever a gap was left open. With gap recovery most
long-lived vesicles are followed for the whole movie; `n_recovered_frames` and
`frac_recovered` say how much of each track came from recovery rather than direct
detection, and `n_gaps`, `longest_gap` and `frac_observed` describe what is still
missing.

`is_censored` marks vesicles present in the first or last frame: their true lifetime
was not observed, and pooling them with complete observations biases mean lifetime
downward. Tracked lifetime is a property of the tracker as much as of the vesicle,
so compare it between tracking settings, not between biological conditions.

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
| `span_frames`, `observed_frames` | track span vs frames with a position |
| `n_recovered_frames`, `frac_recovered` | how much of the track came from gap recovery |
| `n_gaps`, `longest_gap`, `frac_observed` | how much of the span is still missing |
| `is_censored` | present in the first or last frame, so its true lifetime is unknown |
| `sigma_px`, `sigma_deconv_px`, `at_diffraction_limit` | size (see the caveats above) |
| `roi`, `roi_frac`, `roi_changed` | which region, and whether it moved between regions |
| `passes_filter`, `filter_reason` | filter label and the rule failed; nothing is removed |
| `klass` | `mover` / `confined` / `excluded` / `invalid` |
| `persistence` | lag-1 direction cosine; ≈ −0.5 for pure localisation noise |
| `rg`, `aniso` | radius of gyration, anisotropy |

`invalid` means a non-finite coordinate; no biological claim is made about it.
`excluded` is a tracking-quality judgement rather than biology: a single large jump or
repeated large steps is the signature of an identity swap between nearby vesicles, or
of recovered points snapping between grid cells. Leaving those in inflates the mover
count.

---

## Caveats

- The detector was trained by its authors on diffraction-limited spots. It works on
  bright, round vesicles out of the box; a different marker or optics may need one of
  deepBLINK's other pretrained models (`detect.model`) or a model you train with
  deepBLINK's own tools.
- Recovered positions snap to the network's 4 px grid, so a stationary vesicle whose
  recovered frames alternate between two cells shows a 4 px back-and-forth that is not
  motion. Net and directed distances are unaffected; gross path and per-frame speeds
  are inflated by it, which is one reason they are not the primary readouts.
- Tracking is `trackpy` nearest-neighbour linking plus the merge rules above. It has
  no explicit model of merging/splitting vesicles; fusion events appear as one track
  ending and another beginning.
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
├── environment.yml            conda env: Python 3.11 + pip install of everything
├── config/
│   └── default.yaml           every parameter, commented. Copy and edit this
├── docs/
│   └── PARAMETERS.md          full reference: units, defaults, which way to tune
├── notebooks/
│   └── vesicle_analysis.ipynb walkthrough: run → inspect → tune → batch
├── examples/
│   └── make_synthetic.py      ground-truth movie (24 static + 6 movers)
├── tests/                     pytest -q
│   ├── conftest.py            the synthetic movie, a config scaled to it, a map cache
│   ├── test_smoke.py          does it get the known answer right?
│   ├── test_robustness.py     edge cases, odd inputs, every switch, recovery, merging
│   └── test_docs.py           docs cannot drift from the code
└── src/vesicletrack/
    ├── models/vesicle.h5      deepBLINK's pretrained vesicle model
    ├── config.py              parameters, validation, YAML/JSON round-trip
    ├── io.py                  load stacks (tif/nd2/npy), sample sheets, write tables
    ├── preprocess.py          stage drift correction
    ├── detect.py              deepBLINK probability maps and detections
    ├── linking.py             trackpy linking, gap recovery, fragment merging
    ├── recover.py             gap recovery from the probability maps
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

## License

MIT. The bundled `vesicle.h5` model is the pretrained model published with deepBLINK
(Eichenberger et al., 2021) and is redistributed under its license; please cite that
paper when you use it.
