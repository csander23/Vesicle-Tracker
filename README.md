# vesicletrack

Detect, track and score intracellular vesicles in time-lapse microscopy. One config
file decides everything; the only other input is the movie.

```python
from vesicletrack import Config, analyse

cfg = Config.load("config/default.yaml", dt_seconds=0.0446, um_per_px=0.107)
res = analyse("cell.tif", cfg)

res.counts()        # mover / confined / excluded
res.vesicles        # one row per vesicle, all metrics
res.save()          # tables + figures + videos
```

Or from the shell:

```bash
vesicletrack run "data/*.tif" -c config/default.yaml -o output --videos
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
pytest -q                      # 29 tests, ~1 min
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

Recovers all 30 vesicles and exactly the 6 planted movers, p = 0.01 against p = 1.00
for the static ones:

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

Both work, interchangeably — JSON is a subset of YAML, so a `.json` config loads with
no special handling:

```python
Config.load("config/default.yaml")     # shipped, commented
Config.load("config/default.json")     # same settings, machine-friendly
Config.load(None, dt_seconds=0.05)     # all defaults, overridden inline
```

```bash
vesicletrack run movie.tif -c config/default.json
```

`save()` writes whichever format the extension asks for, so a config generated in a
notebook can be handed straight to another program:

```python
cfg.save("runs/exp1.json")     # JSON
cfg.save("runs/exp1.yaml")     # YAML
cfg.to_json()                  # or just the string
```

**Prefer YAML when a person maintains the file** — it takes comments, and the comments
in `config/default.yaml` are half of what makes the parameters usable. **Prefer JSON
when a program generates it** (a sweep, a LIMS, a web front end).

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

## Output

```
output/<name>/
  config_used.yaml     exact parameters — a figure can always be traced to its run
  tracks.parquet       particle, frame, x, y  (every vesicle, every frame)
  vesicles.csv         one row per vesicle: all metrics + class
  summary.json         counts, drift span, runtime, invariant violations
  three_panel.png      RAW | ALL VESICLES | CLASSIFIED
  distances.png        net / directed / gross distributions, directed vs net
  vesicles/*.png       per vesicle: trajectory + the three distances, drawn and printed
  vesicle_videos/*.mp4 per-vesicle crops with a trail        (render.per_vesicle_videos)
  overview.mp4         whole field, coloured by class        (render.overview_video)
```

The three panels answer the two questions worth asking before trusting any number:
did detection *find* the vesicles (middle), and did classification *label* them
sensibly (right).

Per-vesicle output is capped by `render.max_vesicle_outputs` (default 25, movers
first) so a dense field cannot emit thousands of files. The cap is recorded in
`summary.json` rather than applied silently.

### Key columns in `vesicles.csv`

| column | meaning |
|---|---|
| `net`, `gross`, `directed` | the three distances, px (`*_um` when `um_per_px` set) |
| `net_coarse` | \|c_K − c_1\|; the quantity the ordering theorem bounds |
| `*_rate` | per second |
| `directed_p` | permutation p-value against the vesicle's own null |
| `directed_excess` | observed − null mean |
| `klass` | `mover` / `confined` / `excluded` |
| `persistence` | lag-1 direction cosine; ≈ −0.5 for pure localisation noise |
| `rg`, `aniso` | radius of gyration, anisotropy |

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
src/vesicletrack/
  config.py      parameters, validation, YAML round-trip
  io.py          stack loading (tif / nd2 / npy), table writing
  preprocess.py  stage drift correction
  detect.py      per-frame spot detection + NMS
  linking.py     trackpy linking, short-track removal, co-located merge
  metrics.py     net / gross / directed, permutation null, classification
  render.py      three-panel, per-vesicle images, videos
  pipeline.py    analyse() / analyse_many() / Result
  cli.py         command line
notebooks/vesicle_analysis.ipynb    walkthrough, parameter tuning, batch
examples/make_synthetic.py          ground-truth movie generator
```

## Licence

MIT.
