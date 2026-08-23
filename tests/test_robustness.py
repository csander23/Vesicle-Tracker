"""Edge cases, alternative input shapes, and the ways a user will actually break this.

Grouped by what is being defended:
  inputs      shapes and file types the loader must accept or clearly refuse
  degenerate  blank / tiny / single-vesicle movies that must not crash
  config      overrides, round-trip, and rejection of nonsense
  options     permutation off, units off, render off - every switch actually works
  batch       one bad file must not kill the run
  determinism same seed, same answer
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesicletrack import Config, analyse, analyse_many  # noqa: E402
from vesicletrack import io as vio  # noqa: E402

MOVIE = ROOT / "examples" / "synthetic.tif"


def base_cfg(**kw):
    d = {"dt_seconds": 0.05, "metrics.tau_frames": 10,
         "metrics.tau_directed_frames": 40, "link.min_length_frames": 30,
         "metrics.n_permutations": 50,
         "render.three_panel": False, "render.per_vesicle_images": False}
    d.update(kw)
    return Config.load(ROOT / "config" / "default.yaml", **d)


@pytest.fixture(scope="module")
def movie():
    if not MOVIE.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic.py"),
                        str(MOVIE)], check=True)
    return MOVIE


# ------------------------------------------------------------------- inputs
def test_npy_input(movie, tmp_path):
    p = tmp_path / "m.npy"
    np.save(p, tifffile.imread(movie)[:120])
    r = analyse(p, base_cfg(**{"link.min_length_frames": 20}), verbose=False)
    assert r.summary["n_vesicles"] > 0


def test_channel_axis_is_selectable(movie, tmp_path):
    """A (T, C, Y, X) stack: the caller says which channel; we never guess."""
    s = tifffile.imread(movie)[:120]
    both = np.stack([s, np.zeros_like(s)], axis=1)          # channel 1 is empty
    p = tmp_path / "tc.tif"
    tifffile.imwrite(p, both)
    good = analyse(p, base_cfg(**{"link.min_length_frames": 20}), channel=0,
                   verbose=False)
    empty = analyse(p, base_cfg(**{"link.min_length_frames": 20}), channel=1,
                    verbose=False)
    assert good.summary["n_vesicles"] > 0
    assert empty.summary["n_vesicles"] == 0


def test_z_projection(movie, tmp_path):
    s = tifffile.imread(movie)[:120]
    z = np.stack([s, s], axis=1)
    p = tmp_path / "tz.tif"
    tifffile.imwrite(p, z)
    r = analyse(p, base_cfg(**{"link.min_length_frames": 20}), z_project="max",
                verbose=False)
    assert r.summary["n_vesicles"] > 0


def test_ambiguous_4d_is_refused_not_guessed(movie, tmp_path):
    s = tifffile.imread(movie)[:20]
    p = tmp_path / "amb.tif"
    tifffile.imwrite(p, np.stack([s, s], axis=1))
    with pytest.raises(ValueError, match="Refusing to guess"):
        vio.load_stack(p)


def test_unsupported_extension(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError, match="unsupported file type"):
        vio.load_stack(p)


def test_too_few_frames(tmp_path):
    p = tmp_path / "s.tif"
    tifffile.imwrite(p, np.zeros((2, 32, 32), np.uint16))
    with pytest.raises(ValueError, match="nothing to track"):
        vio.load_stack(p)


# --------------------------------------------------------------- degenerate
def test_blank_movie_yields_nothing_and_does_not_crash(tmp_path):
    p = tmp_path / "blank.tif"
    tifffile.imwrite(p, np.full((60, 64, 64), 300, np.uint16))
    r = analyse(p, base_cfg(**{"link.min_length_frames": 10}), verbose=False)
    assert r.summary["n_vesicles"] == 0
    assert len(r.vesicles) == 0
    assert r.counts().empty
    r.save(tmp_path / "out")          # must not raise on an empty result


def test_pure_noise_makes_no_movers(tmp_path):
    rng = np.random.default_rng(1)
    p = tmp_path / "noise.tif"
    tifffile.imwrite(p, rng.normal(300, 30, (80, 64, 64)).clip(0)
                     .astype(np.uint16))
    r = analyse(p, base_cfg(**{"link.min_length_frames": 20}), verbose=False)
    assert int(r.counts().get("mover", 0)) == 0


def test_directed_is_nan_not_zero_when_unmeasurable(movie):
    """A track spanning under two tau-windows CANNOT have a directed value.

    It must come back NaN, never 0.0. Returning 0.0 is indistinguishable from a vesicle
    that genuinely did not move: a 60-frame track travelling 20 px in a straight line
    reported directed=0.00 while net said 20.00, and because net_coarse was 0 too the
    ordering check passed. This test previously asserted notna().all() and passed
    BECAUSE of that bug.
    """
    r = analyse(movie, base_cfg(**{"link.min_length_frames": 5,
                                   "metrics.tau_directed_frames": 40,
                                   "filters.min_observed_frames": 5}),
                verbose=False)
    v = r.vesicles
    assert len(v) > 0
    short = v[~v.directed_measurable]
    long_ = v[v.directed_measurable]
    assert len(short), "fixture should contain tracks too short to score"
    assert short.directed.isna().all()          # unmeasurable -> NaN
    assert short.net_coarse.isna().all()
    assert long_.directed.notna().all()         # measurable -> a real number
    assert (v.directed == 0).sum() == 0, "0.0 must never stand in for unmeasurable"


def test_straight_mover_shorter_than_tau_is_not_reported_as_zero():
    """The concrete case that exposed the bug, asserted directly on the metric."""
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 90,
                      "filters.min_observed_frames": 180})
    f = np.arange(60); x = np.linspace(0, 20, 60); y = np.zeros(60)
    m = M.metrics_for_track(x, y, f, cfg, np.random.default_rng(0))
    assert m["net"] > 19                        # it plainly moved
    assert np.isnan(m["directed"])              # but directed is not measurable
    assert not m["directed_measurable"]


# ------------------------------------------------------------------- config
def test_dotted_override_and_copy():
    c = base_cfg()
    c2 = c.copy(**{"detect.threshold_sigma": 9.9})
    assert c2.detect.threshold_sigma == 9.9
    assert c.detect.threshold_sigma != 9.9          # original untouched


def test_roundtrip_preserves_everything(tmp_path):
    c = base_cfg(**{"um_per_px": 0.107, "detect.threshold_sigma": 2.75})
    p = tmp_path / "c.yaml"
    c.save(p)
    back = Config.load(p)
    assert back.to_dict() == c.to_dict()


def test_unknown_keys_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("detect:\n  thresold_sigma: 3.0\n")     # typo
    with pytest.raises(ValueError, match="unknown key"):
        Config.load(p)
    p.write_text("nonsense: 1\n")
    with pytest.raises(ValueError, match="unknown config key"):
        Config.load(p)


def test_validation_catches_bad_combinations():
    with pytest.raises(ValueError, match="tau_directed_frames"):
        Config.load(None, **{"metrics.tau_frames": 90,
                             "metrics.tau_directed_frames": 10})
    with pytest.raises(ValueError, match="n_permutations"):
        Config.load(None, **{"classify.use_permutation": True,
                             "metrics.n_permutations": 5})
    with pytest.raises(ValueError, match="p_threshold"):
        Config.load(None, **{"classify.p_threshold": 1.5})
    with pytest.raises(ValueError, match="um_per_px"):
        Config.load(None, **{"um_per_px": -1})


# ------------------------------------------------------------------ options
def test_permutations_off(movie):
    r = analyse(movie, base_cfg(**{"metrics.n_permutations": 0,
                                   "classify.use_permutation": False}),
                verbose=False)
    assert r.vesicles.runs_p.isna().all()
    assert int(r.counts().get("mover", 0)) > 0      # rate rule still classifies


def test_units_appear_only_when_scale_given(movie):
    a = analyse(movie, base_cfg(), verbose=False)
    b = analyse(movie, base_cfg(**{"um_per_px": 0.107}), verbose=False)
    assert "net_um" not in a.vesicles.columns
    assert "net_um" in b.vesicles.columns
    np.testing.assert_allclose(b.vesicles.net_um, b.vesicles.net * 0.107)


def test_drift_can_be_disabled(movie):
    r = analyse(movie, base_cfg(**{"drift.enabled": False}), verbose=False)
    assert r.summary["drift_span_px"] == 0.0


def test_uncorrected_drift_inflates_movement(movie):
    """The reason drift correction is on by default, stated as a test."""
    on = analyse(movie, base_cfg(), verbose=False)
    off = analyse(movie, base_cfg(**{"drift.enabled": False}), verbose=False)
    conf_on = on.vesicles.query("klass == 'confined'").net.median()
    conf_off = off.vesicles.query("klass == 'confined'").net.median()
    assert conf_off > conf_on


def test_mask_restricts_detection(movie):
    s = tifffile.imread(movie)
    m = np.zeros(s.shape[1:], bool)
    m[: s.shape[1] // 2] = True
    full = analyse(movie, base_cfg(), verbose=False)
    half = analyse(movie, base_cfg(), mask=m, verbose=False)
    assert half.summary["n_vesicles"] < full.summary["n_vesicles"]
    assert half.tracks.y.max() <= s.shape[1] / 2 + 2


def test_render_switches_produce_expected_files(movie, tmp_path):
    r = analyse(movie, base_cfg(**{"render.three_panel": True,
                                   "render.per_vesicle_images": True,
                                   "render.max_vesicle_outputs": 2}),
                name="rr", verbose=False)
    w = r.save(tmp_path)
    assert (tmp_path / "rr" / "three_panel.png").exists()
    assert len(list((tmp_path / "rr" / "vesicles").glob("*.png"))) == 2
    assert (tmp_path / "rr" / "config_used.yaml").exists()


# -------------------------------------------------------------------- batch
def test_batch_survives_a_bad_file(movie, tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tiff")
    df = analyse_many([movie, bad], base_cfg(), output_dir=tmp_path / "o",
                      verbose=False)
    assert len(df) == 2
    assert df.error.notna().sum() == 1
    assert df.n_vesicles.notna().sum() == 1


# -------------------------------------------------------------- determinism
def test_same_seed_same_answer(movie):
    a = analyse(movie, base_cfg(), verbose=False).vesicles
    b = analyse(movie, base_cfg(), verbose=False).vesicles
    pd.testing.assert_frame_equal(a, b)


def test_seed_changes_only_the_null(movie):
    a = analyse(movie, base_cfg(**{"metrics.random_seed": 0}), verbose=False).vesicles
    b = analyse(movie, base_cfg(**{"metrics.random_seed": 7}), verbose=False).vesicles
    np.testing.assert_allclose(a.directed, b.directed)      # measurement is fixed
    assert not np.allclose(a.runs_null, b.runs_null)


# --------------------------------------------------------------- json config
def test_json_config_loads_identically(tmp_path):
    """JSON is a subset of YAML, so both are accepted with no special handling.

    Only the YAML is shipped - it is the single source of truth, and it carries the
    comments. JSON round-trips through save(), it is not maintained in parallel.
    """
    y = Config.load(ROOT / "config" / "default.yaml")
    p = tmp_path / "exported.json"
    y.save(p)
    assert Config.load(p).to_dict() == y.to_dict()


def test_save_format_follows_extension(tmp_path):
    """A .json file must contain JSON. Writing YAML under a .json name is a lie the
    next program to read it will not survive."""
    import json as _json
    c = Config.load(ROOT / "config" / "default.yaml", **{"dt_seconds": 0.02})
    pj, py = tmp_path / "c.json", tmp_path / "c.yaml"
    c.save(pj)
    c.save(py)
    assert _json.loads(pj.read_text())["dt_seconds"] == 0.02      # real JSON
    assert not py.read_text().lstrip().startswith("{")            # real YAML
    assert Config.load(pj).to_dict() == Config.load(py).to_dict()


def test_json_config_rejects_typos(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text('{"detect": {"thresold_sigma": 3.0}}')
    with pytest.raises(ValueError, match="unknown key"):
        Config.load(p)


def test_non_mapping_config_is_refused(tmp_path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="mapping of settings"):
        Config.load(p)


def test_cli_accepts_json_config(tmp_path, movie):
    import subprocess
    cj = tmp_path / "cfg.json"
    Config.load(ROOT / "config" / "default.yaml").save(cj)
    out = subprocess.run(
        [sys.executable, "-m", "vesicletrack.cli", str(movie),
         "-c", str(cj), "--dt", "0.05",
         "-o", str(tmp_path / "o"), "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "o" / "batch_summary.csv").exists()


# ------------------------------------------------- regressions from the audit
def test_directed_does_not_depend_on_when_the_track_started(movie):
    """Absolute window grid: the same trajectory must score the same whenever it began.

    Binning relative to each track's own first frame made `directed` depend on the
    arbitrary phase against the window grid - the same trajectory gave 0.000 at one
    length and 4.550 one frame later.
    """
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 40,
                      "filters.min_observed_frames": 80})
    vals = []
    for start in range(5):
        f = np.arange(start, start + 400)
        m = M.metrics_for_track(np.linspace(0, 20, 400), np.zeros(400), f, cfg,
                                np.random.default_rng(0))
        vals.append(m["directed"])
    assert max(vals) - min(vals) < 0.05 * np.mean(vals)


def test_sparse_windows_do_not_inflate_directed():
    """Occupancy floor: a window holding one frame carries full localisation noise."""
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 40,
                      "filters.min_observed_frames": 80})
    rng = np.random.default_rng(3)
    f_full = np.arange(400)
    x = rng.normal(0, 0.15, 400); y = rng.normal(0, 0.15, 400)   # stationary
    full = M.metrics_for_track(x, y, f_full, cfg, np.random.default_rng(0))["directed"]
    keep = np.sort(rng.choice(400, 168, replace=False))          # 42% observed
    sparse = M.metrics_for_track(x[keep], y[keep], f_full[keep], cfg,
                                 np.random.default_rng(0))["directed"]
    assert sparse < 2.5 * full, (full, sparse)


def test_degenerate_null_reports_infinite_z_not_zero():
    """A null with no spread is MAXIMUM evidence, not none."""
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 40,
                      "filters.min_observed_frames": 80})
    f = np.arange(200)
    m = M.metrics_for_track(f * 0.006, np.zeros(200), f, cfg, np.random.default_rng(0))
    floor = 1.0 / (cfg.metrics.n_permutations + 1)
    assert m["runs_p"] == pytest.approx(floor)   # nothing in the null ever beat it
    assert np.isinf(m["runs_z"]) and m["runs_z"] > 0


def test_nonfinite_coordinate_is_invalid_not_confined():
    """A NaN must not yield a confident biological label."""
    import numpy as np
    import pandas as pd
    from vesicletrack import metrics as M
    cfg = base_cfg()
    x = np.linspace(0, 10, 200); x[57] = np.nan
    tr = pd.DataFrame({"particle": 0, "frame": np.arange(200), "x": x, "y": 0.0})
    v = M.score_tracks(tr, cfg, n_movie_frames=400)
    assert v.klass.iloc[0] == "invalid"
    assert bool(v.invalid.iloc[0]) and int(v.n_nonfinite.iloc[0]) == 1


def test_pvalue_does_not_depend_on_other_tracks_in_the_table():
    """Per-track RNG: a vesicle's class must not change because a neighbour exists."""
    import numpy as np
    import pandas as pd
    from vesicletrack import metrics as M
    cfg = base_cfg()
    solo = pd.DataFrame({"particle": 7, "frame": np.arange(200),
                         "x": np.arange(200) * 0.006, "y": 0.0})
    pair = pd.concat([pd.DataFrame({"particle": 1, "frame": np.arange(200),
                                    "x": 0.0, "y": 0.0}), solo])
    a = M.score_tracks(solo, cfg).set_index("particle").runs_p[7]
    b = M.score_tracks(pair, cfg).set_index("particle").runs_p[7]
    assert a == b


def test_duplicate_particle_frame_rows_are_refused():
    import pandas as pd
    import numpy as np
    from vesicletrack import metrics as M
    tr = pd.DataFrame({"particle": [0, 0], "frame": [1, 1], "x": [1.0, 2.0],
                       "y": [0.0, 0.0]})
    with pytest.raises(ValueError, match="duplicate"):
        M.score_tracks(tr, base_cfg())


def test_empty_result_keeps_its_schema():
    import pandas as pd
    from vesicletrack import metrics as M
    v = M.score_tracks(pd.DataFrame(columns=["particle", "frame", "x", "y"]),
                       base_cfg())
    assert "gross" in v.columns and "runs_p" in v.columns
    assert len(M.check_ordering(v)) == 0        # must not raise


def test_zstack_is_refused_by_its_own_metadata(tmp_path):
    """A z-stack has the same 3-D shape as a time series; the file says which it is."""
    import numpy as np
    import tifffile
    p = tmp_path / "zstack.tif"
    tifffile.imwrite(p, np.zeros((12, 64, 64), np.uint16),
                     metadata={"axes": "ZYX"}, imagej=True)
    with pytest.raises(ValueError, match="not time"):
        vio.load_stack(p)


def test_config_rejects_unreachable_p_threshold():
    with pytest.raises(ValueError, match="permutation floor"):
        Config.load(None, **{"classify.p_threshold": 0.001,
                             "metrics.n_permutations": 200})


def test_config_rejects_nan_and_scalar_sections(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        Config.load(None, **{"dt_seconds": float("nan")})
    p = tmp_path / "s.yaml"
    p.write_text("drift: false\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        Config.load(p)


def test_render_false_writes_no_figures(movie, tmp_path):
    r = analyse(movie, base_cfg(**{"render.three_panel": True,
                                   "render.per_vesicle_images": True}),
                name="nr", verbose=False)
    r.save(tmp_path, render=False)
    assert not list((tmp_path / "nr").glob("*.png"))
    assert not (tmp_path / "nr" / "vesicles").exists()


def test_cli_end_to_end_with_sheet_and_aggregation(tmp_path, movie):
    """The documented full command: metadata from a sheet, every level of CSV out."""
    import subprocess
    import os
    sheet = tmp_path / "sheet.csv"
    sheet.write_text(f"file,genotype\n{Path(movie).name},WT\n")
    out = subprocess.run(
        [sys.executable, "-m", "vesicletrack.cli", str(movie), "--dt", "0.05",
         "--sheet", str(sheet), "--by", "genotype", "--no-figures",
         "-o", str(tmp_path / "o"), "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    o = tmp_path / "o"
    for f in ("vesicles_all.csv", "vesicles_filtered.csv", "per_video.csv",
              "per_genotype.csv", "long.csv"):
        assert (o / f).exists(), f
    import pandas as pd
    assert (pd.read_csv(o / "vesicles_all.csv").genotype == "WT").all()
    assert not list(o.rglob("*.png")), "--no-figures still wrote images"


def test_directed_requirement_is_on_span_not_observed_frames():
    """The gate that nearly made the defaults useless.

    Real vesicles are only ~40% observed, so one spanning 400 frames holds ~163
    detections. Gating `directed` on 2*tau OBSERVED frames therefore discarded the
    genuine vesicles while appearing to protect the metric. The requirement is on SPAN.
    """
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 90})
    rng = np.random.default_rng(0)
    # the real case: spans 400 frames, 42% observed -> 168 detections, which is BELOW
    # the 2*tau = 180 observed-frame gate that used to reject it.
    f = np.sort(rng.choice(400, 168, replace=False))
    m = M.metrics_for_track(np.linspace(0, 8, 168), np.zeros(168), f, cfg,
                            np.random.default_rng(0))
    assert m["observed_frames"] < 2 * cfg.metrics.tau_directed_frames
    assert m["span_frames"] >= 2 * cfg.metrics.tau_directed_frames
    assert m["directed_measurable"], "span covers two tau-windows: must be measurable"
    assert np.isfinite(m["directed"])


def test_occupancy_floor_keeps_realistically_sparse_windows():
    """A floor of tau/2 sat above what a 42%-observed vesicle puts in a window."""
    import numpy as np
    from vesicletrack import metrics as M
    rng = np.random.default_rng(1)
    f = np.sort(rng.choice(400, 168, replace=False))          # 42% observed
    x = np.linspace(0, 8, 168); y = np.zeros(168)
    strict = M.coarse(x, y, f, 90, occupancy_frac=0.5)[0]
    default = M.coarse(x, y, f, 90, occupancy_frac=0.25)[0]
    assert len(default) > len(strict)
    assert len(default) >= 2, "the shipped floor must leave a measurable track"


def test_shipped_defaults_leave_the_filtered_set_usable():
    """Every vesicle that passes the default filters must have a usable `directed`."""
    import numpy as np
    import pandas as pd
    from vesicletrack import metrics as M, filters as F
    cfg = base_cfg(**{"metrics.tau_directed_frames": 40})
    rng = np.random.default_rng(2)
    rows = []
    for p, span in enumerate([20, 60, 150, 400, 400]):
        f = np.sort(rng.choice(span, max(4, int(span * 0.42)), replace=False))
        rows.append(pd.DataFrame({"particle": p, "frame": f,
                                  "x": np.linspace(0, 4, len(f)), "y": 0.0}))
    v = F.apply_filters(M.score_tracks(pd.concat(rows), cfg, n_movie_frames=400), cfg)
    passing = v[v.passes_filter]
    assert len(passing), "the defaults must not reject everything"
    assert passing.directed_measurable.all(), (
        "a vesicle that passes the filters must have a measurable directed value")
    assert passing.directed.notna().all()


def test_frame_count_thresholds_cannot_guarantee_measurability():
    """Why require_directed_measurable exists rather than a span/observed threshold.

    Whether `directed` exists depends on how many observed frames land in each
    tau-window. A track can clear both a span gate and an observed-frames gate and
    still be unmeasurable, because its detections clustered into too few windows.
    """
    import numpy as np
    from vesicletrack import metrics as M
    cfg = base_cfg(**{"metrics.tau_directed_frames": 90})
    # 200-frame span, 60 observed - but all crammed into one 90-frame window
    f = np.concatenate([np.arange(0, 55), np.arange(195, 200)])
    m = M.metrics_for_track(np.linspace(0, 5, len(f)), np.zeros(len(f)), f, cfg,
                            np.random.default_rng(0))
    assert m["span_frames"] >= 180          # clears a span gate
    assert m["observed_frames"] >= 40       # clears an observed-frames gate
    assert not m["directed_measurable"]     # and is STILL unmeasurable
