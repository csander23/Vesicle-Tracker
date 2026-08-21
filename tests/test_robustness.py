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


def test_short_tracks_have_nan_metrics_not_crashes(movie):
    """min_length below the coarse window: too few windows to score direction."""
    r = analyse(movie, base_cfg(**{"link.min_length_frames": 5,
                                   "metrics.tau_directed_frames": 40}),
                verbose=False)
    assert len(r.vesicles) > 0
    assert r.vesicles.directed.notna().all()       # directed is always defined


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
    assert r.vesicles.directed_p.isna().all()
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
    assert not np.allclose(a.directed_null, b.directed_null)


# --------------------------------------------------------------- json config
def test_json_config_loads_identically(tmp_path):
    """JSON is a subset of YAML, so both are accepted with no special handling."""
    y = Config.load(ROOT / "config" / "default.yaml")
    j = Config.load(ROOT / "config" / "default.json")
    assert y.to_dict() == j.to_dict()


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
    out = subprocess.run(
        [sys.executable, "-m", "vesicletrack.cli", str(movie),
         "-c", str(ROOT / "config" / "default.json"), "--dt", "0.05",
         "-o", str(tmp_path / "o"), "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert (tmp_path / "o" / "batch_summary.csv").exists()
