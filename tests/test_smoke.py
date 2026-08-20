"""Smoke tests against the synthetic movie, where the answer is known.

These are the checks worth running after any change: they assert the invariant that
the directed metric rests on, and that the six planted movers are still recovered.

    pytest -q
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesicletrack import Config, analyse  # noqa: E402
from vesicletrack.metrics import check_ordering  # noqa: E402

MOVIE = ROOT / "examples" / "synthetic.tif"
N_MOVER, N_TOTAL = 6, 46


@pytest.fixture(scope="module")
def result():
    if not MOVIE.exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_synthetic.py"),
                        str(MOVIE)], check=True)
    cfg = Config.load(ROOT / "config" / "default.yaml", **{
        "dt_seconds": 0.05, "metrics.tau_frames": 10,
        "metrics.tau_directed_frames": 40, "link.min_length_frames": 30,
        "metrics.n_permutations": 100})
    return analyse(MOVIE, cfg, name="test", verbose=False)


def test_finds_most_vesicles(result):
    assert result.summary["n_vesicles"] >= N_TOTAL - 3


def test_recovers_the_planted_movers(result):
    assert int(result.counts().get("mover", 0)) == N_MOVER


def test_ordering_invariant_holds(result):
    """gross >= directed >= net_coarse, for every vesicle. Geometry, not tuning."""
    assert len(check_ordering(result.vesicles)) == 0


def test_movers_separate_from_confined(result):
    v = result.vesicles
    assert v[v.klass == "mover"].directed_p.max() < 0.05
    assert v[v.klass == "confined"].directed_p.median() > 0.5


def test_gross_is_noise_dominated(result):
    """The reason the directed metric exists: gross cannot tell the classes apart."""
    g = result.vesicles.groupby("klass").gross.median()
    assert abs(g["mover"] - g["confined"]) / g["confined"] < 0.3


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        Config.load(None, **{"metrics.min_run_steps": 1})
    with pytest.raises(ValueError):
        Config.load(None, **{"dt_seconds": 0})
    with pytest.raises(ValueError):
        Config.load(None, **{"detect.method": "nope"})


def test_config_roundtrip(tmp_path):
    cfg = Config.load(ROOT / "config" / "default.yaml", **{"dt_seconds": 0.1})
    p = tmp_path / "c.yaml"
    cfg.save(p)
    assert Config.load(p).dt_seconds == 0.1
