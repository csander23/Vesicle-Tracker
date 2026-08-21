"""Documentation cannot silently drift from the code.

Every parameter the code defines must appear in the shipped config AND in the
parameter reference. Adding a field without documenting it fails here, which is the
only reliable way to keep a reference honest over time.
"""
import dataclasses
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vesicletrack import config as vcfg  # noqa: E402

YAML = ROOT / "config" / "default.yaml"
REF = ROOT / "docs" / "PARAMETERS.md"

SECTIONS = {"drift": vcfg.DriftConfig, "detect": vcfg.DetectConfig,
            "link": vcfg.LinkConfig, "metrics": vcfg.MetricsConfig,
            "classify": vcfg.ClassifyConfig, "render": vcfg.RenderConfig}


def _all_fields():
    """(section, name) for every parameter, sections flattened as ('', name)."""
    out = [("", f.name) for f in dataclasses.fields(vcfg.Config)
           if f.name not in SECTIONS]
    for sec, cls in SECTIONS.items():
        out += [(sec, f.name) for f in dataclasses.fields(cls)]
    return out


def test_every_parameter_is_in_the_shipped_config():
    raw = yaml.safe_load(YAML.read_text())
    missing = []
    for sec, name in _all_fields():
        present = name in (raw.get(sec, {}) if sec else raw)
        if not present:
            missing.append(f"{sec}.{name}" if sec else name)
    assert not missing, f"not in config/default.yaml: {missing}"


def test_shipped_config_has_no_extra_keys():
    """The reverse: a key in the YAML that the code does not know about."""
    vcfg.Config.load(YAML)          # load() raises on unknown keys


def test_every_parameter_is_documented():
    text = REF.read_text()
    undocumented = [name for _, name in _all_fields() if f"`{name}`" not in text]
    assert not undocumented, f"not in docs/PARAMETERS.md: {undocumented}"


def test_defaults_agree_between_code_and_shipped_config():
    """The YAML must not quietly ship different values than the dataclass defaults."""
    raw = yaml.safe_load(YAML.read_text())
    code = vcfg.Config().to_dict()
    diffs = []
    for sec, name in _all_fields():
        want = code[sec][name] if sec else code[name]
        got = (raw.get(sec, {}) if sec else raw).get(name)
        if isinstance(want, tuple):
            want = list(want)
        if got != want:
            diffs.append(f"{sec + '.' if sec else ''}{name}: yaml={got} code={want}")
    assert not diffs, "shipped config disagrees with code defaults: " + "; ".join(diffs)


def test_readme_points_at_the_reference():
    assert "docs/PARAMETERS.md" in (ROOT / "README.md").read_text()
