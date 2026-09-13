"""Documentation cannot silently drift from the code.

Every parameter the code defines must appear in the shipped config AND in the
parameter reference, with the same default. Adding a field without documenting it
fails here, which is the only reliable way to keep a reference honest over time.

The section list comes from config.SECTIONS, not from a copy here: a copy once
omitted `size` and `filters`, and their parameters went unchecked.
"""
import dataclasses

import yaml

from conftest import ROOT
from vesicletrack import config as vcfg

YAML = ROOT / "config" / "default.yaml"
REF = ROOT / "docs" / "PARAMETERS.md"


def _all_fields():
    """(section, name) for every parameter; top-level fields have section ''."""
    out = [("", f.name) for f in dataclasses.fields(vcfg.Config)
           if f.name not in vcfg.SECTIONS]
    for sec, cls in vcfg.SECTIONS.items():
        out += [(sec, f.name) for f in dataclasses.fields(cls)]
    return out


def test_sections_cover_every_config_dataclass():
    """A new section must be registered in config.SECTIONS, or it is unloadable."""
    nested = {f.name for f in dataclasses.fields(vcfg.Config)
              if dataclasses.is_dataclass(vcfg.Config().__getattribute__(f.name))}
    assert nested == set(vcfg.SECTIONS)


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
        if got != want:
            diffs.append(f"{sec + '.' if sec else ''}{name}: yaml={got} code={want}")
    assert not diffs, "shipped config disagrees with code defaults: " + "; ".join(diffs)


def _doc_value(cell: str):
    """Parse a `default` cell: null / true / false / number / [list] / else raw text."""
    v = cell.strip().strip("`").strip("*").strip()
    if v.startswith("["):
        v = v[: v.index("]") + 1]
        return [_doc_value(x) for x in v[1:-1].split(",") if x.strip()]
    v = v.split()[0] if v else ""
    if v == "null":
        return None
    if v in ("true", "false"):
        return v == "true"
    try:
        return float(v)
    except ValueError:
        return v


def _norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, (int, float)):
        return float(v)
    return str(v)


def test_documented_defaults_match_the_code():
    """The `default` column of each table in PARAMETERS.md is the real default.

    Tables sit under a "## `section`" heading, so the same name in two sections
    (detect.psf_sigma_px vs size.psf_sigma_px) is checked against its own default.
    """
    code = vcfg.Config().to_dict()
    sec, bad = "", []
    for line in REF.read_text().splitlines():
        if line.startswith("## "):
            head = line[3:].strip()
            sec = head.strip("`").split("`")[0] if head.startswith("`") else ""
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("`") or "/" in cells[0]:
            continue
        name = cells[0].strip("`")
        table = code.get(sec, {}) if sec else code
        if name not in table:
            continue
        want, doc = table[name], _doc_value(cells[1])
        if _norm(doc) != _norm(want):
            bad.append(f"{sec + '.' if sec else ''}{name}: docs say {cells[1]!r}, "
                       f"code default is {want!r}")
    assert not bad, "\n".join(bad)


def test_readme_points_at_the_reference():
    assert "docs/PARAMETERS.md" in (ROOT / "README.md").read_text()
