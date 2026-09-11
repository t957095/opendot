"""Tests for OPENDOT.md snapshot rule parsing."""

from opendot.reversibility.rules import load_rules, parse_rules_text


def test_no_block_is_defaults():
    r = parse_rules_text("# My project\n\nSome guidance for the agent.\n")
    assert r.force_include == set()
    assert r.extra_skip == set()


def test_snapshot_and_skip_directives():
    text = """\
# Project

```opendot
snapshot: node_modules, dist
skip: data, secrets/
```
"""
    r = parse_rules_text(text)
    assert r.force_include == {"node_modules", "dist"}
    assert r.extra_skip == {"data", "secrets"}  # trailing slash stripped


def test_glob_skip_pattern_actually_matches_filenames():
    """OPENDOT.md documents `skip: *.log`, but skipped() was pure set membership,
    so the pattern only ever matched a file literally named `*.log` — a user's
    .log files were snapshotted (and clobbered on restore) despite the rule."""
    r = parse_rules_text("```opendot\nskip: *.log, data\n```\n")
    assert r.skipped("app.log")
    assert r.skipped("debug.log")
    assert r.skipped("data")  # literal entries still work
    assert not r.skipped("notes.txt")


def test_glob_skip_does_not_override_force_include():
    r = parse_rules_text("```opendot\nsnapshot: app.log\nskip: *.log\n```\n")
    assert not r.skipped("app.log")  # force_include still wins
    assert r.skipped("other.log")


def test_aliases_and_comments():
    text = """\
```opendot
# force these in
include: venv
# and skip these
exclude: *.log
ignore: tmp
```
"""
    r = parse_rules_text(text)
    assert "venv" in r.force_include
    assert {"*.log", "tmp"} <= r.extra_skip


def test_force_include_beats_default_skip_via_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    (tmp_path / "OPENDOT.md").write_text("```opendot\nsnapshot: venv\n```")
    rules = load_rules(tmp_path)
    assert rules.skipped("venv") is False  # normally skipped, now forced in
    assert rules.skipped("node_modules") is True  # still default-skipped


def test_missing_file_is_defaults(tmp_path):
    rules = load_rules(tmp_path)  # no OPENDOT.md
    assert rules.force_include == set()
