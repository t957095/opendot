"""Irreversibility classifier — the honesty layer.

Before opendot runs a shell command, this decides whether the command's effects
stay *inside* the workspace (so a snapshot can undo it) or *escape* it (network,
sudo, package installs, destructive DB, deletes/writes outside the working dir).
Escaping commands are marked not-reversible so the caller confirms first and the
ledger records them honestly.

Design stance: CONSERVATIVE / fail-safe. When in doubt, treat a command as
irreversible. A few extra confirmations are fine; a silent un-undoable surprise
is not. This is a heuristic on shell text, not a guarantee — the reason string
explains the call so the user can judge.

It is an *explainer*, not a security boundary. Reading shell text can't fully
account for an opaque subprocess (`python foo.py` is as unknowable as `python -c`,
so interpreters are confirm-first) and can't win a TOCTOU symlink race against
hostile input. Kernel-enforced isolation is the real boundary: open-time
containment for the built-in file tools ships today; a sandbox/overlay for
unattended runs is planned (see issue #130).
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path

# Commands whose whole job is contained, read-only, or trivially reversible.
# Only these auto-run without confirmation.
_SAFE_COMMANDS = {
    "ls",
    "cat",
    "head",
    "tail",
    "pwd",
    "echo",
    "grep",
    "rg",
    "find",
    "wc",
    "diff",
    "tree",
    "stat",
    "file",
    "which",
    "env",
    "date",
    "whoami",
    "clear",
    "cls",  # harmless read-only terminal commands
    "pytest",  # test runner: in-workspace, snapshot covers any files it writes
    "touch",
    "mkdir",
    "cp",
    "mv",  # in-workspace fs ops (snapshot covers them)
    "sed",
    "awk",
    "sort",
    "uniq",
    "cut",
    "tr",
    # read-only system/info commands (never mutate the fs or reach the network)
    "ps",
    "df",
    "du",
    "uname",
    "id",
    "groups",
    "printenv",
    "basename",
    "dirname",
    "realpath",
}

# Interpreters / launchers whose effects can't be read from the command text: a
# script (or -c snippet) can open sockets, write outside the workspace, or exec
# anything. `python foo.py` is exactly as opaque as `python -c "..."`, so we
# can't safely auto-run any of them — they are confirm-first, not blocked.
# (#130: treat interpreters as capabilities, not commands.)
_OPAQUE_INTERPRETERS = {
    "python",
    "python3",
    "node",
    "deno",
    "bun",
    "ruby",
    "perl",
    "php",
    "sh",
    "bash",
    "zsh",
    "fish",
    "make",
    "docker",
    "docker-compose",
    "kubectl",
    "go",
    "cargo",  # `go run` / `cargo run` execute arbitrary code
}

# Signals that a command escapes the workspace or is hard/impossible to undo.
_NETWORK = {"curl", "wget", "ssh", "scp", "rsync", "ftp", "nc", "telnet"}
_VCS_REMOTE = ("git push", "git pull", "git fetch", "git clone")
_PKG_INSTALL = (
    "pip install",
    "pip3 install",
    "npm install",
    "npm i ",
    "yarn add",
    "apt install",
    "apt-get install",
    "brew install",
    "uv pip install",
    "uv add",
    "cargo install",
    "gem install",
)
_PRIVILEGE = {"sudo", "doas", "su"}
_DESTRUCTIVE_DB = re.compile(r"\b(drop|truncate|delete)\s+(table|database|from)\b", re.I)

# Directory names snapshots SKIP by default (mirrors snapshots._DEFAULT_IGNORE_DIRS).
# Deleting one of these is NOT undoable — it was never captured — so an `rm`
# that targets them must be treated as irreversible, not auto-run.
_NOT_SNAPSHOTTED = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".opendot",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".cache",
}


def _hits_unsnapshotted_path(command: str) -> str | None:
    """If the command references a directory that snapshots skip (so it can't be
    restored), return that name; else None.

    Tokens are the runs of path characters between whitespace, separators, quotes
    and shell operators — the same character class ``_mentions_outside_path``
    uses, plus ``/`` so nested components are seen. Anything glued to the name
    therefore can't hide it: ``rm -rf ".git"``, ``rm -rf .git>out`` and
    ``rm -rf node_modules/.cache`` all read the same as the bare path."""
    for tok in re.findall(r"[^\s/'\"|&;<>()]+", command):
        if tok in _NOT_SNAPSHOTTED:
            return tok
    return None


@dataclass
class Verdict:
    reversible: bool
    reason: str = ""


def _first_word(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.strip().split()
    return parts[0] if parts else ""


def _mentions_outside_path(command: str, workdir: str) -> bool:
    """Heuristic: does the command reference an absolute path or parent-escape
    that leaves the workspace?

    Path tokens are resolved with symlinks followed (``Path.resolve()``), so a
    symlink *inside* the workspace that points out is caught too. This is a
    best-effort text check, not a TOCTOU-proof guarantee — a path can be swapped
    between this check and the command actually running. Real containment needs
    open-time enforcement / a sandbox (see #130); this only decides whether to
    confirm.
    """
    wd = Path(workdir).resolve()
    # any ../ (or ..\) that could climb out
    if ".." in command:
        return True
    # writing to $HOME / ~ is outside the workspace
    if re.search(r"(^|\s)~(/|\s|$)|\$HOME", command):
        return True
    # Check every path-like token: absolute paths (POSIX / or Windows drive/UNC),
    # and relative tokens containing a separator, resolved (through symlinks) and
    # tested for containment with relative_to so path separators are handled
    # correctly on every platform (not a "/"-only string prefix).
    for tok in re.findall(r"[^\s'\"|&;<>]+", command):
        p = Path(tok)
        if not (p.is_absolute() or "/" in tok or "\\" in tok):
            continue  # not path-like
        try:
            resolved = (p if p.is_absolute() else (wd / tok)).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            resolved.relative_to(wd)
        except ValueError:
            return True  # resolves outside the workspace
    return False


def _split_segments(command: str) -> list[str]:
    """Split a command line into independently-run segments on shell operators,
    without splitting inside quoted strings (single or double). It does NOT parse
    $(...) subshells or backticks, so an operator inside one is treated as a real
    split point — that only ever over-splits, which fails safe (extra confirms,
    never a missed one). Falls back to the whole command if tokenization fails."""
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        list(lex)  # validate it tokenizes; we only need the parse to not raise
    except ValueError:
        return [command]
    # shlex validated quoting; now split on top-level operators only. We re-scan
    # the raw text but skip operators inside single/double quotes.
    segments: list[str] = []
    buf: list[str] = []
    i = 0
    quote: str | None = None
    while i < len(command):
        c = command[i]
        # A backslash escapes the next character (outside single quotes, where the
        # shell treats backslash literally). Consume both so an escaped quote
        # (\") can't open a fake quoted span and swallow a following operator,
        # and an escaped operator (\&) isn't treated as a split point.
        if c == "\\" and quote != "'" and i + 1 < len(command):
            buf.append(c)
            buf.append(command[i + 1])
            i += 2
            continue
        if quote:
            buf.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            buf.append(c)
            i += 1
            continue
        # operator?
        two = command[i : i + 2]
        if two in ("&&", "||"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if c in (";", "|", "\n"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def _classify_segment(cmd: str, workdir: str) -> Verdict:
    """Classify a single command segment (no shell operators)."""
    low = cmd.lower()
    head = _first_word(cmd)

    # Hard irreversible / escaping signals (checked first; fail safe).
    if head in _PRIVILEGE:
        return Verdict(False, "runs with elevated privileges (sudo)")
    if head in _NETWORK:
        return Verdict(False, f"network access ({head})")
    if any(low.startswith(v) or f" {v}" in low for v in _VCS_REMOTE):
        return Verdict(False, "interacts with a remote (git push/pull/clone)")
    if any(p in low for p in _PKG_INSTALL):
        return Verdict(False, "installs packages (changes system/env state)")
    if _DESTRUCTIVE_DB.search(cmd):
        return Verdict(False, "destructive database operation")
    if "rm " in low or head == "rm":
        # rm inside the workspace is covered by the snapshot; rm outside isn't.
        if _mentions_outside_path(cmd, workdir):
            return Verdict(False, "deletes files outside the workspace")
        # Deleting a path snapshots SKIP (e.g. .git, node_modules) can't be
        # undone — it was never captured. Flag it rather than silently run it.
        skipped = _hits_unsnapshotted_path(cmd)
        if skipped:
            return Verdict(
                False,
                f"deletes '{skipped}', which opendot doesn't snapshot — this can't be undone",
            )
        # in-workspace rm is reversible via snapshot
        return Verdict(True)
    if _mentions_outside_path(cmd, workdir):
        return Verdict(False, "touches paths outside the working directory")

    # Opaque interpreters/launchers: their effects can't be read from the text, so
    # confirm rather than auto-run (a script can do anything a -c snippet can).
    if head in _OPAQUE_INTERPRETERS:
        return Verdict(False, f"runs '{head}', whose effects can't be verified from the command")

    # Known-safe, workspace-contained commands auto-run.
    if head in _SAFE_COMMANDS:
        return Verdict(True)

    # Unknown command: fail safe — confirm, since we can't vouch for its effects.
    return Verdict(False, f"unrecognized command '{head}' — effects unknown")


def classify(command: str, workdir: str) -> Verdict:
    """Return a Verdict. reversible=True => safe to snapshot+run silently.

    A command line may chain several commands (``a && b``, ``a | b``, ``a; b``).
    Each segment is classified independently and the whole line is reversible
    only if *every* segment is — the first escaping segment makes the line
    non-reversible and supplies the reason. This stops a safe leading command
    from smuggling a dangerous one past confirmation (#128).
    """
    if not command.strip():
        return Verdict(True)
    for segment in _split_segments(command):
        verdict = _classify_segment(segment, workdir)
        if not verdict.reversible:
            return verdict
    return Verdict(True)
