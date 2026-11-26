"""Probe mode: squeeze a single file down to the tactic line.

Tries deleting things one at a time (unused declarations, `#check`s,
`open`s, tactic lines, `simp [...]` arguments) and keeps a deletion only if
the Lean compiler says the file still builds, every target theorem has
*exactly* the same statement (compared with `pp.all`, so hidden instances
count), and no target picked up a `sorry`.  Costs one compile per attempt,
so it's for one file at a time; use graph mode for whole projects.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .spans import lemma_args, split_commands, tactic_units

_MSG = re.compile(r"^(.*?):(\d+):(\d+): (error|warning|info)(?:\([^)]*\))?: ?(.*)$")


@dataclass
class Verdict:
    ok: bool
    statements: dict[str, str]
    sorry: dict[str, bool]
    errors: list[str]


class Oracle:
    def __init__(self, path: Path, targets: list[str], lean_cmd: list[str], cwd: Path, timeout: float):
        self.path, self.targets, self.lean_cmd, self.cwd, self.timeout = path, targets, lean_cmd, cwd, timeout
        self.calls = 0
        self.seconds = 0.0
        self.cache: dict[str, Verdict] = {}
        self.baseline: Verdict | None = None

    def probe_text(self, n_lines: int) -> str:
        cmds = []
        for t in self.targets:
            cmds.append(f"set_option pp.all true in\n#check @{t}\n#print axioms {t}")
        return "\n\n-- slop_destroyer probe\n" + "\n".join(cmds) + "\n"

    def run(self, text: str) -> Verdict:
        if text in self.cache:
            return self.cache[text]
        probe_start = text.count("\n") + 1
        full = text + self.probe_text(probe_start)
        fd, tmp = tempfile.mkstemp(suffix=".lean", prefix=".slop_", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(full)
            t0 = time.time()
            try:
                r = subprocess.run(self.lean_cmd + [tmp], cwd=self.cwd, capture_output=True, text=True,
                                   timeout=self.timeout)
                out, code = r.stdout + r.stderr, r.returncode
            except subprocess.TimeoutExpired:
                out, code = "error: timeout", 1
            self.seconds += time.time() - t0
            self.calls += 1
        finally:
            os.unlink(tmp)
        v = self._parse(out, code, probe_start)
        self.cache[text] = v
        return v

    def _parse(self, out: str, code: int, probe_start: int) -> Verdict:
        msgs: list[tuple[int, str, str]] = []
        for line in out.split("\n"):
            m = _MSG.match(line)
            if m:
                msgs.append((int(m.group(2)), m.group(4), m.group(5)))
            elif msgs:
                ln, sev, body = msgs[-1]
                msgs[-1] = (ln, sev, body + "\n" + line)
        errors = [f"{ln}: {body}" for ln, sev, body in msgs if sev == "error"]
        probe = [body for ln, sev, body in msgs if ln > probe_start and sev == "info"]
        statements: dict[str, str] = {}
        sorry: dict[str, bool] = {}
        for t in self.targets:
            # The `#check` answer starts with the (pp.all) name; the axioms answer mentions 't'.
            short = t.rsplit(".", 1)[-1]
            for body in probe:
                if "depends on axioms" in body or "does not depend on any axioms" in body:
                    if f"'{t}'" in body or f"'{short}'" in body:
                        sorry[t] = "sorryAx" in body
                elif body.lstrip("@").startswith((t, short)) and t not in statements:
                    statements[t] = body.strip()
        ok = code == 0 and not errors and all(t in statements and t in sorry for t in self.targets)
        return Verdict(ok, statements, sorry, errors)

    def accepts(self, text: str) -> bool:
        v = self.run(text)
        b = self.baseline
        return (v.ok and v.statements == b.statements
                and all(v.sorry[t] <= b.sorry[t] for t in self.targets))


def _drop_lines(text: str, ranges: list[tuple[int, int]]) -> str:
    lines = text.split("\n")
    drop = {k for a, b in ranges for k in range(a, b + 1)}
    return re.sub(r"\n{3,}", "\n\n", "\n".join(ln for k, ln in enumerate(lines, 1) if k not in drop))


def _try_each(oracle: Oracle, text: str, candidates, budget: list[int], label: str, log: list[str]) -> str:
    """Greedy: attempt candidates in order, keeping each deletion the oracle accepts.
    `candidates(text)` is re-evaluated after every success (line numbers shift)."""
    idx = 0
    while budget[0] > 0:
        cands = candidates(text)
        if idx >= len(cands):
            return text
        desc, new = cands[idx]
        budget[0] -= 1
        if new != text and oracle.accepts(new):
            log.append(f"{label}: {desc}")
            text = new
            # same index now points at the next candidate (list shrank by one)
        else:
            idx += 1
    return text


def destroy(path: Path, targets: list[str], lean_cmd: list[str], cwd: Path, *, tactics: bool = True,
            max_compiles: int = 500, timeout: float = 600, keep_debug: bool = False) -> tuple[str, list[str], Oracle]:
    text = path.read_text(encoding="utf-8")
    oracle = Oracle(path, targets, lean_cmd, cwd, timeout)
    base = oracle.run(text)
    if not base.ok:
        problems = base.errors or [f"target not found: {t}" for t in targets if t not in base.statements]
        sys.exit("the original file doesn't pass the checks, so there's nothing safe to compare against:\n  "
                 + "\n  ".join(problems[:10]))
    oracle.baseline = base
    budget = [max_compiles]
    log: list[str] = []

    def is_target(s) -> bool:
        return bool(s.name) and any(t == s.name or t.endswith("." + s.name) for t in targets)

    # 1. whole commands, bottom-up (later things depend on earlier ones)
    def command_cands(t: str):
        out = []
        for s in reversed(split_commands(t)):
            if s.kind == "structural" or is_target(s):
                continue
            if s.kind == "debug" and keep_debug:
                continue
            out.append((f"{s.head} {s.name}".strip() + f" (line {s.start})", _drop_lines(t, [(s.start, s.end)])))
        return out

    text = _try_each(oracle, text, command_cands, budget, "deleted", log)

    if tactics:
        # 2. single tactic lines inside the proofs that are left
        def tactic_cands(t: str):
            lines = t.split("\n")
            out = []
            for s in reversed(split_commands(t)):
                for a, b in reversed(tactic_units(s, lines)):
                    out.append((f"tactic `{lines[a - 1].strip()[:60]}` (line {a})", _drop_lines(t, [(a, b)])))
            return out

        text = _try_each(oracle, text, tactic_cands, budget, "deleted", log)

        # 3. single arguments of simp [...] / linarith [...] / ...
        def arg_cands(t: str):
            lines = t.split("\n")
            out = []
            for k, line in enumerate(lines):
                for a, b in lemma_args(line):
                    new_line = line[:a] + line[b:]
                    new = "\n".join(lines[:k] + [new_line] + lines[k + 1:])
                    out.append((f"argument `{line[a:b].strip(' ,')}` (line {k + 1})", new))
            return out

        text = _try_each(oracle, text, arg_cands, budget, "removed", log)

        # 4. deleting arguments can make more declarations unused
        text = _try_each(oracle, text, command_cands, budget, "deleted", log)

    if budget[0] <= 0:
        log.append(f"(stopped early: compile budget of {max_compiles} used up)")
    return text, log, oracle
