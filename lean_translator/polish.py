"""Optional last pass: a local language model (via Ollama) smooths the wording.

The rules do all the organising and all the mathematics.  The model only sees
the English, with every formula, reference and Lean name replaced by a
placeholder ([M1], [R2], [K3]).  Its rewrite is accepted only if every
placeholder comes back exactly once, unchanged, and nothing new was added;
otherwise the rule-based text is kept.  So the model can make sentences
smoother but cannot change a formula or a citation.

    from lean_translator.polish import Polisher
    p = Polisher("qwen2.5:3b")                # Ollama at localhost:11434
    p.statement(marked) / p.proof(steps)      # -> polished marked text, or None
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

SYSTEM = (
    "You are the copy editor of a mathematics research paper. You rewrite text into smooth, formal, concise "
    "mathematical English, in the style of a published paper.\n"
    "Rules:\n"
    "- Placeholders like [M1], [R2], [K3] stand for formulas, references to other results, and names. Copy every "
    "placeholder exactly once and unchanged, and never invent new ones. Do not put them in quotes or backticks.\n"
    "- Do not add, drop or change any mathematical claim, hypothesis or step. Do not add explanations, motivation "
    "or opinions.\n"
    "- Output only the rewritten text, with no preamble and no markdown."
)
STATEMENT_PROMPT = "Rewrite this theorem statement as it would appear in a paper:\n\n{text}"
PROOF_PROMPT = ("Rewrite these proof steps as the proof in a paper: one or a few flowing paragraphs, no numbered "
                "list. Keep the order of the steps; case splits may become 'In the first case, ...'.\n\n{text}")

_SEG = re.compile("\x01(.*?)\x02|`([^`]+)`|\x03(.*?)\x04|\x05(.*?)\x06", re.S)
_PH = re.compile(r"\[([MRK])(\d+)\]")


def mask(marked: str) -> tuple[str, dict[str, str]]:
    """Replace every formula / code / citation segment by a placeholder; return text and the table."""
    table: dict[str, str] = {}
    counts = {"M": 0, "R": 0, "K": 0}

    def sub(m: re.Match) -> str:
        kind = "M" if m.group(1) is not None else "K" if m.group(2) is not None else "R"
        counts[kind] += 1
        key = f"[{kind}{counts[kind]}]"
        table[key] = m.group(0)
        return key

    return _SEG.sub(sub, marked), table


def unmask(text: str, table: dict[str, str]) -> str:
    return _PH.sub(lambda m: table.get(m.group(0), m.group(0)), text)


def acceptable(original: str, rewritten: str, table: dict[str, str]) -> bool:
    """Every placeholder exactly once, none invented, sane length, no chatter."""
    found = _PH.findall(rewritten)
    keys = [f"[{k}{n}]" for k, n in found]
    if sorted(keys) != sorted(table) or len(set(keys)) != len(keys):
        return False
    if not rewritten.strip() or any(ch in rewritten for ch in "\x01\x02\x03\x04\x05\x06`$\\"):
        return False
    ratio = len(rewritten) / max(1, len(original))
    if ratio < 0.4 or ratio > 2.2:
        return False
    low = rewritten.lower()
    if low.startswith(("here is", "sure", "certainly", "rewritten")) or "placeholder" in low:
        return False
    return True


class Polisher:
    def __init__(self, model: str, host: str = "http://localhost:11434", cache: str | None = None,
                 budget_s: float = 0, chat=None) -> None:
        self.model, self.host = model, host
        self.cache_path = Path(cache) if cache else None
        self.cache: dict[str, str | None] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        self.deadline = time.time() + budget_s if budget_s else None
        self.stats = {"accepted": 0, "rejected": 0, "cached": 0, "skipped (time budget)": 0, "errors": 0}
        if chat is None:
            from .refiner.distill import ollama_chat

            chat = lambda system, user: ollama_chat(self.host, self.model, system, user, timeout=300)  # noqa: E731
        self.chat = chat

    def _ask(self, prompt: str, masked: str, table: dict[str, str]) -> str | None:
        key = hashlib.sha1(f"{self.model}\n{prompt}".encode()).hexdigest()
        if key in self.cache:
            self.stats["cached"] += 1
            out = self.cache[key]
            return out if out is None or acceptable(masked, out, table) else None
        if self.deadline and time.time() > self.deadline:
            self.stats["skipped (time budget)"] += 1
            return None
        try:
            out = self.chat(SYSTEM, prompt).strip().strip('"').strip()
        except Exception:
            self.stats["errors"] += 1
            return None
        ok = acceptable(masked, out, table)
        self.stats["accepted" if ok else "rejected"] += 1
        self.cache[key] = out if ok else None
        return out if ok else None

    def statement(self, marked: str) -> str | None:
        masked, table = mask(marked)
        if len(masked.split()) < 6:
            return None
        out = self._ask(STATEMENT_PROMPT.format(text=masked), masked, table)
        return unmask(out, table) if out else None

    def proof(self, steps: list[tuple[int, str]]) -> str | None:
        """Numbered steps (depth, marked sentence) -> one marked prose text, or None."""
        lines, counters = [], [0] * 10
        for depth, s in steps:
            depth = min(depth, 9)
            counters[depth] += 1
            for k in range(depth + 1, 10):
                counters[k] = 0
            lines.append("   " * depth + f"{counters[depth]}. {s}")
        joined = "\n".join(lines)
        masked, table = mask(joined)
        out = self._ask(PROOF_PROMPT.format(text=masked), masked, table)
        return unmask(out, table) if out else None

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
