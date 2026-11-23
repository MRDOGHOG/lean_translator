"""One `translate(texts)` interface over both model kinds:

- a from-scratch checkpoint (`refiner.pt`, see `model.py` / `train.py`), and
- a fine-tuned Hugging Face seq2seq model directory (see `train_hf.py`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

META_FILE = "lean_translator.json"
_GLUE_BEFORE = re.compile(r"(?<=\w)(?=\[T\d+\])")
_GLUE_AFTER = re.compile(r"(?<=\[T)(\d+\])(?=\w)")


def _fix_slot_spacing(text: str) -> str:
    """Some tokenizers glue added tokens to neighbouring words ("[T0]equals")."""
    return _GLUE_AFTER.sub(r"\1 ", _GLUE_BEFORE.sub(" ", text))


class Translator:
    meta: dict

    def translate(self, texts: list[str], batch_size: int = 64) -> list[str]:  # pragma: no cover
        raise NotImplementedError


class ScratchTranslator(Translator):
    def __init__(self, path: str, device: str):
        from tokenizers import Tokenizer

        from .model import Config, Seq2Seq

        ck = torch.load(path, map_location=device, weights_only=False)
        self.model = Seq2Seq(Config(**ck["cfg"]))
        self.model.load_state_dict(ck["state"])
        self.model.to(device).eval()
        self.tok = Tokenizer.from_str(ck["tokenizer"])
        self.meta = ck.get("meta", {})
        self.device = device

    @torch.no_grad()
    def translate(self, texts: list[str], batch_size: int = 256) -> list[str]:
        from .tokenizer import decode, encode
        from .train import _pad

        out: list[str] = []
        for k in range(0, len(texts), batch_size):
            ids = encode(self.tok, texts[k : k + batch_size], self.model.cfg.max_len)
            gen = self.model.greedy(_pad(ids).to(self.device), max_len=128)
            out.extend(_fix_slot_spacing(t) for t in decode(self.tok, gen.tolist()))
        return out


class HFTranslator(Translator):
    def __init__(self, path: str, device: str, num_beams: int = 4):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device).eval()
        meta_path = Path(path) / META_FILE
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.device = device
        self.num_beams = num_beams

    @torch.no_grad()
    def translate(self, texts: list[str], batch_size: int = 64) -> list[str]:
        out: list[str] = []
        max_len = self.meta.get("max_len", 256)
        for k in range(0, len(texts), batch_size):
            enc = self.tok(texts[k : k + batch_size], return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=160, num_beams=self.num_beams,
                                      no_repeat_ngram_size=3, early_stopping=True)
            out.extend(_fix_slot_spacing(t.strip()) for t in self.tok.batch_decode(gen, skip_special_tokens=True))
        return out


def load_translator(path: str, device: str | None = None) -> Translator:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if Path(path).is_dir():
        return HFTranslator(path, device)
    return ScratchTranslator(path, device)
