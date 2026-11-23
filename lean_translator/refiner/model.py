"""A small encoder-decoder Transformer, the same architecture used by
classic neural machine translation systems (Vaswani et al. 2017, Marian/OPUS-MT),
scaled down to a few million parameters so it runs fast on CPU and
very fast on GPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

PAD, BOS, EOS, UNK = 0, 1, 2, 3


@dataclass
class Config:
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 4
    dec_layers: int = 3
    ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class Seq2Seq(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len)
        self.drop = nn.Dropout(cfg.dropout)
        self.transformer = nn.Transformer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            num_encoder_layers=cfg.enc_layers, num_decoder_layers=cfg.dec_layers,
            dim_feedforward=cfg.ff, dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.out.weight = self.embed.weight  # tied embeddings, as in Marian
        # With tied weights the default N(0, 1) init gives huge initial logits;
        # use the Transformer paper's d_model^-0.5 scale instead.
        nn.init.normal_(self.embed.weight, mean=0.0, std=cfg.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()
        self.scale = math.sqrt(cfg.d_model)

    def _emb(self, ids: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pos(self.embed(ids) * self.scale))

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = src == PAD
        mem = self.transformer.encoder(self._emb(src), src_key_padding_mask=mask)
        return mem, mask

    def decode(self, tgt_in: torch.Tensor, mem: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        T = tgt_in.size(1)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=tgt_in.device), diagonal=1)
        h = self.transformer.decoder(
            self._emb(tgt_in), mem, tgt_mask=causal,
            tgt_key_padding_mask=tgt_in == PAD, memory_key_padding_mask=mem_mask,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        mem, mask = self.encode(src)
        return self.decode(tgt_in, mem, mask)

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, max_len: int = 128, no_repeat_ngram: int = 3) -> torch.Tensor:
        """Batched greedy decoding (re-runs the decoder each step; fine for short outputs).

        `no_repeat_ngram` forbids emitting any n-gram twice, the standard
        fix for small models getting stuck in loops ("new new new ...")."""
        mem, mask = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(ys, mem, mask)[:, -1]
            if no_repeat_ngram and ys.size(1) >= no_repeat_ngram:
                n = no_repeat_ngram
                rows = ys.tolist()
                for b, seq in enumerate(rows):
                    prefix = tuple(seq[len(seq) - n + 1 :])
                    banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix]
                    if banned:
                        logits[b, banned] = float("-inf")
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]

    def save(self, path: str, tokenizer_json: str, meta: dict | None = None) -> None:
        torch.save({"cfg": asdict(self.cfg), "state": self.state_dict(), "tokenizer": tokenizer_json,
                    "meta": meta or {}}, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> tuple["Seq2Seq", str]:
        ck = torch.load(path, map_location=device, weights_only=False)
        model = Seq2Seq(Config(**ck["cfg"]))
        model.load_state_dict(ck["state"])
        return model.to(device).eval(), ck["tokenizer"]
