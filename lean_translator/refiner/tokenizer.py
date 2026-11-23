"""Shared byte-level BPE vocabulary for source and target (HF `tokenizers`)."""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .model import BOS, EOS, PAD, UNK

SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]
assert [PAD, BOS, EOS, UNK] == [0, 1, 2, 3]


def train_tokenizer(texts: Iterable[str], vocab_size: int = 8000) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), min_frequency=2,
    )
    tok.train_from_iterator(texts, trainer)
    return tok


def encode(tok: Tokenizer, texts: list[str], max_len: int) -> list[list[int]]:
    return [[BOS] + e.ids[: max_len - 2] + [EOS] for e in tok.encode_batch(texts)]


def decode(tok: Tokenizer, ids: list[list[int]]) -> list[str]:
    out = []
    for seq in ids:
        if EOS in seq:
            seq = seq[: seq.index(EOS)]
        out.append([i for i in seq if i not in (PAD, BOS)])
    return [s.strip() for s in tok.decode_batch(out)]
