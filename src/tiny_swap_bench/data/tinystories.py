"""TinyStories tokenization with deterministic validation split."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import tiktoken
import torch
from datasets import Dataset, load_dataset


def _stable_bucket(text: str, buckets: int = 100) -> int:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % buckets


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, T) int64
    labels: torch.Tensor  # (B, T) int64


class TinyStoriesTokenizer:
    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.enc = tiktoken.get_encoding(encoding_name)

    def encode_text(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})


def load_train_val_rows(
    dataset_path: str,
    split_name: str,
    val_fraction: float,
    *,
    max_examples: int | None = None,
) -> tuple[Dataset, Dataset]:
    """Split examples into train/val by deterministic hash bucket (≈ ``val_fraction``).

    ``max_examples`` optionally truncates to the first N rows (used for fast smoke runs).
    """
    ds = load_dataset(dataset_path, split=split_name)
    if max_examples is not None:
        ds = ds.select(range(min(len(ds), max_examples)))
    bucket_cut = max(1, int(100 * val_fraction))
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for row in ds:
        text = row.get("text") or row.get("story") or ""
        if not isinstance(text, str):
            raise ValueError(f"Unexpected row format: keys={list(row.keys())}")
        if _stable_bucket(text) < bucket_cut:
            val_rows.append({"text": text})
        else:
            train_rows.append({"text": text})
    return Dataset.from_list(train_rows), Dataset.from_list(val_rows)


def infinite_train_texts(train_ds: Dataset, seed: int) -> Iterator[str]:
    """Infinite shuffle-with-seed iteration over training texts."""
    n = len(train_ds)
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=rng).tolist()
    pos = 0
    while True:
        idx = order[pos % n]
        yield train_ds[int(idx)]["text"]
        pos += 1


def text_stream_to_tokens(stream: Iterator[str], tok: TinyStoriesTokenizer) -> Iterator[int]:
    for text in stream:
        yield from tok.encode_text(text)


def build_batches(
    token_stream: Iterator[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[Batch]:
    """Pack tokens into ``(B, T)`` LM batches from an int token iterator."""
    buf: list[int] = []
    needed = batch_size * (seq_len + 1)
    while True:
        try:
            while len(buf) < needed:
                buf.append(next(token_stream))
        except StopIteration:
            return
        chunk = buf[:needed]
        del buf[:needed]
        x = torch.tensor(chunk, dtype=torch.long, device=device)  # ((B*(T+1)),)
        x = x.view(batch_size, seq_len + 1)
        inputs = x[:, :-1].contiguous()  # (B, T)
        targets = x[:, 1:].contiguous()  # (B, T)
        yield Batch(input_ids=inputs, labels=targets)


def make_train_token_iterator(train_ds: Dataset, tok: TinyStoriesTokenizer, seed: int) -> Iterator[int]:
    return text_stream_to_tokens(infinite_train_texts(train_ds, seed), tok)


def make_val_token_iterator(val_ds: Dataset, tok: TinyStoriesTokenizer) -> Iterator[int]:
    texts = (val_ds[i]["text"] for i in range(len(val_ds)))
    return text_stream_to_tokens(texts, tok)
