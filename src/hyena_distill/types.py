from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class TokenSequence:
    source_id: str
    title: str
    url: str
    text_sha256: str
    chunk_index: int
    split: SplitName
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActivationExample:
    sequence: TokenSequence
    attention_input: torch.Tensor
    attention_output: torch.Tensor


@dataclass(frozen=True)
class DocumentSplits:
    train: tuple[Document, ...]
    validation: tuple[Document, ...]
    test: tuple[Document, ...]


@dataclass(frozen=True)
class SequenceSplits:
    train: tuple[TokenSequence, ...]
    validation: tuple[TokenSequence, ...]
    test: tuple[TokenSequence, ...]


@dataclass(frozen=True)
class ActivationSplits:
    train: tuple[ActivationExample, ...]
    validation: tuple[ActivationExample, ...]
    test: tuple[ActivationExample, ...]
