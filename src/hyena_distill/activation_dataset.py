from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from hyena_distill.types import (
    ActivationExample,
    ActivationSplits,
    SplitName,
    TokenSequence,
)


@dataclass(frozen=True)
class DatasetProvenance:
    model_id: str
    model_revision: str
    tokenizer_revision: str
    teacher_precision: str
    layer_index: int
    dataset_id: str
    dataset_config: str | None
    dataset_revision: str | None
    source_type: str


@dataclass(frozen=True)
class ActivationExportSummary:
    sequence_count: int
    token_count: int
    shard_count: int
    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    bytes_written: int
    splits: dict[str, dict[str, int]]


_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
_FORMAT_VERSION = 1


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _source_license(provenance: DatasetProvenance) -> str:
    if (
        provenance.source_type == "huggingface-stream"
        and provenance.dataset_id == "wikimedia/wikipedia"
    ):
        return (
            "The Wikimedia Wikipedia dataset card lists CC-BY-SA-3.0 and GFDL. "
            "Review those source terms before redistributing derived activations."
        )
    if provenance.source_type == "local-jsonl":
        return "The local JSONL source license is not recorded by this run."
    return "No upstream license terms were recorded by this run."


def _dataset_card(
    provenance: DatasetProvenance,
    summary: ActivationExportSummary,
    hidden_size: int | None,
) -> str:
    split_lines = "\n".join(
        f"| {name} | {summary.splits[name]['sequence_count']} | {summary.splits[name]['token_count']} |"
        for name in _SPLITS
    )
    widths = "unknown" if hidden_size is None else str(hidden_size)
    input_dtypes = ", ".join(summary.input_dtypes) or "none"
    output_dtypes = ", ".join(summary.output_dtypes) or "none"
    dataset_config = provenance.dataset_config or "not recorded"
    dataset_revision = provenance.dataset_revision or "not recorded"
    return f"""---
task_categories:
- feature-extraction
language:
- en
---

# Captured Qwen attention activations

This dataset contains token IDs and native-dtype hidden activations captured from `{provenance.model_id}` at resolved revision `{provenance.model_revision}`. The tokenizer revision is `{provenance.tokenizer_revision}`. The target is `model.layers[{provenance.layer_index}].self_attn`. Its input is after `input_layernorm`; its output is the first attention return value before residual addition. These tensors are hidden activations, not attention maps.

The teacher precision was `{provenance.teacher_precision}`. The recorded input dtypes are {input_dtypes}; output dtypes are {output_dtypes}. Activation shape is `[tokens, {widths}]`.

The source is `{provenance.dataset_id}` with configuration `{dataset_config}` at revision `{dataset_revision}` (`{provenance.source_type}`). {_source_license(provenance)}

| Split | Sequences | Tokens |
| --- | ---: | ---: |
{split_lines}

`data/*.safetensors` stores concatenated token IDs, attention inputs, attention outputs, and sequence offsets. `index.jsonl` maps each complete sequence to its split, source identifiers, text hash, chunk, and offsets. Raw article text and Qwen teacher weights are not included.

This is a small implementation experiment on one attention submodule. It does not establish general Qwen quality, full-model perplexity, generation quality, or a general attention replacement. The test split is included. Publishing it makes those targets public; future independent evaluations need a fresh held-out set.
"""


def _validate_example(split: SplitName, example: ActivationExample) -> int:
    sequence = example.sequence
    if sequence.split != split:
        raise ValueError(
            f"Activation sequence {sequence.source_id!r} is marked {sequence.split!r} "
            f"but appears in the {split!r} split"
        )
    length = len(sequence.token_ids)
    input_tensor = example.attention_input
    output_tensor = example.attention_output
    if input_tensor.ndim != 2 or output_tensor.ndim != 2:
        raise ValueError("Activation tensors must have [tokens, hidden_size] shape")
    if input_tensor.shape[0] != length or output_tensor.shape[0] != length:
        raise ValueError("Activation tensor lengths must match the sequence token IDs")
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("Attention input and output activation shapes must match")
    if length < 1:
        raise ValueError("Activation sequences must contain at least one token")
    return length


def write_activation_dataset(
    destination: Path,
    splits: ActivationSplits,
    *,
    provenance: DatasetProvenance,
    max_shard_bytes: int = 256 * 1024 * 1024,
) -> ActivationExportSummary:
    if max_shard_bytes < 1:
        raise ValueError("max_shard_bytes must be positive")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(
            f"Activation dataset destination already exists: {destination}"
        )
    destination.mkdir(parents=True)
    (destination / "data").mkdir()

    split_counts = {name: {"sequence_count": 0, "token_count": 0} for name in _SPLITS}
    input_dtypes: set[str] = set()
    output_dtypes: set[str] = set()
    shard_count = 0
    hidden_size: int | None = None
    index_path = destination / "index.jsonl"

    with index_path.open("w", encoding="utf-8") as index_file:
        for split in _SPLITS:
            examples = getattr(splits, split)
            pending: list[tuple[ActivationExample, torch.Tensor, torch.Tensor]] = []
            pending_bytes = 0
            pending_dtypes: tuple[str, str] | None = None
            shard_number = 0

            def flush(split_name: SplitName = split) -> None:
                nonlocal \
                    pending, \
                    pending_bytes, \
                    pending_dtypes, \
                    shard_number, \
                    shard_count
                if not pending:
                    return
                shard_relpath = (
                    Path("data") / f"{split_name}-{shard_number:05d}.safetensors"
                )
                token_ids: list[torch.Tensor] = []
                inputs: list[torch.Tensor] = []
                outputs: list[torch.Tensor] = []
                offsets = [0]
                rows: list[dict[str, Any]] = []
                for row, (example, input_tensor, output_tensor) in enumerate(pending):
                    sequence = example.sequence
                    start = offsets[-1]
                    stop = start + len(sequence.token_ids)
                    token_ids.append(
                        torch.tensor(sequence.token_ids, dtype=torch.int64)
                    )
                    inputs.append(input_tensor)
                    outputs.append(output_tensor)
                    offsets.append(stop)
                    rows.append(
                        {
                            "format_version": _FORMAT_VERSION,
                            "split": split_name,
                            "shard": shard_relpath.as_posix(),
                            "row": row,
                            "source_id": sequence.source_id,
                            "title": sequence.title,
                            "url": sequence.url,
                            "text_sha256": sequence.text_sha256,
                            "chunk_index": sequence.chunk_index,
                            "token_length": len(sequence.token_ids),
                            "start": start,
                            "stop": stop,
                            "input_dtype": _dtype_name(input_tensor),
                            "output_dtype": _dtype_name(output_tensor),
                        }
                    )
                save_file(
                    {
                        "token_ids": torch.cat(token_ids).contiguous(),
                        "attention_input": torch.cat(inputs, dim=0).contiguous(),
                        "attention_output": torch.cat(outputs, dim=0).contiguous(),
                        "offsets": torch.tensor(offsets, dtype=torch.int64),
                    },
                    str(destination / shard_relpath),
                )
                index_file.writelines(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    for row in rows
                )
                shard_number += 1
                shard_count += 1
                pending = []
                pending_bytes = 0
                pending_dtypes = None

            for example in examples:
                length = _validate_example(split, example)
                input_tensor = example.attention_input.detach().to("cpu").contiguous()
                output_tensor = example.attention_output.detach().to("cpu").contiguous()
                if hidden_size is None:
                    hidden_size = input_tensor.shape[1]
                elif input_tensor.shape[1] != hidden_size:
                    raise ValueError(
                        "All activation examples must have the same hidden size"
                    )
                input_dtypes.add(_dtype_name(input_tensor))
                output_dtypes.add(_dtype_name(output_tensor))
                dtypes = (_dtype_name(input_tensor), _dtype_name(output_tensor))
                estimated_bytes = (
                    length * 8
                    + input_tensor.numel() * input_tensor.element_size()
                    + output_tensor.numel() * output_tensor.element_size()
                )
                if pending and (
                    dtypes != pending_dtypes
                    or pending_bytes + estimated_bytes > max_shard_bytes
                ):
                    flush()
                pending.append((example, input_tensor, output_tensor))
                pending_bytes += estimated_bytes
                pending_dtypes = dtypes
                split_counts[split]["sequence_count"] += 1
                split_counts[split]["token_count"] += length
            flush()

    summary = ActivationExportSummary(
        sequence_count=sum(row["sequence_count"] for row in split_counts.values()),
        token_count=sum(row["token_count"] for row in split_counts.values()),
        shard_count=shard_count,
        input_dtypes=tuple(sorted(input_dtypes)),
        output_dtypes=tuple(sorted(output_dtypes)),
        bytes_written=0,
        splits=split_counts,
    )
    (destination / "README.md").write_text(
        _dataset_card(provenance, summary, hidden_size), encoding="utf-8"
    )
    bytes_written = sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    return ActivationExportSummary(
        sequence_count=summary.sequence_count,
        token_count=summary.token_count,
        shard_count=summary.shard_count,
        input_dtypes=summary.input_dtypes,
        output_dtypes=summary.output_dtypes,
        bytes_written=bytes_written,
        splits=summary.splits,
    )


def _load_index(directory: Path, split: SplitName | None) -> list[dict[str, Any]]:
    index_path = directory / "index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Activation dataset index is missing: {index_path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in activation index line {line_number}"
            ) from error
        if not isinstance(row, dict) or row.get("format_version") != _FORMAT_VERSION:
            raise ValueError(
                f"Unsupported activation index entry on line {line_number}"
            )
        if row.get("split") not in _SPLITS:
            raise ValueError(f"Invalid split in activation index line {line_number}")
        shard = row.get("shard")
        if (
            not isinstance(shard, str)
            or Path(shard).is_absolute()
            or ".." in Path(shard).parts
        ):
            raise ValueError(
                f"Invalid shard path in activation index line {line_number}"
            )
        if split is None or row["split"] == split:
            rows.append(row)
    return rows


def iter_activation_examples(
    directory: Path, *, split: SplitName | None = None
) -> Iterator[ActivationExample]:
    directory = Path(directory)
    if split is not None and split not in _SPLITS:
        raise ValueError(f"Unknown activation split: {split!r}")
    rows = _load_index(directory, split)
    rows_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_shard[row["shard"]].append(row)

    current_shard = None
    tensors: dict[str, torch.Tensor] | None = None
    for row in rows:
        shard = row["shard"]
        if shard != current_shard:
            tensors = load_file(str(directory / shard), device="cpu")
            current_shard = shard
            shard_rows = rows_by_shard[shard]
            offsets = tensors.get("offsets")
            token_ids = tensors.get("token_ids")
            activation_input = tensors.get("attention_input")
            activation_output = tensors.get("attention_output")
            if (
                offsets is None
                or token_ids is None
                or activation_input is None
                or activation_output is None
            ):
                raise ValueError(f"Shard {shard!r} is missing a required tensor")
            if offsets.dtype != torch.int64 or offsets.ndim != 1:
                raise ValueError(
                    f"Shard {shard!r} offsets must be a one-dimensional int64 tensor"
                )
            if token_ids.dtype != torch.int64 or token_ids.ndim != 1:
                raise ValueError(
                    f"Shard {shard!r} token_ids must be a one-dimensional int64 tensor"
                )
            if activation_input.ndim != 2 or activation_output.ndim != 2:
                raise ValueError(f"Shard {shard!r} activation tensors must be rank two")
            if activation_input.shape != activation_output.shape:
                raise ValueError(f"Shard {shard!r} activation shapes do not match")
            if activation_input.shape[0] != token_ids.numel():
                raise ValueError(f"Shard {shard!r} tensors have different token counts")
            if offsets.numel() != len(shard_rows) + 1 or int(offsets[0]) != 0:
                raise ValueError(
                    f"Shard {shard!r} offsets do not cover all indexed rows"
                )
            if int(offsets[-1]) != token_ids.numel() or not torch.all(
                offsets[1:] >= offsets[:-1]
            ):
                raise ValueError(
                    f"Shard {shard!r} offsets are not contiguous token boundaries"
                )
            ordered_rows = sorted(shard_rows, key=lambda item: item.get("row", -1))
            if [item.get("row") for item in ordered_rows] != list(
                range(len(ordered_rows))
            ):
                raise ValueError(
                    f"Shard {shard!r} index rows must be unique and contiguous"
                )
            if any(item.get("split") != row["split"] for item in ordered_rows):
                raise ValueError(f"Shard {shard!r} cannot mix data splits")

        assert tensors is not None
        offsets = tensors["offsets"]
        row_index = row.get("row")
        if (
            not isinstance(row_index, int)
            or row_index < 0
            or row_index + 1 >= offsets.numel()
        ):
            raise ValueError(f"Index row has an invalid row number for shard {shard!r}")
        start = int(offsets[row_index])
        stop = int(offsets[row_index + 1])
        if row.get("start") != start or row.get("stop") != stop:
            raise ValueError(
                f"Index offsets do not match shard {shard!r} row {row_index}"
            )
        if row.get("token_length") != stop - start or stop <= start:
            raise ValueError(
                f"Index token length does not match shard {shard!r} row {row_index}"
            )
        input_tensor = tensors["attention_input"][start:stop].contiguous()
        output_tensor = tensors["attention_output"][start:stop].contiguous()
        if row.get("input_dtype") != _dtype_name(input_tensor):
            raise ValueError(
                f"Input dtype does not match shard {shard!r} row {row_index}"
            )
        if row.get("output_dtype") != _dtype_name(output_tensor):
            raise ValueError(
                f"Output dtype does not match shard {shard!r} row {row_index}"
            )
        sequence = TokenSequence(
            source_id=row["source_id"],
            title=row["title"],
            url=row["url"],
            text_sha256=row["text_sha256"],
            chunk_index=row["chunk_index"],
            split=row["split"],
            token_ids=tuple(
                int(token) for token in tensors["token_ids"][start:stop].tolist()
            ),
        )
        yield ActivationExample(sequence, input_tensor, output_tensor)
