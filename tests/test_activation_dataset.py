from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from hyena_distill.activation_dataset import (
    DatasetProvenance,
    iter_activation_examples,
    write_activation_dataset,
)
from hyena_distill.types import ActivationExample, ActivationSplits, TokenSequence


def _examples(
    split: str, prefix: int, dtype: torch.dtype
) -> tuple[ActivationExample, ...]:
    result = []
    for row, length in enumerate((3, 2, 4)):
        source_id = f"{split}-{row}"
        sequence = TokenSequence(
            source_id=source_id,
            title=f"Title {source_id}",
            url=f"https://example.test/{source_id}",
            text_sha256=f"hash-{source_id}",
            chunk_index=row,
            split=split,
            token_ids=tuple(range(prefix + row * 10, prefix + row * 10 + length)),
        )
        values = torch.arange(length * 4, dtype=torch.float32).reshape(length, 4)
        result.append(
            ActivationExample(
                sequence,
                values.to(dtype=dtype),
                (values + 1).to(dtype=torch.float32),
            )
        )
    return tuple(result)


def _provenance() -> DatasetProvenance:
    return DatasetProvenance(
        model_id="Qwen/Qwen2.5-7B",
        model_revision="model-commit",
        tokenizer_revision="tokenizer-commit",
        teacher_precision="bf16",
        layer_index=2,
        dataset_id="local-jsonl",
        dataset_config="articles.jsonl",
        dataset_revision=None,
        source_type="local-jsonl",
    )


def test_dataset_round_trips_all_splits_across_native_dtype_shards(
    tmp_path: Path,
) -> None:
    train = _examples("train", 100, torch.bfloat16)
    mixed_train = (
        train[0],
        ActivationExample(
            train[1].sequence,
            train[1].attention_input.to(torch.float16),
            train[1].attention_output,
        ),
        train[2],
    )
    original = ActivationSplits(
        train=mixed_train,
        validation=_examples("validation", 200, torch.float16),
        test=_examples("test", 300, torch.bfloat16),
    )
    destination = tmp_path / "activation_dataset"

    summary = write_activation_dataset(
        destination,
        original,
        provenance=_provenance(),
        max_shard_bytes=160,
    )

    assert summary.sequence_count == 9
    assert summary.token_count == 27
    assert summary.shard_count > 3
    assert summary.input_dtypes == ("bfloat16", "float16")
    assert summary.output_dtypes == ("float32",)
    assert summary.bytes_written == sum(
        path.stat().st_size for path in destination.rglob("*") if path.is_file()
    )
    assert (
        "local JSONL source license is not recorded"
        in (destination / "README.md").read_text()
    )
    assert "Wikimedia" not in (destination / "README.md").read_text()

    restored = tuple(iter_activation_examples(destination))
    expected = original.train + original.validation + original.test
    assert len(restored) == len(expected)
    assert [example.sequence for example in restored] == [
        example.sequence for example in expected
    ]
    seen_token_ids = []
    for actual, wanted in zip(restored, expected, strict=True):
        assert actual.sequence.token_ids == wanted.sequence.token_ids
        assert actual.sequence.title == wanted.sequence.title
        assert actual.sequence.url == wanted.sequence.url
        assert actual.sequence.text_sha256 == wanted.sequence.text_sha256
        assert actual.sequence.chunk_index == wanted.sequence.chunk_index
        assert actual.attention_input.dtype == wanted.attention_input.dtype
        assert actual.attention_output.dtype == wanted.attention_output.dtype
        torch.testing.assert_close(actual.attention_input, wanted.attention_input)
        torch.testing.assert_close(actual.attention_output, wanted.attention_output)
        seen_token_ids.extend(actual.sequence.token_ids)
    assert len(seen_token_ids) == summary.token_count
    assert len(set(seen_token_ids)) == summary.token_count
    assert len(tuple(iter_activation_examples(destination, split="test"))) == 3

    index = [
        json.loads(line)
        for line in (destination / "index.jsonl").read_text().splitlines()
    ]
    assert {row["split"] for row in index} == {"train", "validation", "test"}
    assert all(row["format_version"] == 1 for row in index)
    assert all(row["token_length"] == row["stop"] - row["start"] for row in index)
    for shard_path in {row["shard"] for row in index}:
        rows = sorted(
            (row for row in index if row["shard"] == shard_path),
            key=lambda row: row["row"],
        )
        tensors = load_file(str(destination / shard_path))
        assert tensors["offsets"].tolist() == [0] + [row["stop"] for row in rows]


def test_reader_rejects_an_index_offset_mismatch(tmp_path: Path) -> None:
    destination = tmp_path / "activation_dataset"
    write_activation_dataset(
        destination,
        ActivationSplits(
            train=_examples("train", 100, torch.float32)[:1],
            validation=(),
            test=(),
        ),
        provenance=_provenance(),
    )
    index_path = destination / "index.jsonl"
    row = json.loads(index_path.read_text())
    row["start"] = 1
    index_path.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="offsets do not match"):
        tuple(iter_activation_examples(destination))


def test_custom_hugging_face_source_has_no_inferred_wikimedia_license(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "activation_dataset"
    provenance = replace(
        _provenance(),
        dataset_id="example/custom-corpus",
        source_type="huggingface-stream",
    )

    write_activation_dataset(
        destination,
        ActivationSplits(train=(), validation=(), test=()),
        provenance=provenance,
    )

    card = (destination / "README.md").read_text()
    assert "No upstream license terms were recorded by this run." in card
    assert "Wikimedia" not in card
    assert "local JSONL" not in card
