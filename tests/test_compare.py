from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hyena_distill.compare import load_run, main, write_comparison_report


def _write_run(
    path: Path,
    *,
    precision: str,
    source_ids: list[str],
    legacy: bool = False,
    text_hashes: dict[str, str] | None = None,
    sequence_hashes: dict[str, str] | None = None,
    backend: str = "rocm",
    max_length: int = 512,
    group_learning_rate: float = 1e-3,
) -> Path:
    path.mkdir(parents=True)
    text_hashes = text_hashes or {
        source_id: hashlib.sha256(f"text-{source_id}".encode()).hexdigest()
        for source_id in source_ids
    }
    model = {
        "id": "Qwen/test-model",
        "revision": "model-revision-a",
        "tokenizer_revision": "tokenizer-revision-a",
        "quantization": "4-bit NF4" if precision == "nf4" else "BF16",
    }
    config = {
        "model_id": "Qwen/test-model",
        "model_revision": "model-revision-a",
        "layer_index": 2,
        "max_length": max_length,
        "max_chunks_per_document": 1,
        "max_documents": len(source_ids) + 4,
        "dataset_id": "test/dataset",
        "dataset_config": "en",
        "dataset_revision": "data-revision-a",
        "seed": 43,
        "train_steps": 100,
    }
    dataset = {
        "dataset_id": "test/dataset",
        "dataset_config": "en",
        "revision": "data-revision-a",
    }
    metadata: dict = {
        "config": config,
        "model": model,
        "target": {"layer_index": 2},
        "dataset": dataset,
        "document_splits": {"test": source_ids},
        "source_documents": {
            source_id: {"text_sha256": text_hashes[source_id]}
            for source_id in source_ids
        },
        "splits": {
            "test": {
                "document_count": len(source_ids),
                "sequence_count": len(source_ids),
                "token_count": len(source_ids) * 10,
                "source_ids": source_ids,
            }
        },
        "settings": {
            "max_length": max_length,
            "max_chunks_per_document": 1,
            "seed": 43,
            "batch_size": 1,
            "optimizer_parameter_groups": [
                {
                    "learning_rate": group_learning_rate,
                    "weight_decay": 0.01,
                    "parameter_count": 10,
                    "parameter_names": ["filter.weight"],
                }
            ],
        },
        "training": {"configured_steps": 100, "selected_step": 100},
        "runtime": {"backend": backend} if not legacy else {"cuda_available": True},
    }
    if not legacy:
        config["teacher_precision"] = precision
        model["teacher_precision"] = precision
    activation = {
        "hyena": {
            "mse": 0.004,
            "relative_l2_error": 0.7,
            "cosine_similarity": 0.6,
            "token_count": len(source_ids) * 10,
        },
        "zero": {"mse": 0.01},
        "training_mean": {"mse": 0.008},
    }
    metrics: dict = {
        "activation": {"test": activation},
        "replacement": {
            "teacher_next_token_loss": 2.0,
            "hyena_next_token_loss": 2.01,
            "next_token_loss_delta": 0.01,
            "predicted_token_count": len(source_ids) * 9,
        },
    }
    if not legacy:
        hashes = sequence_hashes or {
            source_id: f"tokens-{source_id}" for source_id in source_ids
        }
        metrics["activation_test_examples"] = [
            {
                "source_id": source_id,
                "chunk_index": 0,
                "token_count": 10,
                "input_token_ids_sha256": hashlib.sha256(
                    hashes[source_id].encode()
                ).hexdigest(),
                "relative_l2_error": 0.5 + 0.1 * index,
            }
            for index, source_id in enumerate(source_ids)
        ]
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    if legacy:
        (path / "activation_metadata.json").write_text(
            json.dumps(
                {
                    "items": [
                        {"source_id": source_id, "split": "test"}
                        for source_id in source_ids[:2]
                    ]
                }
            ),
            encoding="utf-8",
        )
    return path


def test_legacy_run_uses_aggregate_metrics_and_keeps_missing_provenance_unknown(
    tmp_path: Path,
) -> None:
    legacy = _write_run(
        tmp_path / "t4-nf4",
        precision="nf4",
        source_ids=["article-a", "article-b"],
        legacy=True,
    )

    run = load_run(legacy)

    assert (
        run.provenance["teacher_precision"] == "nf4 (inferred from legacy quantization)"
    )
    assert run.provenance["backend"] == "not recorded"
    assert run.test_sequence_hashes is None


def test_comparison_marks_matching_and_different_test_sequences(tmp_path: Path) -> None:
    reference = _write_run(
        tmp_path / "nf4",
        precision="nf4",
        source_ids=["article-a", "article-b"],
    )
    matched = _write_run(
        tmp_path / "bf16-matched",
        precision="bf16",
        source_ids=["article-a", "article-b"],
        backend="rocm",
        max_length=256,
    )
    larger = _write_run(
        tmp_path / "bf16-large",
        precision="bf16",
        source_ids=["article-a", "article-b", "article-c"],
        backend="rocm",
    )
    original_metadata = (reference / "metadata.json").read_bytes()
    output_dir = tmp_path / "comparison"

    report = write_comparison_report(
        [reference, matched, larger],
        output_dir,
        labels=["T4 NF4", "Matched BF16", "Larger BF16"],
    )

    assert report["comparisons"][0]["input"]["status"] == "verified same"
    assert report["comparisons"][1]["input"]["status"] == "verified different"
    assert (
        report["comparisons"][1]["input"]["document_membership_status"]
        == "verified different"
    )
    assert report["comparisons"][1]["input"]["document_text_status"] == "verified same"
    assert any(
        difference.startswith("teacher_precision:")
        for difference in report["comparisons"][0]["setup_differences"]
    )
    assert "max_length: 512 → 256" in report["comparisons"][0]["setup_differences"]
    assert (reference / "metadata.json").read_bytes() == original_metadata
    assert all(
        (output_dir / name).is_file()
        for name in ("run_comparison.png", "run_comparison.md", "run_comparison.json")
    )
    markdown = (output_dir / "run_comparison.md").read_text(encoding="utf-8")
    assert "Runs with verified different token sequences are descriptive" in markdown
    assert "Positive next-token loss delta means" in markdown
    saved = json.loads((output_dir / "run_comparison.json").read_text())
    assert saved["comparisons"][0]["input"]["status"] == "verified same"


def test_legacy_sample_is_not_used_as_full_sequence_evidence(tmp_path: Path) -> None:
    legacy = _write_run(
        tmp_path / "legacy",
        precision="nf4",
        source_ids=["article-a", "article-b"],
        legacy=True,
    )
    current = _write_run(
        tmp_path / "current",
        precision="bf16",
        source_ids=["article-a", "article-b"],
    )

    report = write_comparison_report([legacy, current], tmp_path / "comparison")

    assert report["comparisons"][0]["input"]["status"] == "unknown"
    assert (
        "fingerprints are unavailable"
        in report["comparisons"][0]["input"]["token_sequence_reason"]
    )
    assert (
        report["comparisons"][0]["input"]["document_membership_status"]
        == "verified same"
    )


def test_changed_text_does_not_claim_different_token_sequences(
    tmp_path: Path,
) -> None:
    reference = _write_run(
        tmp_path / "reference",
        precision="nf4",
        source_ids=["article-a"],
        legacy=True,
        text_hashes={
            "article-a": hashlib.sha256(b"same text").hexdigest(),
        },
    )
    changed = _write_run(
        tmp_path / "changed-text",
        precision="bf16",
        source_ids=["article-a"],
        legacy=True,
        text_hashes={
            "article-a": hashlib.sha256(b"different text").hexdigest(),
        },
    )

    report = write_comparison_report([reference, changed], tmp_path / "comparison")

    comparison = report["comparisons"][0]["input"]
    assert comparison["token_sequence_status"] == "unknown"
    assert comparison["document_membership_status"] == "verified same"
    assert comparison["document_text_status"] == "verified different"
    assert (
        "shared source-document text hashes differ"
        in comparison["document_text_reason"]
    )


def test_matching_documents_with_changed_token_sequences_are_verified_different(
    tmp_path: Path,
) -> None:
    reference = _write_run(
        tmp_path / "reference",
        precision="nf4",
        source_ids=["article-a"],
    )
    changed = _write_run(
        tmp_path / "changed-tokens",
        precision="bf16",
        source_ids=["article-a"],
        sequence_hashes={"article-a": "different-token-sequence"},
    )

    report = write_comparison_report([reference, changed], tmp_path / "comparison")

    assert report["comparisons"][0]["input"]["status"] == "verified different"
    assert (
        "token-sequence hashes differ"
        in report["comparisons"][0]["input"]["token_sequence_reason"]
    )
    assert (
        report["comparisons"][0]["input"]["document_membership_status"]
        == "verified same"
    )


def test_sequence_fingerprints_need_an_independent_expected_count(
    tmp_path: Path,
) -> None:
    recorded = _write_run(
        tmp_path / "recorded",
        precision="bf16",
        source_ids=["article-a"],
    )
    missing_count = _write_run(
        tmp_path / "missing-count",
        precision="bf16",
        source_ids=["article-a"],
    )
    metadata_path = missing_count / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    del metadata["splits"]["test"]["sequence_count"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = write_comparison_report([recorded, missing_count], tmp_path / "comparison")

    comparison = report["comparisons"][0]["input"]
    assert comparison["token_sequence_status"] == "unknown"
    assert load_run(missing_count).test_sequence_hashes is None


def test_optimizer_parameter_group_changes_are_reported(tmp_path: Path) -> None:
    reference = _write_run(
        tmp_path / "reference",
        precision="bf16",
        source_ids=["article-a"],
        group_learning_rate=1e-3,
    )
    changed_rate = _write_run(
        tmp_path / "changed-rate",
        precision="bf16",
        source_ids=["article-a"],
        group_learning_rate=2e-3,
    )

    report = write_comparison_report([reference, changed_rate], tmp_path / "comparison")

    differences = report["comparisons"][0]["setup_differences"]
    assert any(
        difference.startswith("optimizer_parameter_groups:")
        and "lr=0.001" in difference
        and "lr=0.002" in difference
        for difference in differences
    )


@pytest.mark.parametrize(
    ("remove_field", "expected_path"),
    [
        ("replacement.next_token_loss_delta", "replacement.next_token_loss_delta"),
        ("activation.test.hyena.mse", "activation.test.hyena.mse"),
    ],
)
def test_incomplete_metric_fields_fail_with_a_path_specific_error(
    tmp_path: Path, remove_field: str, expected_path: str
) -> None:
    run = _write_run(
        tmp_path / "incomplete",
        precision="bf16",
        source_ids=["article-a"],
    )
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    current = metrics
    keys = remove_field.split(".")
    for key in keys[:-1]:
        current = current[key]
    del current[keys[-1]]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_path):
        load_run(run)


def test_comparison_output_cannot_be_written_inside_a_run(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        precision="nf4",
        source_ids=["article-a"],
    )
    other = _write_run(
        tmp_path / "other",
        precision="bf16",
        source_ids=["article-a"],
    )

    with pytest.raises(ValueError, match="outside the input run directory"):
        write_comparison_report([run, other], run / "comparison")


def test_comparison_rejects_duplicate_runs(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path / "run",
        precision="nf4",
        source_ids=["article-a"],
    )

    with pytest.raises(ValueError, match="only appear once"):
        write_comparison_report([run, run], tmp_path / "comparison")


def test_compare_cli_writes_report_from_saved_run_paths(tmp_path: Path, capsys) -> None:
    nf4 = _write_run(
        tmp_path / "nf4",
        precision="nf4",
        source_ids=["article-a"],
    )
    bf16 = _write_run(
        tmp_path / "bf16",
        precision="bf16",
        source_ids=["article-a"],
    )
    output = tmp_path / "comparison"

    result = main(
        [
            str(nf4),
            str(bf16),
            "--labels",
            "NF4",
            "BF16",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert "Comparison report:" in capsys.readouterr().out
    assert (output / "run_comparison.png").is_file()


def test_comparison_import_does_not_load_torch() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hyena_distill.compare; print('torch' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == "False"
