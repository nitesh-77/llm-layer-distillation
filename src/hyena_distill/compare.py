from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunSummary:
    path: Path
    label: str
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    test_source_ids: frozenset[str] | None
    test_document_hashes: dict[str, str] | None
    test_sequence_hashes: dict[tuple[str, int], str] | None
    test_sequence_count: int | None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read run metadata at {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _test_source_ids(metadata: dict[str, Any]) -> frozenset[str] | None:
    split_ids = _mapping(metadata.get("document_splits")).get("test")
    if split_ids is None:
        split_ids = _mapping(_mapping(metadata.get("splits")).get("test")).get(
            "source_ids"
        )
    if not isinstance(split_ids, list) or not all(
        isinstance(source_id, (str, int)) for source_id in split_ids
    ):
        return None
    return frozenset(str(source_id) for source_id in split_ids)


def _document_hashes(
    metadata: dict[str, Any], source_ids: frozenset[str] | None
) -> dict[str, str] | None:
    if source_ids is None:
        return None
    source_documents = _mapping(metadata.get("source_documents"))
    if not source_documents:
        return None
    hashes = {}
    for source_id in source_ids:
        source = source_documents.get(source_id)
        text_hash = source.get("text_sha256") if isinstance(source, dict) else None
        if not _is_sha256(text_hash):
            return None
        hashes[source_id] = text_hash
    return hashes


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _sequence_hashes(
    metrics: dict[str, Any],
    expected_count: int | None,
    source_ids: frozenset[str] | None,
) -> dict[tuple[str, int], str] | None:
    examples = metrics.get("activation_test_examples")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count <= 0
        or not isinstance(examples, list)
        or not examples
    ):
        return None
    if len(examples) != expected_count:
        return None
    hashes: dict[tuple[str, int], str] = {}
    for example in examples:
        if not isinstance(example, dict):
            return None
        source_id = example.get("source_id")
        chunk_index = example.get("chunk_index")
        token_hash = example.get("input_token_ids_sha256")
        if (
            not isinstance(source_id, str)
            or not isinstance(chunk_index, int)
            or isinstance(chunk_index, bool)
            or chunk_index < 0
            or not _is_sha256(token_hash)
            or (source_ids is not None and source_id not in source_ids)
        ):
            return None
        key = (source_id, chunk_index)
        if key in hashes:
            return None
        hashes[key] = token_hash
    return hashes


def _optimizer_groups(settings: dict[str, Any]) -> list[dict[str, Any]] | None:
    groups = settings.get("optimizer_parameter_groups")
    if not isinstance(groups, list) or not groups:
        return None
    normalized = []
    for group in groups:
        if not isinstance(group, dict):
            return None
        learning_rate = group.get("learning_rate")
        weight_decay = group.get("weight_decay")
        parameter_count = group.get("parameter_count")
        parameter_names = group.get("parameter_names")
        numeric_values = (learning_rate, weight_decay, parameter_count)
        if (
            any(not _is_finite_number(value) for value in numeric_values)
            or parameter_count < 0
            or not isinstance(parameter_names, list)
            or not all(isinstance(name, str) for name in parameter_names)
        ):
            return None
        normalized.append(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_count": parameter_count,
                "parameter_names": sorted(parameter_names),
            }
        )
    return sorted(
        normalized,
        key=lambda group: (
            group["learning_rate"],
            group["weight_decay"],
            group["parameter_names"],
        ),
    )


def _required_metric_number(
    run_dir: Path,
    metrics: dict[str, Any],
    metric_path: tuple[str, ...],
    *,
    nonnegative: bool = False,
) -> None:
    value: Any = metrics
    for key in metric_path:
        value = value.get(key) if isinstance(value, dict) else None
    if not _is_finite_number(value) or (nonnegative and value < 0):
        name = ".".join(metric_path)
        requirement = "finite nonnegative number" if nonnegative else "finite number"
        raise ValueError(
            f"{run_dir / 'metrics.json'} is missing or has an invalid {name}; "
            f"expected a {requirement}"
        )


def _precision(model: dict[str, Any], config: dict[str, Any]) -> str:
    precision = _coalesce(
        _first(model, "teacher_precision"), _first(config, "teacher_precision")
    )
    if precision is not None:
        return str(precision)
    quantization = str(model.get("quantization", "")).lower()
    if "nf4" in quantization:
        return "nf4 (inferred from legacy quantization)"
    if "bf16" in quantization or "bfloat16" in quantization:
        return "bf16 (inferred from legacy quantization)"
    return "not recorded"


def load_run(run_dir: Path, *, label: str | None = None) -> RunSummary:
    run_dir = run_dir.expanduser()
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    if not metadata_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"{run_dir} must contain both metadata.json and metrics.json")
    metadata = _read_json(metadata_path)
    metrics = _read_json(metrics_path)
    activation_summary = _mapping(metrics.get("activation"))
    activation = _mapping(activation_summary.get("test"))
    if any(
        not isinstance(activation.get(key), dict)
        for key in ("hyena", "zero", "training_mean")
    ):
        raise TypeError(f"{run_dir} is missing aggregate held-out activation metrics")
    replacement = _mapping(metrics.get("replacement"))
    if not replacement:
        raise TypeError(f"{run_dir} is missing replacement loss metrics")
    for baseline in ("hyena", "zero", "training_mean"):
        _required_metric_number(
            run_dir,
            metrics,
            ("activation", "test", baseline, "mse"),
            nonnegative=True,
        )
    _required_metric_number(
        run_dir,
        metrics,
        ("replacement", "next_token_loss_delta"),
    )

    model = _mapping(metadata.get("model"))
    config = _mapping(metadata.get("config"))
    settings = _mapping(metadata.get("settings"))
    dataset = _mapping(metadata.get("dataset"))
    target = _mapping(metadata.get("target"))
    runtime = _mapping(metadata.get("runtime"))
    training = _mapping(metadata.get("training"))
    export = _mapping(metadata.get("activation_export"))
    split_summary = _mapping(_mapping(metadata.get("splits")).get("test"))
    source_ids = _test_source_ids(metadata)
    sequence_count = split_summary.get("sequence_count")
    if (
        not isinstance(sequence_count, int)
        or isinstance(sequence_count, bool)
        or sequence_count <= 0
    ):
        sequence_count = None
    document_hashes = _document_hashes(metadata, source_ids)

    provenance = {
        "model_id": _coalesce(
            _first(model, "id"), _first(config, "model_id"), "not recorded"
        ),
        "model_revision": _coalesce(
            _first(model, "revision"),
            _first(config, "model_revision"),
            "not recorded",
        ),
        "tokenizer_revision": _coalesce(
            _first(model, "tokenizer_revision"), "not recorded"
        ),
        "teacher_precision": _precision(model, config),
        "backend": _first(runtime, "backend") or "not recorded",
        "gpu": _first(runtime, "gpu") or "not recorded",
        "dataset_id": _coalesce(
            _first(dataset, "dataset_id"), _first(config, "dataset_id"), "not recorded"
        ),
        "dataset_config": _coalesce(
            _first(dataset, "dataset_config"),
            _first(config, "dataset_config"),
            "not recorded",
        ),
        "dataset_revision": _coalesce(
            _first(dataset, "revision"),
            _first(config, "dataset_revision"),
            "not recorded",
        ),
        "target_layer": target.get(
            "layer_index", config.get("layer_index", "not recorded")
        ),
        "target_module": target.get("module", "not recorded"),
        "target_input": target.get("input", "not recorded"),
        "target_output": target.get("output", "not recorded"),
        "max_length": _coalesce(
            _first(settings, "max_length"),
            _first(config, "max_length"),
            "not recorded",
        ),
        "max_chunks_per_document": _coalesce(
            _first(settings, "max_chunks_per_document"),
            _first(config, "max_chunks_per_document"),
            "not recorded",
        ),
        "max_documents": _coalesce(_first(config, "max_documents"), "not recorded"),
        "seed": _coalesce(
            _first(settings, "seed"), _first(config, "seed"), "not recorded"
        ),
        "batch_size": _coalesce(
            _first(settings, "batch_size"),
            _first(config, "batch_size"),
            "not recorded",
        ),
        "configured_steps": _coalesce(
            _first(training, "configured_steps"),
            _first(config, "train_steps"),
            "not recorded",
        ),
        "selected_step": _coalesce(_first(training, "selected_step"), "not recorded"),
        "learning_rate": _coalesce(
            _first(settings, "learning_rate"),
            _first(config, "learning_rate"),
            "not recorded",
        ),
        "weight_decay": _coalesce(
            _first(settings, "weight_decay"),
            _first(config, "weight_decay"),
            "not recorded",
        ),
        "optimizer": _coalesce(_first(settings, "optimizer"), "not recorded"),
        "order": _coalesce(
            _first(settings, "order"), _first(config, "order"), "not recorded"
        ),
        "filter_order": _coalesce(
            _first(settings, "filter_order"),
            _first(config, "filter_order"),
            "not recorded",
        ),
        "activation_export_mode": _first(export, "mode") or "not recorded",
        "activation_export_bytes": _first(export, "bytes_written")
        or _first(export, "sample_bytes_written")
        or "not recorded",
        "test_document_count": len(source_ids)
        if source_ids is not None
        else split_summary.get("document_count", "not recorded"),
        "test_sequence_count": sequence_count or "not recorded",
        "test_activation_tokens": activation["hyena"].get(
            "token_count", "not recorded"
        ),
        "predicted_token_count": replacement.get(
            "predicted_token_count", "not recorded"
        ),
        "test_ids": sorted(source_ids) if source_ids is not None else None,
        "test_document_hashes": document_hashes,
        "optimizer_parameter_groups": _optimizer_groups(settings),
        "test_sequence_hashes": None,
    }
    sequence_hashes = _sequence_hashes(metrics, sequence_count, source_ids)
    provenance["test_sequence_hashes"] = (
        [
            {
                "source_id": source_id,
                "chunk_index": chunk_index,
                "sha256": token_hash,
            }
            for (source_id, chunk_index), token_hash in sorted(sequence_hashes.items())
        ]
        if sequence_hashes is not None
        else None
    )
    return RunSummary(
        path=run_dir,
        label=label or run_dir.name,
        metadata=metadata,
        metrics=metrics,
        provenance=provenance,
        test_source_ids=source_ids,
        test_document_hashes=document_hashes,
        test_sequence_hashes=sequence_hashes,
        test_sequence_count=sequence_count,
    )


def _input_comparison(reference: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    reference_sequences = reference.test_sequence_hashes
    candidate_sequences = candidate.test_sequence_hashes
    if reference_sequences is not None and candidate_sequences is not None:
        if reference_sequences == candidate_sequences:
            token_status = "verified same"
            token_reason = "complete test token-sequence hashes match"
        else:
            token_status = "verified different"
            token_reason = "complete test token-sequence hashes differ"
    else:
        token_status = "unknown"
        token_reason = (
            "complete test token-sequence fingerprints are unavailable; "
            "document evidence cannot establish exact token identity"
        )

    if reference.test_source_ids is None or candidate.test_source_ids is None:
        membership_status = "unknown"
    elif reference.test_source_ids == candidate.test_source_ids:
        membership_status = "verified same"
    else:
        membership_status = "verified different"

    reference_text_hashes = reference.test_document_hashes
    candidate_text_hashes = candidate.test_document_hashes
    if reference_text_hashes is None or candidate_text_hashes is None:
        text_status = "unknown"
        text_reason = "complete source-document text hashes are unavailable"
    else:
        shared_source_ids = reference_text_hashes.keys() & candidate_text_hashes.keys()
        if not shared_source_ids:
            text_status = "unknown"
            text_reason = "the runs have no shared source documents to compare"
        elif any(
            reference_text_hashes[source_id] != candidate_text_hashes[source_id]
            for source_id in shared_source_ids
        ):
            text_status = "verified different"
            text_reason = "one or more shared source-document text hashes differ"
        else:
            text_status = "verified same"
            text_reason = (
                f"text hashes match for all {len(shared_source_ids)} shared documents"
            )

    if reference.test_source_ids is None or candidate.test_source_ids is None:
        overlap = "unknown"
    else:
        shared = len(reference.test_source_ids & candidate.test_source_ids)
        overlap = (
            f"{shared}/{len(reference.test_source_ids)} reference docs"
            f" ({len(candidate.test_source_ids)} candidate docs)"
        )
    return {
        "reference": reference.label,
        "candidate": candidate.label,
        "status": token_status,
        "token_sequence_status": token_status,
        "token_sequence_reason": token_reason,
        "document_membership_status": membership_status,
        "document_text_status": text_status,
        "document_text_reason": text_reason,
        "source_document_overlap": overlap,
    }


_SETUP_FIELDS = (
    "model_id",
    "model_revision",
    "tokenizer_revision",
    "dataset_id",
    "dataset_config",
    "dataset_revision",
    "target_layer",
    "target_module",
    "target_input",
    "target_output",
    "max_length",
    "max_chunks_per_document",
    "max_documents",
    "seed",
    "batch_size",
    "configured_steps",
    "backend",
    "gpu",
    "learning_rate",
    "weight_decay",
    "optimizer",
    "order",
    "filter_order",
    "activation_export_mode",
    "activation_export_bytes",
)


def _setup_differences(reference: RunSummary, candidate: RunSummary) -> list[str]:
    differences = []
    for field in _SETUP_FIELDS:
        reference_value = reference.provenance[field]
        candidate_value = candidate.provenance[field]
        if "not recorded" in (reference_value, candidate_value):
            differences.append(f"{field}: not fully recorded")
        elif reference_value != candidate_value:
            differences.append(f"{field}: {reference_value!s} → {candidate_value!s}")
    reference_groups = reference.provenance["optimizer_parameter_groups"]
    candidate_groups = candidate.provenance["optimizer_parameter_groups"]
    if reference_groups is None or candidate_groups is None:
        differences.append("optimizer_parameter_groups: not fully recorded")
    elif reference_groups != candidate_groups:
        differences.append(
            "optimizer_parameter_groups: "
            f"{_describe_optimizer_groups(reference_groups)} → "
            f"{_describe_optimizer_groups(candidate_groups)}"
        )
    if (
        reference.provenance["teacher_precision"]
        != candidate.provenance["teacher_precision"]
    ):
        differences.append(
            "teacher_precision: "
            f"{reference.provenance['teacher_precision']} → "
            f"{candidate.provenance['teacher_precision']}"
        )
    return differences


def _describe_optimizer_groups(groups: list[dict[str, Any]]) -> str:
    summaries = []
    for group in groups:
        names_json = json.dumps(group["parameter_names"], separators=(",", ":"))
        names_hash = hashlib.sha256(names_json.encode("utf-8")).hexdigest()[:10]
        summaries.append(
            "(lr={:g}, wd={:g}, parameters={}, names={})".format(
                group["learning_rate"],
                group["weight_decay"],
                group["parameter_count"],
                names_hash,
            )
        )
    return "[" + ", ".join(summaries) + "]"


def _positive_scale(axis: Any, values: list[float]) -> str:
    if all(value > 0 for value in values):
        axis.set_yscale("log")
        return "log scale"
    axis.set_yscale("symlog", linthresh=1e-8)
    return "symmetric log scale (includes zero)"


def _plot_comparison(runs: list[RunSummary], path: Path) -> None:
    figure, (mse_axis, delta_axis) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    run_positions = list(range(len(runs)))
    width = 0.22
    metrics = (
        ("hyena", "Hyena", "#4c78a8"),
        ("zero", "Zero output", "#f58518"),
        ("training_mean", "Training mean", "#54a24b"),
    )
    all_mse_values = []
    for index, (key, label, color) in enumerate(metrics):
        values = [float(run.metrics["activation"]["test"][key]["mse"]) for run in runs]
        all_mse_values.extend(values)
        offsets = [position + (index - 1) * width for position in run_positions]
        mse_axis.scatter(offsets, values, label=label, color=color, s=65, zorder=3)
    scale = _positive_scale(mse_axis, all_mse_values)
    mse_axis.set_xticks(run_positions, [run.label for run in runs], rotation=15)
    mse_axis.set_ylabel(f"Held-out activation MSE ({scale})")
    mse_axis.set_title("Activation prediction baselines")
    mse_axis.legend()
    mse_axis.grid(axis="y", alpha=0.25)

    deltas = [
        float(run.metrics["replacement"]["next_token_loss_delta"]) for run in runs
    ]
    colors = ["#e45756" if value > 0 else "#54a24b" for value in deltas]
    bars = delta_axis.bar(run_positions, deltas, color=colors, width=0.58)
    delta_axis.axhline(0, color="#333333", linewidth=1)
    delta_limit = max((max(abs(value) for value in deltas) * 1.3), 1e-8)
    delta_axis.set_ylim(-delta_limit, delta_limit)
    delta_axis.set_xticks(run_positions, [run.label for run in runs], rotation=15)
    delta_axis.set_ylabel("Change in next-token loss (Hyena − teacher)")
    delta_axis.set_title("Downstream replacement effect")
    delta_axis.text(
        0.5,
        0.97,
        "Positive means loss increased",
        ha="center",
        va="top",
        transform=delta_axis.transAxes,
    )
    delta_axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, deltas, strict=True):
        delta_axis.annotate(
            f"{value:+.3g}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4 if value >= 0 else -4),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _format(value: Any, precision: int = 5) -> str:
    if value is None or value == "not recorded":
        return "not recorded"
    if isinstance(value, (float, int)):
        return f"{value:.{precision}g}"
    return str(value)


def _markdown_report(runs: list[RunSummary], comparisons: list[dict[str, Any]]) -> str:
    lines = [
        "# Saved-run comparison",
        "",
        "This report compares measured run summaries. It does not claim a causal precision effect. Exact test token-sequence identity is `verified same` or `verified different` only when both runs have complete sequence fingerprints. Source-document membership and source text hashes are reported separately and cannot prove token identity.",
        "",
        "Positive next-token loss delta means the Hyena replacement increased loss. Activation-example counts and predicted-token counts have different denominators.",
        "",
        "## Run metrics",
        "",
        "| Run | Teacher precision | Test docs / sequences | Activation tokens | Hyena MSE | Zero MSE | Training-mean MSE | Relative L2 | Cosine | Teacher → Hyena loss | Δ loss | Predicted tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        activation = run.metrics["activation"]["test"]
        replacement = run.metrics["replacement"]
        provenance = run.provenance
        test_size = (
            f"{provenance['test_document_count']} / {provenance['test_sequence_count']}"
        )
        losses = (
            f"{_format(replacement.get('teacher_next_token_loss'))} → "
            f"{_format(replacement.get('hyena_next_token_loss'))}"
        )
        lines.append(
            "| {label} | {precision} | {test_size} | {tokens} | {mse} | {zero} | {mean} | {relative} | {cosine} | {losses} | {delta} | {predicted} |".format(
                label=run.label,
                precision=provenance["teacher_precision"],
                test_size=test_size,
                tokens=_format(provenance["test_activation_tokens"]),
                mse=_format(activation["hyena"].get("mse")),
                zero=_format(activation.get("zero", {}).get("mse")),
                mean=_format(activation.get("training_mean", {}).get("mse")),
                relative=_format(activation["hyena"].get("relative_l2_error")),
                cosine=_format(activation["hyena"].get("cosine_similarity")),
                losses=losses,
                delta=_format(replacement.get("next_token_loss_delta")),
                predicted=_format(provenance["predicted_token_count"]),
            )
        )

    reference = runs[0]
    lines.extend(
        [
            "",
            "## Input and setup comparison",
            "",
            f"Reference run: **{reference.label}**.",
            "",
            "| Run | Test token sequences vs reference | Test document membership | Shared source text hashes | Source-document overlap | Setup differences vs reference |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for comparison in comparisons:
        differences = "; ".join(comparison["setup_differences"]) or "none recorded"
        lines.append(
            f"| {comparison['candidate']} | {comparison['input']['token_sequence_status']}: "
            f"{comparison['input']['token_sequence_reason']} | "
            f"{comparison['input']['document_membership_status']} | "
            f"{comparison['input']['document_text_status']}: "
            f"{comparison['input']['document_text_reason']} | "
            f"{comparison['input']['source_document_overlap']} | {differences} |"
        )
    lines.extend(
        [
            "",
            "Only runs marked `verified same` under test token sequences have verified matching inputs. Runs with verified different token sequences are descriptive; `unknown` token inputs cannot support paired claims. Even verified matching token sequences do not isolate precision if model revisions, backend, target, or training settings also differ; inspect the setup differences above.",
            "",
            "## Recorded provenance",
            "",
            "| Run | Model revision | Tokenizer revision | Dataset revision | Target layer | Backend / GPU | Configured / selected steps | Activation export | Export bytes |",
            "| --- | --- | --- | --- | ---: | --- | ---: | --- | ---: |",
        ]
    )
    for run in runs:
        provenance = run.provenance
        lines.append(
            f"| {run.label} | {provenance['model_revision']} | "
            f"{provenance['tokenizer_revision']} | {provenance['dataset_revision']} | "
            f"{provenance['target_layer']} | {provenance['backend']} / {provenance['gpu']} | "
            f"{provenance['configured_steps']} / {provenance['selected_step']} | "
            f"{provenance['activation_export_mode']} | "
            f"{_format(provenance['activation_export_bytes'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _serializable_run(run: RunSummary) -> dict[str, Any]:
    activation = run.metrics["activation"]["test"]
    replacement = run.metrics["replacement"]
    return {
        "label": run.label,
        "path": str(run.path),
        "provenance": run.provenance,
        "metrics": {
            "activation_test": activation,
            "replacement": replacement,
        },
    }


def write_comparison_report(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two run directories to compare")
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError(
            "The number of --labels must match the number of run directories"
        )
    runs = [
        load_run(path, label=labels[index] if labels is not None else None)
        for index, path in enumerate(run_dirs)
    ]
    resolved_runs = [run.path.resolve() for run in runs]
    if len(set(resolved_runs)) != len(resolved_runs):
        raise ValueError("A run directory can only appear once in a comparison")
    if len({run.label for run in runs}) != len(runs):
        raise ValueError("Comparison labels must be unique")
    resolved_output = output_dir.expanduser().resolve()
    for run in runs:
        resolved_run = run.path.resolve()
        if resolved_output == resolved_run or resolved_output.is_relative_to(
            resolved_run
        ):
            raise ValueError(
                f"Comparison output must be outside the input run directory {run.path}"
            )

    reference = runs[0]
    comparisons = [
        {
            "input": _input_comparison(reference, run),
            "setup_differences": _setup_differences(reference, run),
        }
        for run in runs[1:]
    ]
    for comparison, run in zip(comparisons, runs[1:], strict=True):
        comparison["candidate"] = run.label

    resolved_output.mkdir(parents=True, exist_ok=True)
    png_path = resolved_output / "run_comparison.png"
    _plot_comparison(runs, png_path)
    markdown_path = resolved_output / "run_comparison.md"
    markdown_path.write_text(_markdown_report(runs, comparisons), encoding="utf-8")
    report = {
        "format_version": 1,
        "reference_run": reference.label,
        "runs": [_serializable_run(run) for run in runs],
        "comparisons": comparisons,
        "artifacts": [png_path.name, markdown_path.name],
    }
    (resolved_output / "run_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyena-distill-compare",
        description="Compare saved Hyena distillation runs from JSON summaries only.",
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories to compare")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Display labels in the same order as the run directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/experiment-comparison"),
        help="Comparison report directory, kept outside the run directories",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if len(args.runs) < 2:
        parser.error("provide at least two run directories")
    if args.labels is not None and len(args.labels) != len(args.runs):
        parser.error("the number of --labels must match the number of run directories")
    try:
        write_comparison_report(args.runs, args.output, labels=args.labels)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"Comparison report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
