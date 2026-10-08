from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional

from hyena_distill.hyena import HyenaOperator
from hyena_distill.teacher import (
    Teacher,
    capture_attention_outputs,
    evaluate_replacement,
    load_qwen_teacher,
    release_teacher,
    tokenize_sequences,
)
from hyena_distill.types import (
    ActivationExample,
    ActivationSplits,
    Document,
    DocumentSplits,
    SequenceSplits,
    TokenSequence,
)


@dataclass(frozen=True)
class DemoConfig:
    model_id: str = "Qwen/Qwen2.5-7B"
    model_revision: str = "main"
    layer_index: int = 2
    input_path: Path | None = None
    output_dir: Path = Path("runs/qwen2.5-7b-layer-2")
    max_length: int = 512
    max_documents: int = 18
    max_chunks_per_document: int = 1
    train_steps: int = 25
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 17
    order: int = 4
    filter_order: int = 64
    dataset_id: str = "wikimedia/wikipedia"
    dataset_config: str = "20231101.en"
    dataset_revision: str = "ad5752b"


@dataclass(frozen=True)
class DemoResult:
    output_dir: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class _TrainingRun:
    operator: HyenaOperator
    history: tuple[dict[str, float | int], ...]
    selected_step: int
    best_validation_mse: float
    optimizer_parameter_groups: tuple[dict[str, Any], ...]


TeacherFactory = Callable[[DemoConfig], Teacher]


def _validate_config(config: DemoConfig) -> None:
    if config.layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if config.max_length < 2 or config.max_length > 512:
        raise ValueError("max_length must be between 2 and 512 tokens")
    if config.max_documents < 6:
        raise ValueError(
            "max_documents must be at least 6 for document-disjoint splits"
        )
    if config.max_chunks_per_document < 1:
        raise ValueError("max_chunks_per_document must be positive")
    if config.train_steps < 1 or config.batch_size < 1:
        raise ValueError("train_steps and batch_size must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError(
            "learning_rate must be positive and weight_decay cannot be negative"
        )
    if config.order < 2 or config.filter_order < 1:
        raise ValueError("order must be at least 2 and filter_order must be positive")
    if config.input_path is not None and not config.input_path.is_file():
        raise FileNotFoundError(f"Input JSONL file does not exist: {config.input_path}")
    if config.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {config.output_dir}")


def _read_jsonl(
    path: Path, max_documents: int
) -> tuple[list[Document], dict[str, Any]]:
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    file_hash = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            file_hash.update(raw_line)
            if len(documents) >= max_documents:
                continue
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error
            source_id = str(row.get("source_id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not source_id or not text:
                raise ValueError(
                    f"Line {line_number} must have non-empty source_id and text"
                )
            if source_id in seen_ids:
                raise ValueError(f"Duplicate source_id {source_id!r} in {path}")
            normalized_text = re.sub(r"\s+", " ", text).casefold()
            if normalized_text in seen_text:
                raise ValueError(
                    f"Duplicate article text in {path} would leak across splits"
                )
            seen_ids.add(source_id)
            seen_text.add(normalized_text)
            documents.append(
                Document(
                    source_id=source_id,
                    title=str(row.get("title") or source_id),
                    url=str(row.get("url") or ""),
                    text=text,
                )
            )

    return documents, {
        "type": "local-jsonl",
        "dataset_id": "local-jsonl",
        "dataset_config": path.name,
        "path": path.name,
        "sha256": file_hash.hexdigest(),
        "revision": None,
        "license": None,
    }


def _load_wikipedia(config: DemoConfig) -> tuple[list[Document], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install the demo dependencies with `pip install -e '.[demo]'`."
        ) from error

    stream = load_dataset(
        config.dataset_id,
        config.dataset_config,
        split="train",
        streaming=True,
        revision=config.dataset_revision,
    )
    stream = stream.shuffle(seed=config.seed, buffer_size=1000)
    documents: list[Document] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for row in stream:
        source_id = str(row.get("id", "")).strip()
        text = str(row.get("text", "")).strip()
        if not source_id or not text or source_id in seen_ids:
            continue
        normalized_text = re.sub(r"\s+", " ", text).casefold()
        if normalized_text in seen_text:
            continue
        seen_ids.add(source_id)
        seen_text.add(normalized_text)
        title = str(row.get("title") or source_id)
        documents.append(
            Document(
                source_id=source_id,
                title=title,
                url=str(row.get("url") or ""),
                text=text,
            )
        )
        if len(documents) >= config.max_documents:
            break

    if len(documents) < 6:
        raise ValueError(
            f"The Wikipedia stream returned only {len(documents)} usable documents"
        )
    return documents, {
        "type": "huggingface-stream",
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "revision": config.dataset_revision,
        "license": ["CC-BY-SA-3.0", "GFDL"],
        "selection": "streaming shuffle before taking max_documents",
        "seed": config.seed,
    }


def _load_documents(config: DemoConfig) -> tuple[list[Document], dict[str, Any]]:
    if config.input_path is not None:
        documents, source = _read_jsonl(config.input_path, config.max_documents)
    else:
        documents, source = _load_wikipedia(config)
    if len(documents) < 6:
        raise ValueError(
            f"At least 6 unique source documents are required, found {len(documents)}"
        )
    return documents, source


def _split_documents(documents: Sequence[Document], seed: int) -> DocumentSplits:
    if len({document.source_id for document in documents}) != len(documents):
        raise ValueError("Source document identifiers must be unique")
    ordered = sorted(
        documents,
        key=lambda document: hashlib.sha256(
            f"{seed}:{document.source_id}".encode()
        ).hexdigest(),
    )
    test_count = max(1, math.ceil(len(ordered) * 0.15))
    validation_count = max(1, math.ceil(len(ordered) * 0.15))
    if len(ordered) - test_count - validation_count < 1:
        raise ValueError(
            "Document set is too small for train, validation, and test splits"
        )
    return DocumentSplits(
        train=tuple(ordered[test_count + validation_count :]),
        validation=tuple(ordered[test_count : test_count + validation_count]),
        test=tuple(ordered[:test_count]),
    )


def _document_split_map(splits: DocumentSplits) -> dict[str, tuple[Document, ...]]:
    return {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }


def _capture_splits(
    teacher: Teacher,
    document_splits: DocumentSplits,
    config: DemoConfig,
) -> tuple[SequenceSplits, ActivationSplits]:
    sequences_by_split = tokenize_sequences(
        teacher.tokenizer,
        _document_split_map(document_splits),
        max_length=config.max_length,
        max_chunks_per_document=config.max_chunks_per_document,
    )
    for split_name in ("train", "validation", "test"):
        if not sequences_by_split[split_name]:
            raise ValueError(
                f"The {split_name} split has no token sequences with at least two tokens"
            )
    sequence_splits = SequenceSplits(
        train=sequences_by_split["train"],
        validation=sequences_by_split["validation"],
        test=sequences_by_split["test"],
    )
    activations = ActivationSplits(
        train=capture_attention_outputs(
            teacher,
            sequence_splits.train,
            layer_index=config.layer_index,
            max_length=config.max_length,
        ),
        validation=capture_attention_outputs(
            teacher,
            sequence_splits.validation,
            layer_index=config.layer_index,
            max_length=config.max_length,
        ),
        test=capture_attention_outputs(
            teacher,
            sequence_splits.test,
            layer_index=config.layer_index,
            max_length=config.max_length,
        ),
    )
    return sequence_splits, activations


def _activation_mse(
    examples: Sequence[ActivationExample], operator: HyenaOperator, device: torch.device
) -> float:
    squared_error = torch.zeros((), dtype=torch.float64)
    element_count = 0
    operator.eval()
    with torch.inference_mode():
        for example in examples:
            inputs = example.attention_input.unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            target = example.attention_output.to(dtype=torch.float32)
            prediction = operator(inputs).squeeze(0).cpu()
            squared_error += (prediction.double() - target.double()).square().sum()
            element_count += target.numel()
    return float((squared_error / max(1, element_count)).item())


def _optimizer_parameter_groups(
    operator: HyenaOperator, config: DemoConfig
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[float, float], dict[str, Any]] = {}
    for name, parameter in operator.named_parameters():
        hints = getattr(parameter, "_optim", {})
        learning_rate = float(hints.get("lr", config.learning_rate))
        weight_decay = float(
            config.weight_decay
            if config.weight_decay
            else hints.get("weight_decay", 0.0)
        )
        key = (learning_rate, weight_decay)
        group = grouped.setdefault(
            key,
            {"params": [], "parameter_names": []},
        )
        group["params"].append(parameter)
        group["parameter_names"].append(name)

    parameter_groups: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for (learning_rate, weight_decay), group in grouped.items():
        parameters = group["params"]
        parameter_groups.append(
            {
                "params": parameters,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        )
        metadata.append(
            {
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "parameter_names": group["parameter_names"],
            }
        )
    return parameter_groups, tuple(metadata)


def _fit_hyena(
    train: Sequence[ActivationExample],
    validation: Sequence[ActivationExample],
    *,
    width: int,
    config: DemoConfig,
    device: torch.device,
) -> _TrainingRun:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    operator = HyenaOperator(
        d_model=width,
        l_max=config.max_length,
        order=config.order,
        filter_order=config.filter_order,
    ).to(device=device, dtype=torch.float32)
    optimizer_groups, optimizer_parameter_groups = _optimizer_parameter_groups(
        operator, config
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
    grouped: dict[int, list[ActivationExample]] = {}
    for example in train:
        grouped.setdefault(example.attention_input.shape[0], []).append(example)
    if not grouped:
        raise ValueError("Hyena training requires at least one train activation")

    rng = random.Random(config.seed)
    lengths = list(grouped)
    weights = [len(grouped[length]) for length in lengths]
    validation_every = max(1, config.train_steps // 10)
    best_validation_mse = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    history: list[dict[str, float | int]] = []

    for step in range(1, config.train_steps + 1):
        sequence_length = rng.choices(lengths, weights=weights, k=1)[0]
        candidates = grouped[sequence_length]
        batch_count = min(config.batch_size, len(candidates))
        batch = rng.sample(candidates, k=batch_count)
        inputs = torch.stack([example.attention_input for example in batch]).to(
            device=device, dtype=torch.float32
        )
        targets = torch.stack([example.attention_output for example in batch]).to(
            device=device, dtype=torch.float32
        )

        operator.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = operator(inputs)
        loss = functional.mse_loss(prediction, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Hyena training produced a non-finite loss at step {step}"
            )
        loss.backward()
        optimizer.step()
        row: dict[str, float | int] = {
            "step": step,
            "train_mse": float(loss.detach().item()),
        }

        if step % validation_every == 0 or step == config.train_steps:
            validation_mse = _activation_mse(validation, operator, device)
            row["validation_mse"] = validation_mse
            if validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_step = step
                best_state = {
                    name: value.detach().cpu().contiguous().clone()
                    for name, value in operator.state_dict().items()
                }
        history.append(row)

    if best_state is None:
        raise RuntimeError("Hyena training did not produce a validation checkpoint")
    operator.load_state_dict(best_state)
    operator.to(device="cpu", dtype=torch.float32)
    return _TrainingRun(
        operator=operator,
        history=tuple(history),
        selected_step=best_step,
        best_validation_mse=best_validation_mse,
        optimizer_parameter_groups=optimizer_parameter_groups,
    )


def _predict_examples(
    examples: Sequence[ActivationExample],
    operator: HyenaOperator,
    device: torch.device,
) -> dict[tuple[str, int], torch.Tensor]:
    predictions: dict[tuple[str, int], torch.Tensor] = {}
    operator.eval()
    with torch.inference_mode():
        for example in examples:
            key = (example.sequence.source_id, example.sequence.chunk_index)
            inputs = example.attention_input.unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            predictions[key] = (
                operator(inputs).squeeze(0).cpu().to(dtype=torch.float16).contiguous()
            )
    return predictions


def _metric_values(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, float | int | None]:
    prediction = prediction.float()
    target = target.float()
    error = prediction - target
    squared_error = error.double().square().sum()
    target_squared = target.double().square().sum()
    target_norm = float(target_squared.sqrt().item())
    relative_error = (
        None if target_norm == 0 else float(squared_error.sqrt().item() / target_norm)
    )
    token_target_norms = torch.linalg.vector_norm(target, dim=-1)
    token_prediction_norms = torch.linalg.vector_norm(prediction, dim=-1)
    valid_cosines = (token_target_norms > 1e-12) & (token_prediction_norms > 1e-12)
    if valid_cosines.any():
        cosine = functional.cosine_similarity(
            prediction[valid_cosines],
            target[valid_cosines],
            dim=-1,
        ).mean()
        cosine_similarity = float(cosine.item())
    else:
        cosine_similarity = None
    mse = float(error.double().square().mean().item())
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "relative_l2_error": relative_error,
        "cosine_similarity": cosine_similarity,
        "target_l2_norm": target_norm,
        "token_count": int(target.shape[0]),
        "element_count": int(target.numel()),
    }


def _evaluate_activations(
    train: Sequence[ActivationExample],
    test: Sequence[ActivationExample],
    operator: HyenaOperator,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, float | int | None]],
    dict[tuple[str, int], torch.Tensor],
    dict[tuple[str, int], float],
]:
    if not train or not test:
        raise ValueError(
            "Activation evaluation requires non-empty train and test splits"
        )
    predictions = _predict_examples(test, operator, device)
    train_targets = torch.cat(
        [example.attention_output.float() for example in train], dim=0
    )
    test_targets = torch.cat(
        [example.attention_output.float() for example in test], dim=0
    )
    test_predictions = torch.cat(
        [
            predictions[
                (example.sequence.source_id, example.sequence.chunk_index)
            ].float()
            for example in test
        ],
        dim=0,
    )
    train_mean = train_targets.mean(dim=0)
    metrics = {
        "hyena": _metric_values(test_predictions, test_targets),
        "zero": _metric_values(torch.zeros_like(test_targets), test_targets),
        "training_mean": _metric_values(
            train_mean.unsqueeze(0).expand_as(test_targets), test_targets
        ),
    }
    per_example_error: dict[tuple[str, int], float] = {}
    for example in test:
        key = (example.sequence.source_id, example.sequence.chunk_index)
        relative_error = _metric_values(predictions[key], example.attention_output)[
            "relative_l2_error"
        ]
        per_example_error[key] = (
            math.inf if relative_error is None else float(relative_error)
        )
    return metrics, predictions, per_example_error


def _select_display_examples(
    activations: ActivationSplits,
    per_example_error: dict[tuple[str, int], float],
) -> tuple[ActivationExample, ActivationExample]:
    ordered = sorted(
        activations.test,
        key=lambda example: (
            per_example_error[
                (example.sequence.source_id, example.sequence.chunk_index)
            ],
            example.sequence.source_id,
            example.sequence.chunk_index,
        ),
    )
    if not ordered:
        raise ValueError("Cannot select comparison figures without test examples")
    return ordered[(len(ordered) - 1) // 2], ordered[-1]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _save_operator(
    operator: HyenaOperator, output_dir: Path, config: DemoConfig, width: int
) -> dict[str, Any]:
    weights = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in operator.state_dict().items()
    }
    save_file(weights, str(output_dir / "hyena.safetensors"))
    student_config = {
        "d_model": width,
        "l_max": config.max_length,
        "order": config.order,
        "filter_order": config.filter_order,
    }
    _write_json(output_dir / "hyena_config.json", student_config)
    return student_config


def _plot_loss(history: Sequence[dict[str, float | int]], path: Path) -> None:
    steps = [int(row["step"]) for row in history]
    train = [float(row["train_mse"]) for row in history]
    validation_steps = [int(row["step"]) for row in history if "validation_mse" in row]
    validation = [
        float(row["validation_mse"]) for row in history if "validation_mse" in row
    ]
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(steps, train, label="Train minibatch MSE", alpha=0.8)
    axis.plot(validation_steps, validation, marker="o", label="Validation MSE")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Mean squared error")
    axis.set_title("Hyena fit on captured Qwen attention outputs")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_baselines(
    metrics: dict[str, dict[str, float | int | None]], path: Path
) -> None:
    names = ["Hyena", "Zero output", "Training mean"]
    keys = ["hyena", "zero", "training_mean"]
    values = [float(metrics[key]["mse"]) for key in keys]
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = axis.bar(names, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_yscale("symlog", linthresh=1e-8)
    axis.set_ylabel("Held-out mean squared error (symmetric log scale)")
    axis.set_title("Attention-output prediction baselines")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            f"{value:.3g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
        )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_comparison(
    example: ActivationExample,
    prediction: torch.Tensor,
    error: float,
    label: str,
    path: Path,
) -> None:
    target = example.attention_output.float().numpy().T
    prediction_array = prediction.float().numpy().T
    difference = (example.attention_output.float() - prediction.float()).numpy().T
    output_limit = max(
        float(abs(target).max()), float(abs(prediction_array).max()), 1e-8
    )
    difference_limit = max(float(abs(difference).max()), 1e-8)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (
        (target, "Qwen attention output", output_limit, "RdBu_r"),
        (prediction_array, "Hyena output", output_limit, "RdBu_r"),
        (difference, "Qwen minus Hyena", difference_limit, "RdBu_r"),
    )
    for axis, (values, title, limit, palette) in zip(axes, panels, strict=True):
        image = axis.imshow(
            values,
            aspect="auto",
            interpolation="nearest",
            cmap=palette,
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel("Token position")
        axis.set_ylabel("Hidden channel")
        figure.colorbar(image, ax=axis, label="Activation value", shrink=0.82)
    sequence = example.sequence
    figure.suptitle(
        f"{label} held-out example, relative L2 error={error:.4f}\n"
        f"{sequence.title} ({sequence.source_id}), chunk {sequence.chunk_index}"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _activation_artifacts(
    splits: ActivationSplits,
    predictions: dict[tuple[str, int], torch.Tensor],
    display_examples: tuple[ActivationExample, ActivationExample],
    per_example_error: dict[tuple[str, int], float],
    operator: HyenaOperator,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    median, worst = display_examples
    selected: list[ActivationExample] = [splits.train[0], splits.validation[0], median]
    if (worst.sequence.source_id, worst.sequence.chunk_index) != (
        median.sequence.source_id,
        median.sequence.chunk_index,
    ):
        selected.append(worst)

    tensor_values: dict[str, torch.Tensor] = {}
    metadata_items: list[dict[str, Any]] = []
    operator.eval()
    with torch.inference_mode():
        for index, example in enumerate(selected):
            sequence = example.sequence
            key = (sequence.source_id, sequence.chunk_index)
            if key in predictions:
                prediction = predictions[key]
            else:
                inputs = example.attention_input.unsqueeze(0).to(
                    device=device, dtype=torch.float32
                )
                prediction = (
                    operator(inputs)
                    .squeeze(0)
                    .cpu()
                    .to(dtype=torch.float16)
                    .contiguous()
                )
            prefix = f"sample_{index:03d}"
            tensor_keys = {
                "token_ids_key": f"{prefix}_token_ids",
                "input_key": f"{prefix}_attention_input",
                "target_key": f"{prefix}_attention_output",
                "prediction_key": f"{prefix}_hyena_output",
            }
            tensor_values[tensor_keys["token_ids_key"]] = torch.tensor(
                sequence.token_ids, dtype=torch.long
            )
            tensor_values[tensor_keys["input_key"]] = example.attention_input.to(
                dtype=torch.float16
            ).contiguous()
            tensor_values[tensor_keys["target_key"]] = example.attention_output.to(
                dtype=torch.float16
            ).contiguous()
            tensor_values[tensor_keys["prediction_key"]] = prediction.to(
                dtype=torch.float16
            ).contiguous()
            metadata_items.append(
                {
                    **tensor_keys,
                    "source_id": sequence.source_id,
                    "title": sequence.title,
                    "url": sequence.url,
                    "text_sha256": sequence.text_sha256,
                    "split": sequence.split,
                    "chunk_index": sequence.chunk_index,
                    "token_length": len(sequence.token_ids),
                    "tensor_shape": list(example.attention_input.shape),
                    "storage_dtype": "float16",
                    "relative_l2_error": per_example_error.get(key),
                }
            )
    save_file(tensor_values, str(output_dir / "activations.safetensors"))
    return metadata_items


def _metric_verdict(metrics: dict[str, Any]) -> str:
    activation = metrics["activation"]["test"]
    hyena_mse = activation["hyena"]["mse"]
    zero_mse = activation["zero"]["mse"]
    mean_mse = activation["training_mean"]["mse"]
    beats_zero = hyena_mse < zero_mse
    beats_mean = hyena_mse < mean_mse
    if beats_zero and beats_mean:
        activation_result = (
            "Hyena beat both activation baselines on this held-out sample."
        )
    elif beats_zero:
        activation_result = "Hyena beat the zero-output baseline but not the training-mean baseline on this held-out sample."
    elif beats_mean:
        activation_result = "Hyena beat the training-mean baseline but not the zero-output baseline on this held-out sample."
    else:
        activation_result = (
            "Hyena did not beat either activation baseline on this held-out sample."
        )
    delta = metrics["replacement"]["next_token_loss_delta"]
    if delta < 0:
        replacement_result = "The substituted model had lower next-token loss on the same test sequences."
    elif delta > 0:
        replacement_result = "The substituted model had higher next-token loss on the same test sequences."
    else:
        replacement_result = "The substituted model had the same next-token loss on the same test sequences."
    return f"{activation_result} {replacement_result}"


def _write_readme(
    output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any]
) -> None:
    activation = metrics["activation"]["test"]
    replacement = metrics["replacement"]
    content = f"""# One-layer Qwen attention-to-Hyena demo

This run trained a Hyena operator to approximate the output of `self_attn` in Qwen decoder block {metadata["target"]["layer_index"]} (the {_ordinal(metadata["target"]["layer_ordinal"])} block, using zero-based indexing). The captured input is after the block's input normalization. The captured target is the attention output before the residual addition.

## Measured result

{_metric_verdict(metrics)}

| Held-out activation predictor | MSE | RMSE | Relative L2 error | Mean cosine similarity | Target L2 norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hyena | {activation["hyena"]["mse"]:.6g} | {activation["hyena"]["rmse"]:.6g} | {_format_metric(activation["hyena"]["relative_l2_error"])} | {_format_metric(activation["hyena"]["cosine_similarity"])} | {activation["hyena"]["target_l2_norm"]:.6g} |
| Zero output | {activation["zero"]["mse"]:.6g} | {activation["zero"]["rmse"]:.6g} | {_format_metric(activation["zero"]["relative_l2_error"])} | {_format_metric(activation["zero"]["cosine_similarity"])} | {activation["zero"]["target_l2_norm"]:.6g} |
| Training mean | {activation["training_mean"]["mse"]:.6g} | {activation["training_mean"]["rmse"]:.6g} | {_format_metric(activation["training_mean"]["relative_l2_error"])} | {_format_metric(activation["training_mean"]["cosine_similarity"])} | {activation["training_mean"]["target_l2_norm"]:.6g} |

MSE is the mean squared error across all held-out activation elements. Relative L2 error is `||prediction - target||₂ / ||target||₂`. Cosine similarity averages token-level values and excludes tokens with a zero prediction or target norm. The zero-output baseline's cosine is undefined.

Validation MSE selects the checkpoint. The run scores the final test split after selection and does not pass test captures to the optimizer or checkpoint selector.

The original teacher's held-out next-token loss was {replacement["teacher_next_token_loss"]:.6g}. The replaced model's loss was {replacement["hyena_next_token_loss"]:.6g}. The change was {replacement["next_token_loss_delta"]:+.6g} over {replacement["predicted_token_count"]} predicted tokens.

## Figures

![Training and validation loss](loss_curve.png)

![Held-out baseline comparison](baseline_metrics.png)

![Median-error held-out example](comparison_median.png)

![Highest-error held-out example](comparison_worst.png)

## Files

- `hyena.safetensors` and `hyena_config.json` contain the trained operator.
- `activations.safetensors` contains a small sample of token IDs, captured inputs, teacher outputs, and Hyena outputs. `activation_metadata.json` describes each tensor and its source article.
- `metrics.json` and `metadata.json` record the measured results, split membership, revisions, settings, and runtime.
- `loss_curve.png` shows minibatch training loss and validation loss.
- `baseline_metrics.png` compares held-out MSE with zero-output and training-mean predictors.
- `comparison_median.png` and `comparison_worst.png` show deterministic held-out examples.
- `model_card.md` and `dataset_card.md` are drafts for separate Hugging Face repositories.

## Limits

This run measures a small document sample and one forward-only replacement. It does not establish general Qwen quality, a full perplexity result, a speedup, or multi-layer replacement. The Qwen teacher weights are not saved in this directory.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _write_cards(
    output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any]
) -> None:
    activation = metrics["activation"]["test"]
    replacement = metrics["replacement"]
    dataset = metadata["dataset"]
    if dataset["type"] == "huggingface-stream":
        source_license = "The source dataset card lists CC-BY-SA-3.0 and GFDL. Review those terms before redistributing derived activations."
    else:
        source_license = (
            "The local input file has no upstream dataset license recorded in this run."
        )
    model_card = f"""---
library_name: pytorch
tags:
- hyena
- qwen2
- attention-distillation
---

# Hyena approximation of one Qwen attention submodule

This repository contains only a trained Hyena operator. The operator approximates the output of `self_attn` in Qwen decoder block {metadata["target"]["layer_index"]} (the {_ordinal(metadata["target"]["layer_ordinal"])} block) of `{metadata["model"]["id"]}`. The target input follows the block's input normalization. The target output precedes the residual addition.

The run used model revision `{metadata["model"]["revision"]}`, {metadata["model"]["quantization"]} loading, hidden width {metadata["model"]["hidden_size"]}, and a maximum sequence length of {metadata["settings"]["max_length"]} tokens. The run selected Hyena optimizer step {metadata["training"]["selected_step"]} using validation MSE.

On the untouched test split, Hyena MSE was {activation["hyena"]["mse"]:.6g}; the zero-output MSE was {activation["zero"]["mse"]:.6g}; the training-mean MSE was {activation["training_mean"]["mse"]:.6g}. The original Qwen next-token loss was {replacement["teacher_next_token_loss"]:.6g}, and the substituted loss was {replacement["hyena_next_token_loss"]:.6g}.

This checkpoint is not a Qwen model and does not include Qwen teacher weights. The result covers a small sample and one forward-only attention replacement. It does not establish general quality preservation or speed.

## Figures

![Held-out baseline comparison](baseline_metrics.png)

![Median-error held-out example](comparison_median.png)

![Highest-error held-out example](comparison_worst.png)

Load the operator with the project code:

```python
import json
from safetensors.torch import load_file
from hyena_distill.hyena import HyenaOperator

config = json.load(open("hyena_config.json"))
operator = HyenaOperator(**config)
operator.load_state_dict(load_file("hyena.safetensors"))
```
"""
    dataset_card = f"""---
task_categories:
- text-generation
language:
- en
---

# Captured Qwen attention activations

This artifact contains a compact sample of token IDs, the input to one Qwen `self_attn` module, its returned output, and the trained Hyena prediction. It does not contain raw article text or Qwen model weights.

The teacher was `{metadata["model"]["id"]}` at revision `{metadata["model"]["revision"]}`. The target was decoder block {metadata["target"]["layer_index"]} (the {_ordinal(metadata["target"]["layer_ordinal"])} block). Activations have shape `[tokens, {metadata["model"]["hidden_size"]}]` and are stored as float16. `activation_metadata.json` maps tensor names to source article IDs, URLs, text hashes, chunk indices, token lengths, and document splits.

The source was `{dataset["dataset_id"]}` with configuration `{dataset["dataset_config"]}` at revision `{dataset["revision"]}`. {source_license}

The test split had Hyena MSE {activation["hyena"]["mse"]:.6g}, zero-output MSE {activation["zero"]["mse"]:.6g}, and training-mean MSE {activation["training_mean"]["mse"]:.6g}. The corresponding replaced-model next-token loss delta was {replacement["next_token_loss_delta"]:+.6g}.

The run uses document-disjoint train, validation, and test splits. It is a small implementation demo, not a broad evaluation of Qwen quality or a claim that Hyena replaces attention in general.
"""
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    (output_dir / "dataset_card.md").write_text(dataset_card, encoding="utf-8")


def _write_metrics_plot(metrics: dict[str, Any], output_dir: Path) -> None:
    test_metrics = metrics["activation"]["test"]
    _plot_baselines(test_metrics, output_dir / "baseline_metrics.png")


def _json_config(config: DemoConfig) -> dict[str, Any]:
    value = asdict(config)
    value["input_path"] = None if config.input_path is None else config.input_path.name
    value["output_dir"] = config.output_dir.name
    return value


def _runtime_metadata() -> dict[str, Any]:
    packages = {}
    for package in (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "einops",
        "matplotlib",
        "safetensors",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
    }


def _split_summary(
    doc_splits: DocumentSplits, sequence_splits: SequenceSplits
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        documents = getattr(doc_splits, name)
        sequences: Sequence[TokenSequence] = getattr(sequence_splits, name)
        result[name] = {
            "document_count": len(documents),
            "sequence_count": len(sequences),
            "token_count": sum(len(sequence.token_ids) for sequence in sequences),
            "source_ids": [document.source_id for document in documents],
        }
    return result


def _write_run(
    config: DemoConfig,
    output_dir: Path,
    source: dict[str, Any],
    doc_splits: DocumentSplits,
    sequence_splits: SequenceSplits,
    activations: ActivationSplits,
    teacher_info: dict[str, Any],
    train_run: _TrainingRun,
    student_config: dict[str, Any],
    activation_metrics: dict[str, dict[str, float | int | None]],
    predictions: dict[tuple[str, int], torch.Tensor],
    per_example_error: dict[tuple[str, int], float],
    display_examples: tuple[ActivationExample, ActivationExample],
    replacement_metrics: dict[str, float | int],
    operator: HyenaOperator,
    device: torch.device,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _json_config(config),
        "model": teacher_info,
        "target": {
            "layer_index": config.layer_index,
            "layer_ordinal": config.layer_index + 1,
            "module": "model.layers[layer_index].self_attn",
            "input": "after input_layernorm",
            "output": "first self_attn return value before residual addition",
        },
        "dataset": source,
        "document_splits": {
            name: [document.source_id for document in getattr(doc_splits, name)]
            for name in ("train", "validation", "test")
        },
        "source_documents": {
            document.source_id: {
                "title": document.title,
                "url": document.url,
                "text_sha256": hashlib.sha256(
                    document.text.encode("utf-8")
                ).hexdigest(),
            }
            for document in doc_splits.train + doc_splits.validation + doc_splits.test
        },
        "splits": _split_summary(doc_splits, sequence_splits),
        "settings": {
            "max_length": config.max_length,
            "max_chunks_per_document": config.max_chunks_per_document,
            "seed": config.seed,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_parameter_groups": list(train_run.optimizer_parameter_groups),
            "order": config.order,
            "filter_order": config.filter_order,
        },
        "training": {
            "configured_steps": config.train_steps,
            "selected_step": train_run.selected_step,
            "best_validation_mse": train_run.best_validation_mse,
            "history": list(train_run.history),
        },
        "student": student_config,
        "runtime": _runtime_metadata(),
    }
    metrics = {
        "activation": {"test": activation_metrics},
        "replacement": replacement_metrics,
        "training_history": list(train_run.history),
        "definitions": {
            "mse": "Mean squared error across all activation elements in the held-out split.",
            "rmse": "Square root of held-out MSE.",
            "relative_l2_error": "L2 norm of prediction minus target divided by target L2 norm; undefined for a zero target norm.",
            "cosine_similarity": "Mean token-level cosine similarity over tokens whose prediction and target norms are non-zero; undefined when no such token exists.",
            "training_mean_baseline": "Per-channel mean of train-split attention outputs, computed without validation or test targets.",
            "next_token_loss": "Token-weighted shifted cross entropy with the Qwen cache disabled.",
        },
        "verdict": _metric_verdict(
            {
                "activation": {"test": activation_metrics},
                "replacement": replacement_metrics,
            }
        ),
    }

    _plot_loss(train_run.history, output_dir / "loss_curve.png")
    _write_metrics_plot(metrics, output_dir)
    median, worst = display_examples
    median_key = (median.sequence.source_id, median.sequence.chunk_index)
    worst_key = (worst.sequence.source_id, worst.sequence.chunk_index)
    _plot_comparison(
        median,
        predictions[median_key],
        per_example_error[median_key],
        "Median-error",
        output_dir / "comparison_median.png",
    )
    _plot_comparison(
        worst,
        predictions[worst_key],
        per_example_error[worst_key],
        "Highest-error",
        output_dir / "comparison_worst.png",
    )
    activation_items = _activation_artifacts(
        activations,
        predictions,
        display_examples,
        per_example_error,
        operator,
        device,
        output_dir,
    )
    activation_metadata = {
        "format_version": 1,
        "description": "Small sample of token IDs, Qwen self_attn inputs and outputs, and Hyena predictions.",
        "storage_dtype": "float16",
        "tensor_layout": "[tokens, hidden_size]",
        "items": activation_items,
    }
    _write_json(output_dir / "activation_metadata.json", activation_metadata)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "metadata.json", metadata)
    _write_readme(output_dir, metadata, metrics)
    _write_cards(output_dir, metadata, metrics)
    return metrics


def run_demo(
    config: DemoConfig,
    *,
    teacher_factory: TeacherFactory | None = None,
) -> DemoResult:
    _validate_config(config)
    output_dir = config.output_dir.expanduser().resolve()
    config = replace(config, output_dir=output_dir)
    factory = teacher_factory or load_qwen_teacher
    if teacher_factory is None and not torch.cuda.is_available():
        raise RuntimeError(
            "The real Qwen2.5-7B run requires a CUDA GPU. Use a Lightning T4 Studio."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    teacher: Teacher | None = None
    reloaded_teacher: Teacher | None = None
    try:
        documents, source = _load_documents(config)
        document_splits = _split_documents(documents, config.seed)
        teacher = factory(config)
        model_config = getattr(teacher.model, "config", None)
        if model_config is None or getattr(model_config, "model_type", None) != "qwen2":
            raise ValueError("Teacher must be a Qwen2 causal model")
        width = int(model_config.hidden_size)
        layer_count = len(getattr(getattr(teacher.model, "model", None), "layers", ()))
        if config.layer_index >= layer_count:
            raise ValueError(
                f"Layer index {config.layer_index} is outside the model's {layer_count} decoder layers"
            )

        sequence_splits, activations = _capture_splits(teacher, document_splits, config)
        teacher_info = {
            "id": config.model_id,
            "revision": teacher.model_revision,
            "tokenizer_revision": teacher.tokenizer_revision,
            "quantization": teacher.quantization,
            "hidden_size": width,
            "decoder_layers": layer_count,
        }
        resolved_config = replace(config, model_revision=teacher.model_revision)
        release_teacher(teacher)
        teacher = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_run = _fit_hyena(
            activations.train,
            activations.validation,
            width=width,
            config=config,
            device=device,
        )
        student_config = _save_operator(
            train_run.operator, temporary_dir, config, width
        )
        student = HyenaOperator(**student_config)
        student.load_state_dict(
            load_file(str(temporary_dir / "hyena.safetensors"), device="cpu")
        )
        student.to(device=device, dtype=torch.float32).eval()
        activation_metrics, test_predictions, per_example_error = _evaluate_activations(
            activations.train,
            activations.test,
            student,
            device,
        )
        display_examples = _select_display_examples(activations, per_example_error)

        reloaded_teacher = factory(resolved_config)
        if reloaded_teacher.model_revision != teacher_info["revision"]:
            raise RuntimeError(
                "Reloaded Qwen model revision does not match the captured teacher revision"
            )
        replacement_metrics = evaluate_replacement(
            reloaded_teacher,
            sequence_splits.test,
            layer_index=config.layer_index,
            operator=student,
        )
        metrics = _write_run(
            config,
            temporary_dir,
            source,
            document_splits,
            sequence_splits,
            activations,
            teacher_info,
            train_run,
            student_config,
            activation_metrics,
            test_predictions,
            per_example_error,
            display_examples,
            replacement_metrics,
            student,
            device,
        )
        release_teacher(reloaded_teacher)
        reloaded_teacher = None
        _write_json(
            temporary_dir / "complete.json",
            {
                "status": "complete",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifact_files": sorted(
                    path.name for path in temporary_dir.iterdir() if path.is_file()
                ),
            },
        )
        if output_dir.exists():
            raise FileExistsError(
                f"Output directory appeared during the run: {output_dir}"
            )
        os.rename(temporary_dir, output_dir)
        return DemoResult(output_dir=output_dir, metrics=metrics)
    except Exception:
        if teacher is not None:
            release_teacher(teacher)
        if reloaded_teacher is not None:
            release_teacher(reloaded_teacher)
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _format_metric(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.6g}"
