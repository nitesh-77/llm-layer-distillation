import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from hyena_distill import DemoConfig, run_demo
from hyena_distill import demo as demo_module
from hyena_distill.demo import (
    _evaluate_activations,
    _load_documents,
    _metric_values,
)
from hyena_distill.hyena import HyenaOperator
from hyena_distill.teacher import Teacher, temporary_attention_replacement
from hyena_distill.types import ActivationExample, TokenSequence


class ByteTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [byte + 2 for byte in text.encode("utf-8")]
        return ([1] + tokens) if add_special_tokens else tokens


@pytest.fixture
def tiny_qwen_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-qwen"
    config = Qwen2Config(
        vocab_size=258,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    Qwen2ForCausalLM(config).save_pretrained(model_dir, safe_serialization=True)
    return model_dir


def make_teacher_factory(model_dir: Path):
    def load(_config: DemoConfig) -> Teacher:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        return Teacher(
            model=model,
            tokenizer=ByteTokenizer(),
            model_revision="tiny-local-fixture",
            tokenizer_revision="tiny-local-fixture",
            quantization="none (CPU test fixture)",
        )

    return load


def write_documents(
    path: Path,
    *,
    test_suffix: str = "",
    changed_source_ids: set[str] | None = None,
) -> None:
    changed_source_ids = changed_source_ids or set()
    rows = [
        {
            "source_id": f"article-{index}",
            "title": f"Article {index}",
            "url": f"https://example.test/{index}",
            "text": (
                f"Article {index} describes a sequence model, a residual stream, "
                f"and local computation. "
                f"{test_suffix if f'article-{index}' in changed_source_ids else ''}"
            ),
        }
        for index in range(6)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def demo_config(input_path: Path, output_dir: Path) -> DemoConfig:
    return DemoConfig(
        model_id="Qwen/tiny-local-fixture",
        model_revision="tiny-local-fixture",
        input_path=input_path,
        output_dir=output_dir,
        max_length=32,
        max_documents=6,
        max_chunks_per_document=1,
        train_steps=2,
        batch_size=1,
        learning_rate=1e-3,
        filter_order=8,
        seed=17,
    )


def test_run_demo_captures_trains_saves_and_replaces_one_qwen_attention(
    tmp_path: Path,
    tiny_qwen_dir: Path,
) -> None:
    documents_path = tmp_path / "documents.jsonl"
    write_documents(documents_path)
    output_dir = tmp_path / "run"

    result = run_demo(
        demo_config(documents_path, output_dir),
        teacher_factory=make_teacher_factory(tiny_qwen_dir),
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    activation_metadata = json.loads(
        (output_dir / "activation_metadata.json").read_text()
    )
    activations = load_file(output_dir / "activations.safetensors")
    assert result.output_dir == output_dir
    assert metadata["target"]["layer_index"] == 2
    assert metadata["target"]["layer_ordinal"] == 3
    assert metadata["model"]["hidden_size"] == 16
    assert metadata["model"]["quantization"] == "none (CPU test fixture)"
    assert set(metadata["document_splits"]) == {"train", "validation", "test"}
    split_ids = [
        set(metadata["document_splits"][name])
        for name in ("train", "validation", "test")
    ]
    assert all(
        split_ids[i].isdisjoint(split_ids[j]) for i in range(3) for j in range(i + 1, 3)
    )
    assert metrics["activation"]["test"]["hyena"]["mse"] >= 0
    assert metrics["activation"]["test"]["zero"]["mse"] >= 0
    assert metrics["activation"]["test"]["training_mean"]["mse"] >= 0
    assert metrics["activation"]["test"]["zero"]["cosine_similarity"] is None
    assert metrics["replacement"]["teacher_next_token_loss"] >= 0
    assert metrics["replacement"]["hyena_next_token_loss"] >= 0
    assert metrics["replacement"]["mean_abs_next_token_logit_delta"] > 0
    assert len(metrics["training_history"]) == 2

    required_artifacts = (
        "hyena.safetensors",
        "hyena_config.json",
        "activations.safetensors",
        "activation_metadata.json",
        "loss_curve.png",
        "baseline_metrics.png",
        "comparison_median.png",
        "comparison_worst.png",
        "metrics.json",
        "metadata.json",
        "model_card.md",
        "dataset_card.md",
        "README.md",
        "complete.json",
    )
    assert all((output_dir / name).is_file() for name in required_artifacts)
    run_readme = (output_dir / "README.md").read_text()
    assert "![Training and validation loss](loss_curve.png)" in run_readme
    assert "![Held-out baseline comparison](baseline_metrics.png)" in run_readme
    assert "![Median-error held-out example](comparison_median.png)" in run_readme
    assert "![Highest-error held-out example](comparison_worst.png)" in run_readme

    artifact = activation_metadata["items"][0]
    saved_input = activations[artifact["input_key"]].float()
    saved_output = activations[artifact["target_key"]].float()
    saved_prediction = activations[artifact["prediction_key"]].float()
    model = AutoModelForCausalLM.from_pretrained(
        tiny_qwen_dir, local_files_only=True
    ).eval()
    ids = activations[artifact["token_ids_key"]].long().unsqueeze(0)
    block = model.model.layers[2]
    attention_input = []
    attention_output = []
    block_input = []
    block_output = []

    def hidden_states(args, kwargs):
        return kwargs.get("hidden_states", args[0] if args else None)

    def capture_attention_input(_module, args, kwargs):
        attention_input.append(hidden_states(args, kwargs).detach().squeeze(0))

    def capture_attention_output(_module, _args, _kwargs, output):
        attention_output.append(output[0].detach().squeeze(0))

    def capture_block_input(_module, args, kwargs):
        block_input.append(hidden_states(args, kwargs).detach().squeeze(0))

    def capture_block_output(_module, _args, _kwargs, output):
        block_output.append(output[0].detach().squeeze(0))

    handles = [
        block.register_forward_pre_hook(capture_block_input, with_kwargs=True),
        block.register_forward_hook(capture_block_output, with_kwargs=True),
        block.self_attn.register_forward_pre_hook(
            capture_attention_input, with_kwargs=True
        ),
        block.self_attn.register_forward_hook(
            capture_attention_output, with_kwargs=True
        ),
    ]
    try:
        with torch.inference_mode():
            model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    torch.testing.assert_close(
        saved_input, attention_input[0].float(), atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(
        saved_output, attention_output[0].float(), atol=2e-3, rtol=2e-3
    )
    assert not torch.allclose(saved_input, block_input[0].float())
    assert not torch.allclose(saved_output, block_output[0].float())

    student_config = json.loads((output_dir / "hyena_config.json").read_text())
    reloaded = HyenaOperator(**student_config)
    reloaded.load_state_dict(load_file(output_dir / "hyena.safetensors"))
    reloaded.eval()
    with torch.inference_mode():
        actual_prediction = reloaded(saved_input.unsqueeze(0)).squeeze(0)
    torch.testing.assert_close(
        saved_prediction, actual_prediction, atol=1e-4, rtol=1e-4
    )


def test_training_checkpoint_does_not_depend_on_final_test_text(
    tmp_path: Path,
    tiny_qwen_dir: Path,
) -> None:
    first_documents = tmp_path / "first.jsonl"
    second_documents = tmp_path / "second.jsonl"
    write_documents(first_documents)
    factory = make_teacher_factory(tiny_qwen_dir)

    first = run_demo(
        demo_config(first_documents, tmp_path / "first-run"),
        teacher_factory=factory,
    )
    split = json.loads((first.output_dir / "metadata.json").read_text())[
        "document_splits"
    ]
    write_documents(
        second_documents,
        test_suffix="replaced final test content",
        changed_source_ids=set(split["test"]),
    )
    second = run_demo(
        demo_config(second_documents, tmp_path / "second-run"),
        teacher_factory=factory,
    )

    assert (first.output_dir / "hyena.safetensors").read_bytes() == (
        second.output_dir / "hyena.safetensors"
    ).read_bytes()


def test_wikipedia_loader_streams_a_bounded_fixed_revision(
    monkeypatch, tmp_path: Path
) -> None:
    import datasets

    observed = {}

    class FakeStream:
        def shuffle(self, *, seed: int, buffer_size: int):
            observed["seed"] = seed
            observed["buffer_size"] = buffer_size
            return self

        def __iter__(self):
            for index in range(100):
                observed["yielded"] = index + 1
                yield {
                    "id": str(index),
                    "title": f"Article {index}",
                    "url": f"https://example.test/{index}",
                    "text": f"Wikipedia article number {index} has distinct text.",
                }

    def fake_load_dataset(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return FakeStream()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    config = DemoConfig(
        max_documents=6,
        dataset_revision="fixed-wiki-commit",
        output_dir=tmp_path / "unused-run",
    )

    documents, source = _load_documents(config)

    assert len(documents) == 6
    assert observed["yielded"] == 6
    assert observed["seed"] == config.seed
    assert observed["buffer_size"] == 1000
    assert observed["args"] == (config.dataset_id, config.dataset_config)
    assert observed["kwargs"]["streaming"] is True
    assert observed["kwargs"]["revision"] == "fixed-wiki-commit"
    assert source["revision"] == "fixed-wiki-commit"


def test_hyena_is_causal_and_has_finite_gradients() -> None:
    torch.manual_seed(7)
    layer = HyenaOperator(d_model=4, l_max=8, order=4, filter_order=8)
    layer.eval()
    original = torch.randn(1, 6, 4)
    changed_future = original.clone()
    changed_future[:, 4:] += 5

    with torch.no_grad():
        expected_prefix = layer(original)[:, :4]
        actual_prefix = layer(changed_future)[:, :4]
    torch.testing.assert_close(expected_prefix, actual_prefix)

    layer(original).square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_activation_metrics_match_hand_calculated_values() -> None:
    target = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    prediction = torch.tensor([[0.0, 4.0], [0.0, 0.0]])

    metrics = _metric_values(prediction, target)

    assert metrics["mse"] == pytest.approx(3.25)
    assert metrics["rmse"] == pytest.approx(3.25**0.5)
    assert metrics["relative_l2_error"] == pytest.approx((13 / 29) ** 0.5)
    assert metrics["cosine_similarity"] == pytest.approx(0.8)
    assert metrics["target_l2_norm"] == pytest.approx(29**0.5)


def test_held_out_metrics_aggregate_the_complete_test_split() -> None:
    train = (
        ActivationExample(
            TokenSequence("train-1", "Train 1", "", "a", 0, "train", (1, 2)),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        ),
        ActivationExample(
            TokenSequence("train-2", "Train 2", "", "b", 0, "train", (3, 4)),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[4.0, 0.0], [0.0, 4.0]]),
        ),
    )
    test = (
        ActivationExample(
            TokenSequence("test-1", "Test 1", "", "c", 0, "test", (5, 6)),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[2.0, 1.0], [1.0, 2.0]]),
        ),
        ActivationExample(
            TokenSequence("test-2", "Test 2", "", "d", 0, "test", (7, 8)),
            torch.tensor([[2.0, 1.0]]),
            torch.tensor([[4.0, 3.0]]),
        ),
    )

    metrics, predictions, _ = _evaluate_activations(
        train, test, torch.nn.Identity(), torch.device("cpu")
    )
    test_targets = torch.cat([example.attention_output for example in test])
    test_predictions = torch.cat(
        [predictions[(example.sequence.source_id, 0)] for example in test]
    )
    train_mean = torch.cat([example.attention_output for example in train]).mean(dim=0)

    assert metrics["hyena"]["mse"] == pytest.approx(
        (test_predictions - test_targets).square().mean().item()
    )
    assert metrics["zero"]["mse"] == pytest.approx(test_targets.square().mean().item())
    assert metrics["training_mean"]["mse"] == pytest.approx(
        (train_mean - test_targets).square().mean().item()
    )


def test_plots_label_the_compared_quantities(monkeypatch, tmp_path: Path) -> None:
    figures = []
    original_subplots = demo_module.plt.subplots

    def capture_subplots(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append((figure, axes))
        return figure, axes

    monkeypatch.setattr(demo_module.plt, "subplots", capture_subplots)
    demo_module._plot_loss(
        [
            {"step": 1, "train_mse": 2.0, "validation_mse": 1.5},
            {"step": 2, "train_mse": 1.0, "validation_mse": 0.75},
        ],
        tmp_path / "loss.png",
    )
    demo_module._plot_baselines(
        {
            "hyena": {"mse": 2.0},
            "zero": {"mse": 1.0},
            "training_mean": {"mse": 0.5},
        },
        tmp_path / "baselines.png",
    )
    sequence = TokenSequence("test", "Test", "", "hash", 0, "test", (1, 2, 3))
    example = ActivationExample(
        sequence,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]]),
    )
    demo_module._plot_comparison(
        example,
        example.attention_output + 0.1,
        0.2,
        "Median-error",
        tmp_path / "comparison.png",
    )

    loss_axes = figures[0][1]
    baseline_axes = figures[1][1]
    comparison_axes = figures[2][1]
    assert loss_axes.get_xlabel() == "Optimizer step"
    assert loss_axes.get_ylabel() == "Mean squared error"
    assert loss_axes.get_legend() is not None
    assert baseline_axes.get_ylabel().startswith("Held-out mean squared error")
    assert [label.get_text() for label in baseline_axes.get_xticklabels()] == [
        "Hyena",
        "Zero output",
        "Training mean",
    ]
    assert [axis.get_title() for axis in comparison_axes.flat] == [
        "Qwen attention output",
        "Hyena output",
        "Qwen minus Hyena",
    ]
    assert all(axis.get_xlabel() == "Token position" for axis in comparison_axes.flat)
    assert all(axis.get_ylabel() == "Hidden channel" for axis in comparison_axes.flat)
    assert all(
        (tmp_path / name).is_file()
        for name in ("loss.png", "baselines.png", "comparison.png")
    )
    for figure, _ in figures:
        demo_module.plt.close(figure)


def test_temporary_attention_replacement_restores_module_on_error(
    tiny_qwen_dir: Path,
) -> None:
    model = AutoModelForCausalLM.from_pretrained(tiny_qwen_dir, local_files_only=True)
    original = model.model.layers[2].self_attn
    student = HyenaOperator(d_model=16, l_max=32, order=4, filter_order=8)

    with (
        pytest.raises(RuntimeError, match="forced failure"),
        temporary_attention_replacement(model, 2, student),
    ):
        assert model.model.layers[2].self_attn is not original
        raise RuntimeError("forced failure")

    assert model.model.layers[2].self_attn is original
