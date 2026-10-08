from __future__ import annotations

import gc
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from hyena_distill.hyena import HyenaOperator
from hyena_distill.types import ActivationExample, TokenSequence


@dataclass
class Teacher:
    model: nn.Module
    tokenizer: Any
    model_revision: str
    tokenizer_revision: str
    quantization: str


def _layers(model: nn.Module) -> Sequence[nn.Module]:
    base_model = getattr(model, "model", None)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Expected a Qwen causal model with model.layers")
    return layers


def _model_input_device(model: nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    return embeddings.weight.device


def load_qwen_teacher(config: Any) -> Teacher:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The real Qwen2.5-7B run requires a CUDA GPU. Use a Lightning T4 Studio."
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        raise RuntimeError(
            "Install the demo dependencies with `pip install -e '.[demo]'`."
        ) from error

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    model_type = getattr(model.config, "model_type", "")
    if model_type != "qwen2":
        raise ValueError(
            f"Expected a Qwen2 causal model, got model_type={model_type!r}"
        )
    if _model_input_device(model).type != "cuda":
        raise RuntimeError("The 4-bit Qwen teacher must load onto the CUDA GPU")

    revision = getattr(model.config, "_commit_hash", None) or config.model_revision
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or revision
    return Teacher(
        model=model,
        tokenizer=tokenizer,
        model_revision=revision,
        tokenizer_revision=tokenizer_revision,
        quantization="4-bit NF4",
    )


def tokenize_sequences(
    tokenizer: Any,
    documents_by_split: dict[str, Sequence[Any]],
    *,
    max_length: int,
    max_chunks_per_document: int,
) -> dict[str, tuple[TokenSequence, ...]]:
    import hashlib

    sequences: dict[str, tuple[TokenSequence, ...]] = {}
    for split_name, documents in documents_by_split.items():
        split_sequences: list[TokenSequence] = []
        for document in documents:
            token_ids = tokenizer.encode(document.text, add_special_tokens=True)
            text_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
            chunks = [
                token_ids[offset : offset + max_length]
                for offset in range(0, len(token_ids), max_length)
            ][:max_chunks_per_document]
            for chunk_index, chunk in enumerate(chunks):
                if len(chunk) < 2:
                    continue
                split_sequences.append(
                    TokenSequence(
                        source_id=document.source_id,
                        title=document.title,
                        url=document.url,
                        text_sha256=text_hash,
                        chunk_index=chunk_index,
                        split=split_name,
                        token_ids=tuple(int(token) for token in chunk),
                    )
                )
        sequences[split_name] = tuple(split_sequences)
    return sequences


def _hidden_state(args: tuple[Any, ...], kwargs: dict[str, Any]) -> torch.Tensor:
    value = kwargs.get("hidden_states")
    if value is None and args:
        value = args[0]
    if not torch.is_tensor(value):
        raise TypeError("Qwen self_attn did not receive a hidden_states tensor")
    return value


def _attention_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Qwen self_attn did not return its attention output as the first result"
    )


def capture_attention_outputs(
    teacher: Teacher,
    sequences: Sequence[TokenSequence],
    *,
    layer_index: int,
    max_length: int,
) -> tuple[ActivationExample, ...]:
    layers = _layers(teacher.model)
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(
            f"Layer index {layer_index} is outside the model's {len(layers)} decoder layers"
        )
    attention = getattr(layers[layer_index], "self_attn", None)
    if attention is None:
        raise ValueError(f"Qwen decoder layer {layer_index} has no self_attn module")

    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    def capture_input(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        captured_inputs.append(
            _hidden_state(args, kwargs)
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .squeeze(0)
        )

    def capture_output(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        captured_outputs.append(
            _attention_output(output)
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .squeeze(0)
        )

    handles = [
        attention.register_forward_pre_hook(capture_input, with_kwargs=True),
        attention.register_forward_hook(capture_output, with_kwargs=True),
    ]
    examples: list[ActivationExample] = []
    input_device = _model_input_device(teacher.model)
    teacher.model.eval()
    try:
        with torch.inference_mode():
            for sequence in sequences:
                length = len(sequence.token_ids)
                if length > max_length:
                    raise ValueError(
                        f"Sequence {sequence.source_id}:{sequence.chunk_index} exceeds max_length={max_length}"
                    )
                input_ids = torch.tensor(
                    sequence.token_ids, dtype=torch.long, device=input_device
                ).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids)
                input_count = len(captured_inputs)
                output_count = len(captured_outputs)
                teacher.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
                if (
                    len(captured_inputs) != input_count + 1
                    or len(captured_outputs) != output_count + 1
                ):
                    raise RuntimeError(
                        "Expected exactly one selected self_attn call per token sequence"
                    )

                hidden_input = captured_inputs[-1]
                attention_output = captured_outputs[-1]
                expected_shape = (length, int(teacher.model.config.hidden_size))
                if (
                    tuple(hidden_input.shape) != expected_shape
                    or tuple(attention_output.shape) != expected_shape
                ):
                    raise RuntimeError(
                        f"Captured self_attn tensors must have shape {expected_shape}, got "
                        f"{tuple(hidden_input.shape)} and {tuple(attention_output.shape)}"
                    )
                if (
                    not torch.isfinite(hidden_input).all()
                    or not torch.isfinite(attention_output).all()
                ):
                    raise RuntimeError(
                        "Qwen self_attn produced a non-finite activation"
                    )
                examples.append(
                    ActivationExample(
                        sequence=sequence,
                        attention_input=hidden_input.contiguous(),
                        attention_output=attention_output.contiguous(),
                    )
                )
    finally:
        for handle in handles:
            handle.remove()

    return tuple(examples)


class _HyenaAttentionAdapter(nn.Module):
    def __init__(self, operator: HyenaOperator) -> None:
        super().__init__()
        self.operator = operator

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if hidden_states is None:
            hidden_states = _hidden_state(args, kwargs)
        if (
            kwargs.get("use_cache")
            or kwargs.get("past_key_value") is not None
            or kwargs.get("past_key_values") is not None
        ):
            raise ValueError(
                "The Hyena replacement only supports full-sequence forwards with cache disabled"
            )
        if hidden_states.shape[-2] > self.operator.l_max:
            raise ValueError(
                f"Sequence length exceeds Hyena's configured limit of {self.operator.l_max}"
            )

        parameter = next(self.operator.parameters())
        output = self.operator(
            hidden_states.to(device=parameter.device, dtype=torch.float32)
        )
        return output.to(device=hidden_states.device, dtype=hidden_states.dtype), None


@contextmanager
def temporary_attention_replacement(
    model: nn.Module,
    layer_index: int,
    operator: HyenaOperator,
) -> Iterator[None]:
    layers = _layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(
            f"Layer index {layer_index} is outside the model's {len(layers)} decoder layers"
        )
    original = layers[layer_index].self_attn
    first_parameter = next(original.parameters(), None)
    device = (
        first_parameter.device
        if first_parameter is not None
        else _model_input_device(model)
    )
    operator.to(device=device, dtype=torch.float32)
    layers[layer_index].self_attn = _HyenaAttentionAdapter(operator)
    try:
        yield
    finally:
        layers[layer_index].self_attn = original


def evaluate_replacement(
    teacher: Teacher,
    sequences: Sequence[TokenSequence],
    *,
    layer_index: int,
    operator: HyenaOperator,
) -> dict[str, float | int]:
    if not sequences:
        raise ValueError("The final test split has no token sequences")

    model = teacher.model
    input_device = _model_input_device(model)
    model.eval()

    def collect_losses() -> tuple[float, int, list[torch.Tensor]]:
        weighted_loss = 0.0
        predicted_tokens = 0
        next_token_logits: list[torch.Tensor] = []
        for sequence in sequences:
            input_ids = torch.tensor(
                sequence.token_ids, dtype=torch.long, device=input_device
            ).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                )
            token_count = max(0, input_ids.shape[1] - 1)
            weighted_loss += float(output.loss.float().item()) * token_count
            predicted_tokens += token_count
            next_token_logits.append(output.logits[0, -1].detach().float().cpu())
        if predicted_tokens == 0:
            raise ValueError("The final test sequences contain no next-token targets")
        return weighted_loss / predicted_tokens, predicted_tokens, next_token_logits

    teacher_loss, token_count, teacher_logits = collect_losses()
    with temporary_attention_replacement(model, layer_index, operator):
        hyena_loss, replaced_token_count, hyena_logits = collect_losses()
    if token_count != replaced_token_count:
        raise RuntimeError(
            "Teacher and replacement losses used different next-token counts"
        )

    logit_deltas = [
        float((expected - actual).abs().mean().item())
        for expected, actual in zip(teacher_logits, hyena_logits, strict=True)
    ]
    return {
        "teacher_next_token_loss": teacher_loss,
        "hyena_next_token_loss": hyena_loss,
        "next_token_loss_delta": hyena_loss - teacher_loss,
        "mean_abs_next_token_logit_delta": sum(logit_deltas) / len(logit_deltas),
        "predicted_token_count": token_count,
        "sequence_count": len(sequences),
    }


def release_teacher(teacher: Teacher) -> None:
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
