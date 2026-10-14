from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from hyena_distill.teacher import Teacher, capture_attention_outputs
from hyena_distill.types import TokenSequence


class _Attention(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        return hidden_states * 2, None


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _QwenFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4, dtype=torch.bfloat16)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DecoderLayer()])
        self.config = SimpleNamespace(model_type="qwen2", hidden_size=4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        hidden_states = self.embedding(input_ids)
        return self.model.layers[0].self_attn(hidden_states=hidden_states)


def test_capture_preserves_teacher_bfloat16_dtype() -> None:
    model = _QwenFixture()
    teacher = Teacher(
        model=model,
        tokenizer=None,
        model_revision="fixture",
        tokenizer_revision="fixture",
        quantization="none (full BF16)",
    )
    sequence = TokenSequence(
        source_id="fixture-1",
        title="Fixture",
        url="https://example.test/fixture",
        text_sha256="fixture-hash",
        chunk_index=0,
        split="train",
        token_ids=(1, 2, 3),
    )

    (example,) = capture_attention_outputs(
        teacher, (sequence,), layer_index=0, max_length=4
    )

    assert example.attention_input.shape == (3, 4)
    assert example.attention_output.shape == (3, 4)
    assert example.attention_input.dtype == torch.bfloat16
    assert example.attention_output.dtype == torch.bfloat16
    torch.testing.assert_close(example.attention_output, example.attention_input * 2)
