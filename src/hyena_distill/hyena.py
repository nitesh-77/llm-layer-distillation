from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import nn


def _fft_convolution(
    inputs: torch.Tensor, kernel: torch.Tensor, direct: torch.Tensor
) -> torch.Tensor:
    length = inputs.shape[-1]
    fft_size = 2 * length
    kernel_frequency = torch.fft.rfft(kernel, n=fft_size) / fft_size
    input_frequency = torch.fft.rfft(inputs.to(dtype=kernel.dtype), n=fft_size)
    result = torch.fft.irfft(
        input_frequency * kernel_frequency,
        n=fft_size,
        norm="forward",
    )[..., :length]
    return (result + inputs * direct.unsqueeze(-1)).to(dtype=inputs.dtype)


class _OptimModule(nn.Module):
    def register(
        self, name: str, tensor: torch.Tensor, lr: float | None = None, wd: float = 0.0
    ) -> None:
        if lr == 0.0:
            self.register_buffer(name, tensor)
            return

        parameter = nn.Parameter(tensor)
        if lr is not None:
            parameter._optim = {"lr": lr, "weight_decay": wd}
        self.register_parameter(name, parameter)


class _Sin(nn.Module):
    def __init__(self, dimension: int, frequency: float = 10.0) -> None:
        super().__init__()
        self.frequency = nn.Parameter(frequency * torch.ones(1, dimension))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.frequency * inputs)


class _PositionalEmbedding(_OptimModule):
    def __init__(
        self, embedding_dim: int, sequence_length: int, learning_rate: float = 1e-5
    ) -> None:
        super().__init__()
        if embedding_dim < 3 or embedding_dim % 2 == 0:
            raise ValueError("embedding_dim must be odd and at least 3")

        times = torch.linspace(0, 1, sequence_length)[None, :, None]
        positions = torch.linspace(0, sequence_length - 1, sequence_length)[
            None, :, None
        ]
        angular_positions = 2 * math.pi * positions / sequence_length
        bands = (embedding_dim - 1) // 2
        frequencies = torch.linspace(1e-4, bands - 1, bands)[None, None]
        phases = torch.exp(-1j * frequencies * angular_positions)
        embedding = torch.cat((times, phases.real, phases.imag), dim=-1)
        self.register("embedding", embedding, lr=learning_rate)
        self.register("times", times, lr=0.0)

    def forward(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embedding[:, :length], self.times[:, :length]


class _ExponentialModulation(_OptimModule):
    def __init__(
        self, dimension: int, fast_decay_pct: float = 0.3, slow_decay_pct: float = 1.5
    ) -> None:
        super().__init__()
        max_decay = math.log(1e-2) / fast_decay_pct
        min_decay = math.log(1e-2) / slow_decay_pct
        decay_rates = torch.linspace(min_decay, max_decay, dimension)[None, None]
        self.register("decay_rates", decay_rates, lr=0.0)

    def forward(self, times: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return values * torch.exp(-times * self.decay_rates.abs())


class _HyenaFilter(_OptimModule):
    def __init__(
        self,
        dimension: int,
        *,
        order: int,
        sequence_length: int,
        learning_rate: float = 1e-3,
        position_learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.randn(dimension))
        self.position_embedding = _PositionalEmbedding(
            3, sequence_length, position_learning_rate
        )
        activation = _Sin(order)
        self.implicit_filter = nn.Sequential(
            nn.Linear(3, order),
            activation,
            nn.Linear(order, order),
            activation,
            nn.Linear(order, order),
            activation,
            nn.Linear(order, dimension, bias=False),
        )
        self.modulation = _ExponentialModulation(dimension)
        for module in self.implicit_filter:
            for parameter in module.parameters(recurse=False):
                parameter._optim = {"lr": learning_rate, "weight_decay": weight_decay}
        self.position_embedding.embedding._optim = {
            "lr": position_learning_rate,
            "weight_decay": weight_decay,
        }

    def kernel(self, length: int) -> torch.Tensor:
        positions, times = self.position_embedding(length)
        return self.modulation(times, self.implicit_filter(positions))


class HyenaOperator(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int,
        order: int = 4,
        filter_order: int = 64,
    ) -> None:
        super().__init__()
        if d_model < 1 or l_max < 1 or order < 2 or filter_order < 1:
            raise ValueError(
                "d_model, l_max, and filter_order must be positive and order must be at least 2"
            )

        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(0.0)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.out_proj = nn.Linear(d_model, d_model)
        self.short_filter = nn.Conv1d(
            inner_width,
            inner_width,
            kernel_size=3,
            padding=2,
            groups=inner_width,
        )
        self.filter_fn = _HyenaFilter(
            d_model * (order - 1),
            order=filter_order,
            sequence_length=l_max,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"HyenaOperator expects [batch, length, width], got {tuple(inputs.shape)}"
            )
        if inputs.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected hidden width {self.d_model}, got {inputs.shape[-1]}"
            )
        length = inputs.shape[-2]
        if length < 1 or length > self.l_max:
            raise ValueError(
                f"Sequence length must be between 1 and {self.l_max}, got {length}"
            )

        projected = rearrange(self.in_proj(inputs), "b l d -> b d l")
        short_filtered = self.short_filter(projected)[..., :length]
        *gates, state = short_filtered.split(self.d_model, dim=1)
        filters = rearrange(
            self.filter_fn.kernel(length)[0], "l (o d) -> o d l", o=self.order - 1
        )
        direct = rearrange(self.filter_fn.bias, "(o d) -> o d", o=self.order - 1)

        for index, gate in enumerate(reversed(gates[1:])):
            state = self.dropout(state * gate)
            state = _fft_convolution(state, filters[index], direct[index])

        output = rearrange(state * gates[0], "b d l -> b l d")
        return self.out_proj(output)
