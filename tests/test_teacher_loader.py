from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hyena_distill import demo as demo_module
from hyena_distill.demo import DemoConfig
from hyena_distill.teacher import load_qwen_teacher


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.config = SimpleNamespace(model_type="qwen2", _commit_hash="model-sha")

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


def _install_transformers(monkeypatch, *, model_error: Exception | None = None):
    calls: dict[str, object] = {}
    module = types.ModuleType("transformers")

    class _Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["tokenizer"] = (args, kwargs)
            return SimpleNamespace(_commit_hash="tokenizer-sha")

    class _AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model"] = (args, kwargs)
            if model_error is not None:
                raise model_error
            return _Model()

    def bits_and_bytes(**kwargs):
        calls["bnb"] = kwargs
        return object()

    module.AutoTokenizer = _Tokenizer
    module.AutoModelForCausalLM = _AutoModel
    module.BitsAndBytesConfig = bits_and_bytes
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        "hyena_distill.teacher._model_input_device", lambda model: torch.device("cuda")
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    return calls


def test_bf16_loader_omits_bitsandbytes_and_loads_full_bf16(monkeypatch) -> None:
    calls = _install_transformers(monkeypatch)
    config = DemoConfig(teacher_precision="bf16")

    teacher = load_qwen_teacher(config)

    model_args, kwargs = calls["model"]
    assert model_args == (config.model_id,)
    assert kwargs["revision"] == config.model_revision
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert "quantization_config" not in kwargs
    assert "bnb" not in calls
    assert teacher.quantization == "none (full BF16)"
    assert teacher.model.training is False
    assert all(not parameter.requires_grad for parameter in teacher.model.parameters())


def test_nf4_loader_keeps_existing_bitsandbytes_settings(monkeypatch) -> None:
    calls = _install_transformers(monkeypatch)
    config = DemoConfig()

    teacher = load_qwen_teacher(config)

    model_args, kwargs = calls["model"]
    assert model_args == (config.model_id,)
    assert calls["bnb"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.float16,
    }
    assert kwargs["quantization_config"] is not None
    assert kwargs["torch_dtype"] is torch.float16
    assert teacher.quantization == "4-bit NF4"


def test_bf16_unsupported_fails_without_attempting_model_load(monkeypatch) -> None:
    calls = _install_transformers(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    with pytest.raises(RuntimeError, match="does not report BF16 support"):
        load_qwen_teacher(DemoConfig(teacher_precision="bf16"))

    assert "model" not in calls
    assert "bnb" not in calls


def test_bf16_model_load_failure_is_actionable_without_fallback(monkeypatch) -> None:
    calls = _install_transformers(monkeypatch, model_error=RuntimeError("load failed"))

    with pytest.raises(RuntimeError, match="No precision fallback was attempted"):
        load_qwen_teacher(DemoConfig(teacher_precision="bf16"))

    _args, kwargs = calls["model"]
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert "quantization_config" not in kwargs
    assert "bnb" not in calls


def test_nf4_rocm_load_failure_is_actionable_and_has_no_fallback(monkeypatch) -> None:
    calls = _install_transformers(
        monkeypatch, model_error=RuntimeError("kernel missing")
    )
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)

    with pytest.raises(RuntimeError, match="No BF16 or other precision fallback"):
        load_qwen_teacher(DemoConfig())

    _args, kwargs = calls["model"]
    assert kwargs["torch_dtype"] is torch.float16
    assert "quantization_config" in kwargs


def test_rocm_runtime_metadata_reports_backend_device_and_versions(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "AMD Instinct MI300X"
    )
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)

    runtime = demo_module._runtime_metadata()

    assert runtime["backend"] == "rocm"
    assert runtime["gpu"] == "AMD Instinct MI300X"
    assert runtime["rocm_version"] == "6.2"
    assert runtime["torch_version"] == torch.__version__
    assert "cuda_runtime_version" in runtime


def test_cpu_teacher_runtime_metadata_stays_cpu_on_rocm_build(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)

    runtime = demo_module._runtime_metadata(teacher_device_type="cpu")

    assert runtime["backend"] == "cpu"
    assert runtime["teacher_device_type"] == "cpu"
    assert runtime["gpu"] is None
