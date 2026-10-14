# LLM-Layer-Distillation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)

**GPU used for the BF16 source run:** AMD Instinct MI300X VF on AMD Developer Cloud. The run used ROCm 10.1 and PyTorch `2.14.0+rocm10.1.0`. The later 400-step tuning record does not identify its GPU.

##  Project Overview
LLM-Layer-Distillation is an experimental repository focused on two critical areas of LLM optimization: architectural distillation and grounded dataset generation.

This project uses layer-level knowledge distillation to replace expensive self-attention layers with sub-quadratic Hyena operators (fftconv). Target model activations serve as a cheap supervision signal to train the efficient sequence mixer.

The original notebook reported activation MSE near `0.00057` on its test split after training on 400 samples. Treat that as an earlier experiment. Its score is not directly comparable to the saved-activation results below, which also check what happens to Qwen's next-token loss when Hyena replaces attention.

## Saved-activation tuning results

We tuned the Hyena checkpoint for 400 steps on saved BF16 activation recordings. The run used validation MSE to select the checkpoint at step 400. It used learning-rate scale `0.03` and seed `44`, with 358 training, 77 validation, and 77 test sequences.

| Attention-output prediction on the recorded test split | MSE |
| --- | ---: |
| Original Hyena checkpoint | 0.00406211 |
| Tuned Hyena checkpoint | 0.00386767 |

The tuned checkpoint lowered MSE by 4.79% on the same 77-sequence test split.

| Qwen2.5-7B evaluation | Next-token loss | Increase over unmodified Qwen |
| --- | ---: | ---: |
| Unmodified Qwen2.5-7B | 2.05470741 | 0 |
| Qwen with original Hyena | 2.05611621 | +0.00140880 |
| Qwen with tuned Hyena | 2.05549194 | +0.00078453 |

The replacement comparison used the same 77 sequences and 12,521 predicted tokens. Tuning reduced the extra loss from the Hyena replacement by 44.31%. The tuned replacement still has slightly higher loss than unmodified Qwen. This measures one attention submodule on a previously exposed test split. It does not measure generation quality or speed.

![Validation MSE across three tuning attempts. Lower is better.](docs/experiments/assets/mi300x-validation-curve.svg)

![Extra next-token loss from replacing attention with the original and tuned Hyena checkpoints.](docs/experiments/assets/mi300x-qwen-replacement-gap.svg)

## Why This Project Exists
Standard transformer attention scales quadratically with sequence length, creating a memory and compute bottleneck for long contexts. This project explores whether that bottleneck can be bypassed cheaply — by training a sub-quadratic Hyena operator to mimic an existing attention layer's behaviour using only its saved input/output activations, without retraining the full model.

## 📊 Attention Visualisation

The image below shows attention maps over a 4,096-token context window. The near-uniform distribution across all positions illustrates the challenge standard attention faces with long documents — the signal is spread thin, making it hard for the model to attend to specific evidence.

![comparison of the attention outputs from the original and hyena op](attention.png)

This motivates the use of the Hyena operator, whose implicit convolutional filters can capture sparse, long-range dependencies more efficiently.

## Project Structure

```
.
├── data_download.ipynb     # Download Wikipedia & generate Q/A pairs via GPT-3.5-turbo
├── finetune_gen.ipynb      # Format training data with CONTEXT IDs and HIGHLIGHT tags
├── hyena.ipynb             # Hyena operator implementation (FFT conv, filters, HyenaOperator)
├── train_data.json         # Generated training pairs (context chunk + Q/A)
├── QA_pairs.json           # Raw Q/A pairs with article IDs and chunk indices
└── README.md
```

---
## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch
- **Ollama** installed locally (with a model like `mistral` or `llama3` pulled)
- A Hugging Face account with access to `Qwen/Qwen2.5-7B`

### Installation

```bash
git clone https://github.com/nitesh-77/llm-layer-distillation.git
pip install torch transformers datasets openai einops
```

### Step 1 — Generate Training Data

Open `data_download.ipynb`. This notebook:

1. Downloads the English Wikipedia dataset (6.4M articles) via Hugging Face `datasets`
2. Splits each article into 2,000-character chunks
3. Calls a local state-of-the-art LLM (e.g., Llama 3 or Mistral via Ollama) to generate a specific, searchable Q/A pair for each chunk
4. Saves results incrementally to `QA_pairs.json`


> **Note:** The full Wikipedia dataset is large (~20GB). For quick experimentation, limit the loop in Cell 5 to the first few hundred articles.

### Step 2 — Format Fine-Tuning Batches

Open `finetune_gen.ipynb`. This notebook:

1. Loads the saved `train_data.json`
2. Tokenizes data using the `Qwen/Qwen2.5-7B` tokenizer
3. Formats batches with randomised `[CONTEXT <ID>]` blocks and citation instructions

The prompt format trains the model to answer questions and cite its source with `[HIGHLIGHT]` tags:

```
[CONTEXT 74821]
<wikipedia passage>
[/CONTEXT 74821]

Answer the following questions, providing highlighted text as evidence for your answer.
To cite and highlight, first link the [CONTEXT] block, then add [HIGHLIGHT] tags around the relevant text.
```

### Step 3 — The Hyena Operator

`hyena.ipynb` contains a standalone implementation of the Hyena sequence model layer. Key components:

| Component | Description |
|---|---|
| `fftconv` | FFT-based long convolution kernel |
| `PositionalEmbedding` | Complex exponential positional encoding |
| `ExponentialModulation` | Learnable decay applied to the filter |
| `HyenaFilter` | Implicit MLP-parameterised long filter |
| `HyenaOperator` | Full Hyena layer replacing multi-head attention |

The Hyena operator can be dropped in as a replacement for attention in any transformer-style architecture.

---

## License

MIT
