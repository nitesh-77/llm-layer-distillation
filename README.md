# Qwen attention-to-Hyena demo

This repository runs a bounded implementation demo. It captures one Qwen2.5-7B attention submodule, trains a Hyena operator to predict that submodule's output, and checks one forward-only replacement.

The run measures two separate outcomes. Activation metrics show how closely Hyena predicts the captured attention output. Next-token loss shows how one replacement changes Qwen's predictions on the same held-out sequences. The run can report weak or mixed results. It does not require Hyena to win.

## What the run captures

The default target is `model.layers[2].self_attn`, the third decoder block with zero-based indexing. The captured input follows the block's input normalization. The target is the first value returned by `self_attn`, before the residual addition. The run does not capture attention probabilities or replace the full decoder block.

The teacher is frozen Qwen2.5-7B loaded in 4-bit NF4 mode. The Hyena width comes from the teacher configuration. The default sequence limit is 512 tokens. The default run samples 18 source articles and keeps at most one sequence from each article.

When you omit `--input`, the run streams from the English subset of [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en). It uses dataset revision `ad5752b`, shuffles with seed 17 and a 1,000-article buffer, then assigns whole articles to train, validation, and test before tokenization. The dataset card lists CC-BY-SA-3.0 and GFDL. See the [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) for its provenance and terms. The stream does not download the full corpus. The Hugging Face [streaming guide](https://huggingface.co/docs/datasets/stream) describes the seeded shuffle used by this run.

The old Q&A generation and prompt-formatting notebooks are not part of this workflow. They remain under `legacy/notebooks/` as historical prototypes. The old `attention.png` is also preserved under `legacy/`; it is not a result from this run.

## Run the demo on Lightning AI

Start a Lightning AI Studio with a T4 GPU. Install the project and its Hugging Face dependencies.

```bash
python -m pip install -e ".[demo]"
```

Run the default bounded experiment. Use a new output directory for each run.

```bash
python -m hyena_distill \
  --output runs/qwen-layer-2-demo \
  --max-documents 18 \
  --max-length 512 \
  --steps 25 \
  --batch-size 1
```

The command saves the resolved model revision, dataset revision, split membership, token counts, optimizer settings, and package versions. It releases Qwen after capture, fits Hyena, then reloads the same model revision to compare the original and replaced model.

## Verification status

The CPU integration suite runs a tiny local Qwen2 model and checks the workflow without downloading model weights. It does not exercise Qwen2.5-7B or a T4. No real-model result is included yet. A completed run writes a verdict and embeds its figures in the run directory's `README.md`; use that report for measured claims.

You can pass a local JSONL file with `--input path/to/articles.jsonl`. Each line must contain a unique `source_id` and non-empty `text`. `title` and `url` are optional. The workflow still assigns complete source documents to splits before it chunks their tokens.

## Read the results

Each run directory contains the trained Hyena checkpoint, compact activation sample, metrics, metadata, plots, and draft Hugging Face cards.

- `hyena.safetensors` and `hyena_config.json` contain the Hyena weights and constructor settings.
- `activations.safetensors` contains sample token IDs, captured attention inputs and outputs, and Hyena predictions. `activation_metadata.json` maps tensors to source articles, chunks, token lengths, and split names.
- `metrics.json` reports held-out MSE, RMSE, relative L2 error, cosine similarity, zero-output and training-mean baselines, and teacher-versus-replacement next-token loss.
- `metadata.json` records the model and data revisions, hidden width, target block, source IDs, split membership, runtime versions, and effective AdamW parameter-group settings. The CLI learning rate applies to parameters without a Hyena-specific rate; the filter and positional embedding rates are recorded separately.
- `loss_curve.png` plots minibatch training loss and validation loss.
- `baseline_metrics.png` compares held-out MSE for Hyena, zero output, and the training mean.
- `comparison_median.png` and `comparison_worst.png` show the teacher output, Hyena output, and their difference for deterministic held-out examples.
- `model_card.md` and `dataset_card.md` are drafts for separate Hugging Face repositories.
- `complete.json` appears only after the run writes the full artifact directory.

The zero-output baseline predicts a zero attention contribution. The training-mean baseline predicts the per-channel mean of training targets. The run computes each baseline from the declared split and reports target scale beside the error metrics. A positive next-token loss delta means the Hyena replacement increased loss on this sample.

Validation MSE selects the checkpoint. The run scores the final test split after selection and does not pass test captures to the optimizer or checkpoint selector.

The activation sample stores tensors as float16 to reduce disk use. Hyena training converts each batch to float32. The checkpoint contains Hyena weights only. The run never saves Qwen teacher weights or uploads files to Hugging Face.

## Run the CPU integration test

Install the test extra, then run the suite.

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The integration test builds a tiny local Qwen2 model. It checks the capture boundary, document-disjoint splits, training and checkpoint reload, activation and loss artifacts, and the attention replacement. It does not prove Qwen2.5-7B loading, T4 memory fit, or model fidelity. Run the documented experiment on a Lightning T4 before claiming those results.

## What this demo does not establish

The run covers one attention submodule, a small document sample, and a forward pass with the cache disabled. It does not measure full-model perplexity, multi-layer replacement, generation behavior, latency, throughput, or a general quality guarantee.

The previous README reported train loss near `0.0003` and test loss near `0.00057`. The repository does not contain the teacher-capture code or activation tensors needed to reproduce those values. Treat them as unverified historical claims. The current run writes its own measurements and preserves poor cases in the report.

## License

The repository code uses the MIT License in `LICENSE`. Public activation samples must follow the source dataset's terms, listed in each generated dataset card.

## Project files

```text
src/hyena_distill/      Runnable capture, training, evaluation, and artifact code
tests/                  CPU integration and Hyena behavior tests
legacy/notebooks/       Historical data, prompt-formatting, and Hyena notebooks
legacy/attention.png    Unverified historical figure
```
