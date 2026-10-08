from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hyena_distill.demo import DemoConfig, run_demo


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyena-distill",
        description="Capture one Qwen attention submodule, fit Hyena, and write a reviewable run package.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layer-index", type=int, default=2)
    parser.add_argument(
        "--input",
        type=Path,
        help="JSONL with source_id, text, and optional title and url",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/qwen2.5-7b-layer-2"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-documents", type=int, default=18)
    parser.add_argument("--max-chunks-per-document", type=int, default=1)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--filter-order", type=int, default=64)
    parser.add_argument("--dataset-id", default="wikimedia/wikipedia")
    parser.add_argument("--dataset-config", default="20231101.en")
    parser.add_argument("--dataset-revision", default="ad5752b")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = DemoConfig(
        model_id=args.model,
        model_revision=args.revision,
        layer_index=args.layer_index,
        input_path=args.input,
        output_dir=args.output,
        max_length=args.max_length,
        max_documents=args.max_documents,
        max_chunks_per_document=args.max_chunks_per_document,
        train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        order=args.order,
        filter_order=args.filter_order,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
    )
    result = run_demo(config)
    print(f"Run artifacts: {result.output_dir}")
    print(result.metrics["verdict"])
    return 0
