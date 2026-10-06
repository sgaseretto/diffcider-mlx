"""Small command-line interface for MLX decisions and diffusion generation."""

import argparse
import json
from pathlib import Path

from .engine import Diffcider
from .model import DTYPES


def main():
    """Run individual or batched decisions and text generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="s1", help="s1, base, local directory, or Hugging Face repo ID"
    )
    parser.add_argument("--revision", help="Base model Hub revision")
    parser.add_argument("--adapter", help="PEFT LoRA directory or Hugging Face repo ID")
    parser.add_argument("--adapter-revision", help="Adapter Hub revision")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Use only local files and cached downloads"
    )
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="Accept stale adapter base metadata after independently verifying the base",
    )
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decision", type=Path, help="JSON object with state and question fields")
    group.add_argument(
        "--decisions", type=Path, help="JSON list of state/question objects to batch"
    )
    group.add_argument("--prompt", help="Prompt for the base model's diffusion sampler")
    group.add_argument("--prompts", type=Path, help="JSON list of prompts to generate as one batch")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Per-decision prompt token limit"
    )
    args = parser.parse_args()
    if args.adapter_revision and not args.adapter:
        parser.error("--adapter-revision requires --adapter")
    engine = Diffcider.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=args.dtype,
        adapters={"default": args.adapter} if args.adapter else None,
        adapter_revisions={"default": args.adapter_revision} if args.adapter_revision else None,
        local_files_only=args.local_files_only,
        allow_base_mismatch=args.allow_base_mismatch,
    )
    adapter = "default" if args.adapter else None
    if args.decision:
        payload = json.loads(args.decision.read_text())
        result = engine.decide(
            payload["state"], payload["question"], adapter=adapter, max_length=args.max_length
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.decisions:
        results = engine.decide_batch(
            json.loads(args.decisions.read_text()), adapter=adapter, max_length=args.max_length
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    elif args.prompts:
        results = engine.generate_batch(
            json.loads(args.prompts.read_text()),
            adapter=adapter,
            max_new_tokens=args.max_new_tokens,
            steps=args.steps,
            block_size=args.block_size,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(
            engine.generate(
                args.prompt,
                adapter=adapter,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                block_size=args.block_size,
            )
        )


if __name__ == "__main__":
    main()
