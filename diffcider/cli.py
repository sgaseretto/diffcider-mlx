"""Small command-line interface for MLX decisions and diffusion generation."""

import argparse
import json
from pathlib import Path

from .inference import decide, decision_input, generate, generation_input
from .model import DTYPES, checkpoint_path, load_model


def main():
    """Run one decision JSON file or a text-generation prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="s1", help="s1, base, or local checkpoint directory")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--decision", type=Path, help="JSON object with state and question fields")
    group.add_argument("--prompt", help="Prompt for the base model's diffusion sampler")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()
    model, tokenizer = load_model(checkpoint_path(args.model), args.dtype)
    if args.decision:
        payload = json.loads(args.decision.read_text())
        encoded = decision_input(tokenizer, payload["state"], payload["question"])
        print(json.dumps(decide(model, encoded), indent=2, ensure_ascii=False))
    else:
        ids = generation_input(tokenizer, args.prompt)
        output = generate(
            model, ids, tokenizer.mask_token_id, args.max_new_tokens, args.steps, args.block_size
        )
        print(tokenizer.decode(output[0, len(ids) :].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
