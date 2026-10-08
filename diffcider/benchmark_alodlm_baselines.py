"""Decision baselines on exactly the supported ALoDLM evaluation cases."""

import argparse
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from . import alodlm
from .alodlm_decisions import compile_read, interpret
from .benchmark_alodlm import canonical_question, load_decision_cases, measure, quality
from .inference import decision_input, probabilities
from .model import CHECKPOINTS, checkpoint_path, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=["alodlm-ar", "s1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-cases", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.decision_cases < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("Invalid measurement counts")
    args.backend, args.device = "mlx", "gpu"
    if args.baseline == "s1":
        model, tokenizer = load_model(checkpoint_path("s1"))
        checkpoint = CHECKPOINTS["s1"]
    else:
        model, tokenizer = alodlm.load_model(alodlm.checkpoint_path())
        decoder = alodlm.Decoder(model, tokenizer)
        checkpoint = (alodlm.MODEL_ID, alodlm.MODEL_REVISION)
    cases, digest, skipped = load_decision_cases(args.decision_cases)
    rows = []
    for case in cases:
        q, gold = canonical_question(case)
        if args.baseline == "s1":
            original = next(iter(case["questions"].values()))
            encoded = decision_input(tokenizer, case["state"], original)
            ids = mx.array([encoded.input_ids])
            positions, answer_ids = mx.array(encoded.positions), mx.array(encoded.answer_ids)
            mx.eval(ids, positions, answer_ids)

            def evaluate(ids=ids, positions=positions, answer_ids=answer_ids):
                result = model.score_masks(ids, positions, answer_ids)
                mx.eval(result)  # Synchronizing alone does not execute a lazy MLX graph.
                return result

            logits, samples = measure(evaluate, args)
            probs = probabilities(np.asarray(logits), encoded.kind).tolist()
            tokens = len(encoded.input_ids)
            input_ids = encoded.input_ids
        else:
            compiled = compile_read(tokenizer, case["state"], [q])
            # Standard next-token logits, with no mask and no recurrent refinement.
            first_mask = compiled.tail_ids.index(model.config.mask_token_id)
            input_ids = compiled.prefix_ids + compiled.tail_ids[:first_mask]
            ids = mx.array(input_ids)
            labels = mx.array(compiled.label_ids)

            def evaluate(ids=ids, labels=labels):
                cache = alodlm.PrefixCache(model.config)
                hidden, cos, sin, attention = decoder._inputs(ids, mx.arange(len(ids)), cache)
                _, features = model.step(hidden, cos, sin, attention, cache, {}, 0)
                result = (features[-1:] @ model.backbone.output_weight[labels].T).astype(mx.float32)
                mx.eval(result)
                return result

            logits, samples = measure(evaluate, args)
            probs = list(
                interpret(compiled, np.asarray(logits))[q["name"]]["probabilities"].values()
            )
            tokens = len(ids)
        rows.append(
            {
                "case_id": case["case_id"],
                "type": q["type"],
                "gold": gold,
                "probabilities": probs,
                "passes": 1,
                "readout": "letters" if args.baseline == "alodlm-ar" else "yesno",
                "prediction": int(np.argmax(probs)),
                "tokens": tokens,
                "input_sha256": hashlib.sha256(json.dumps(input_ids).encode()).hexdigest(),
                "prefill_samples": [0.0],
                "read_samples": samples,
            }
        )
        print(
            f"{args.baseline}: {len(rows)}/{len(cases)} {statistics.median(samples):.3f}s",
            flush=True,
        )
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": args.baseline,
        "dtype": "float32",
        "backend": "mlx",
        "checkpoint": checkpoint,
        "dataset_sha256": digest,
        "excluded": skipped,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "Synchronized full-input model inference, no reusable prefix. Excludes tokenization, loading and probability normalization.",
        "decisions": rows,
        "quality": quality(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
