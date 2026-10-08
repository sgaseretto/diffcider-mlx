"""Inspect full-vocabulary mass of decision labels on the first six evaluation cases.

This does not tune prompts or thresholds. Run after timed GPU comparisons.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from diffcider.alodlm import AloDiffcider
from diffcider.alodlm_decisions import compile_read, interpret
from diffcider.benchmark_alodlm import canonical_question, load_decision_cases


def main():
    engine = AloDiffcider.from_pretrained()
    rows = []
    for case in load_decision_cases(6)[0]:
        question, gold = canonical_question(case)
        for readout in ("letters", "yesno"):
            compiled = compile_read(engine.tokenizer, case["state"], [question], readout=readout)
            cache = engine.decoder.prefill(compiled.prefix_ids)
            reads = engine.decoder.read(compiled.tail_ids, cache, passes=4)
            for depth, output in enumerate(reads, 1):
                logits = output["logits"]
                log_z = mx.logsumexp(logits, axis=-1)
                selected = logits[:, mx.array(compiled.label_ids)]
                label_mass = mx.exp(mx.logsumexp(selected, axis=-1) - log_z)
                top_ids = logits.argmax(axis=-1).tolist()
                p = list(
                    interpret(compiled, np.asarray(selected))[question["name"]][
                        "probabilities"
                    ].values()
                )
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "type": question["type"],
                        "readout": readout,
                        "passes": depth,
                        "allowed_label_mass_by_slot": label_mass.tolist(),
                        "full_vocab_argmax_ids": top_ids,
                        "full_vocab_argmax_tokens": [engine.tokenizer.decode([t]) for t in top_ids],
                        "restricted_probabilities": p,
                        "gold": gold,
                        "correct": int(np.argmax(p)) == int(np.argmax(gold)),
                        "hazard": output["hazard"].tolist(),
                    }
                )
        print(f"diagnostics: {len(rows) // 8}/6", flush=True)
    Path("reports/alodlm-label-mass.json").write_text(
        json.dumps(
            {
                "scope": "First six format-eligible evaluation cases; no fitting or template changes",
                "dtype": "float32",
                "backend": "mlx",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
