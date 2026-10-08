"""Reproducible ALoDLM generation parity and zero-shot decision evaluation.

Run each backend sequentially, then use --compare to retain all mismatches.
PyTorch generation calls the vendored upstream decoder without algorithm changes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from . import reference_alodlm as reference
from .alodlm import (
    MODEL_ID,
    MODEL_REVISION,
    REFERENCE_REVISION,
    DecodeConfig,
    Decoder,
    checkpoint_path,
    load_model,
)
from .alodlm_decisions import compile_read, interpret
from .benchmark import DATA_URL, load_cases

PROMPTS = [
    "What is 7 + 5? Answer with only the number.",
    "Explain binary search in two short sentences.",
    "Write a Python function that returns the larger of two numbers.",
]


def load_decision_cases(count):
    """Select balanced supported cases by format limits, before any model runs."""
    if count == 0:
        return [], None, []
    source, digest = load_cases(max(300, count * 20))
    kinds = ("choice", "noul", "score")
    groups = {kind: [] for kind in kinds}
    skipped = []
    per_type = (count + 2) // 3
    for case in source:
        q = next(iter(case["questions"].values()))
        if len(groups[q["type"]]) >= per_type:
            continue
        if len(q["criteria"]) > 26:
            skipped.append(
                {
                    "case_id": case["case_id"],
                    "options": len(q["criteria"]),
                    "reason": "letter readout supports at most 26 options",
                }
            )
        else:
            groups[q["type"]].append(case)
    if any(len(group) < per_type for group in groups.values()):
        raise ValueError("Insufficient supported cases for the requested balanced comparison")
    cases = [groups[kind][i] for i in range(per_type) for kind in kinds][:count]
    return cases, digest, skipped


def synchronize(backend, device):
    if backend == "mlx":
        mx.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def measure(fn, args):
    """Time completed work, excluding warmups and retaining every measured sample."""
    samples = []
    for index in range(args.warmup + args.repeats):
        synchronize(args.backend, args.device)
        started = time.perf_counter()
        output = fn()
        synchronize(args.backend, args.device)
        if index >= args.warmup:
            samples.append(time.perf_counter() - started)
    return output, samples


def to_numpy(value, backend):
    return np.asarray(value) if backend == "mlx" else value.cpu().float().numpy()


def canonical_question(case):
    """Preserve dataset instructions, option order, rubrics and gold distributions."""
    name, q = next(iter(case["questions"].items()))
    kind, criteria = q["type"], q["criteria"]
    result = {"name": name, "type": kind, "question": q["instructions"]}
    if kind == "score":
        result.update(levels=list(range(len(criteria))), descriptions=criteria)
        keys = [str(i) for i in range(len(criteria))]
    elif kind == "noul":
        keys = ["false", "true"]
        result["descriptions"] = [criteria[k] or "" for k in keys]
    else:
        keys = list(criteria)
        result.update(options=keys, descriptions=[criteria[k] or "" for k in keys])
    gold = [case["gold"][name]["probabilities"][key] for key in keys]
    return result, gold


def quality(rows):
    """Small-sample descriptive metrics; no thresholds or calibration are fitted."""
    groups = {}
    for readout in ("letters", "yesno"):
        for passes in range(1, 5):
            selected = [r for r in rows if r["readout"] == readout and r["passes"] == passes]
            if not selected:
                continue
            confidence, correct, nll, brier, score_mae = [], [], [], [], []
            for row in selected:
                p, g = np.array(row["probabilities"]), np.array(row["gold"])
                confidence.append(float(p.max()))
                correct.append(int(p.argmax() == g.argmax()))
                nll.append(float(-np.dot(g, np.log(np.clip(p, 1e-12, 1)))))
                brier.append(float(np.square(p - g).sum()))
                if row["type"] == "score":
                    score_mae.append(float(abs(np.dot(p - g, np.arange(len(p))))))
            ece = 0.0
            bins = np.minimum((np.array(confidence) * 15).astype(int), 14)
            for b in range(15):
                keep = bins == b
                if keep.any():
                    ece += float(
                        keep.mean()
                        * abs(np.array(confidence)[keep].mean() - np.array(correct)[keep].mean())
                    )
            groups[f"{readout}/depth{passes}"] = {
                "count": len(selected),
                "accuracy": statistics.mean(correct),
                "by_type": {
                    kind: {
                        "count": sum(r["type"] == kind for r in selected),
                        "accuracy": statistics.mean(
                            c for r, c in zip(selected, correct, strict=True) if r["type"] == kind
                        ),
                    }
                    for kind in sorted({r["type"] for r in selected})
                },
                "nll": statistics.mean(nll),
                "brier": statistics.mean(brier),
                "ece15": ece,
                "score_expected_value_mae": statistics.mean(score_mae) if score_mae else None,
                "mean_prefill_seconds": statistics.mean(
                    statistics.median(r["prefill_samples"]) for r in selected
                ),
                "mean_read_seconds": statistics.mean(
                    statistics.median(r["read_samples"]) for r in selected
                ),
            }
    return groups


def run(args):
    path = checkpoint_path()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if args.backend == "mlx":
        model, _ = load_model(path, args.dtype)
        decoder = Decoder(model, tokenizer)
        make_prefix = decoder.prefill
        read = decoder.read
    else:
        model = (
            reference.ALoDLM.from_pretrained(path, dtype=getattr(torch, args.dtype))
            .to(args.device)
            .eval()
        )
        decoder = reference.Decoder(model, tokenizer)
        make_prefix = partial(reference.prefill, decoder)
        read = partial(reference.read, decoder)
    cases, digest, skipped = load_decision_cases(args.decision_cases)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "device": "mlx-gpu" if args.backend == "mlx" else args.device,
        "dtype": args.dtype,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "reference_revision": REFERENCE_REVISION,
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "mlx", "transformers", "numpy")
        },
        "source_hashes": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "alodlm.py",
                "alodlm_decisions.py",
                "reference_alodlm.py",
                "benchmark_alodlm.py",
            )
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "dataset": {
            "url": DATA_URL,
            "sha256": digest,
            "selection": "First supported records per type, interleaved choice/noul/score; at most 26 options; no result-based filtering",
            "count": len(cases),
            "excluded": skipped,
        },
        "generation": [],
        "decisions": [],
    }
    for mode in ("left1", "entropy"):
        for i, prompt in enumerate(PROMPTS[: args.generation_prompts]):
            config = DecodeConfig(
                mode=mode,
                q=0.4 if mode == "left1" else 0.5,
                tau=0.4,
                max_new_tokens=args.max_new_tokens,
            )
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            actual_config = (
                config if args.backend == "mlx" else reference.DecodeConfig(**asdict(config))
            )
            output, samples = measure(partial(decoder.generate, ids, actual_config), args)
            result["generation"].append(
                {
                    "id": f"{mode}/{i}",
                    "prompt": prompt,
                    "input_ids": ids,
                    "config": asdict(config),
                    "samples": samples,
                    "output": output,
                }
            )
            print(
                f"{args.backend}: generation {mode}/{i} {statistics.median(samples):.3f}s",
                flush=True,
            )
    for i, case in enumerate(cases):
        question, gold = canonical_question(case)
        for readout in ("letters", "yesno"):
            compiled = compile_read(
                tokenizer,
                case["state"],
                [question],
                readout=readout,
                mask_token_id=model.config.mask_token_id,
            )
            cache, prefix_samples = measure(partial(make_prefix, compiled.prefix_ids), args)
            for passes in range(1, 5):
                reads, samples = measure(
                    partial(
                        read, compiled.tail_ids, cache, passes=passes, label_ids=compiled.label_ids
                    ),
                    args,
                )
                logits = to_numpy(reads[-1]["logits"], args.backend)
                answer = interpret(compiled, logits)[question["name"]]
                probs = list(answer["probabilities"].values())
                result["decisions"].append(
                    {
                        "case_id": case["case_id"],
                        "type": question["type"],
                        "readout": readout,
                        "passes": passes,
                        "input_sha256": hashlib.sha256(
                            json.dumps(
                                [compiled.prefix_ids, compiled.tail_ids, compiled.label_ids]
                            ).encode()
                        ).hexdigest(),
                        "prefix_tokens": len(compiled.prefix_ids),
                        "tail_ids": compiled.tail_ids,
                        "label_ids": compiled.label_ids,
                        "gold": gold,
                        "probabilities": probs,
                        "prediction": int(np.argmax(probs)),
                        "answer": answer["answer"],
                        "logits": logits.tolist(),
                        "first_pass_halt": to_numpy(reads[0]["hazard"], args.backend).tolist(),
                        "halt_cumulative": to_numpy(
                            reads[-1]["halt_cumulative"], args.backend
                        ).tolist(),
                        "prefill_samples": prefix_samples,
                        "read_samples": samples,
                    }
                )
        print(f"{args.backend}: decisions {i + 1}/{len(cases)}", flush=True)
        # Preserve finished cases if a later measurement is interrupted.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    result["quality"] = quality(result["decisions"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare(torch_path, mlx_path):
    """Compare matching reports, retaining every mismatch and measured latency."""
    a, b = json.loads(torch_path.read_text()), json.loads(mlx_path.read_text())
    if a["backend"] != "torch" or b["backend"] != "mlx":
        raise ValueError("Expected PyTorch report followed by MLX report")
    for key in ("dtype", "model_revision", "reference_revision", "dataset", "warmup", "repeats"):
        if a[key] != b[key]:
            raise ValueError(f"Mismatched report settings: {key}")
    result = {
        "torch_report": str(torch_path),
        "mlx_report": str(mlx_path),
        "generation": [],
        "decisions": [],
    }
    for x, y in zip(a["generation"], b["generation"], strict=True):
        if any(x[k] != y[k] for k in ("id", "input_ids", "config")):
            raise ValueError("Mismatched generation inputs")
        xx, yy = x["output"], y["output"]
        result["generation"].append(
            {
                "id": x["id"],
                "same_tokens": xx["token_ids"] == yy["token_ids"],
                "same_exit_depths": xx["exit_depths"] == yy["exit_depths"],
                "torch_tokens": xx["token_ids"],
                "mlx_tokens": yy["token_ids"],
                "torch_seconds": statistics.median(x["samples"]),
                "mlx_seconds": statistics.median(y["samples"]),
                "speedup": statistics.median(x["samples"]) / statistics.median(y["samples"]),
            }
        )
    for x, y in zip(a["decisions"], b["decisions"], strict=True):
        if any(x[k] != y[k] for k in ("case_id", "readout", "passes", "input_sha256", "gold")):
            raise ValueError("Mismatched decision inputs")
        result["decisions"].append(
            {
                "case_id": x["case_id"],
                "readout": x["readout"],
                "passes": x["passes"],
                "same_decision": x["prediction"] == y["prediction"],
                "max_probability_error": float(
                    np.max(np.abs(np.array(x["probabilities"]) - y["probabilities"]))
                ),
                "max_logit_error": float(np.max(np.abs(np.array(x["logits"]) - y["logits"]))),
                "read_speedup": statistics.median(x["read_samples"])
                / statistics.median(y["read_samples"]),
                "cold_speedup": (
                    statistics.median(x["prefill_samples"]) + statistics.median(x["read_samples"])
                )
                / (statistics.median(y["prefill_samples"]) + statistics.median(y["read_samples"])),
            }
        )
    result["summary"] = {
        "matching_generations": sum(r["same_tokens"] for r in result["generation"]),
        "generation_count": len(result["generation"]),
        "matching_decisions": sum(r["same_decision"] for r in result["decisions"]),
        "decision_count": len(result["decisions"]),
        "max_probability_error": max(
            (r["max_probability_error"] for r in result["decisions"]), default=0
        ),
        "mean_generation_speedup": statistics.mean(r["speedup"] for r in result["generation"])
        if result["generation"]
        else None,
    }
    result["quality"] = {"torch": a.get("quality"), "mlx": b.get("quality")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--decision-cases", type=int, default=30)
    parser.add_argument("--generation-prompts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("TORCH", "MLX"))
    args = parser.parse_args()
    if (
        args.warmup < 0
        or args.repeats < 1
        or args.decision_cases < 0
        or not 0 <= args.generation_prompts <= 3
    ):
        parser.error("Invalid measurement counts")
    with torch.inference_mode():
        if args.compare:
            output = compare(*args.compare)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(output["summary"], indent=2))
        else:
            run(args)


if __name__ == "__main__":
    main()
