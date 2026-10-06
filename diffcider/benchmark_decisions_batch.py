"""Compare S1 batches of up to six with sequential MLX and independent PyTorch."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from . import reference
from .benchmark import DATA_URL, load_cases
from .engine import Diffcider
from .inference import decide, decide_batch, decision_input, decision_result
from .model import CHECKPOINTS, DTYPES


def torch_decide_batch(model, encoded, pad_id):
    """Apply the upstream S1 readout with independent PyTorch padding and row gathers."""
    if len(encoded) == 1:
        item = encoded[0]
        logits = reference.score_masks(
            model,
            torch.tensor([item.input_ids], device="mps"),
            torch.tensor(item.positions, device="mps"),
            torch.tensor(item.answer_ids, device="mps"),
        )
        return [decision_result(logits.cpu().numpy(), item)]
    width = max(len(item.input_ids) for item in encoded)
    ids = torch.tensor(
        [item.input_ids + [pad_id] * (width - len(item.input_ids)) for item in encoded],
        device="mps",
    )
    lengths = torch.tensor([len(item.input_ids) for item in encoded], device="mps")
    valid = torch.arange(width, device="mps")[None, :] < lengths[:, None]
    hidden = model.model(input_ids=ids, attention_mask=valid[:, None, None, :]).last_hidden_state
    # Gather only real candidates, independently of MLX's rectangular position array.
    selected = torch.cat(
        [hidden[i, torch.tensor(item.positions, device="mps")] for i, item in enumerate(encoded)]
    )
    answers = torch.tensor(encoded[0].answer_ids, device="mps")
    logits = (
        torch.nn.functional.linear(selected, model.lm_head.weight[answers]).float().cpu().numpy()
    )
    results, offset = [], 0
    for item in encoded:
        count = len(item.positions)
        results.append(decision_result(logits[offset : offset + count], item))
        offset += count
    return results


def compare(actual, expected):
    """Retain decision agreement and probability/ordinal drift separately."""
    return {
        "matching_decisions": sum(
            a["index"] == b["index"] for a, b in zip(actual, expected, strict=True)
        ),
        "max_probability_error": max(
            float(np.max(np.abs(np.array(a["probabilities"]) - b["probabilities"])))
            for a, b in zip(actual, expected, strict=True)
        ),
        "max_expected_score_error": max(
            (
                abs(a["expected_score"] - b["expected_score"])
                for a, b in zip(actual, expected, strict=True)
                if "expected_score" in a
            ),
            default=0.0,
        ),
        "mismatch_indices": [
            i
            for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            if a["index"] != b["index"]
        ],
    }


def main():
    """Save reproducible batch parity, synchronized timings, and every selected output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="s1")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--decision-cases", type=int, default=60)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.decision_cases, *args.batch_sizes) < 1:
        parser.error("warmup, repeats, case count, and batch sizes must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable MPS CPU fallback before benchmarking")
    if not mx.metal.is_available() or not torch.backends.mps.is_available():
        parser.error("This benchmark requires MLX Metal and PyTorch MPS")
    cases, source_hash = load_cases(args.decision_cases, args.cases)
    if max(args.batch_sizes) > len(cases):
        parser.error("batch size cannot exceed the number of selected cases")
    mx.set_default_device(mx.gpu)
    # Keep varied sequence shapes from retaining a large allocator cache.
    mx.set_cache_limit(1024**3)
    engine = Diffcider.from_pretrained(args.model, dtype=args.dtype)
    oracle = reference.load_reference(engine.path, args.dtype, "mps")
    encoded = [
        decision_input(engine.tokenizer, case["state"], next(iter(case["questions"].values())))
        for case in cases
    ]
    pad_id = engine.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = engine.tokenizer.mask_token_id
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("mlx", "torch", "transformers")
        },
        "checkpoint": CHECKPOINTS[args.model],
        "dtype": args.dtype,
        "torch_device": "mps",
        "mlx_device": "gpu",
        "adapter": None,
        "dataset": {
            "source": str(args.cases) if args.cases else DATA_URL,
            "sha256": source_hash,
            "selection": "First records per type, interleaved in sorted type order; no filtering by result",
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mlx_cache_limit_bytes": 1024**3,
        "allocator_policy": "Clear both unused GPU allocator caches before every variant/sample, outside timing. MLX cache is capped at 1 GiB during each workload. Kernel compilation remains warm; first-use buffer allocation is included in timing.",
        "methodology": "Every variant processes the same ordered workload. One warmup per variant/size; rotating variant order; GPU synchronized before and after each sample. Includes input-array construction, scoring, host transfer, and probability/result readout on both backends. Excludes loading, tokenization, adapter selection, and JSON serialization. The sequential variant completes each request before starting the next. Peak MLX active allocation includes model weights but excludes allocator cache, PyTorch, and other system memory.",
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "inference.py",
                "engine.py",
                "reference.py",
                "benchmark.py",
                "benchmark_decisions_batch.py",
            )
        },
        "cases": [
            {
                "case_id": case["case_id"],
                "type": item.kind,
                "tokens": len(item.input_ids),
                "masks": len(item.positions),
                "input_sha256": hashlib.sha256(json.dumps(item.input_ids).encode()).hexdigest(),
                "options": item.options,
            }
            for case, item in zip(cases, encoded, strict=True)
        ],
        "batches": [],
    }
    with torch.inference_mode():
        for size in args.batch_sizes:
            groups = [encoded[i : i + size] for i in range(0, len(encoded), size)]
            calls = {
                "mlx_sequential": lambda: [decide(engine.model, item) for item in encoded],
                "mlx_batch": lambda groups=groups: [
                    value for group in groups for value in decide_batch(engine.model, group, pad_id)
                ],
                "torch_sequential": lambda: [
                    value
                    for item in encoded
                    for value in torch_decide_batch(oracle, [item], pad_id)
                ],
                "torch_batch": lambda groups=groups: [
                    value for group in groups for value in torch_decide_batch(oracle, group, pad_id)
                ],
            }
            names = list(calls)
            samples, outputs = {name: [] for name in names}, {}
            stable = {name: True for name in names}
            peaks = {name: 0 for name in names if name.startswith("mlx")}
            memory = {name: [] for name in names}
            for iteration in range(args.warmup + args.repeats):
                order = names[iteration % len(names) :] + names[: iteration % len(names)]
                for name in order:
                    mx.synchronize()
                    reference.synchronize("mps")
                    mx.clear_cache()
                    torch.mps.empty_cache()
                    if name.startswith("mlx"):
                        mx.reset_peak_memory()
                    start = time.perf_counter()
                    output = calls[name]()
                    if name.startswith("mlx"):
                        mx.synchronize()
                    else:
                        reference.synchronize("mps")
                    elapsed = time.perf_counter() - start
                    if iteration >= args.warmup:
                        samples[name].append(elapsed)
                        memory[name].append(
                            {
                                "mlx_active": mx.get_active_memory(),
                                "mlx_cache": mx.get_cache_memory(),
                                "torch_active": torch.mps.current_allocated_memory(),
                                "torch_driver": torch.mps.driver_allocated_memory(),
                            }
                        )
                    if name in outputs:
                        stable[name] &= output == outputs[name]
                    outputs[name] = output
                    if name.startswith("mlx"):
                        peaks[name] = max(peaks[name], mx.get_peak_memory())
            medians = {name: statistics.median(values) for name, values in samples.items()}
            record = {
                "batch_size": size,
                "requests": len(encoded),
                "batch_count": len(groups),
                "samples_seconds": samples,
                "median_workload_seconds": medians,
                "decisions_per_second": {
                    name: len(encoded) / value for name, value in medians.items()
                },
                "mlx_batch_speedup_vs_sequential": medians["mlx_sequential"] / medians["mlx_batch"],
                "mlx_batch_speedup_vs_torch_batch": medians["torch_batch"] / medians["mlx_batch"],
                "mlx_peak_active_bytes": peaks,
                "allocator_memory_bytes_after_samples": memory,
                "repeat_outputs_exactly_stable": stable,
                "comparisons": {
                    "mlx_batch_vs_mlx_sequential": compare(
                        outputs["mlx_batch"], outputs["mlx_sequential"]
                    ),
                    "mlx_batch_vs_torch_batch": compare(
                        outputs["mlx_batch"], outputs["torch_batch"]
                    ),
                    "mlx_batch_vs_torch_sequential": compare(
                        outputs["mlx_batch"], outputs["torch_sequential"]
                    ),
                    "torch_batch_vs_torch_sequential": compare(
                        outputs["torch_batch"], outputs["torch_sequential"]
                    ),
                },
                "outputs": outputs,
            }
            result["batches"].append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"Batch {size}: MLX {medians['mlx_batch']:.3f}s, {record['mlx_batch_speedup_vs_sequential']:.2f}x vs sequential; {record['comparisons']}",
                flush=True,
            )
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
