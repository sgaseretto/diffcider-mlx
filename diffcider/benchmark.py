"""Compare the upstream PyTorch implementation with MLX on identical inputs."""

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from . import reference, reference_sampler
from .inference import decision_input, generate, generation_input, probabilities, validate_schedule
from .model import CHECKPOINTS, DTYPES, checkpoint_path, load_model

DATA_REVISION = "92e681da3cc01e6888a246378d418645c54ca0f0"
DATA_URL = f"https://raw.githubusercontent.com/zhouzihao11/jev-dllm/{DATA_REVISION}/datasets/s0/test.jsonl.gz"
PROMPTS = [
    "What is the capital of France? Answer in one short sentence.",
    "Calculate 17 + 25. Answer briefly.",
    "Name three primary colors. Answer briefly.",
    "Write a Python function that adds two numbers.",
]


def load_cases(count: int, path: Path | None = None):
    """Select a deterministic, type-balanced subset of the upstream held-out set.

    Args:
        count: Maximum number of decisions, round-robin across question types.
        path: Optional local gzip or plain canonical JSONL data file.

    Returns:
        Selected source records and a content hash for reproducibility.
    """
    if path is None:
        path = Path(".cache") / f"jev-s0-test-{DATA_REVISION}.jsonl.gz"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(DATA_URL, timeout=60) as response:
                path.write_bytes(response.read())
    data = path.read_bytes()
    contents = gzip.decompress(data) if path.suffix == ".gz" else data
    groups = defaultdict(list)
    for line in contents.splitlines():
        record = json.loads(line)
        if len(record["questions"]) != 1:
            raise ValueError("Expected one decision per canonical source record.")
        question = next(iter(record["questions"].values()))
        groups[question["type"]].append(record)
    selected = []
    for i in range(max(map(len, groups.values()))):
        for kind in sorted(groups):
            if i < len(groups[kind]) and len(selected) < count:
                selected.append(groups[kind][i])
    return selected, hashlib.sha256(data).hexdigest()


def measure_pair(torch_fn, mlx_fn, device: str, warmup: int, repeats: int):
    """Time completed GPU work, alternating backend order and excluding warmups.

    Returns:
        Last outputs and all wall-clock samples, in seconds, for each backend.
    """
    calls = {"torch": torch_fn, "mlx": mlx_fn}
    durations = {name: [] for name in calls}
    outputs = {}
    for iteration in range(warmup + repeats):
        order = ("torch", "mlx") if iteration % 2 == 0 else ("mlx", "torch")
        for name in order:
            if name == "torch":
                reference.synchronize(device)
            else:
                mx.synchronize()
            start = time.perf_counter()
            output = calls[name]()
            if name == "torch":
                reference.synchronize(device)
            else:
                mx.eval(output)
                mx.synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= warmup:
                durations[name].append(elapsed)
            outputs[name] = output
    return outputs, durations


def timing_summary(durations: dict) -> dict:
    """Summarize matched timing samples without discarding slow measurements."""
    medians = {name: statistics.median(values) for name, values in durations.items()}
    return {
        "torch_ms": 1000 * medians["torch"],
        "mlx_ms": 1000 * medians["mlx"],
        "speedup": medians["torch"] / medians["mlx"],
        "samples_seconds": durations,
    }


def decision_benchmark(args, torch_model, mlx_model, tokenizer):
    """Compare logits, probabilities, decisions, and synchronized scoring latency."""
    cases, source_hash = load_cases(args.decision_cases, args.cases)
    records = []
    for index, case in enumerate(cases):
        question = next(iter(case["questions"].values()))
        encoded = decision_input(tokenizer, case["state"], question)
        torch_inputs = [
            torch.tensor(value, device=args.device)
            for value in ([encoded.input_ids], encoded.positions, encoded.answer_ids)
        ]
        mlx_inputs = [
            mx.array(value)
            for value in ([encoded.input_ids], encoded.positions, encoded.answer_ids)
        ]
        mx.eval(*mlx_inputs)
        reference.synchronize(args.device)
        outputs, times = measure_pair(
            partial(reference.score_masks, torch_model, *torch_inputs),
            partial(mlx_model.score_masks, *mlx_inputs),
            args.device,
            args.warmup,
            args.repeats,
        )
        tz = outputs["torch"].cpu().numpy()
        mz = np.asarray(outputs["mlx"])
        tp, mp = probabilities(tz, encoded.kind), probabilities(mz, encoded.kind)
        records.append(
            {
                "case_id": case["case_id"],
                "type": encoded.kind,
                "tokens": len(encoded.input_ids),
                "masks": len(encoded.positions),
                "input_sha256": hashlib.sha256(json.dumps(encoded.input_ids).encode()).hexdigest(),
                "torch_index": int(tp.argmax()),
                "mlx_index": int(mp.argmax()),
                "same_decision": bool(tp.argmax() == mp.argmax()),
                "torch_probabilities": tp.tolist(),
                "mlx_probabilities": mp.tolist(),
                "max_probability_error": float(np.abs(tp - mp).max()),
                "max_logit_error": float(np.abs(tz - mz).max()),
                "mean_logit_error": float(np.abs(tz - mz).mean()),
                **timing_summary(times),
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == len(cases):
            print(f"Decisions: {index + 1}/{len(cases)}", flush=True)
    torch_ms = statistics.mean(r["torch_ms"] for r in records)
    mlx_ms = statistics.mean(r["mlx_ms"] for r in records)
    return {
        "source": str(args.cases) if args.cases else DATA_URL,
        "source_sha256": source_hash,
        "selection": "First records per type, interleaved in sorted type order; no filtering by result",
        "count": len(records),
        "matching_decisions": sum(r["same_decision"] for r in records),
        "max_probability_error": max(r["max_probability_error"] for r in records),
        "mean_torch_ms": torch_ms,
        "mean_mlx_ms": mlx_ms,
        "speedup": torch_ms / mlx_ms,
        "cases": records,
    }


def generation_benchmark(args, torch_model, mlx_model, tokenizer):
    """Compare the original model-card sampler with the MLX sampler at temperature zero."""
    records = []
    prompts = json.loads(args.prompts.read_text()) if args.prompts else PROMPTS
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        raise ValueError("Generation prompts must be a nonempty JSON list of strings.")
    prompts = prompts[: args.generation_cases]
    for prompt in prompts:
        ids = generation_input(tokenizer, prompt)
        tx = torch.tensor([ids], device=args.device)
        lengths = torch.tensor([len(ids)], device=args.device)
        pad_id = tokenizer.pad_token_id
        # Compare one identical masked input before the two trajectories can diverge.
        masked_ids = [ids + [tokenizer.mask_token_id] * args.max_new_tokens]
        with torch.inference_mode():
            tz = (
                torch_model(torch.tensor(masked_ids, device=args.device))
                .logits[0, len(ids) :]
                .float()
            )
        mz = mlx_model(mx.array(masked_ids))[0, len(ids) :].astype(mx.float32)
        mx.eval(mz)
        t_first, m_first = tz.cpu().numpy(), np.asarray(mz)
        first_pass = {
            "max_logit_error": float(np.abs(t_first - m_first).max()),
            "mean_logit_error": float(np.abs(t_first - m_first).mean()),
            "argmax_agreement": float(np.mean(t_first.argmax(-1) == m_first.argmax(-1))),
        }
        del tz, mz, t_first, m_first
        outputs, times = measure_pair(
            partial(
                reference_sampler.generate,
                torch_model,
                tx,
                lengths,
                pad_id=pad_id,
                steps=args.steps,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                temperature=0.0,
                cfg_scale=0.0,
                remasking="low_confidence",
                tokenizer=tokenizer,
            ),
            partial(
                generate,
                mlx_model,
                ids,
                tokenizer.mask_token_id,
                args.max_new_tokens,
                args.steps,
                args.block_size,
            ),
            args.device,
            args.warmup,
            args.repeats,
        )
        t_ids = outputs["torch"][0, len(ids) :].cpu().tolist()
        m_ids = outputs["mlx"][0, len(ids) :].tolist()
        same = t_ids == m_ids
        records.append(
            {
                "prompt": prompt,
                "prompt_tokens": len(ids),
                "same_token_ids": same,
                "matching_token_positions": sum(t == m for t, m in zip(t_ids, m_ids, strict=True)),
                "torch_token_ids": t_ids,
                "mlx_token_ids": m_ids,
                "torch_text": tokenizer.decode(t_ids, skip_special_tokens=True),
                "mlx_text": tokenizer.decode(m_ids, skip_special_tokens=True),
                "same_visible_text": tokenizer.decode(t_ids, skip_special_tokens=True)
                == tokenizer.decode(m_ids, skip_special_tokens=True),
                "first_masked_forward": first_pass,
                **timing_summary(times),
            }
        )
        print(f"Generation {len(records)}/{len(prompts)}: exact tokens={same}", flush=True)
    torch_ms = statistics.mean(r["torch_ms"] for r in records)
    mlx_ms = statistics.mean(r["mlx_ms"] for r in records)
    return {
        "count": len(records),
        "matching_sequences": sum(r["same_token_ids"] for r in records),
        "matching_visible_texts": sum(r["same_visible_text"] for r in records),
        "mean_torch_ms": torch_ms,
        "mean_mlx_ms": mlx_ms,
        "speedup": torch_ms / mlx_ms,
        "max_new_tokens": args.max_new_tokens,
        "steps": args.steps,
        "block_size": args.block_size,
        "temperature": 0,
        "cfg_scale": 0,
        "remasking": "low_confidence",
        "cases": records,
    }


def main():
    """Write reproducible parity and latency results as JSON, including mismatches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--task", choices=("decisions", "generation", "both"), default="both")
    parser.add_argument("--decision-cases", type=int, default=60)
    parser.add_argument("--generation-cases", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--prompts", type=Path, help="Optional JSON list of generation prompts")
    parser.add_argument("--cases", type=Path, help="Optional canonical upstream JSONL(.gz)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("reports/comparison.json"))
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.decision_cases < 1:
        parser.error("warmup, repeats, and decision-cases must be positive")
    validate_schedule(args.max_new_tokens, args.steps, args.block_size)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable PYTORCH_ENABLE_MPS_FALLBACK for an honest GPU comparison")
    if not mx.metal.is_available():
        parser.error("The MLX comparison requires an available Metal GPU")
    mx.set_default_device(mx.gpu)
    torch.manual_seed(0)
    mx.random.seed(0)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "memory_gb": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1e9,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("mlx", "torch", "transformers")
        },
        "dtype": args.dtype,
        "torch_device": args.device,
        "mlx_device": "gpu",
        "torch_attention": "sdpa",
        "mlx_attention": "mlx.core.fast.scaled_dot_product_attention",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "batch_size": 1,
        "timing_scope": "Synchronized warm inference; excludes download, load, tokenization, input transfer and output decoding. Generation includes the sampling loop.",
        "checkpoints": CHECKPOINTS,
        "quantized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task, checkpoint, runner in (
        ("decisions", "s1", decision_benchmark),
        ("generation", "base", generation_benchmark),
    ):
        if args.task not in (task, "both"):
            continue
        path = checkpoint_path(checkpoint)
        print(f"Loading {checkpoint}, {args.dtype}: PyTorch {args.device} and MLX GPU", flush=True)
        torch_model = reference.load_reference(path, args.dtype, args.device)
        mlx_model, tokenizer = load_model(path, args.dtype)
        result[task] = runner(args, torch_model, mlx_model, tokenizer)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps({task: {k: v for k, v in result[task].items() if k != "cases"}}, indent=2),
            flush=True,
        )
        del torch_model, mlx_model, tokenizer
        gc.collect()
        mx.clear_cache()
        if args.device == "mps":
            torch.mps.empty_cache()
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
