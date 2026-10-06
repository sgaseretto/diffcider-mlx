"""Compare batched and sequential generation on the same ordered prompt workload."""

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
import torch

from . import reference, reference_sampler
from .benchmark import PROMPTS
from .engine import Diffcider
from .inference import generate, generate_batch, generation_input, validate_schedule
from .model import CHECKPOINTS, DTYPES, checkpoint_path


class PaddedReference:
    """Supply padding masks to the unchanged upstream PyTorch model and sampler."""

    def __init__(self, model, valid):
        self.model = model
        self.valid = valid
        self.device = model.device

    def __call__(self, input_ids):
        return self.model(input_ids, attention_mask=self.valid)


def main():
    """Record output agreement, throughput, latency, and MLX peak active memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="base")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument(
        "--prompts", type=Path, help="JSON list; defaults to the four existing benchmark prompts"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/batch-float32.json"))
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text()) if args.prompts else PROMPTS
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) for p in prompts):
        parser.error("prompts must be a nonempty JSON list of strings")
    if min(args.warmup, args.repeats, *args.batch_sizes) < 1:
        parser.error("warmup, repeats, and batch sizes must be positive")
    if max(args.batch_sizes) > len(prompts):
        parser.error("batch size cannot exceed the number of prompts in this benchmark")
    validate_schedule(args.max_new_tokens, args.steps, args.block_size)
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable MPS CPU fallback before benchmarking")
    if not mx.metal.is_available() or not torch.backends.mps.is_available():
        parser.error("This benchmark requires MLX Metal and PyTorch MPS")
    mx.set_default_device(mx.gpu)
    path = checkpoint_path(args.model)
    engine = Diffcider.from_pretrained(args.model, dtype=args.dtype)
    model = engine.model
    oracle = reference.load_reference(path, args.dtype, "mps")
    tokenizer = engine.tokenizer
    ids = [generation_input(tokenizer, p) for p in prompts]
    mask_id, pad_id = tokenizer.mask_token_id, tokenizer.pad_token_id
    pad_id = mask_id if pad_id is None else pad_id
    new, steps, block = args.max_new_tokens, args.steps, args.block_size
    if max(map(len, ids)) + new > model.args.max_position_embeddings:
        parser.error("prompt and generation exceed model context")

    def torch_inputs(group):
        lengths = torch.tensor([len(p) for p in group], device="mps")
        width = max(map(len, group))
        prompt = torch.tensor([p + [pad_id] * (width - len(p)) for p in group], device="mps")
        valid = torch.arange(width + new, device="mps")[None, :] < lengths[:, None] + new
        return prompt, lengths, PaddedReference(oracle, valid)

    single_inputs = [torch_inputs([p]) for p in ids]

    def run_torch(inputs):
        outputs = []
        for prompt, lengths, network in inputs:
            output = reference_sampler.generate(
                network,
                prompt,
                lengths,
                pad_id=pad_id,
                max_new_tokens=new,
                steps=steps,
                block_size=block,
                temperature=0.0,
                remasking="low_confidence",
                tokenizer=tokenizer,
            )
            positions = lengths[:, None] + torch.arange(new, device="mps")[None, :]
            outputs.append(output.gather(1, positions))
        return torch.cat(outputs)

    def run_mlx_single():
        return mx.concatenate(
            [generate(model, p, mask_id, new, steps, block)[:, len(p) :] for p in ids]
        )

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
        "prompts": prompts,
        "prompt_token_ids": ids,
        "max_new_tokens": new,
        "steps": steps,
        "block_size": block,
        "temperature": 0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "Same full prompt workload per variant, rotating variant order; synchronized warm inference, excluding loading, chat templating, decoding, and padding-mask preparation for PyTorch. MLX sampler input-array construction is included. Throughput counts the fixed generated token budget, including tokens after EOS.",
        "reference_padding": "The unchanged PyTorch sampler receives a model wrapper that supplies explicit right-padding masks, since the upstream sampler does not pass attention_mask itself. No algorithm modifications.",
        "memory_scope": "MLX peak active bytes per full workload, including resident MLX weights and temporary arrays; excludes the allocator cache, PyTorch allocations, and other system memory. Not total unified-memory usage.",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "inference.py",
                "engine.py",
                "adapters.py",
                "reference.py",
                "reference_sampler.py",
                "benchmark_batch.py",
            )
        },
        "batches": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.batch_sizes:
        groups = [ids[i : i + size] for i in range(0, len(ids), size)]
        batch_inputs = [torch_inputs(group) for group in groups]

        def run_mlx_batch(groups=groups):
            return mx.concatenate(
                [
                    generate_batch(model, group, mask_id, pad_id, new, steps, block)
                    for group in groups
                ]
            )

        calls = {
            "mlx_sequential": run_mlx_single,
            "mlx_batch": run_mlx_batch,
            "torch_sequential": lambda: run_torch(single_inputs),
            "torch_batch": lambda inputs=batch_inputs: run_torch(inputs),
        }
        names = list(calls)
        samples = {name: [] for name in names}
        peaks = {name: [] for name in names if name.startswith("mlx")}
        outputs = {}
        repeat_agreement = {name: True for name in names}
        for iteration in range(args.warmup + args.repeats):
            order = names[iteration % len(names) :] + names[: iteration % len(names)]
            for name in order:
                is_mlx = name.startswith("mlx")
                if is_mlx:
                    mx.synchronize()
                    mx.reset_peak_memory()
                else:
                    torch.mps.synchronize()
                start = time.perf_counter()
                output = calls[name]()
                if is_mlx:
                    mx.eval(output)
                    mx.synchronize()
                else:
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                if iteration >= args.warmup:
                    samples[name].append(elapsed)
                    if is_mlx:
                        peaks[name].append(mx.get_peak_memory())
                tokens = output.tolist() if is_mlx else output.cpu().tolist()
                if name in outputs:
                    repeat_agreement[name] &= outputs[name] == tokens
                outputs[name] = tokens
                del output
            print(
                f"{args.dtype} batch={size}: repetition {iteration + 1}/{args.warmup + args.repeats}",
                flush=True,
            )
        medians = {name: statistics.median(values) for name, values in samples.items()}
        comparisons = {}
        for label, left, right in (
            ("mlx_batch_vs_sequential", "mlx_batch", "mlx_sequential"),
            ("mlx_batch_vs_torch_batch", "mlx_batch", "torch_batch"),
            ("mlx_batch_vs_torch_sequential", "mlx_batch", "torch_sequential"),
            ("torch_batch_vs_sequential", "torch_batch", "torch_sequential"),
        ):
            matches = [a == b for a, b in zip(outputs[left], outputs[right], strict=True)]
            comparisons[label] = {
                "matching_sequences": sum(matches),
                "count": len(matches),
                "per_prompt": matches,
            }
        record = {
            "batch_size": size,
            "actual_batch_sizes": [len(g) for g in groups],
            "padded_sequence_lengths": [max(map(len, g)) + new for g in groups],
            "samples_seconds": samples,
            "median_workload_seconds": medians,
            "generated_tokens_per_second": {
                name: len(prompts) * new / t for name, t in medians.items()
            },
            "mlx_batch_vs_sequential_speedup": medians["mlx_sequential"] / medians["mlx_batch"],
            "mlx_vs_torch_batch_speedup": medians["torch_batch"] / medians["mlx_batch"],
            "mlx_peak_active_bytes": peaks,
            "outputs_stable_across_repetitions": repeat_agreement,
            "comparisons": comparisons,
            "token_ids": outputs,
            "texts": {
                name: tokenizer.batch_decode(tokens, skip_special_tokens=True)
                for name, tokens in outputs.items()
            },
        }
        result["batches"].append(record)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in record.items()
                    if k
                    in (
                        "batch_size",
                        "median_workload_seconds",
                        "mlx_batch_vs_sequential_speedup",
                        "mlx_vs_torch_batch_speedup",
                        "comparisons",
                    )
                }
            ),
            flush=True,
        )
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
