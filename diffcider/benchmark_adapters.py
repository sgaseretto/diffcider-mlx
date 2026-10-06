"""Compare deterministic PEFT LoRA exports with MLX; adapters are synthetic, not trained."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import mlx.core as mx
import numpy as np
import peft
import torch

from . import reference
from .benchmark import decision_benchmark, generation_benchmark
from .engine import Diffcider
from .model import CHECKPOINTS, DTYPES, checkpoint_path


def main():
    """Write adapter-on/off parity, latency, and exact base-restoration measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=CHECKPOINTS, default="base")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--decision-cases", type=int, default=6)
    parser.add_argument("--generation-cases", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/peft-comparison.json"))
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.decision_cases) < 1:
        parser.error("warmup, repeats, and decision-cases must be positive")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("Disable MPS CPU fallback before benchmarking")
    args.device, args.cases, args.prompts = "mps", None, None
    args.max_new_tokens, args.steps, args.block_size = 16, 16, 8
    path = checkpoint_path(args.model)
    engine = Diffcider.from_pretrained(
        CHECKPOINTS[args.model][0],
        revision=CHECKPOINTS[args.model][1],
        dtype=args.dtype,
    )
    torch_model = reference.load_reference(path, args.dtype, args.device)
    torch_model = peft.get_peft_model(
        torch_model,
        peft.LoraConfig(
            r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "gate_proj"], bias="none"
        ),
    )
    torch_model.peft_config["default"].base_model_name_or_path = CHECKPOINTS[args.model][0]
    torch_model.peft_config["default"].revision = CHECKPOINTS[args.model][1]
    generator = torch.Generator(device="cpu").manual_seed(2026)
    with torch.no_grad():
        for name, parameter in torch_model.named_parameters():
            if "lora_" in name:
                values = torch.randn(parameter.shape, generator=generator) * 0.01
                parameter.copy_(values.to(device=parameter.device, dtype=parameter.dtype))
    torch_model.eval()
    probe = mx.array([[100, engine.tokenizer.mask_token_id, 200]])
    positions = mx.array([1])
    answers = mx.array(
        [engine.tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
    )
    baseline = np.asarray(engine.model.score_masks(probe, positions, answers))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "torch", "transformers", "peft")
        },
        "dtype": args.dtype,
        "torch_device": "mps",
        "mlx_device": "gpu",
        "checkpoint": CHECKPOINTS[args.model],
        "adapter_kind": "Synthetic nonzero standard LoRA, not fine-tuned; validates inference compatibility, not quality.",
        "seed": 2026,
        "adapter_initialization": "CPU float32 normal(0, 0.01), copied into PEFT adapter parameters in named_parameters order.",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_scope": "Synchronized warm inference using the existing benchmark runners; excludes loading, tokenization, and adapter selection. Both modes reuse one loaded base model per backend.",
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "model.py",
                "inference.py",
                "engine.py",
                "adapters.py",
                "reference.py",
                "reference_sampler.py",
                "benchmark.py",
                "benchmark_adapters.py",
            )
        },
        "states": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(".cache").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="peft-validation-", dir=".cache") as directory:
        adapter_path = Path(directory)
        torch_model.save_pretrained(
            adapter_path, safe_serialization=True, save_embedding_layers=False
        )
        result["adapter_config"] = json.loads((adapter_path / "adapter_config.json").read_text())
        result["adapter_sha256"] = hashlib.sha256(
            (adapter_path / "adapter_model.safetensors").read_bytes()
        ).hexdigest()
        engine.load_adapter(adapter_path, name="task")
        for state, active in (("base_before", None), ("adapter", "task"), ("base_after", None)):
            context = nullcontext() if active else torch_model.disable_adapter()
            print(f"{args.model} {args.dtype}: {state}", flush=True)
            with context, engine._request(active):
                probe_logits = np.asarray(engine.model.score_masks(probe, positions, answers))
                result["states"][state] = {
                    "probe_logits": probe_logits.tolist(),
                    "probe_max_change_from_base": float(np.abs(probe_logits - baseline).max()),
                    "decisions": decision_benchmark(
                        args, torch_model.get_base_model(), engine.model, engine.tokenizer
                    ),
                    "generation": generation_benchmark(
                        args, torch_model.get_base_model(), engine.model, engine.tokenizer
                    ),
                }
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        before, after = result["states"]["base_before"], result["states"]["base_after"]
        result["base_restored_exactly"] = (
            before["probe_logits"] == after["probe_logits"]
            and all(
                a["mlx_token_ids"] == b["mlx_token_ids"]
                for a, b in zip(
                    before["generation"]["cases"], after["generation"]["cases"], strict=True
                )
            )
            and all(
                a["mlx_probabilities"] == b["mlx_probabilities"]
                for a, b in zip(
                    before["decisions"]["cases"], after["decisions"]["cases"], strict=True
                )
            )
        )
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for state, record in result["states"].items():
        print(
            state,
            {
                task: {
                    k: v
                    for k, v in record[task].items()
                    if k
                    in (
                        "count",
                        "matching_decisions",
                        "matching_sequences",
                        "max_probability_error",
                        "speedup",
                        "mean_torch_ms",
                        "mean_mlx_ms",
                    )
                }
                for task in ("decisions", "generation")
            },
            flush=True,
        )
    print(
        f"Base restored exactly: {result['base_restored_exactly']}; report: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
