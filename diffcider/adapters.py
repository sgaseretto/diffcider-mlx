"""Inference-only loading of standard PEFT LoRA and rsLoRA safetensors."""

import json
import math
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """Keep the base projection intact and optionally add one low-rank update."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.adapters = {}
        self.active = None

    def __call__(self, x):
        result = self.base(x)
        if self.active is None or self.active not in self.adapters:
            return result
        adapter = self.adapters[self.active]
        # PEFT's default loader promotes fp16/bf16 adapter parameters to float32.
        delta = (x.astype(mx.float32) @ adapter["a"].T) @ adapter["b"].T
        return (result + delta * adapter["scale"]).astype(result.dtype)


def _matches(pattern, name):
    if isinstance(pattern, str):
        return re.fullmatch(pattern, name) is not None
    return any(name == key or name.endswith("." + key) for key in pattern or [])


def _pattern_value(patterns, name, default):
    for pattern, value in patterns.items():
        if re.match(rf"(.*\.)?({pattern})$", name):
            return value
    return default


def _validate_config(config):
    if config.get("peft_type") != "LORA":
        raise ValueError("Only PEFT LORA adapters are supported.")
    if config.get("bias", "none") != "none":
        raise ValueError("LoRA bias must be 'none' so disabling an adapter restores the base.")
    if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
        raise ValueError(
            "Adapters with base-modifying initialization must be converted to standard LoRA first."
        )
    supported = {
        "peft_type",
        "peft_version",
        "task_type",
        "inference_mode",
        "auto_mapping",
        "base_model_name_or_path",
        "revision",
        "r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "exclude_modules",
        "layers_to_transform",
        "layers_pattern",
        "rank_pattern",
        "alpha_pattern",
        "use_rslora",
        "init_lora_weights",
        "bias",
    }
    for key, value in config.items():
        if key == "megatron_core" and value == "megatron.core":
            continue
        if key == "qalora_group_size" and not config.get("use_qalora"):
            continue
        if key not in supported and value not in (None, False, [], {}, ""):
            raise ValueError(f"Unsupported PEFT option: {key}={value!r}")
    targets = config.get("target_modules")
    if not isinstance(targets, (str, list)) or not targets:
        raise ValueError("An explicit target_modules list or regex is required.")
    if isinstance(targets, list) and not all(isinstance(t, str) and t for t in targets):
        raise ValueError("target_modules must contain nonempty strings.")
    rank = config.get("r")
    if type(rank) is not int or rank < 1:
        raise ValueError("LoRA rank r must be a positive integer.")


class AdapterManager:
    """Validate and attach named adapters without changing any base weights."""

    def __init__(self, model):
        self.model = model
        self.names = []
        self.layers = {}

    def load(self, path: Path, name: str):
        """Load an adapter atomically after validating all configuration and tensors.

        Args:
            path: Local PEFT directory containing config and adapter safetensors.
            name: Unique, nonempty name used for per-request selection.
        """
        if not isinstance(name, str) or not name or name in self.names:
            raise ValueError(f"Adapter name must be nonempty and unique: {name!r}")
        config = json.loads((path / "adapter_config.json").read_text())
        _validate_config(config)
        modules = {}
        for i, block in enumerate(self.model.model.layers):
            for group, names in (
                ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                ("mlp", ("gate_proj", "up_proj", "down_proj")),
            ):
                parent = getattr(block, group)
                for attr in names:
                    modules[f"model.layers.{i}.{group}.{attr}"] = (parent, attr)
        # Include unsupported heads/embeddings when checking targets, so they fail explicitly.
        candidates = set(modules) | {"lm_head", "model.embed_tokens"}
        expected = set()
        for key in candidates:
            target = config["target_modules"]
            selected = key in modules if target == "all-linear" else _matches(target, key)
            if not selected or _matches(config.get("exclude_modules"), key):
                continue
            layers = config.get("layers_to_transform")
            # PEFT applies layer filtering only to suffix-list matches. Exact
            # module paths (including expanded all-linear targets) bypass it.
            if (
                isinstance(target, list)
                and key not in target
                and layers is not None
                and layers != []
            ):
                patterns = config.get("layers_pattern") or ["layers"]
                if isinstance(patterns, str):
                    patterns = [patterns]
                match = next(
                    (m for p in patterns if (m := re.match(rf".*\.{p}\.(\d+)\.", key))), None
                )
                indices = [layers] if isinstance(layers, int) else layers
                if match is None or int(match[1]) not in indices:
                    continue
            expected.add(key)
        if not expected or expected - modules.keys():
            raise ValueError(
                "LoRA targets must be attention/MLP projections; embeddings and lm_head are unsupported."
            )
        weight_path = path / "adapter_model.safetensors"
        if not weight_path.is_file():
            raise ValueError(
                "Missing adapter_model.safetensors; export with safe_serialization=True."
            )
        tensors = mx.load(str(weight_path))
        grouped = {}
        for key, value in tensors.items():
            match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", key)
            module = match[1] if match else ""
            # PEFT exports a bare Qwen3 backbone without the CausalLM's model prefix.
            if module.startswith("layers."):
                module = "model." + module
            if match is None or module not in expected:
                raise ValueError(f"Unsupported or unexpected adapter tensor: {key}")
            pair = grouped.setdefault(module, {})
            if match[2] in pair:
                raise ValueError(f"Duplicate adapter tensor for {module}: {key}")
            pair[match[2]] = value
        if grouped.keys() != expected:
            raise ValueError(
                f"Adapter target tensors are missing: {sorted(expected - grouped.keys())}"
            )
        prepared = []
        for key, pair in grouped.items():
            if pair.keys() != {"A", "B"}:
                raise ValueError(f"Both LoRA A and B weights are required: {key}")
            rank = _pattern_value(config.get("rank_pattern") or {}, key, config["r"])
            alpha = _pattern_value(
                config.get("alpha_pattern") or {}, key, config.get("lora_alpha", 8)
            )
            if (
                type(rank) is not int
                or rank < 1
                or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha)
            ):
                raise ValueError(f"Invalid rank or alpha for {key}")
            parent, attr = modules[key]
            layer = getattr(parent, attr)
            base = layer.base if isinstance(layer, LoRALinear) else layer
            out_features, in_features = base.weight.shape
            a, b = pair["A"], pair["B"]
            if a.shape != (rank, in_features) or b.shape != (out_features, rank):
                raise ValueError(f"LoRA tensor shape does not match rank/base projection: {key}")
            if any(w.dtype not in (mx.float32, mx.float16, mx.bfloat16) for w in (a, b)):
                raise ValueError(f"LoRA weights must be floating-point tensors: {key}")
            if not (mx.all(mx.isfinite(a)) & mx.all(mx.isfinite(b))).item():
                raise ValueError(f"Non-finite LoRA weights: {key}")
            weights = {
                "a": a.astype(mx.float32),
                "b": b.astype(mx.float32),
                "scale": alpha / (math.sqrt(rank) if config.get("use_rslora") else rank),
            }
            mx.eval(weights)
            prepared.append((key, parent, attr, layer, weights))
        for key, parent, attr, layer, weights in prepared:
            if not isinstance(layer, LoRALinear):
                layer = LoRALinear(layer)
                setattr(parent, attr, layer)
            layer.adapters[name] = weights
            self.layers[key] = layer
        self.names.append(name)

    def activate(self, name: str | None):
        """Select a loaded adapter, or bypass every adapter with ``None``.

        Args:
            name: Loaded adapter name or ``None`` for the unchanged base model.
        """
        if name is not None and name not in self.names:
            raise ValueError(f"Unknown adapter {name!r}; loaded adapters: {self.names}")
        for layer in self.layers.values():
            layer.active = name
