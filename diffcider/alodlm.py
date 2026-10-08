"""ALoDLM recurrent causal inference in MLX.

Ported from amazon-science/ALoDLM at 1da9ddafeca02425480230b9bb7091fdef5672b7.
Modified for MLX and fixed-depth, no-commit decision reads. See THIRD_PARTY.md
and licenses/alodlm-CC-BY-NC-4.0.txt, alodlm-NOTICE.txt and WeDLM.txt.
"""

import hashlib
import json
import tempfile
import time
from dataclasses import dataclass, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .model import DTYPES, Model, ModelArgs

MODEL_ID = "amazon/ALoDLM-1.7B"
MODEL_REVISION = "588c7dc1946e3e09fd496da0b0808bd26f46a758"
REFERENCE_REVISION = "1da9ddafeca02425480230b9bb7091fdef5672b7"


@dataclass(frozen=True)
class LoopConfig:
    """Checkpoint recurrence configuration; layer interval is half-open."""

    loop_start: int = 0
    loop_end: int = 28
    max_depth: int = 4
    mask_token_id: int = 151665

    def validate(self, args):
        if not 0 <= self.loop_start < self.loop_end <= args.num_hidden_layers:
            raise ValueError("Invalid recurrent layer interval")
        if self.max_depth < 1 or not 0 <= self.mask_token_id < args.vocab_size:
            raise ValueError("Invalid recurrent depth or mask token ID")


@dataclass(frozen=True)
class DecodeConfig:
    """Generation controls matching upstream's portable greedy decoder."""

    q: float = 0.5
    tau: float = 0.4
    window_size: int = 16
    max_new_tokens: int = 128
    position_penalty: float | None = None
    mode: str = "entropy"

    def validate(self):
        if not 0 <= self.q <= 1 or not 0 <= self.tau < float("inf"):
            raise ValueError("q must be in [0, 1] and tau finite and nonnegative")
        if any(type(v) is not int or v < 1 for v in (self.window_size, self.max_new_tokens)):
            raise ValueError("Window size and output budget must be positive integers")
        if self.mode not in ("entropy", "left1"):
            raise ValueError("mode must be entropy or left1")
        if self.position_penalty is not None and not 0 <= self.position_penalty < float("inf"):
            raise ValueError("position_penalty must be finite and nonnegative")


class GateHead(nn.Module):
    """Learned conditional halt probability, with a separate bias per depth."""

    def __init__(self, hidden_size, depths):
        super().__init__()
        self.net = nn.Linear(hidden_size, 1)
        self.depth_bias = mx.zeros(depths)

    def __call__(self, features, depth):
        return self.net(features).squeeze(-1) + self.depth_bias[depth]


class PrefixCache:
    """Request-local observed-token K/V, indexed by layer and recurrent depth."""

    def __init__(self, config):
        self.config = config
        self.values = {}
        self.length = 0

    def key(self, layer, depth):
        c = self.config
        return layer, depth if c.loop_start <= layer < c.loop_end else 0

    def complete(self, window, completed):
        for layer in range(self.config.loop_start, self.config.loop_end):
            for depth in range(completed, self.config.max_depth):
                window[layer, depth] = window[layer, completed - 1]

    def append(self, window, rows):
        if not len(rows):
            return
        for key, (k, v) in window.items():
            selected = k[rows], v[rows]
            previous = self.values.get(key)
            self.values[key] = (
                selected
                if previous is None
                else tuple(
                    mx.concatenate((old, new), axis=0)
                    for old, new in zip(previous, selected, strict=True)
                )
            )
        self.length += len(rows)
        mx.eval(self.values)


def rotary_factors(positions, args, dtype):
    """Compute RoPE from logical positions, independent of physical token order."""
    inv = 1.0 / (
        args.rope_theta ** (mx.arange(0, args.head_dim, 2).astype(mx.float32) / args.head_dim)
    )
    angles = positions.astype(mx.float32)[:, None] * inv[None, :]
    angles = mx.concatenate((angles, angles), axis=-1)
    return mx.cos(angles).astype(dtype)[:, None, :], mx.sin(angles).astype(dtype)[:, None, :]


def layer_forward(layer, hidden, cos, sin, mask, prefix=None):
    """Evaluate one Qwen3 layer, retaining only current-token keys and values."""
    h = layer.input_layernorm(hidden)
    attn = layer.self_attn
    length = len(h)
    q = attn.q_norm(attn.q_proj(h).reshape(length, attn.n_heads, -1))
    k = attn.k_norm(attn.k_proj(h).reshape(length, attn.n_kv_heads, -1))
    v = attn.v_proj(h).reshape(length, attn.n_kv_heads, -1)

    def rotate(x):
        a, b = mx.split(x, 2, axis=-1)
        return x * cos + mx.concatenate((-b, a), axis=-1) * sin

    q, k = rotate(q), rotate(k)
    current = k, v
    if prefix is not None:
        k, v = mx.concatenate((prefix[0], k)), mx.concatenate((prefix[1], v))
    out = (
        mx.fast.scaled_dot_product_attention(
            q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None],
            v.transpose(1, 0, 2)[None],
            scale=attn.scale,
            mask=mask,
        )[0]
        .transpose(1, 0, 2)
        .reshape(length, -1)
    )
    hidden = hidden + attn.o_proj(out)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden)), current


class ALoDLM(nn.Module):
    """Shared Qwen3 weights, recurrent execution, and learned exit head."""

    def __init__(self, args: ModelArgs, config: LoopConfig):
        super().__init__()
        config.validate(args)
        if args.tie_word_embeddings:
            raise ValueError("ALoDLM requires the checkpoint's untied output projection")
        self.backbone = Model(args)
        self.config = config
        self.exit_gate = GateHead(args.hidden_size, config.max_depth)

    def step(self, hidden, cos, sin, attention, cache, window, depth):
        """Return recurrent carry and normalized readout for one depth."""
        base, c = self.backbone.model, self.config

        def run(index, h):
            key = cache.key(index, depth)
            h, window[key] = layer_forward(
                base.layers[index], h, cos, sin, attention, cache.values.get(key)
            )
            return h

        if depth == 0:
            for layer in range(c.loop_start):
                hidden = run(layer, hidden)
        for layer in range(c.loop_start, c.loop_end):
            hidden = run(layer, hidden)
        readout = hidden
        for layer in range(c.loop_end, len(base.layers)):
            readout = run(layer, readout)
        return hidden, base.norm(readout)


class Decoder:
    """Greedy generation and read-only recurrence on one loaded backbone."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def _inputs(self, ids, positions, cache):
        hidden = self.model.backbone.model.embed_tokens(ids)
        cos, sin = rotary_factors(positions, self.model.backbone.args, hidden.dtype)
        attention = mx.arange(cache.length + len(ids))[None, :] <= (
            cache.length + mx.arange(len(ids))[:, None]
        )
        return hidden, cos, sin, attention

    def _window(self, ids, positions, cache, mask_rows, config, logical_mask_positions):
        hidden, cos, sin, attention = self._inputs(ids, positions, cache)
        window = {}
        count = len(mask_rows)
        committed = mx.zeros(count, mx.bool_)
        newly_committed = committed
        tokens = mx.zeros(count, mx.int32)
        exit_depths = mx.zeros(count, mx.int32)
        survival = mx.ones(count, mx.float32)
        first_hazard = mx.zeros(count, mx.float32)
        penalty = (
            config.position_penalty if config.position_penalty is not None else config.tau * 0.05
        )
        relative = (
            logical_mask_positions - logical_mask_positions[:1] if count else logical_mask_positions
        )
        base, c = self.model.backbone.model, self.model.config
        for depth in range(c.max_depth):
            if depth and count:
                hidden[mask_rows] = mx.where(
                    newly_committed[:, None], base.embed_tokens(tokens), hidden[mask_rows]
                )
            hidden, features = self.model.step(hidden, cos, sin, attention, cache, window, depth)
            stop = depth == c.max_depth - 1
            if count:
                selected = features[mask_rows]
                logits = self.model.backbone.lm_head(selected).astype(mx.float32)
                predicted = mx.argmax(logits, axis=-1)
                logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                entropy = -(mx.exp(logp) * logp).sum(axis=-1)
                hazard = mx.sigmoid(self.model.exit_gate(selected, depth).astype(mx.float32))
                if depth == 0:
                    first_hazard = hazard
                survival = survival * (1 - hazard)
                adjusted = entropy + relative * penalty
                if config.mode == "left1":
                    agree = mx.zeros(count, mx.bool_)
                    remaining = [i for i, done in enumerate(committed.tolist()) if not done]
                    if remaining:
                        agree[remaining[0]] = True
                else:
                    agree = (adjusted < config.tau) & ~committed
                tokens = mx.where(agree, predicted, tokens)
                exit_depths = mx.where(agree, depth + 1, exit_depths)
                committed = committed | agree
                newly_committed = agree
                residual = ~committed
                residual_count = int(residual.sum().item())
                stop = (
                    stop
                    or residual_count == 0
                    or bool(
                        (
                            mx.where(residual, 1 - survival, 0).sum() / max(residual_count, 1)
                            >= config.q
                        ).item()
                    )
                )
                if stop and not bool(committed.any().item()):
                    index = int(adjusted.argmin().item())
                    tokens[index] = predicted[index]
                    exit_depths[index] = depth + 1
                    committed[index] = True
            if stop:
                cache.complete(window, depth + 1)
                mx.eval(window, committed, tokens, exit_depths, first_hazard)
                return window, committed, tokens, exit_depths, depth + 1, first_hazard
            hidden = base.norm(hidden)
        raise RuntimeError("Recurrent decoding did not terminate")

    def _validate_ids(self, ids):
        if not ids or any(
            type(t) is not int or not 0 <= t < self.model.backbone.args.vocab_size for t in ids
        ):
            raise ValueError("Expected a nonempty list of valid integer token IDs")

    def prefill(self, prompt_ids):
        """Prepare a fresh four-depth prefix cache (all trained depths in general)."""
        self._validate_ids(prompt_ids)
        if len(prompt_ids) > self.model.backbone.args.max_position_embeddings:
            raise ValueError("Prompt exceeds context length")
        cache = PrefixCache(self.model.config)
        cache.owner = self.model
        empty = mx.array([], mx.int32)
        window = self._window(
            mx.array(prompt_ids), mx.arange(len(prompt_ids)), cache, empty, DecodeConfig(), empty
        )[0]
        cache.append(window, mx.arange(len(prompt_ids)))
        return cache

    def read(self, tail_ids, cache, *, passes=4, label_ids=None):
        """Read answer slots without commitments; leave the prefix cache unchanged.

        Args:
            tail_ids: Logical-order tail with configured mask IDs (at most 32 tokens).
            cache: Prefix cache prepared by this decoder with the same weights.
            passes: Fixed number of recurrent passes, from 1 to the trained maximum.
            label_ids: Optional output rows to project; None returns full vocabulary.

        Returns:
            Per-depth logits and gate diagnostics in logical mask-slot order.
        """
        self._validate_ids(tail_ids)
        if getattr(cache, "owner", None) is not self.model:
            raise ValueError("Use a prefix cache prepared by this decoder's model")
        if type(passes) is not int or not 1 <= passes <= self.model.config.max_depth:
            raise ValueError("passes must be between 1 and the trained maximum")
        if (
            len(tail_ids) > 32
            or len(tail_ids) + cache.length > self.model.backbone.args.max_position_embeddings
        ):
            raise ValueError("Decision tail exceeds the 32-token limit or model context")
        if label_ids is not None:
            self._validate_ids(label_ids)
        mask = self.model.config.mask_token_id
        observed = [i for i, t in enumerate(tail_ids) if t != mask]
        masked = [i for i, t in enumerate(tail_ids) if t == mask]
        if not masked:
            raise ValueError("Decision tail must contain answer masks")
        order = observed + masked
        hidden, cos, sin, attention = self._inputs(
            mx.array([tail_ids[i] for i in order]), cache.length + mx.array(order), cache
        )
        window, reads = {}, []
        survival = mx.ones(len(masked), mx.float32)
        weight = self.model.backbone.output_weight
        if label_ids is not None:
            weight = weight[mx.array(label_ids)]
        for depth in range(passes):
            hidden, features = self.model.step(hidden, cos, sin, attention, cache, window, depth)
            selected = features[len(observed) :]
            logits = (selected @ weight.T).astype(mx.float32)
            hazard = mx.sigmoid(self.model.exit_gate(selected, depth).astype(mx.float32))
            survival = survival * (1 - hazard)
            mx.eval(logits, hazard, survival)
            reads.append({"logits": logits, "hazard": hazard, "halt_cumulative": 1 - survival})
            hidden = self.model.backbone.model.norm(hidden)
        return reads

    def generate(self, prompt_ids, config=None):
        """Generate greedily, preserving upstream adaptive window/cache semantics."""
        config = config or DecodeConfig()
        config.validate()
        self._validate_ids(prompt_ids)
        if (
            len(prompt_ids) + config.max_new_tokens
            > self.model.backbone.args.max_position_embeddings
        ):
            raise ValueError("Prompt and generation budget exceed the model context")
        eos = self.tokenizer.eos_token_id
        stop_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        for token in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            if token in self.tokenizer.get_vocab():
                stop_ids.add(self.tokenizer.convert_tokens_to_ids(token))
        mx.synchronize()
        started = time.perf_counter()
        cache = self.prefill(prompt_ids)
        prefill_seconds = time.perf_counter() - started
        generated, depths, hazards = [], [], []
        tokens, flags, window_depths, window_hazards = [], [], [], []
        steps = passes = loop_token_passes = 0
        stop_reason = "length"
        while len(generated) < config.max_new_tokens:
            prefix = next((i for i, flag in enumerate(flags) if flag), len(flags))
            finished = len(generated) + prefix >= config.max_new_tokens or any(
                t in stop_ids for t in tokens[:prefix]
            )
            if finished:
                for i in range(prefix):
                    if tokens[i] in stop_ids:
                        stop_reason = "eos"
                        break
                    generated.append(tokens[i])
                    depths.append(window_depths[i])
                    hazards.append(window_hazards[i])
                break
            desired = min(config.window_size, config.max_new_tokens - len(generated) - prefix)
            grow = max(0, prefix + desired - len(tokens))
            tokens.extend([self.model.config.mask_token_id] * grow)
            flags.extend([True] * grow)
            window_depths.extend([0] * grow)
            window_hazards.extend([0.0] * grow)
            observed = [i for i, flag in enumerate(flags) if not flag]
            masked = [i for i, flag in enumerate(flags) if flag]
            order = observed + masked
            order_tensor = mx.array(order)
            window, agreed, predicted, committed_depth, executed, halt = self._window(
                mx.array([tokens[i] for i in order]),
                cache.length + order_tensor,
                cache,
                mx.arange(len(observed), len(order)),
                config,
                mx.array(masked),
            )
            steps += 1
            passes += executed
            loop_token_passes += len(order) * executed
            cache.append(window, mx.argsort(order_tensor)[:prefix])
            generated.extend(tokens[:prefix])
            depths.extend(window_depths[:prefix])
            hazards.extend(window_hazards[:prefix])
            for j, original in enumerate(masked):
                if bool(agreed[j].item()):
                    tokens[original] = int(predicted[j].item())
                    flags[original] = False
                    window_depths[original] = int(committed_depth[j].item())
                    window_hazards[original] = float(halt[j].item())
            tokens, flags = tokens[prefix:], flags[prefix:]
            window_depths, window_hazards = window_depths[prefix:], window_hazards[prefix:]
        mx.synchronize()
        elapsed = time.perf_counter() - started
        count = len(generated)
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "token_ids": generated,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": count,
            "wall_seconds": elapsed,
            "prefill_seconds": prefill_seconds,
            "tokens_per_second": count / elapsed,
            "mean_exit_depth": sum(depths) / count if count else None,
            "loop_token_passes": loop_token_passes,
            "loop_token_passes_per_output_token": loop_token_passes / count if count else None,
            "outer_steps": steps,
            "recurrent_passes": passes,
            "exit_depths": depths,
            "first_pass_halt_probabilities": hazards,
            "stop_reason": stop_reason,
        }


def checkpoint_path(source=MODEL_ID, revision=None):
    """Resolve a local directory or download a pinned Hugging Face checkpoint."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    return Path(
        snapshot_download(
            str(source),
            revision=revision or (MODEL_REVISION if str(source) == MODEL_ID else None),
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.pt",
                "*.jinja",
                "*.txt",
                "LICENSE",
                "NOTICE",
            ],
        )
    )


def _gate_weights(path):
    safe = path / "exit_gate.safetensors"
    if safe.is_file():
        return mx.load(str(safe))
    original = path / "exit_gate.pt"
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    cache = Path.home() / ".cache" / "diffcider" / "alodlm-gates" / f"{digest}.safetensors"
    if not cache.is_file():
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "First-time conversion of upstream exit_gate.pt needs the benchmark extra "
                "(PyTorch), or supply exit_gate.safetensors in the checkpoint directory."
            ) from error
        weights = torch.load(original, map_location="cpu", weights_only=True)
        converted = {key: mx.array(value.float().numpy()) for key, value in weights.items()}
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Atomic rename avoids a partially written gate during concurrent loads.
        with tempfile.NamedTemporaryFile(
            dir=cache.parent, suffix=".safetensors", delete=False
        ) as temp:
            temporary = Path(temp.name)
        try:
            mx.save_safetensors(str(temporary), converted)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    return mx.load(str(cache))


def load_model(path, dtype="float32"):
    """Load the causal backbone and learned gate without changing source files."""
    path = Path(path)
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    raw = json.loads((path / "config.json").read_text())
    if raw.get("model_type") != "qwen3" or raw.get("tie_word_embeddings", True):
        raise ValueError("Expected an untied Qwen3 ALoDLM checkpoint")
    if raw.get("rope_scaling") or raw.get("use_sliding_window") or raw.get("quantization_config"):
        raise ValueError("Scaled RoPE, sliding attention and quantization are unsupported")
    if raw.get("attention_bias") or raw.get("hidden_act", "silu") != "silu":
        raise ValueError("Expected bias-free Qwen3 attention with SiLU activation")
    args = ModelArgs(**{f.name: raw[f.name] for f in fields(ModelArgs) if f.name in raw})
    metadata = json.loads((path / "alodlm_config.json").read_text())
    loop = LoopConfig(**{f.name: metadata[f.name] for f in fields(LoopConfig)})
    model = ALoDLM(args, loop)
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    weights = {}
    for shard in sorted(set(index.values())) if index else ["model.safetensors"]:
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Invalid safetensors shard path")
        values = mx.load(str(path / shard))
        if weights.keys() & values.keys() or (index and any(index.get(k) != shard for k in values)):
            raise ValueError("Duplicate weights or inconsistent shard index")
        weights.update(values)
    if index is not None and weights.keys() != index.keys():
        raise ValueError("Missing weights from shard index")
    state = {f"backbone.{key}": value.astype(DTYPES[dtype]) for key, value in weights.items()}
    state.update(
        {
            f"exit_gate.{key}": value.astype(DTYPES[dtype])
            for key, value in _gate_weights(path).items()
        }
    )
    model.load_weights(list(state.items()))
    model.eval()
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if max(tokenizer.get_vocab().values()) >= args.vocab_size:
        raise ValueError("Tokenizer exceeds checkpoint vocabulary")
    return model, tokenizer


class AloDiffcider:
    """Experimental ALoDLM generation and zero-shot decisions; API may change."""

    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.decoder = Decoder(model, tokenizer)

    @classmethod
    def from_pretrained(cls, source=MODEL_ID, *, revision=None, dtype="float32"):
        """Load a local checkpoint or the pinned upstream model from the Hub."""
        return cls(*load_model(checkpoint_path(source, revision), dtype))

    def generate(self, prompt, *, config=None, thinking=False):
        """Generate a chat response with upstream greedy decoding semantics."""
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        return self.decoder.generate(ids, config)

    def decide(self, state, questions, *, passes=4, readout="letters"):
        """Return uncalibrated structured decisions without text generation."""
        from .alodlm_decisions import compile_read, interpret

        compiled = compile_read(
            self.tokenizer,
            state,
            questions,
            readout=readout,
            mask_token_id=self.model.config.mask_token_id,
        )
        mx.synchronize()
        started = time.perf_counter()
        cache = self.decoder.prefill(compiled.prefix_ids)
        prefill_seconds = time.perf_counter() - started
        reads = self.decoder.read(
            compiled.tail_ids, cache, passes=passes, label_ids=compiled.label_ids
        )
        result = interpret(compiled, np.asarray(reads[-1]["logits"]))
        mx.synchronize()
        return {
            "answers": result,
            "passes": passes,
            "calibrated": False,
            "first_pass_halt": reads[0]["hazard"].tolist(),
            "halt_cumulative": reads[-1]["halt_cumulative"].tolist(),
            "prefill_seconds": prefill_seconds,
            "wall_seconds": time.perf_counter() - started,
        }
