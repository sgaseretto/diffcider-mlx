"""Bidirectional Qwen3, with the checkpoint's original parameter names.

Architecture follows dLLM's A2DQwen3 and Apple's mlx-lm Qwen3 implementation.
See THIRD_PARTY.md for sources and licenses.
"""

import json
from dataclasses import dataclass, fields
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

CHECKPOINTS = {
    "s1": (
        "SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1",
        "1f1c29ff9fc6f6e9dc066b03089878a7dab8b6a0",
    ),
    "base": (
        "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1",
        "c8d24a3f4adaeef46881b450e1bf7d1005203bd7",
    ),
}
DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}


def checkpoint_path(model: str) -> Path:
    """Resolve a local checkpoint or download a pinned supported checkpoint.

    Args:
        model: ``s1``, ``base``, or a local Hugging Face checkpoint directory.

    Returns:
        Directory containing the config, tokenizer, and safetensors weights.
    """
    if model in CHECKPOINTS:
        repo, revision = CHECKPOINTS[model]
        return Path(snapshot_download(repo, revision=revision))
    path = Path(model).expanduser()
    if not (path / "config.json").is_file():
        raise ValueError(f"Unknown checkpoint: {model}. Use s1, base, or a local directory.")
    return path


@dataclass
class ModelArgs:
    """Configuration for the supported dense, bidirectional Qwen3 architecture."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 40960

    @classmethod
    def from_config(cls, config: dict) -> "ModelArgs":
        """Read a checkpoint config, rejecting unsupported architectural changes."""
        if config.get("model_type") != "a2d-qwen3":
            raise ValueError("Expected a2d-qwen3, a bidirectional masked language model.")
        if config.get("rope_scaling") or config.get("use_sliding_window"):
            raise ValueError("Scaled RoPE and sliding attention are not supported.")
        if config.get("attention_bias") or config.get("hidden_act", "silu") != "silu":
            raise ValueError("Expected bias-free attention and SiLU activation.")
        return cls(**{f.name: config[f.name] for f in fields(cls) if f.name in config})


class Attention(nn.Module):
    """Grouped-query self-attention with no causal mask or persistent KV cache."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = args.head_dim**-0.5
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * args.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * args.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * args.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * args.head_dim, args.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(args.head_dim, traditional=False, base=args.rope_theta)

    def __call__(self, x, mask=None):
        batch, length, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(batch, length, self.n_heads, -1))
        k = self.k_norm(self.k_proj(x).reshape(batch, length, self.n_kv_heads, -1))
        v = self.v_proj(x).reshape(batch, length, self.n_kv_heads, -1)
        q, k = self.rope(q.transpose(0, 2, 1, 3)), self.rope(k.transpose(0, 2, 1, 3))
        out = mx.fast.scaled_dot_product_attention(
            q, k, v.transpose(0, 2, 1, 3), scale=self.scale, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class MLP(nn.Module):
    """Qwen3's gated SiLU feed-forward layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-normalized attention and feed-forward residual block."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    """Bidirectional transformer with optional right-padding key masks."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, input_ids, attention_mask=None):
        h = self.embed_tokens(input_ids)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same [batch, length] shape as input_ids."
                )
            mask = attention_mask.astype(mx.bool_)[:, None, None, :]
        for layer in self.layers:
            h = layer(h, mask)
        return self.norm(h)


class Model(nn.Module):
    """Masked language model exposing full-vocabulary and selected-token readouts."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model = Backbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def output_weight(self):
        """Return the output projection, respecting tied word embeddings."""
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.weight
        return self.lm_head.weight

    def __call__(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask) @ self.output_weight.T

    def score_masks(self, input_ids, positions, answer_ids, attention_mask=None):
        """Project mask states onto Yes/No rows without full-vocabulary logits.

        Args:
            input_ids: Token IDs with shape ``[1, length]``.
            positions: One-dimensional array of mask positions.
            answer_ids: The two token IDs in ``[Yes, No]`` order.
            attention_mask: Optional boolean valid-token mask.

        Returns:
            Float32 logits with shape ``[number_of_masks, 2]``.
        """
        hidden = self.model(input_ids, attention_mask)[0, positions]
        return (hidden @ self.output_weight[answer_ids].T).astype(mx.float32)


def load_model(path: Path, dtype: str = "float32"):
    """Load original HF safetensors directly into MLX, without quantization.

    Args:
        path: Local checkpoint directory.
        dtype: Weight/activation precision; float32 is the parity-first default.

    Returns:
        The evaluated MLX model and original tokenizer.
    """
    args = ModelArgs.from_config(json.loads((path / "config.json").read_text()))
    model = Model(args)
    weights = {}
    for shard in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    if not weights:
        raise ValueError(f"No safetensors weights in {path}")
    if args.tie_word_embeddings and "lm_head.weight" in weights:
        # Never silently choose between inconsistent copies of tied weights.
        if not mx.array_equal(
            weights["lm_head.weight"], weights["model.embed_tokens.weight"]
        ).item():
            raise ValueError("Checkpoint contains unequal copies of tied embedding weights.")
        weights.pop("lm_head.weight")
    model.load_weights([(name, value.astype(DTYPES[dtype])) for name, value in weights.items()])
    model.eval()
    mx.eval(model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return model, tokenizer
