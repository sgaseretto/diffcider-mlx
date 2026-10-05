"""The original Hugging Face PyTorch model, used only for verification."""

import importlib.util
import sys
from pathlib import Path

import torch


def load_reference(path: Path, dtype: str, device: str):
    """Load the checkpoint's unmodified Python model implementation.

    Importing the inspected, pinned local file directly avoids Transformers'
    dependency scanner treating the ``if __name__ == '__main__'`` dllm demo
    import as a runtime dependency. The model implementation is not patched.

    Args:
        path: Directory resolved by ``checkpoint_path``.
        dtype: Torch dtype name shared with the MLX comparison.
        device: Explicit ``mps``, ``cpu``, or ``cuda`` backend.

    Returns:
        The upstream model in inference mode, using SDPA without KV caching.
    """
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable; choose --device cpu explicitly.")
    name = "_diffcider_upstream_qwen3"
    spec = importlib.util.spec_from_file_location(name, path / "modeling_qwen3.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    config = module.A2DQwen3Config.from_pretrained(path, local_files_only=True)
    config.use_cache = False
    model = module.A2DQwen3LMHeadModel.from_pretrained(
        path,
        config=config,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    return model.to(device).eval()


@torch.inference_mode()
def score_masks(model, input_ids, positions, answer_ids):
    """Run Jev-DLLM's backbone, mask gather, and two-row projection."""
    # An explicit 4D mask also handles the upstream BlockMask=Tensor fallback.
    mask = torch.ones((1, 1, 1, input_ids.shape[1]), dtype=torch.bool, device=input_ids.device)
    hidden = model.model(input_ids=input_ids, attention_mask=mask).last_hidden_state[0, positions]
    return torch.nn.functional.linear(hidden, model.lm_head.weight[answer_ids]).float()


def synchronize(device: str):
    """Wait for GPU execution so benchmark timings include completed work."""
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
