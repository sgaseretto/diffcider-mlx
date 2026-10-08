"""Independent PyTorch no-commit read alongside unchanged upstream generation.

The read intervention is adapted from the pinned upstream Decoder._window;
it keeps latent recurrence but removes both commitment and forced commitment.
See THIRD_PARTY.md for the ALoDLM/WeDLM licenses and exact source revision.
"""

import torch

from ._vendor.alodlm.inference import DecodeConfig, Decoder, PrefixCache
from ._vendor.alodlm.model import ALoDLM, layer_forward

__all__ = ["ALoDLM", "DecodeConfig", "Decoder", "PrefixCache", "prefill", "read"]


@torch.inference_mode()
def prefill(decoder, prompt_ids):
    """Use the upstream decoder unchanged to prepare all prefix depths."""
    device = decoder.model.backbone.device
    prompt = torch.tensor(prompt_ids, device=device)
    empty = torch.empty(0, dtype=torch.long, device=device)
    cache = PrefixCache(decoder.model.config)
    window = decoder._window(
        prompt, torch.arange(len(prompt), device=device), cache, empty, DecodeConfig(), empty
    )[0]
    cache.append(window, torch.arange(len(prompt), device=device))
    return cache


@torch.inference_mode()
def read(decoder, tail_ids, cache, *, passes=4, label_ids=None):
    """Return per-depth logits/gates for the same no-commit decision specification."""
    model, c = decoder.model, decoder.model.config
    base, device = model.backbone.model, model.backbone.device
    if not 1 <= passes <= c.max_depth:
        raise ValueError("Invalid pass budget")
    observed = [i for i, t in enumerate(tail_ids) if t != c.mask_token_id]
    masked = [i for i, t in enumerate(tail_ids) if t == c.mask_token_id]
    if not masked or len(tail_ids) > 32:
        raise ValueError("Expected answer masks in a tail of at most 32 tokens")
    order = observed + masked
    ids = torch.tensor([tail_ids[i] for i in order], device=device)
    positions = cache.length + torch.tensor(order, device=device)
    hidden = base.embed_tokens(ids)
    cos, sin = base.rotary_emb(hidden.unsqueeze(0), positions.unsqueeze(0))
    cos, sin = cos.squeeze(0), sin.squeeze(0)
    length = len(ids)
    attention = torch.ones(length, cache.length + length, dtype=torch.bool, device=device)
    attention[:, cache.length :] = torch.ones(
        length, length, dtype=torch.bool, device=device
    ).tril()

    def run(layer, h, depth):
        return layer_forward(
            base.layers[layer], h, cos, sin, attention, cache.values.get(cache.key(layer, depth))
        )[0]

    for layer in range(c.loop_start):
        hidden = run(layer, hidden, 0)
    survival, reads = torch.ones(len(masked), device=device), []
    weight = model.backbone.lm_head.weight
    if label_ids is not None:
        weight = weight[torch.tensor(label_ids, device=device)]
    for depth in range(passes):
        for layer in range(c.loop_start, c.loop_end):
            hidden = run(layer, hidden, depth)
        readout = hidden
        for layer in range(c.loop_end, len(base.layers)):
            readout = run(layer, readout, depth)
        selected = base.norm(readout)[len(observed) :]
        logits = (selected @ weight.T).float()
        hazard = model.exit_gate(selected, depth).float().sigmoid()
        survival = survival * (1 - hazard)
        reads.append({"logits": logits, "hazard": hazard, "halt_cumulative": 1 - survival})
        hidden = base.norm(hidden)
    return reads
