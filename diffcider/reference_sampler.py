"""Original PyTorch sampler from the pinned dLLM Hugging Face model card.

Source: dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1,
revision c8d24a3f4adaeef46881b450e1bf7d1005203bd7, README.md (Apache-2.0).
Only API change: tokenizer is passed explicitly instead of being a module global.
Kept separately to provide an independent, reproducible generation baseline.
"""

import numpy as np
import torch
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    """Apply the upstream perturbation, leaving logits unchanged at zero temperature."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Distribute the initial mask count across the requested denoising steps."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(
    model,
    prompt,
    prompt_lens,
    pad_id,
    steps=128,
    max_new_tokens=128,
    block_size=64,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="random",
    tokenizer=None,
):
    """Run the original model-card sampler as the independent PyTorch baseline.

    Args:
        model: The checkpoint's original PyTorch model.
        prompt: Padded prompt token tensor.
        prompt_lens: Length of each unpadded prompt.
        pad_id: Padding token ID.
        steps: Number of denoising steps.
        max_new_tokens: Fixed output length per prompt.
        block_size: Number of generated positions per block.
        temperature: Gumbel perturbation temperature; comparison uses zero.
        cfg_scale: Classifier-free guidance scale; comparison uses zero.
        remasking: Upstream confidence strategy.
        tokenizer: Original tokenizer, passed explicitly instead of globally.

    Returns:
        Token tensor containing the prompts and generated sequences.
    """
    mask_id = tokenizer.mask_token_id
    batch_size = prompt.size(0)
    total_length = int(prompt_lens.max().item() + max_new_tokens)
    x = torch.full((batch_size, total_length), pad_id, dtype=torch.long, device=model.device)
    for i, length in enumerate(prompt_lens.tolist()):
        x[i, :length] = prompt[i, :length]
        x[i, length : length + max_new_tokens] = mask_id

    prompt_index = torch.arange(total_length, device=x.device).unsqueeze(0) < prompt_lens.unsqueeze(
        1
    )
    positions = torch.arange(total_length, device=x.device)

    assert max_new_tokens % block_size == 0
    num_blocks = max_new_tokens // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt_lens + num_block * block_size
        block_end = block_start + block_size
        init_block_mask = (
            (positions.unsqueeze(0) >= block_start.unsqueeze(1))
            & (positions.unsqueeze(0) < block_end.unsqueeze(1))
            & (x == mask_id)
        )
        num_transfer_tokens = get_num_transfer_tokens(init_block_mask, steps_per_block)

        for i in range(steps_per_block):
            block_mask = (
                (positions.unsqueeze(0) >= block_start.unsqueeze(1))
                & (positions.unsqueeze(0) < block_end.unsqueeze(1))
                & (x == mask_id)
            )

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise NotImplementedError(remasking)

            confidence = torch.full_like(x0_p, -np.inf)
            confidence = torch.where(block_mask, x0_p, confidence)

            x0 = torch.where(block_mask, x0, x)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i].item())
                if k == 0:
                    continue
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
