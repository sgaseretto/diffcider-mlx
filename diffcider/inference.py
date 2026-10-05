"""Prompt construction, Shared Yes/No scoring, and greedy MDLM generation."""

import json
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class DecisionInput:
    """A tokenized decision with explicit mask locations and output labels."""

    input_ids: list[int]
    positions: list[int]
    answer_ids: list[int]
    options: list[str]
    kind: str


def render_value(value):
    """Render structured state or criterion values using the upstream JSON format."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def decision_input(tokenizer, state, question: dict, max_length: int = 4096) -> DecisionInput:
    """Build the original Jev-DLLM prompt, including explicit mask token IDs.

    Args:
        tokenizer: The original checkpoint tokenizer.
        state: Text or a JSON-serializable state.
        question: Upstream question with type, instructions, and criteria fields.
        max_length: Reject longer prompts instead of silently truncating candidates.

    Returns:
        The encoded prompt and metadata required for decision scoring.
    """
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = {item: None for item in criteria}
        options = [
            key if value is None or value == "" else f"{key}: {render_value(value)}"
            for key, value in criteria.items()
        ]
    elif kind == "score":
        options = [f"level {i}: {render_value(value)}" for i, value in enumerate(criteria)]
    elif kind == "noul":
        options = ["No", "Yes"]
    else:
        raise ValueError(f"Unsupported decision type: {kind}")
    if not options:
        raise ValueError("A decision needs at least one candidate.")
    answer_ids = []
    for word in ("Yes", "No"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1 or ids[0] in tokenizer.all_special_ids:
            raise ValueError(f"{word} must be a single, non-special token.")
        answer_ids.append(ids[0])
    mask_id, mask_text = tokenizer.mask_token_id, tokenizer.mask_token
    if mask_id is None or tokenizer.encode(mask_text, add_special_tokens=False) != [mask_id]:
        raise ValueError("Tokenizer must expose its original mask token.")
    state_text = render_value(state)
    instruction = question["instructions"]
    if not isinstance(instruction, str):
        instruction = json.dumps(instruction)
    if any(mask_text in text for text in [state_text, instruction, *options]):
        raise ValueError("Input text must not contain the reserved mask token.")
    user = f"State:\n{state_text}\n\nQuestion:\n{instruction}\n\n"
    if kind == "noul":
        user += "Answer Yes or No.\nAnswer:" + mask_text
        count = 1
    else:
        user += "For each option, mark Yes if it answers the question, otherwise No.\n"
        user += "\n".join(f"{option}: {mask_text}" for option in options)
        count = len(options)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    parts = prompt.split(mask_text)
    if len(parts) != count + 1:
        raise ValueError("Chat template changed the number of masks.")
    ids, positions = [], []
    for i, part in enumerate(parts):
        ids.extend(tokenizer.encode(part, add_special_tokens=False, truncation=False))
        if i < count:
            positions.append(len(ids))
            ids.append(mask_id)
    if len(ids) > max_length:
        raise ValueError(f"Prompt has {len(ids)} tokens; limit is {max_length}.")
    return DecisionInput(ids, positions, answer_ids, options, kind)


def probabilities(logits, kind: str):
    """Apply the original readout; binary output order is [No, Yes].

    Args:
        logits: Array of ``[Yes, No]`` logits for each mask.
        kind: ``choice``, ``score``, or ``noul``.

    Returns:
        A normalized NumPy probability vector in candidate order.
    """
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 2 or not np.isfinite(z).all():
        raise ValueError("Expected finite Yes/No logits with shape [masks, 2].")
    if kind == "noul":
        if z.shape[0] != 1:
            raise ValueError("A binary decision must have one mask.")
        scores = z[0, ::-1]
    elif kind in ("choice", "score"):
        scores = z[:, 0] - z[:, 1]
    else:
        raise ValueError(f"Unsupported decision type: {kind}")
    values = np.exp(scores - scores.max())
    return values / values.sum()


def decide(model, encoded: DecisionInput) -> dict:
    """Score one decision and return its selected option and probabilities."""
    logits = model.score_masks(
        mx.array([encoded.input_ids]), mx.array(encoded.positions), mx.array(encoded.answer_ids)
    )
    mx.eval(logits)
    values = probabilities(logits, encoded.kind)
    index = int(values.argmax())
    result = {"index": index, "option": encoded.options[index], "probabilities": values.tolist()}
    if encoded.kind == "score":
        result["expected_score"] = float(values @ np.arange(len(values)))
    return result


def generation_input(tokenizer, prompt: str) -> list[int]:
    """Apply the checkpoint's chat template with thinking disabled."""
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def validate_schedule(max_new_tokens: int, steps: int, block_size: int):
    """Validate the upstream semi-autoregressive diffusion schedule."""
    if min(max_new_tokens, steps, block_size) < 1:
        raise ValueError("Token count, steps, and block size must be positive.")
    if max_new_tokens % block_size:
        raise ValueError("max_new_tokens must be divisible by block_size.")
    if steps % (max_new_tokens // block_size):
        raise ValueError("steps must be divisible by the number of blocks.")


def generate(
    model,
    prompt_ids: list[int],
    mask_id: int,
    max_new_tokens: int = 64,
    steps: int = 64,
    block_size: int = 32,
):
    """Run greedy, low-confidence MDLM sampling for a single prompt.

    Mirrors the model card at temperature=0, cfg_scale=0 and
    remasking='low_confidence'. No EOS early-stop: the fixed token budget matches
    the reference. The model sees the whole sequence on every iteration.

    Args:
        model: Loaded MLX masked language model.
        prompt_ids: Already-tokenized prompt; preserved throughout sampling.
        mask_id: Checkpoint mask token ID.
        max_new_tokens: Fixed generation length.
        steps: Total denoising iterations, distributed over blocks.
        block_size: Number of generated positions in each block.

    Returns:
        MLX token array of shape ``[1, prompt_length + max_new_tokens]``.
    """
    validate_schedule(max_new_tokens, steps, block_size)
    prompt_length = len(prompt_ids)
    if prompt_length + max_new_tokens > model.args.max_position_embeddings:
        raise ValueError("Prompt and generation exceed the model's context length.")
    x = mx.array([prompt_ids + [mask_id] * max_new_tokens])
    positions = mx.arange(x.shape[1])[None, :]
    iterations = steps // (max_new_tokens // block_size)
    transfers = [
        block_size // iterations + (i < block_size % iterations) for i in range(iterations)
    ]
    for start in range(prompt_length, x.shape[1], block_size):
        for count in transfers:
            if count == 0:
                continue
            active = (positions >= start) & (positions < start + block_size) & (x == mask_id)
            logits = model(x)
            predicted = mx.argmax(logits, axis=-1)
            # Match PyTorch's softmax output precision before gathering confidence.
            probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(logits.dtype)
            confidence = mx.take_along_axis(probs, predicted[..., None], axis=-1)[..., 0]
            confidence = mx.where(active, confidence.astype(mx.float32), -mx.inf)
            selected = mx.argpartition(-confidence[0], kth=count - 1)[:count]
            transfer = mx.zeros(x.shape, dtype=mx.bool_)
            transfer[0, selected] = True
            x = mx.where(transfer, predicted, x)
            # Bound the lazy graph to one denoising iteration.
            mx.eval(x)
    return x
