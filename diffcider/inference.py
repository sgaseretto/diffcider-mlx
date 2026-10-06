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
    return decision_result(logits, encoded)


def decision_result(logits, encoded: DecisionInput) -> dict:
    """Convert one decision's Yes/No logits to its public result dictionary."""
    values = probabilities(logits, encoded.kind)
    index = int(values.argmax())
    result = {"index": index, "option": encoded.options[index], "probabilities": values.tolist()}
    if encoded.kind == "score":
        result["expected_score"] = float(values @ np.arange(len(values)))
    return result


def decide_batch(model, encoded: list[DecisionInput], pad_id: int) -> list[dict]:
    """Score tokenized decisions in one forward pass, preserving input order.

    Args:
        model: Shared bidirectional model.
        encoded: Nonempty list of decisions from the same tokenizer. Prompt
            lengths, question types, and candidate counts may differ.
        pad_id: Token used for right padding, excluded from attention by length.

    Returns:
        One decision result per input, normalized over that input's options.
    """
    if not encoded:
        raise ValueError("Decision batch must not be empty.")
    if len(encoded) == 1:
        return [decide(model, encoded[0])]
    answer_ids = encoded[0].answer_ids
    if any(item.answer_ids != answer_ids for item in encoded):
        raise ValueError("Decisions must use the same Yes/No token IDs.")
    lengths = [len(item.input_ids) for item in encoded]
    width, masks = max(lengths), max(len(item.positions) for item in encoded)
    ids = mx.array([item.input_ids + [pad_id] * (width - len(item.input_ids)) for item in encoded])
    valid = mx.arange(width)[None, :] < mx.array(lengths)[:, None]
    # Repeat a valid position for unused candidate slots, then discard their logits.
    positions = mx.array(
        [item.positions + [item.positions[0]] * (masks - len(item.positions)) for item in encoded]
    )
    logits = model.score_masks(ids, positions, mx.array(answer_ids), attention_mask=valid)
    mx.eval(logits)
    values = np.asarray(logits)
    return [
        decision_result(values[i, : len(item.positions)], item) for i, item in enumerate(encoded)
    ]


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
    iterations = steps // (max_new_tokens // block_size)
    transfers = [
        block_size // iterations + (i < block_size % iterations) for i in range(iterations)
    ]
    pending = None
    for start in range(prompt_length, x.shape[1], block_size):
        stop = start + block_size
        for count in transfers:
            if count == 0:
                continue
            block = x[:, start:stop]
            logits = model(x, logit_positions=slice(start, stop))
            predicted = mx.argmax(logits, axis=-1)
            # Match PyTorch's softmax output precision before gathering confidence.
            probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(logits.dtype)
            confidence = mx.take_along_axis(probs, predicted[..., None], axis=-1)[..., 0]
            # Keep global selection indices and precision, including confidence ties.
            scores = mx.full(x.shape, -mx.inf)
            scores[:, start:stop] = mx.where(
                block == mask_id, confidence.astype(mx.float32), -mx.inf
            )
            selected = mx.argpartition(-scores[0], kth=count - 1)[:count]
            transfer = mx.zeros(x.shape, dtype=mx.bool_)
            transfer[0, selected] = True
            block = mx.where(transfer[:, start:stop], predicted, block)
            x = mx.concatenate((x[:, :start], block, x[:, stop:]), axis=1)
            # Overlap CPU dispatch with GPU work, with at most two steps in flight.
            mx.async_eval(x)
            if pending is not None:
                mx.eval(pending)
            pending = x
    mx.eval(x)
    return x


def generate_batch(
    model,
    prompt_ids: list[list[int]],
    mask_id: int,
    pad_id: int | None,
    max_new_tokens: int = 64,
    steps: int = 64,
    block_size: int = 32,
):
    """Generate independent sequences together with one shared greedy schedule.

    Right padding follows each prompt's entire generation region and is excluded
    from attention. Single-item batches use the existing single-prompt sampler.

    Args:
        model: Loaded MLX masked language model.
        prompt_ids: Nonempty list of tokenized prompts, possibly different lengths.
        mask_id: Checkpoint mask token ID.
        pad_id: Padding ID; the masked-out padding uses mask_id if absent.
        max_new_tokens: Fixed number of generated tokens per prompt.
        steps: Shared denoising schedule, distributed over blocks.
        block_size: Generated positions per block for every prompt.

    Returns:
        Completed token array of shape ``[batch, max_new_tokens]``, excluding
        prompts and padding but including special tokens and tokens after EOS.
    """
    validate_schedule(max_new_tokens, steps, block_size)
    if not prompt_ids:
        raise ValueError("At least one tokenized prompt is required.")
    lengths = [len(ids) for ids in prompt_ids]
    if max(lengths) + max_new_tokens > model.args.max_position_embeddings:
        raise ValueError("Prompt and generation exceed the model's context length.")
    if len(prompt_ids) == 1:
        return generate(model, prompt_ids[0], mask_id, max_new_tokens, steps, block_size)[
            :, lengths[0] :
        ]
    width = max(lengths) + max_new_tokens
    pad_id = mask_id if pad_id is None else pad_id
    x = mx.array(
        [
            ids + [mask_id] * max_new_tokens + [pad_id] * (max(lengths) - len(ids))
            for ids in prompt_ids
        ]
    )
    starts = mx.array(lengths)[:, None]
    rows = mx.arange(len(prompt_ids))[:, None]
    valid = mx.arange(width)[None, :] < starts + max_new_tokens
    attention_mask = valid if min(lengths) != max(lengths) else None
    iterations = steps // (max_new_tokens // block_size)
    transfers = [
        block_size // iterations + (i < block_size % iterations) for i in range(iterations)
    ]
    pending = None
    for offset in range(0, max_new_tokens, block_size):
        positions = starts + mx.arange(offset, offset + block_size)[None, :]
        for count in transfers:
            if count == 0:
                continue
            block = x[rows, positions]
            logits = model(x, attention_mask=attention_mask, logit_positions=positions)
            predicted = mx.argmax(logits, axis=-1)
            probs = mx.softmax(logits.astype(mx.float32), axis=-1).astype(logits.dtype)
            confidence = mx.take_along_axis(probs, predicted[..., None], axis=-1)[..., 0]
            scores = mx.full(x.shape, -mx.inf)
            scores[rows, positions] = mx.where(
                block == mask_id, confidence.astype(mx.float32), -mx.inf
            )
            selected = mx.argpartition(-scores, kth=count - 1, axis=-1)[:, :count]
            transfer = mx.zeros(x.shape, dtype=mx.bool_)
            transfer[rows, selected] = True
            # Keep pending bound to the previous step's array, as in generate().
            updated = mx.array(x)
            updated[rows, positions] = mx.where(transfer[rows, positions], predicted, block)
            x = updated
            mx.async_eval(x)
            if pending is not None:
                mx.eval(pending)
            pending = x
    result = x[rows, starts + mx.arange(max_new_tokens)[None, :]]
    mx.eval(result)
    return result
