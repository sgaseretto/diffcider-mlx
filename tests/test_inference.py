"""Contract checks for decision probabilities and the masked generation schedule."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from diffcider.inference import generate, probabilities, validate_schedule


def test_binary_order_is_no_yes():
    """A larger Yes logit must choose index 1 in the public binary output."""
    p = probabilities([[4.0, 1.0]], "noul")
    assert p[1] > 0.95
    assert p[0] < 0.05


def test_candidates_rank_by_log_odds():
    """The largest absolute Yes logit is not necessarily the winning candidate."""
    p = probabilities([[100.0, 99.0], [4.0, 0.0]], "choice")
    assert p.argmax() == 1
    assert p.sum() == pytest.approx(1)


@pytest.mark.parametrize("schedule", [(0, 4, 4), (8, 0, 4), (9, 8, 4), (8, 3, 4)])
def test_invalid_schedules_fail(schedule):
    """Reject invalid schedules before launching any model computation."""
    with pytest.raises(ValueError):
        validate_schedule(*schedule)


def test_sampler_preserves_prompt_and_resolves_all_masks():
    """A deterministic model must fill every block without overwriting the prompt."""

    class KnownModel:
        args = SimpleNamespace(max_position_embeddings=128)

        def __call__(self, x, *, logit_positions=None):
            # Different confidence at each position avoids unspecified top-k ties.
            logits = mx.zeros((*x.shape, 8))
            logits[:, :, 6] = mx.arange(x.shape[1]) + 1.0
            return logits[:, logit_positions] if logit_positions is not None else logits

    result = generate(KnownModel(), [2, 3, 4], mask_id=1, max_new_tokens=8, steps=4, block_size=4)
    np.testing.assert_array_equal(np.asarray(result), [[2, 3, 4] + [6] * 8])


def test_context_overflow_fails():
    """Do not silently exceed the checkpoint's positional embedding limit."""
    model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=5))
    with pytest.raises(ValueError, match="context"):
        generate(model, [2, 3, 4], mask_id=1, max_new_tokens=4, steps=4, block_size=4)


@pytest.mark.parametrize("steps,block_size", [(4, 4), (6, 4), (16, 4), (8, 8), (8, 1)])
def test_sampler_matches_pytorch_with_changing_predictions(steps, block_size):
    """Check multi-token transfers, uneven schedules, zero transfers, and block boundaries."""
    torch = pytest.importorskip("torch")
    from diffcider.reference_sampler import generate as torch_generate

    class TorchModel:
        device = "cpu"

        def __call__(self, x):
            positions = torch.arange(x.shape[1])
            # Every update changes predictions elsewhere, including earlier positions.
            choices = (x.sum() + positions) % 6 + 2
            logits = torch.zeros((*x.shape, 8))
            logits[0, positions, choices] = 1.0 + positions / 10
            return SimpleNamespace(logits=logits)

    class MLXModel:
        args = SimpleNamespace(max_position_embeddings=128)

        def __call__(self, x, *, logit_positions=None):
            positions = mx.arange(x.shape[1])
            choices = (x.sum() + positions) % 6 + 2
            logits = mx.zeros((*x.shape, 8))
            logits[0, positions, choices] = 1.0 + positions / 10
            return logits[:, logit_positions]

    prompt = [2, 3, 4]
    expected = torch_generate(
        TorchModel(),
        torch.tensor([prompt]),
        torch.tensor([len(prompt)]),
        pad_id=0,
        steps=steps,
        max_new_tokens=8,
        block_size=block_size,
        remasking="low_confidence",
        tokenizer=SimpleNamespace(mask_token_id=1),
    )
    actual = generate(MLXModel(), prompt, 1, 8, steps, block_size)
    np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
