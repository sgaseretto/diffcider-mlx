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

        def __call__(self, x):
            # Different confidence at each position avoids unspecified top-k ties.
            logits = mx.zeros((*x.shape, 8))
            logits[:, :, 6] = mx.arange(x.shape[1]) + 1.0
            return logits

    result = generate(KnownModel(), [2, 3, 4], mask_id=1, max_new_tokens=8, steps=4, block_size=4)
    np.testing.assert_array_equal(np.asarray(result), [[2, 3, 4] + [6] * 8])


def test_context_overflow_fails():
    """Do not silently exceed the checkpoint's positional embedding limit."""
    model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=5))
    with pytest.raises(ValueError, match="context"):
        generate(model, [2, 3, 4], mask_id=1, max_new_tokens=4, steps=4, block_size=4)
