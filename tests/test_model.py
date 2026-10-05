"""Numerical and masking checks against independently implemented PyTorch Qwen3."""

import mlx.core as mx
import numpy as np
import pytest

from diffcider.model import Model, ModelArgs

torch = pytest.importorskip("torch")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402


@pytest.fixture(params=[True, False], ids=["tied", "untied"])
def models(request):
    """Build small matching networks without downloading a checkpoint."""
    torch.manual_seed(17)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        tie_word_embeddings=request.param,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=False,
        rms_norm_eps=1e-6,
    )
    config._attn_implementation = "sdpa"
    reference = Qwen3ForCausalLM(config).eval()
    args = ModelArgs.from_config({**config.to_dict(), "model_type": "a2d-qwen3"})
    model = Model(args)
    weights = reference.state_dict()
    if config.tie_word_embeddings:
        weights.pop("lm_head.weight")
    model.load_weights([(key, mx.array(value.numpy())) for key, value in weights.items()])
    model.eval()
    return reference, model


def test_logits_match_pytorch_with_bidirectional_padding(models):
    """Verify Q/K normalization, nonstandard head size, RoPE, GQA, and padding."""
    reference, model = models
    ids = [[2, 8, 4, 5, 0], [7, 3, 9, 0, 0]]
    valid = [[True, True, True, True, False], [True, True, True, False, False]]
    with torch.inference_mode():
        expected = reference(
            torch.tensor(ids), attention_mask=torch.tensor(valid)[:, None, None, :]
        ).logits.numpy()
    actual = np.asarray(model(mx.array(ids), mx.array(valid)))
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_padding_cannot_change_real_tokens(models):
    """Adding right padding must leave all existing token logits unchanged."""
    _, model = models
    short = np.asarray(model(mx.array([[2, 8, 4]])))
    padded = np.asarray(model(mx.array([[2, 8, 4, 0, 0]]), mx.array([[1, 1, 1, 0, 0]], mx.bool_)))
    np.testing.assert_allclose(short, padded[:, :3], atol=2e-5, rtol=2e-4)


def test_attention_sees_future_tokens(models):
    """Catch an accidental causal mask, even if causal logits otherwise look plausible."""
    _, model = models
    a = np.asarray(model(mx.array([[2, 8, 4]]), logit_positions=slice(0, 1)))
    b = np.asarray(model(mx.array([[2, 8, 30]]), logit_positions=slice(0, 1)))
    assert np.max(np.abs(a[0, 0] - b[0, 0])) > 1e-4


@pytest.mark.parametrize("positions", [slice(1, 4), [3, 1]])
def test_position_projection_matches_full_logits(models, positions):
    """Selecting output positions must preserve full-context and padding semantics."""
    _, model = models
    ids = mx.array([[2, 8, 4, 5, 0], [7, 3, 9, 0, 0]])
    valid = mx.array([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    selected = mx.array(positions) if isinstance(positions, list) else positions
    actual = np.asarray(model(ids, valid, logit_positions=selected))
    expected = np.asarray(model(ids, valid))[:, positions]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_selected_projection_matches_full_vocabulary(models):
    """Validate the memory-saving Yes/No readout against the ordinary output head."""
    _, model = models
    ids, positions, answers = mx.array([[2, 8, 4]]), mx.array([0, 2]), mx.array([10, 20])
    actual = np.asarray(model.score_masks(ids, positions, answers))
    expected = np.asarray(model(ids))[0, [0, 2]][:, [10, 20]]
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)


def test_integer_padding_mask_has_boolean_semantics(models):
    """Accept tokenizer-style 0/1 masks without treating them as additive biases."""
    _, model = models
    ids = mx.array([[2, 8, 4, 0]])
    integers = mx.array([[1, 1, 1, 0]])
    actual = np.asarray(model(ids, integers))
    expected = np.asarray(model(ids, integers.astype(mx.bool_)))
    np.testing.assert_array_equal(actual, expected)
