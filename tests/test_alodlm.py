"""Parity against the unmodified, pinned portable ALoDLM implementation."""

from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from diffcider.alodlm import ALoDLM, DecodeConfig, Decoder, LoopConfig, PrefixCache, load_model
from diffcider.model import ModelArgs

torch = pytest.importorskip("torch")
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from diffcider import reference_alodlm as reference  # noqa: E402
from diffcider._vendor.alodlm.config import ModelConfig  # noqa: E402


class Tokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {}

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens))


@pytest.fixture(params=[(0, 3), (1, 2)], ids=["full-loop", "prelude-coda"])
def pair(request):
    torch.manual_seed(42)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    backbone = Qwen3ForCausalLM(config)
    start, end = request.param
    loop = LoopConfig(start, end, 4, 3)
    pt = reference.ALoDLM(backbone, ModelConfig(**asdict(loop), compile_layers=False)).eval()
    # Nontrivial gate: catch lost learned weights/depth biases and wrong features.
    with torch.no_grad():
        pt.exit_gate.net.weight.normal_(0, 0.1)
        pt.exit_gate.net.bias.fill_(0.2)
    args = ModelArgs.from_config({**config.to_dict(), "model_type": "a2d-qwen3"})
    model = ALoDLM(args, loop)
    model.load_weights(
        [(name, mx.array(value.detach().numpy())) for name, value in pt.state_dict().items()]
    )
    model.eval()
    return reference.Decoder(pt, Tokenizer()), Decoder(model, Tokenizer())


def assert_cache(a, b):
    assert a.keys() == b.keys()
    for key in a:
        for x, y in zip(a[key], b[key], strict=True):
            np.testing.assert_allclose(np.asarray(y), x.detach().numpy(), atol=3e-5, rtol=3e-4)


def test_prefill_and_read_parity_and_cache_immutability(pair):
    pt, mlx = pair
    a, b = reference.prefill(pt, [4, 8, 6, 11]), mlx.prefill([4, 8, 6, 11])
    assert_cache(a.values, b.values)
    before = {key: tuple(np.asarray(t).copy() for t in value) for key, value in b.values.items()}
    # Observed tokens after masks require physical reordering with original RoPE positions.
    tail = [12, 3, 3, 9, 3, 7]
    expected = reference.read(pt, tail, a)
    actual = mlx.read(tail, b)
    for x, y in zip(expected, actual, strict=True):
        for key in x:
            np.testing.assert_allclose(np.asarray(y[key]), x[key].numpy(), atol=3e-5, rtol=3e-4)
    for passes in range(1, 5):
        projected = mlx.read(tail, b, passes=passes, label_ids=[5, 9])[-1]["logits"]
        np.testing.assert_allclose(
            np.asarray(projected), np.asarray(actual[passes - 1]["logits"])[:, [5, 9]], atol=3e-5
        )
    assert b.length == 4
    for key in before:
        for x, y in zip(before[key], b.values[key], strict=True):
            np.testing.assert_array_equal(x, np.asarray(y))


@pytest.mark.parametrize(
    "config",
    [
        DecodeConfig(mode="left1", q=0.0, max_new_tokens=11, window_size=4),
        DecodeConfig(mode="left1", q=1.0, max_new_tokens=11, window_size=4),
        DecodeConfig(
            mode="entropy", q=0.0, tau=0.0, position_penalty=0, max_new_tokens=11, window_size=4
        ),
        DecodeConfig(mode="entropy", q=0.8, tau=10.0, max_new_tokens=11, window_size=4),
    ],
)
def test_generation_windows_match_reference(pair, config):
    pt, mlx = pair
    pt_windows, mlx_windows = [], []
    original_pt, original_mlx = pt._window, mlx._window

    def capture_pt(*args):
        result = original_pt(*args)
        pt_windows.append(result)
        return result

    def capture_mlx(*args):
        result = original_mlx(*args)
        mlx_windows.append(result)
        return result

    pt._window, mlx._window = capture_pt, capture_mlx
    with torch.inference_mode():
        expected = pt.generate([4, 9, 7], reference.DecodeConfig(**asdict(config)))
    actual = mlx.generate([4, 9, 7], config)
    for key in (
        "token_ids",
        "exit_depths",
        "outer_steps",
        "recurrent_passes",
        "loop_token_passes",
        "stop_reason",
    ):
        assert actual[key] == expected[key], key
    assert len(mlx_windows) == len(pt_windows)
    for x, y in zip(pt_windows, mlx_windows, strict=True):
        assert_cache(x[0], y[0])
        for index in (1, 2, 3):
            np.testing.assert_array_equal(x[index].numpy(), np.asarray(y[index]))
        np.testing.assert_allclose(x[5].numpy(), np.asarray(y[5]), atol=3e-5)


def test_cache_completion_preserves_computed_depths(pair):
    _, mlx = pair
    c = mlx.model.config
    cache = PrefixCache(c)
    window = {
        (layer, depth): (mx.array([depth]), mx.array([depth]))
        for layer in range(c.loop_start, c.loop_end)
        for depth in range(2)
    }
    cache.complete(window, 2)
    for layer in range(c.loop_start, c.loop_end):
        assert window[layer, 0][0].item() == 0
        assert [window[layer, d][0].item() for d in (1, 2, 3)] == [1, 1, 1]


def test_input_validation(pair):
    _, mlx = pair
    for ids in ([], [-1], [64], [1.5]):
        with pytest.raises(ValueError):
            mlx.generate(ids)
    with pytest.raises(ValueError, match="context"):
        mlx.generate([4], DecodeConfig(max_new_tokens=128))
    cache = mlx.prefill([4])
    for passes in (0, 5, 1.5):
        with pytest.raises(ValueError):
            mlx.read([3], cache, passes=passes)
    for tail in ([4], [3] * 33):
        with pytest.raises(ValueError):
            mlx.read(tail, cache)
    with pytest.raises(ValueError, match="cache"):
        mlx.read([3], PrefixCache(mlx.model.config))


def test_checkpoint_and_gate_roundtrip(pair, tmp_path, monkeypatch):
    pt, original = pair
    pt.model.save_pretrained(tmp_path)
    tokenizer = Tokenizer()
    tokenizer.get_vocab = lambda: {"dummy": 4}
    monkeypatch.setattr(
        "diffcider.alodlm.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    model, _ = load_model(tmp_path)
    restored = Decoder(model, tokenizer)
    expected = original.read([3, 9, 3], original.prefill([4, 8]))
    actual = restored.read([3, 9, 3], restored.prefill([4, 8]))
    np.testing.assert_array_equal(
        np.asarray(expected[-1]["logits"]), np.asarray(actual[-1]["logits"])
    )
    converted = list((tmp_path / ".cache/diffcider/alodlm-gates").glob("*.safetensors"))
    assert len(converted) == 1
    # The second load must not need PyTorch's checkpoint reader again.
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("gate was not cached"))
    load_model(tmp_path)
