"""PEFT-exported adapters and complete checkpoints, with independent Torch oracles."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from diffcider import Diffcider
from diffcider.adapters import LoRALinear
from diffcider.inference import decision_input, generation_input, probabilities
from diffcider.model import CHECKPOINTS, checkpoint_path, load_model

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from diffcider import reference_sampler  # noqa: E402


@pytest.fixture
def bundle(tmp_path):
    """Export a tiny full model and two nonzero adapters using the actual PEFT writer."""
    torch.manual_seed(17)
    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        tie_word_embeddings=True,
        rope_theta=1_000_000,
        attention_dropout=0.0,
        use_cache=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
    )
    config._attn_implementation = "sdpa"
    base = Qwen3ForCausalLM(config).eval()
    path = tmp_path / "base"
    base.save_pretrained(path, safe_serialization=True)
    saved = json.loads((path / "config.json").read_text())
    saved["model_type"] = "a2d-qwen3"
    (path / "config.json").write_text(json.dumps(saved))
    words = ["[PAD]", "[MASK]", "[UNK]", "[BOS]", "[EOS]", "Yes", "No", "hello", "world"]
    vocab = {word: i for i, word in enumerate(words + [f"t{i}" for i in range(9, 64)])}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}",
    )
    tokenizer.save_pretrained(path)
    configs = [
        peft.LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj", "o_proj", "gate_proj", "down_proj"],
            rank_pattern={"model.layers.1.self_attn.q_proj": 3},
            alpha_pattern={"model.layers.1.self_attn.q_proj": 7},
            lora_dropout=0.2,
        ),
        peft.LoraConfig(r=3, lora_alpha=5, target_modules=["v_proj", "up_proj"], use_rslora=True),
    ]
    adapters = []
    for i, cfg in enumerate(configs):
        torch.manual_seed(40 + i)
        adapted = peft.get_peft_model(copy.deepcopy(base), cfg)
        adapted.peft_config["default"].base_model_name_or_path = str(path)
        with torch.no_grad():
            for name, value in adapted.named_parameters():
                if "lora_" in name:
                    value.uniform_(-0.1, 0.1)
        directory = tmp_path / f"adapter-{i}"
        adapted.save_pretrained(directory, safe_serialization=True)
        adapters.append(directory)
    oracle = peft.PeftModel.from_pretrained(copy.deepcopy(base), adapters[0]).eval()
    oracle.load_adapter(adapters[1], adapter_name="second")
    oracle.eval()
    return SimpleNamespace(
        path=path, base=base, oracle=oracle, tokenizer=tokenizer, adapters=adapters
    )


def torch_logits(model, ids):
    """Run the independent Qwen3 implementation with explicitly bidirectional attention."""
    x = torch.tensor(ids)
    with torch.inference_mode():
        return model(
            x, attention_mask=torch.ones((x.shape[0], 1, 1, x.shape[1]), dtype=torch.bool)
        ).logits


def test_peft_parity_and_switching(bundle):
    """Nonzero LoRA/rsLoRA must match PEFT and disabling must recover exact MLX base logits."""
    engine = Diffcider.from_pretrained(
        bundle.path, adapters=dict(zip(("first", "second"), bundle.adapters, strict=True))
    )
    ids = [[7, 8, 1, 1]]
    base = np.asarray(engine.model(mx.array(ids)))
    for name in ("first", None, "second", "first", None):
        with engine._request(name):
            actual = np.asarray(engine.model(mx.array(ids)))
        if name is None:
            np.testing.assert_array_equal(actual, base)
            with bundle.oracle.disable_adapter():
                expected = torch_logits(bundle.oracle, ids)
        else:
            bundle.oracle.set_adapter("default" if name == "first" else "second")
            bundle.oracle.eval()
            expected = torch_logits(bundle.oracle, ids)
            assert np.max(np.abs(actual - base)) > 1e-4
        np.testing.assert_allclose(actual, expected.numpy(), atol=3e-5, rtol=3e-4)
        assert all(layer.active is None for layer in engine._adapters.layers.values())


def test_alternating_decision_and_generation(bundle):
    """Exercise the public engine against PEFT scoring and the original diffusion sampler."""
    engine = Diffcider.from_pretrained(bundle.path, adapters={"first": bundle.adapters[0]})
    question = {"type": "noul", "instructions": "Is this good?", "criteria": "Good"}
    encoded = decision_input(bundle.tokenizer, "hello world", question)
    for name in ("first", None, "first"):
        context = bundle.oracle.disable_adapter() if name is None else torch.inference_mode()
        with context:
            expected = torch_logits(bundle.oracle, [encoded.input_ids])[0, encoded.positions][
                :, encoded.answer_ids
            ]
            output = engine.decide("hello world", question, adapter=name)
            np.testing.assert_allclose(
                output["probabilities"], probabilities(expected.numpy(), "noul"), atol=2e-5
            )

            class Bidirectional:
                device = "cpu"

                def __call__(self, x):
                    return SimpleNamespace(logits=torch_logits(bundle.oracle, x.tolist()))

            ids = generation_input(bundle.tokenizer, "hello world")
            expected_ids = reference_sampler.generate(
                Bidirectional(),
                torch.tensor([ids]),
                torch.tensor([len(ids)]),
                pad_id=0,
                max_new_tokens=4,
                steps=4,
                block_size=2,
                remasking="low_confidence",
                tokenizer=bundle.tokenizer,
            )[0, len(ids) :].tolist()
            actual_ids = engine.generate_tokens(
                "hello world", adapter=name, max_new_tokens=4, steps=4, block_size=2
            )
            assert actual_ids == expected_ids
            assert engine.generate(
                "hello world", adapter=name, max_new_tokens=4, steps=4, block_size=2
            ) == bundle.tokenizer.decode(actual_ids, skip_special_tokens=True)


def test_failure_and_concurrent_requests_restore_base(bundle):
    """Failed or overlapping requests must not leave an adapter enabled for another request."""
    engine = Diffcider.from_pretrained(bundle.path, adapters={"task": bundle.adapters[0]})
    original = np.asarray(engine.model(mx.array([[7, 1, 8]])))
    with pytest.raises(ValueError):
        engine.generate("hello", adapter="task", steps=3, max_new_tokens=4, block_size=2)
    with pytest.raises(ValueError, match="Unknown adapter"):
        engine.generate("hello", adapter="missing")
    np.testing.assert_array_equal(np.asarray(engine.model(mx.array([[7, 1, 8]]))), original)

    def run(name):
        return engine.generate_tokens(
            "hello world", adapter=name, max_new_tokens=4, steps=4, block_size=2
        )

    expected = {name: run(name) for name in (None, "task")}
    names = ["task", None, "task", None]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, names)) == [expected[name] for name in names]


@pytest.mark.parametrize(
    "option,value",
    [
        ("bias", "all"),
        ("use_dora", True),
        ("modules_to_save", ["lm_head"]),
        ("target_modules", ["lm_head"]),
        ("fan_in_fan_out", True),
        ("init_lora_weights", "pissa"),
        ("future_variant", True),
    ],
)
def test_unsupported_adapters_fail_atomically(bundle, option, value):
    """Unsupported PEFT behavior must fail before modifying the model."""
    path = bundle.adapters[0]
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config[option] = value
    config_path.write_text(json.dumps(config))
    engine = Diffcider.from_pretrained(bundle.path)
    with pytest.raises(ValueError):
        engine.load_adapter(path)
    assert engine.loaded_adapters == ()
    assert not isinstance(engine.model.model.layers[0].self_attn.q_proj, LoRALinear)


def test_wrong_base_and_tokenizer_rejected(bundle):
    """Metadata overrides cannot bypass tensor or tokenizer compatibility checks."""
    path = bundle.adapters[0]
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "another/base"
    config_path.write_text(json.dumps(config))
    engine = Diffcider.from_pretrained(bundle.path)
    with pytest.raises(ValueError, match="declares base"):
        engine.load_adapter(path)
    engine.load_adapter(path, name="verified", allow_base_mismatch=True)
    tokenizer = copy.deepcopy(bundle.tokenizer)
    tokenizer.add_tokens(["new-token"])
    tokenizer.save_pretrained(path)
    with pytest.raises(ValueError, match="tokenizer"):
        engine.load_adapter(path, name="bad", allow_base_mismatch=True)
    assert engine.loaded_adapters == ("verified",)


def test_missing_and_wrong_adapter_tensors_fail_atomically(bundle):
    """Reject corrupted exports instead of silently loading a partial adapter."""
    path = bundle.adapters[0]
    weights_path = path / "adapter_model.safetensors"
    weights = mx.load(str(weights_path))
    key = next(k for k in weights if k.endswith("lora_B.weight"))
    weights[key] = mx.zeros((1, 1))
    mx.save_safetensors(str(weights_path), weights)
    engine = Diffcider.from_pretrained(bundle.path)
    with pytest.raises(ValueError, match="shape"):
        engine.load_adapter(path)
    assert engine.loaded_adapters == ()
    weights.pop(key)
    mx.save_safetensors(str(weights_path), weights)
    with pytest.raises(ValueError, match="Both LoRA"):
        engine.load_adapter(path)
    assert engine.loaded_adapters == ()


def test_custom_hub_model_and_adapter_one_call(bundle, monkeypatch):
    """Exercise resolver arguments without downloading fixtures or trusting remote code."""
    calls = []
    config_path = bundle.adapters[0] / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = "test/custom-model"
    config_path.write_text(json.dumps(config))

    def snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(bundle.adapters[0] if repo == "test/custom-adapter" else bundle.path)

    monkeypatch.setattr("diffcider.model.snapshot_download", snapshot)
    engine = Diffcider.from_pretrained(
        "test/custom-model",
        revision="base-revision",
        adapters={"task": "test/custom-adapter"},
        adapter_revisions={"task": "adapter-revision"},
        local_files_only=True,
    )
    assert engine.loaded_adapters == ("task",)
    assert [(repo, kwargs["revision"], kwargs["local_files_only"]) for repo, kwargs in calls] == [
        ("test/custom-model", "base-revision", True),
        ("test/custom-adapter", "adapter-revision", True),
    ]
    calls.clear()
    checkpoint_path("base", local_files_only=True)
    assert calls[0][0] == CHECKPOINTS["base"][0]
    assert calls[0][1]["revision"] == CHECKPOINTS["base"][1]


def test_merged_and_sharded_full_checkpoint(bundle, tmp_path):
    """Merged PEFT weights load as a full model, including HF's sharded safetensors format."""
    bundle.oracle.set_adapter("default")
    merged = bundle.oracle.merge_and_unload().eval()
    path = tmp_path / "merged"
    merged.save_pretrained(path, safe_serialization=True, max_shard_size="5KB")
    config = json.loads((path / "config.json").read_text())
    config["model_type"] = "a2d-qwen3"
    (path / "config.json").write_text(json.dumps(config))
    bundle.tokenizer.save_pretrained(path)
    model, _ = load_model(path)
    np.testing.assert_allclose(
        np.asarray(model(mx.array([[7, 1, 8]]))),
        torch_logits(merged, [[7, 1, 8]]).numpy(),
        atol=3e-5,
        rtol=3e-4,
    )
    assert (path / "model.safetensors.index.json").is_file()


def test_missing_local_path_is_not_treated_as_hub_id(tmp_path):
    """Fail locally for explicit paths and malformed local revisions."""
    with pytest.raises(ValueError, match="does not exist"):
        checkpoint_path(tmp_path / "missing")
    with pytest.raises(ValueError, match="local directories"):
        checkpoint_path(tmp_path, revision="main")


@pytest.mark.parametrize(
    "selection",
    [
        {
            "target_modules": ["q_proj", "v_proj"],
            "layers_to_transform": [1],
            "layers_pattern": "layers",
        },
        {"target_modules": r"model\.layers\.0\.self_attn\.(q_proj|v_proj)"},
        {"target_modules": "all-linear", "exclude_modules": ["k_proj"]},
        {
            "target_modules": ["model.layers.0.self_attn.q_proj"],
            "layers_to_transform": [1],
        },
    ],
)
def test_peft_target_selection(bundle, tmp_path, selection):
    """Load real exports for regex, layer-filtered, and all-linear PEFT targeting."""
    oracle = peft.get_peft_model(
        copy.deepcopy(bundle.base), peft.LoraConfig(r=2, lora_alpha=3, **selection)
    )
    oracle.peft_config["default"].base_model_name_or_path = str(bundle.path)
    with torch.no_grad():
        for name, param in oracle.named_parameters():
            if "lora_" in name:
                param.uniform_(-0.1, 0.1)
    oracle.eval()
    path = tmp_path / "selection"
    oracle.save_pretrained(path, safe_serialization=True)
    engine = Diffcider.from_pretrained(bundle.path, adapters={"selected": path})
    with engine._request("selected"):
        actual = np.asarray(engine.model(mx.array([[7, 8, 1, 1]])))
    np.testing.assert_allclose(
        actual, torch_logits(oracle, [[7, 8, 1, 1]]).numpy(), atol=3e-5, rtol=3e-4
    )


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_lora_precision_matches_peft(dtype):
    """Match PEFT's float32 adapters and cast-back behavior on reduced-precision bases."""
    torch.manual_seed(4)
    base = torch.nn.Sequential(torch.nn.Linear(8, 5, bias=False)).to(getattr(torch, dtype))
    oracle = peft.get_peft_model(
        base, peft.LoraConfig(r=2, lora_alpha=3, target_modules=["0"])
    ).eval()
    layer = oracle.get_base_model()[0]
    with torch.no_grad():
        layer.lora_B["default"].weight.uniform_(-0.1, 0.1)
    mlx_base = nn.Linear(8, 5, bias=False)
    mlx_base.weight = mx.array(layer.base_layer.weight.float().detach().numpy()).astype(
        getattr(mx, dtype)
    )
    wrapped = LoRALinear(mlx_base)
    wrapped.adapters["test"] = {
        "a": mx.array(layer.lora_A["default"].weight.detach().float().numpy()),
        "b": mx.array(layer.lora_B["default"].weight.detach().float().numpy()),
        "scale": 1.5,
    }
    x = np.random.default_rng(9).normal(size=(3, 8)).astype(np.float32)
    mx_x = mx.array(x).astype(getattr(mx, dtype))
    wrapped.active = "test"
    actual = wrapped(mx_x)
    with torch.inference_mode():
        expected = oracle(torch.tensor(x).to(getattr(torch, dtype))).float().numpy()
    assert actual.dtype == getattr(mx, dtype)
    np.testing.assert_allclose(
        np.asarray(actual.astype(mx.float32)),
        expected,
        atol=0.008 if dtype == "bfloat16" else 0.001 if dtype == "float16" else 1e-6,
        rtol=0,
    )
    wrapped.active = None
    np.testing.assert_array_equal(
        np.asarray(wrapped(mx_x).astype(mx.float32)), np.asarray(mlx_base(mx_x).astype(mx.float32))
    )


def test_base_revision_and_duplicate_adapter_checks(bundle):
    """Reject known revision mismatches and duplicate names before modifying loaded adapters."""
    engine = Diffcider.from_pretrained(bundle.path)
    engine.source = "test/model"
    engine.revision = "a" * 40
    config_path = bundle.adapters[0] / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path="test/model", revision="b" * 40)
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="revision"):
        engine.load_adapter(bundle.adapters[0], local_files_only=True)
    assert engine.loaded_adapters == ()
    config["revision"] = engine.revision
    config_path.write_text(json.dumps(config))
    engine.load_adapter(bundle.adapters[0], local_files_only=True)
    with pytest.raises(ValueError, match="unique"):
        engine.load_adapter(bundle.adapters[0])
    assert engine.loaded_adapters == ("default",)
