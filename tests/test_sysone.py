"""Saved-head arithmetic, calibration, prompt boundaries and sysone sampling."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from diffcider.sysone import (
    KINDS,
    ROW,
    SysoneDiffcider,
    SysoneInput,
    answer,
    sysone_input,
    validate_metadata,
)


class Tokenizer:
    mask_token_id, pad_token_id, eos_token_id = 1, 2, 3

    def encode(self, text, **kwargs):
        return [ord(c) + 10 for c in text]

    def apply_chat_template(self, messages, **kwargs):
        return "".join(m["content"] for m in messages)

    def decode(self, tokens, **kwargs):
        return ",".join(map(str, tokens))


def template():
    return {
        "row": ROW,
        "kinds": KINDS,
        "mask": "<|mask|>",
        "truncate": "right",
        "max_len": 4096,
        "head_max_len": 2048,
        "option_tokens": 512,
    }


def test_row_preserves_anchors_and_literal_tails_when_truncated():
    tok, t = Tokenizer(), template()
    t.update(max_len=500, head_max_len=180, option_tokens=64)
    q = {
        "type": "choice",
        "instructions": "question " * 100,
        "criteria": {"a": "option " * 100, "b": "<|mask|> other " * 100},
    }
    row = sysone_input(tok, t, "state " * 500, q)
    assert len(row.input_ids) <= 500
    assert [i for i, tid in enumerate(row.input_ids) if tid == 1] == row.positions
    for at in row.positions:
        assert row.input_ids[at - 2 : at] == tok.encode(": ")
    assert row.input_ids[-len(tok.encode("<|im_end|>\n")) :] == tok.encode("<|im_end|>\n")
    binary = sysone_input(tok, t, "state", {"type": "noul", "instructions": "Ready?"})
    assert binary.positions[0] == binary.positions[1]
    assert binary.input_ids.count(1) == 1


def test_calibration_and_binary_answer_order():
    row = SysoneInput([], [], "choice", ["a", "b"], {})
    got = answer(row, [0.0, 2.0], {"choice": 1.0, "choice:2": 2.0})
    assert got["choice"] == "b"
    assert got["probabilities"]["b"] == pytest.approx(1 / (1 + np.exp(-1)))
    binary = SysoneInput([], [], "noul", ["false", "true"], None)
    assert answer(binary, [0.0, -2.0], {})["noul"] < 0.5
    assert answer(binary, [0.0, 2.0], {})["noul"] > 0.5
    with pytest.raises(ValueError, match="Non-finite"):
        answer(row, [0.0, np.nan], {})


def test_saved_head_uses_delta_scale_and_type_embedding_then_restores_adapter():
    class Engine:
        tokenizer = Tokenizer()
        active = None
        model = SimpleNamespace(
            args=SimpleNamespace(max_position_embeddings=4096),
            model=lambda ids, attention_mask: mx.stack([ids, ids + 1], axis=-1),
        )

        @contextmanager
        def _request(self, name):
            assert name == "decision"
            self.active = name
            try:
                yield
            finally:
                self.active = None

    head = {
        "scorer.w0": mx.array([1.0, 2.0]),
        "scorer.delta": mx.array([3.0, -1.0]),
        "scorer.log_scale": mx.array(np.log(2)),
        "type_emb.weight": mx.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
    }
    model = SysoneDiffcider(Engine(), {"template": template()}, head, Path("."))
    rows = [
        SysoneInput([1, 2, 3], [1, 2], "choice", ["a", "b"], {}),
        SysoneInput([7, 8], [1, 1], "noul", ["false", "true"], None),
    ]
    got = model.logits(rows)
    np.testing.assert_allclose(got[0], [2 * (4 * 12 + 23), 2 * (4 * 13 + 24)])
    np.testing.assert_allclose(got[1], [0, 2 * (4 * 58 + 69)])
    assert model.engine.active is None
    with pytest.raises(ValueError, match="nonempty"):
        model.logits([])


def test_sysone_sampler_suppresses_mask_preserves_prompt_and_stops_decode():
    calls = []

    class Network:
        args = SimpleNamespace(max_position_embeddings=1000)

        def __call__(self, ids, *, logit_positions):
            calls.append(ids.tolist())
            # Mask is the largest logit; it must never become a generated answer.
            logits = mx.zeros((1, logit_positions.stop - logit_positions.start, 8))
            logits[..., 1] = 100
            logits[..., 3] = 10  # EOS
            return logits

    class Engine:
        tokenizer = Tokenizer()
        model = Network()
        active = None

        @contextmanager
        def _request(self, name):
            self.active = name
            try:
                yield
            finally:
                self.active = None

    model = SysoneDiffcider(Engine(), {}, {}, Path("."))
    ids = model.generate_tokens("hi", max_new_tokens=13, steps=7, block_size=8)
    assert ids == [3] * 13
    assert len(calls) == 6  # floor(7 / 2) steps per block
    assert all(c[0][:2] == Tokenizer().encode("hi") for c in calls)
    assert model.engine.active is None
    assert model.generate("hi", max_new_tokens=2, steps=2) == ""
    calls.clear()
    assert model.generate_tokens("hi", max_new_tokens=24, steps=24, block_size=32) == [3] * 24
    assert len(calls) == 24  # Exactly 24 passes, even when every generated token is EOS.


def test_unsupported_metadata_fails_explicitly():
    meta = {
        "encoder": {"family": "a2d-qwen3", "weights": "adapter"},
        "head": {
            "layers": 0,
            "scorer": "yesno",
            "standardize": False,
            "readout": "anchor",
            "query": None,
            "act": False,
        },
        "template": template(),
    }
    validate_metadata(meta)
    for key, value in [("layers", 1), ("scorer", "mlp"), ("act", True)]:
        bad = copy.deepcopy(meta)
        bad["head"][key] = value
        with pytest.raises(ValueError):
            validate_metadata(bad)
    meta["tfms"] = [{"kind": "unknown"}]
    with pytest.raises(ValueError, match="Transforms"):
        validate_metadata(meta)


def test_saved_reference_rows_match_current_renderer():
    """Compare tokenization to independent sysone rows when the tokenizer is cached."""
    from huggingface_hub.errors import LocalEntryNotFoundError
    from transformers import AutoTokenizer

    from diffcider.sysone import BROWSER_MODEL, bundle_path

    try:
        path = bundle_path(BROWSER_MODEL, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("Run the browser comparison once to cache its tokenizer.")
    tok = AutoTokenizer.from_pretrained(path / "tokenizer", local_files_only=True)
    meta = json.loads((path / "sysone.json").read_text())
    root = Path(__file__).parents[1]
    cases = json.loads((root / "examples/browser_cases.json").read_text())
    oracle = json.loads((root / "reports/browser-torch-fp32.json").read_text())
    for group, expected in zip(cases["decisions"], oracle["decisions"], strict=True):
        rows = [
            sysone_input(tok, meta["template"], item["state"], item["question"])
            for item in group["requests"]
        ]
        assert [row.input_ids for row in rows] == expected["input_ids"]
        assert [row.positions for row in rows] == expected["positions"]


def test_bundle_checksums_reject_corruption_and_path_escape(tmp_path):
    from diffcider.export_sysone import sha256, verify_bundle

    (tmp_path / "weights").write_bytes(b"weights")
    manifest = {
        "format": "diffcider-sysone-v1",
        "sha256": {"weights": sha256(tmp_path / "weights")},
    }
    (tmp_path / "mlx_bundle.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path) == manifest
    (tmp_path / "weights").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)
    manifest["sha256"] = {"../outside": "hash"}
    (tmp_path / "mlx_bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(tmp_path)
