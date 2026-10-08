"""Answer mapping, prompt information retention and reference provenance checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from diffcider.alodlm_decisions import compile_read, interpret


class Tokenizer:
    def encode(self, text, **kwargs):
        # Single character labels, with single-token Yes/No, including in context.
        text = text.replace("Yes", "\u00c8").replace("No", "\u00c9")
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        self.prompt = messages[0]["content"]
        return self.encode(self.prompt)


def test_readouts_and_binary_order():
    q = [
        {"name": "intent", "options": ["a", "b", "c"]},
        {"name": "urgent", "type": "noul"},
        {"name": "progress", "type": "score", "levels": [0, 2, 4]},
    ]
    letters = compile_read(Tokenizer(), "state", q)
    result = interpret(letters, np.array([[0, 2, 0], [0, 2, 0], [0, 0, 2]]))
    assert result["intent"]["answer"] == "b"
    assert result["urgent"]["answer"] is True
    assert result["progress"]["answer"] == 4
    assert 3 < result["progress"]["expected_value"] < 4
    yn = compile_read(Tokenizer(), "state", q, readout="yesno")
    result = interpret(yn, np.array([[0, 0], [0, 2], [0, 0], [0, 2], [0, 0], [0, 0], [0, 2]]))
    assert result["intent"]["answer"] == "b"
    assert result["urgent"]["answer"] is True
    assert list(result["urgent"]["probabilities"]) == ["No", "Yes"]
    assert result["urgent"]["probabilities"]["Yes"] == pytest.approx(1 / (1 + np.exp(-2)))


def test_compiler_keeps_rubrics_and_uses_real_mask_ids():
    tokenizer = Tokenizer()
    q = [
        {
            "name": "ready",
            "type": "score",
            "levels": [0, 1],
            "question": "Which requirement holds?",
            "descriptions": ["Failed inspection", "Passed inspection"],
        }
    ]
    compiled = compile_read(tokenizer, {"inspection": "failed"}, q, mask_token_id=12345)
    assert compiled.tail_ids.count(12345) == 1
    assert "Failed inspection" in tokenizer.prompt
    assert '"inspection": "failed"' in tokenizer.prompt
    assert "Which requirement holds?" in tokenizer.prompt


def test_validation():
    tokenizer = Tokenizer()
    question = {"name": "q", "options": ["a", "b"]}
    for questions in (
        [],
        [question, question],
        [{"name": "q", "options": ["a", "a"]}],
        [{"name": "q", "type": "score", "levels": [0, float("nan")]}],
    ):
        with pytest.raises(ValueError):
            compile_read(tokenizer, "state", questions)
    with pytest.raises(ValueError, match="32"):
        compile_read(tokenizer, "state", [{**question, "name": str(i)} for i in range(10)])
    with pytest.raises(ValueError, match="Descriptions"):
        compile_read(tokenizer, "state", [{**question, "descriptions": ["one"]}])


def test_vendored_reference_is_byte_identical_to_pinned_source():
    root = Path(__file__).parents[1] / "diffcider/_vendor/alodlm"
    manifest = json.loads((root / "UPSTREAM.json").read_text())
    assert manifest["revision"] == "1da9ddafeca02425480230b9bb7091fdef5672b7"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
