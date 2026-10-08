"""Experimental answer-slot formats for zero-shot ALoDLM decisions."""

import json
import string
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CompiledRead:
    """Concrete tokens and slot mappings shared by independent inference backends."""

    prefix_ids: list[int]
    tail_ids: list[int]
    label_ids: list[int]
    questions: list[dict]
    readout: str


def compile_read(tokenizer, state, questions, *, readout="letters", mask_token_id=151665):
    """Compile typed questions into one observed prefix and a short masked tail.

    Args:
        tokenizer: Original checkpoint tokenizer, including its chat template.
        state: Text to classify or structured input serialized as JSON.
        questions: List of named choice/noul/score dictionaries. Choice uses an
            ``options`` list; score uses numeric ``levels``; noul is No/Yes.
        readout: ``letters`` (one slot per question) or ``yesno`` (one per option).
        mask_token_id: Mask ID from the checkpoint's recurrent metadata.

    Returns:
        Tokens and exact mappings; raises if the tail exceeds 32 tokens.
    """
    if readout not in ("letters", "yesno"):
        raise ValueError("readout must be letters or yesno")
    if not questions:
        raise ValueError("At least one question is required")
    names, normalized = set(), []
    for question in questions:
        name, kind = question.get("name"), question.get("type", "choice")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Questions require unique, nonempty names")
        names.add(name)
        if kind == "noul":
            options = ["No", "Yes"]
        elif kind == "score":
            options = question.get("levels", [])
            if any(
                isinstance(x, bool) or not isinstance(x, (float, int)) or not np.isfinite(x)
                for x in options
            ):
                raise ValueError("Score levels must be finite numbers")
        elif kind == "choice":
            options = question.get("options", [])
            if any(not isinstance(x, str) or not x for x in options):
                raise ValueError("Choice options must be nonempty strings")
        else:
            raise ValueError("Question type must be choice, noul or score")
        if not 2 <= len(options) <= 26 or len(set(options)) != len(options):
            raise ValueError("Questions require 2–26 distinct options/levels")
        normalized.append(
            {
                "name": name,
                "type": kind,
                "question": question.get("question", name),
                "options": list(options),
                "descriptions": question.get("descriptions", [""] * len(options)),
            }
        )
        if len(normalized[-1]["descriptions"]) != len(options):
            raise ValueError("Descriptions must align with the answer options")
    labels = (
        list(string.ascii_uppercase[: max(len(q["options"]) for q in normalized)])
        if readout == "letters"
        else ["No", "Yes"]
    )
    ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
    if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(ids):
        raise ValueError("Answer labels must be distinct single tokens")
    label_ids = [x[0] for x in ids]
    if mask_token_id in label_ids:
        raise ValueError("A mask token cannot also be an answer label")
    lines = ["Read the state and answer every question using only the specified answer slots."]
    if readout == "letters":
        lines.append(
            "For each question, put the letter of the single best option in its answer slot."
        )
    else:
        lines.append(
            "For each question, each answer slot refers to an option in the listed order. Put Yes if that option is correct, otherwise No."
        )
    for i, q in enumerate(normalized, 1):
        lines.append(f"Question {i}: {q['question']}")
        if readout == "yesno" and q["type"] == "noul":
            lines.append("This is a binary question: its single slot is Yes if true, otherwise No.")
            lines.extend(
                f"{option}: {description}"
                for option, description in zip(q["options"], q["descriptions"], strict=True)
                if description
            )
        else:
            lines.extend(
                f"{string.ascii_uppercase[j]}. {option}"
                + (f": {q['descriptions'][j]}" if q["descriptions"][j] else "")
                for j, option in enumerate(q["options"])
            )
    lines.append(
        "State:\n" + (state if isinstance(state, str) else json.dumps(state, ensure_ascii=False))
    )
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines)}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix += tokenizer.encode("Answers:\n", add_special_tokens=False)
    tail = []
    for i, q in enumerate(normalized, 1):
        anchor = tokenizer.encode(f"{i}:\n", add_special_tokens=False)
        for label, label_id in zip(labels, label_ids, strict=True):
            if tokenizer.encode(f"{i}:\n{label}\n", add_special_tokens=False) != anchor + [
                label_id
            ] + tokenizer.encode("\n", add_special_tokens=False):
                raise ValueError("Answer label tokenization changes at the template boundary")
        tail += anchor
        # A noul question in yesno mode needs only the Yes-vs-No read for its proposition.
        slots = 1 if readout == "letters" or q["type"] == "noul" else len(q["options"])
        tail += [mask_token_id] * slots
        tail += tokenizer.encode("\n", add_special_tokens=False)
    if len(tail) > 32:
        raise ValueError("Questions require more than 32 tail tokens; use smaller groups")
    return CompiledRead(prefix, tail, label_ids, normalized, readout)


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    probs = np.exp(values - values.max())
    return probs / probs.sum()


def interpret(compiled, logits):
    """Map slot logits to uncalibrated choices, binary probabilities and scores."""
    result, row = {}, 0
    for q in compiled.questions:
        options = q["options"]
        if compiled.readout == "letters":
            probs = _softmax(logits[row, : len(options)])
            row += 1
        elif q["type"] == "noul":
            probs = _softmax(logits[row])  # [No, Yes]; equals sigmoid(Yes - No).
            row += 1
        else:
            group = logits[row : row + len(options)]
            probs = _softmax(group[:, 1] - group[:, 0])
            row += len(options)
        best = int(probs.argmax())
        answer = bool(best) if q["type"] == "noul" else options[best]
        entry = {
            "answer": answer,
            "probabilities": {str(k): float(v) for k, v in zip(options, probs, strict=True)},
        }
        if q["type"] == "score":
            entry["expected_value"] = float(np.dot(np.asarray(options), probs))
        result[q["name"]] = entry
    if row != len(logits):
        raise ValueError("Logit rows do not match the compiled answer slots")
    return result
