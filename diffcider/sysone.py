"""MLX inference for the text-only, zero-layer Yes/No sysone export format.

Prompt budgeting, the trained head and sampling follow sysonelib (MIT).
See THIRD_PARTY.md for the exact source revision and attribution.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .engine import Diffcider

BROWSER_MODEL = "sgaseretto/diffcider-browser"
BROWSER_REVISION = "f8059b1a532f42ce62179e13cf0a0d61a5c626ea"
QTYPES = {"choice": 0, "score": 1, "noul": 2}
ROW = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nState:\n{state}\n\nQuestion:\n{instructions}\n\n{options}<|im_end|>\n"
)
OPTION = {
    "header": "For each option, mark Yes if it answers the question, otherwise No.\n",
    "option": "{text}: {mask}",
    "sep": "\n",
}
KINDS = {
    "choice": OPTION,
    "score": OPTION,
    "noul": {"header": "Answer Yes or No.\nAnswer:", "option": "{mask}", "sep": ""},
}


def bundle_path(source, revision=None, local_files_only=False):
    """Resolve a local export or download only its declarative files and weights."""
    path = Path(source).expanduser()
    if not path.is_dir():
        path = Path(
            snapshot_download(
                str(source),
                revision=revision or (BROWSER_REVISION if source == BROWSER_MODEL else None),
                local_files_only=local_files_only,
                allow_patterns=[
                    "sysone.json",
                    "mlx_bundle.json",
                    "head.safetensors",
                    "adapter/*",
                    "tokenizer/*",
                    "encoder/*.json",
                    "encoder/*.safetensors",
                    "encoder/*.jinja",
                ],
            )
        )
    if not (path / "sysone.json").is_file():
        raise ValueError("Expected a sysone export with sysone.json.")
    return path


def validate_metadata(meta):
    """Reject architectures or templates whose semantics this port cannot preserve."""
    h, t, enc = meta["head"], meta["template"], meta["encoder"]
    supported = {
        "layers": 0,
        "scorer": "yesno",
        "standardize": False,
        "readout": "anchor",
        "query": None,
        "act": False,
    }
    if any(h.get(k) != v for k, v in supported.items()):
        raise ValueError("Only zero-layer, unstandardized, anchor Yes/No heads are supported.")
    if enc.get("family") != "a2d-qwen3" or enc.get("weights") != "adapter":
        raise ValueError("Expected an a2d-qwen3 encoder with an unmerged adapter.")
    if meta.get("tfms") or meta.get("streams") or meta.get("act_threshold") is not None:
        raise ValueError("Transforms, side streams and action thresholds are unsupported.")
    if (
        t.get("row") != ROW
        or t.get("kinds") != KINDS
        or t.get("mask") != "<|mask|>"
        or t.get("truncate") != "right"
        or any(t.get(k) for k in ("start", "sep", "end", "image"))
    ):
        raise ValueError("Expected the saved text-only Jev template with right truncation.")
    if any(
        type(t.get(k)) is not int or t[k] < 16 for k in ("max_len", "head_max_len", "option_tokens")
    ):
        raise ValueError("Invalid template token budgets.")


def render(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def cut(ids, count, tail):
    return ids if len(ids) <= count else ids[: max(0, count - len(tail))] + tail


@dataclass
class SysoneInput:
    """A row with the saved head's type ID, option keys and anchor locations."""

    input_ids: list[int]
    positions: list[int]
    kind: str
    keys: list
    criteria: object


def sysone_input(tokenizer, template, state, question):
    """Render the saved Jev row, preserving sysone's segment boundaries and budgets.

    Args:
        tokenizer: Export tokenizer with the original mask token.
        template: Validated Jev template metadata.
        state: Text or a JSON-serializable browser state.
        question: A choice, score or noul question in sysone's dictionary format.

    Returns:
        Token IDs, anchors and answer metadata for one decision.
    """
    kind, criteria = question.get("type"), question.get("criteria")
    if kind not in QTYPES or question.get("instructions") is None:
        raise ValueError("A question needs a supported type and instructions.")
    if kind == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("Choice criteria must be a nonempty mapping or list of labels.")
        keys = list(criteria)
        options = [
            str(k) if v is None or v == "" else f"{k}: {render(v)}" for k, v in criteria.items()
        ]
    elif kind == "score":
        if not isinstance(criteria, list) or not criteria or None in criteria:
            raise ValueError("Score criteria must be a nonempty list of non-null levels.")
        keys = [str(i) for i in range(len(criteria))]
        options = [f"level {i}: {render(v)}" for i, v in enumerate(criteria)]
    else:
        if criteria is not None and (
            not isinstance(criteria, dict) or set(criteria) - {"false", "true"}
        ):
            raise ValueError("Noul criteria use only false and true keys.")
        keys, options = ["false", "true"], []

    def scrub(s):
        return render(s).replace(template["mask"], " ")

    def encode(s):
        return tokenizer.encode(s, add_special_tokens=False, truncation=False)

    prefix = encode(ROW.split("{state}")[0])
    suffix = encode("<|im_end|>\n")
    state_tail, option_tail = encode("\n\n"), encode(": ")
    state_ids = encode(scrub(state) + "\n\n")
    instructions = encode("Question:\n" + scrub(question["instructions"]) + "\n\n")
    header = encode(KINDS[kind]["header"])
    sep = encode(KINDS[kind]["sep"])
    chunks = [cut(encode(scrub(o) + ": "), template["option_tokens"], option_tail) for o in options]
    if kind == "noul":
        chunks = [[]]

    def size():
        return len(header) + sum(len(c) + 1 for c in chunks) + len(sep) * (len(chunks) - 1)

    budget = template["head_max_len"] - size()
    if budget < 16:
        per = max(4, (template["head_max_len"] - 16 - len(header)) // len(chunks))
        chunks = [cut(c, max(len(option_tail), per - 1), option_tail) for c in chunks]
        budget = template["head_max_len"] - size()
    instructions = cut(instructions, max(8, budget), state_tail)
    room = max(0, template["max_len"] - len(prefix) - len(instructions) - len(suffix) - size())
    ids = prefix + cut(state_ids, room, state_tail) + instructions + header
    positions = []
    for i, chunk in enumerate(chunks):
        if i:
            ids += sep
        ids += chunk
        positions.append(len(ids))
        ids.append(tokenizer.mask_token_id)
    ids += suffix
    if kind == "noul":
        positions *= 2
    if len(ids) > template["max_len"] or any(p >= len(ids) for p in positions):
        raise ValueError("Options do not fit the saved template's token budget.")
    return SysoneInput(ids, positions, kind, keys, criteria)


def answer(row, logits, temperatures):
    """Calibrate logits and return sysone's answer fields without rounding."""
    count = len(row.keys)
    bucket = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    temp = temperatures.get(f"{row.kind}:{bucket}", temperatures.get(row.kind, 1.0))
    try:
        temp = float(temp)
    except (TypeError, ValueError):
        temp = 1.0
    temp = min(5.0, max(0.5, temp)) if math.isfinite(temp) else 1.0
    z = np.asarray(logits, dtype=np.float64) / temp
    if not np.isfinite(z).all():
        raise ValueError("Non-finite decision logits.")
    p = np.exp(z - z.max())
    p /= p.sum()
    confidence = (
        1.0
        if count == 1
        else float(np.clip(1 + (p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(count), 0, 1))
    )
    result = {"type": row.kind, "confidence": confidence, "answer_confidence": float(p.max())}
    if row.kind == "noul":
        result.update(noul=float(p[1]), confidence=float(max(p)))
    else:
        result["probabilities"] = dict(zip(row.keys, p.tolist(), strict=True))
        if row.kind == "choice":
            result["choice"] = row.keys[int(p.argmax())]
        else:
            result.update(score=float(np.arange(count) @ p), legend=dict(enumerate(row.criteria)))
            result["legend"] = {str(k): v for k, v in result["legend"].items()}
    return result


class SysoneDiffcider:
    """One MLX backbone: trained adapter/head for decisions, base weights for writing."""

    def __init__(self, engine, metadata, head, path):
        self.engine, self.metadata, self.head, self.path = engine, metadata, head, path
        self.tokenizer = engine.tokenizer

    @classmethod
    def from_pretrained(
        cls, model=BROWSER_MODEL, *, revision=None, dtype="float32", local_files_only=False
    ):
        """Load the original sysone export or a standalone diffcider MLX bundle.

        Args:
            model: Local export directory or Hugging Face repository ID.
            revision: Hub revision; the browser model defaults to a pinned commit.
            dtype: Backbone precision. Float32 preserves the training precision.
            local_files_only: Require all files in local directories or Hub cache.

        Returns:
            A resident model with the decision adapter, head and calibration loaded.
        """
        path = bundle_path(model, revision, local_files_only)
        meta = json.loads((path / "sysone.json").read_text())
        validate_metadata(meta)
        ref = meta["encoder"]
        bundled = (path / "mlx_bundle.json").is_file()
        if bundled:
            manifest = json.loads((path / "mlx_bundle.json").read_text())
            if manifest.get("format") != "diffcider-sysone-v1" or manifest.get("encoder") != {
                "id": ref["id"],
                "revision": ref["revision"],
            }:
                raise ValueError("MLX bundle provenance does not match sysone.json.")
            source = path / "encoder"
        else:
            if (path / "encoder").exists():
                raise ValueError("Bundled encoders require a diffcider mlx_bundle.json manifest.")
            source = ref["id"]
        engine = Diffcider.from_pretrained(
            source,
            revision=None if bundled else ref.get("revision"),
            dtype=dtype,
            local_files_only=local_files_only,
        )
        if bundled:
            engine.source, engine.revision = ref["id"], ref["revision"]
        tok = AutoTokenizer.from_pretrained(
            path / "tokenizer", local_files_only=True, trust_remote_code=False
        )
        if tok.get_vocab() != engine.tokenizer.get_vocab() or any(
            getattr(tok, key) != getattr(engine.tokenizer, key)
            for key in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id")
        ):
            raise ValueError("Export tokenizer vocabulary/special tokens differ from the base.")
        engine.tokenizer = tok  # generation uses the export's own chat template
        engine.load_adapter(path / "adapter", name="decision", local_files_only=local_files_only)
        width = engine.model.args.hidden_size
        head = mx.load(str(path / "head.safetensors"))
        expected = {
            "scorer.w0": (width,),
            "scorer.delta": (width,),
            "scorer.log_scale": (),
            "type_emb.weight": (3, width),
        }
        if head.keys() != expected.keys() or meta["head"]["width"] != width:
            raise ValueError("Unexpected decision head tensors or width.")
        for key, shape in expected.items():
            if (
                head[key].shape != shape
                or not mx.issubdtype(head[key].dtype, mx.floating)
                or not mx.all(mx.isfinite(head[key])).item()
            ):
                raise ValueError(f"Invalid decision head tensor: {key}")
            head[key] = head[key].astype(mx.float32)
        mx.eval(head)
        return cls(engine, meta, head, path)

    def encode(self, state, question):
        """Return the saved template's row for inspecting or comparing tokenization."""
        return sysone_input(self.tokenizer, self.metadata["template"], state, question)

    def logits(self, rows):
        """Score encoded rows in one padded forward pass with the trained adapter."""
        if not rows:
            raise ValueError("A decision batch must be nonempty.")
        lengths = [len(r.input_ids) for r in rows]
        width, count = max(lengths), max(len(r.positions) for r in rows)
        if width > self.engine.model.args.max_position_embeddings:
            raise ValueError("Decision exceeds the model context.")
        ids = mx.array(
            [r.input_ids + [self.tokenizer.pad_token_id] * (width - len(r.input_ids)) for r in rows]
        )
        valid = mx.arange(width)[None, :] < mx.array(lengths)[:, None]
        pos = mx.array([r.positions + [0] * (count - len(r.positions)) for r in rows])
        with self.engine._request("decision"):
            h = self.engine.model.model(ids, attention_mask=valid)
            h = mx.take_along_axis(h, pos[..., None], axis=1).astype(mx.float32)
            h = (
                h
                + self.head["type_emb.weight"][mx.array([QTYPES[r.kind] for r in rows])][:, None, :]
            )
            z = mx.sum(h * (self.head["scorer.w0"] + self.head["scorer.delta"]), axis=-1) * mx.exp(
                self.head["scorer.log_scale"]
            )
            mx.eval(z)
            values = np.array(z)
        result = []
        for r, v in zip(rows, values, strict=True):
            result.append(np.array([0.0, v[1]]) if r.kind == "noul" else v[: len(r.keys)])
        return result

    def decide_batch(self, requests):
        """Answer a nonempty list of {state, question} requests in one forward pass."""
        rows = [self.encode(item["state"], item["question"]) for item in requests]
        return [
            answer(r, z, self.metadata.get("temperature") or {})
            for r, z in zip(rows, self.logits(rows), strict=True)
        ]

    def decide(self, state, question):
        """Answer one question with the trained adapter, learned head and calibration."""
        return self.decide_batch([{"state": state, "question": question}])[0]

    def predict(self, state, questions):
        """Answer a mapping of named questions about one state in a single batch."""
        results = self.decide_batch([{"state": state, "question": q} for q in questions.values()])
        return dict(zip(questions, results, strict=True))

    def generation_input(self, prompt, system=None):
        """Apply the saved chat template, with an optional explicit system message."""
        messages = [{"role": "user", "content": prompt}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate_tokens(
        self, prompt, *, system=None, max_new_tokens=16, steps=8, block_size=32, adapter=False
    ):
        """Generate with sysone's greedy, mask-suppressed diffusion schedule.

        Args:
            prompt: User text to complete.
            system: Optional system instruction, e.g. exact form-value output.
            max_new_tokens: Output budget, including EOS and subsequent tokens.
            steps: Requested steps, divided among blocks as in sysone.
            block_size: Maximum generated positions in each block.
            adapter: Explicitly enable the decision adapter; normally keep False.

        Returns:
            All generated token IDs, including the fixed-budget tail after EOS.
        """
        if any(type(n) is not int or n < 1 for n in (max_new_tokens, steps, block_size)):
            raise ValueError("Token budget, steps and block size must be positive integers.")
        ids = self.generation_input(prompt, system)
        mask = self.tokenizer.mask_token_id
        if mask in ids:
            raise ValueError("Generation prompts cannot contain the reserved mask token.")
        if len(ids) + max_new_tokens > self.engine.model.args.max_position_embeddings:
            raise ValueError("Prompt and generation exceed the model context.")
        per = max(1, steps // math.ceil(max_new_tokens / block_size))
        with self.engine._request("decision" if adapter else None):
            x = mx.array([ids + [mask] * max_new_tokens])
            for start in range(len(ids), x.shape[1], block_size):
                end = min(start + block_size, x.shape[1])
                remaining = end - start
                for step in range(per):
                    if not remaining:
                        break
                    z = self.engine.model(x, logit_positions=slice(start, end)).astype(mx.float32)
                    z[:, :, mask] = -mx.inf
                    pick = mx.argmax(z, axis=-1)
                    conf = mx.take_along_axis(mx.softmax(z, axis=-1), pick[..., None], axis=-1)[
                        0, :, 0
                    ]
                    conf = mx.where(x[0, start:end] == mask, conf, -mx.inf)
                    count = math.ceil(remaining / (per - step))
                    chosen = mx.argpartition(-conf, kth=count - 1)[:count] + start
                    x[0, chosen] = pick[0, chosen - start]
                    mx.eval(x)
                    remaining -= count
            return x[0, len(ids) :].tolist()

    def generate(self, prompt, **kwargs):
        """Generate text using generate_tokens options, ending at the first EOS/pad."""
        tokens = self.generate_tokens(prompt, **kwargs)
        stop = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        end = next((i for i, token in enumerate(tokens) if token in stop), len(tokens))
        return self.tokenizer.decode(tokens[:end], skip_special_tokens=True)
