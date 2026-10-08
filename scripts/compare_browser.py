"""Independent sysone/MPS and MLX runs; use separate environments, sequentially."""

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
    parser.add_argument("--cases", default="examples/browser_cases.json")
    parser.add_argument("--model", default="sgaseretto/diffcider-browser")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    cases = json.loads(Path(args.cases).read_text())
    report = {
        "backend": args.backend,
        "platform": platform.platform(),
        "repeats": args.repeats,
        "source_revision": "f8059b1a532f42ce62179e13cf0a0d61a5c626ea",
        "dtype": "float32",
        "sysone_revision": "55a8c9a38d1bdf8f42394295016b8e75a1277aff",
        "decisions": [],
        "generation": [],
    }
    if args.backend == "torch":
        import torch
        from sysone.core import Question, fill_answer
        from sysone.inference import Decider, adapters_off
        from sysone.mdlm import mdlm_sample

        model = Decider.load(
            args.model, revision=report["source_revision"], device="mps", digits=None
        )
        report["versions"] = {
            n: importlib.metadata.version(n) for n in ["torch", "transformers", "peft", "sysone"]
        }
        sync = torch.mps.synchronize
        clear = torch.mps.empty_cache

        def encode(item):
            return model.builder.build(item["state"], item["question"])

        def token_ids(row):
            return list(row.ids)

        def positions(row):
            return list(row.markers)

        def score(rows):
            return model._scores(rows)[0]

        def answers(items, rows, z):
            return [
                dict(fill_answer(q, model._softmax(q, v), digits=None))
                for q, v in zip([Question.from_dict(i["question"]) for i in items], z, strict=True)
            ]

        def generate(item):
            ids = model._prompt_ids(item["prompt"], item.get("system"))
            enc = model.model.encoder
            with adapters_off(enc), torch.inference_mode():
                result = mdlm_sample(
                    enc.get_base_model().config,
                    enc,
                    torch.tensor([ids], device="mps"),
                    tokenizer=model.builder.tok,
                    **item["settings"],
                )
            return result[0, len(ids) :].tolist()

        tok = model.builder.tok
    else:
        import mlx.core as mx

        from diffcider import SysoneDiffcider
        from diffcider.sysone import answer

        mx.set_cache_limit(1024**3)
        model = SysoneDiffcider.from_pretrained(args.model)
        report["versions"] = {n: importlib.metadata.version(n) for n in ["mlx", "transformers"]}
        sync, clear = mx.synchronize, mx.clear_cache

        def encode(item):
            return model.encode(item["state"], item["question"])

        def token_ids(row):
            return row.input_ids

        def positions(row):
            return row.positions

        score = model.logits

        def answers(items, rows, z):
            return [
                answer(r, v, model.metadata["temperature"]) for r, v in zip(rows, z, strict=True)
            ]

        def generate(item):
            return model.generate_tokens(
                item["prompt"], system=item.get("system"), **item["settings"]
            )

        tok = model.tokenizer

    def timed(fn):
        fn()  # warmup; allocator cleanup always outside the timer
        times = []
        result = None
        for _ in range(args.repeats):
            sync()
            clear()
            start = time.perf_counter()
            result = fn()
            sync()
            times.append(time.perf_counter() - start)
        return result, times

    for group in cases["decisions"]:
        items = group["requests"]
        rows = [encode(i) for i in items]
        logits, times = timed(lambda rows=rows: score(rows))
        count = [len(positions(r)) for r in rows]
        entry = {
            "name": group["name"],
            "input_ids": [token_ids(r) for r in rows],
            "positions": [positions(r) for r in rows],
            "logits": [np.asarray(v)[:k].tolist() for v, k in zip(logits, count, strict=True)],
            "answers": answers(items, rows, logits),
            "seconds": times,
        }
        report["decisions"].append(entry)
        print(
            group["name"],
            [a.get("choice", a.get("score", a.get("noul"))) for a in entry["answers"]],
            times,
            flush=True,
        )
    for item in cases["generation"]:
        tokens, times = timed(lambda item=item: generate(item))
        stop = next(
            (i for i, t in enumerate(tokens) if t in (tok.eos_token_id, tok.pad_token_id)),
            len(tokens),
        )
        entry = {
            **item,
            "tokens": tokens,
            "text": tok.decode(tokens[:stop], skip_special_tokens=True),
            "seconds": times,
        }
        report["generation"].append(entry)
        print(item["name"], entry["text"], times, flush=True)
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())
        checks = []
        for got, expected in zip(report["decisions"], ref["decisions"], strict=True):
            errors, picks = [], []
            for a, b in zip(got["answers"], expected["answers"], strict=True):
                ap, bp = (
                    a.get("probabilities", {"true": a.get("noul")}),
                    b.get("probabilities", {"true": b.get("noul")}),
                )
                errors.append(max(abs(ap[k] - bp[k]) for k in ap))
                picks.append(
                    a.get("choice", np.argmax(list(ap.values())))
                    == b.get("choice", np.argmax(list(bp.values())))
                )
            checks.append(
                {
                    "name": got["name"],
                    "tokens_equal": got["input_ids"] == expected["input_ids"],
                    "positions_equal": got["positions"] == expected["positions"],
                    "max_probability_error": max(errors),
                    "answers_equal": all(picks),
                    "speedup": float(np.median(expected["seconds"]) / np.median(got["seconds"])),
                }
            )
        report["comparison"] = {
            "decisions": checks,
            "generation": [
                {
                    "name": a["name"],
                    "tokens_equal": a["tokens"] == b["tokens"],
                    "text_equal": a["text"] == b["text"],
                    "speedup": float(np.median(b["seconds"]) / np.median(a["seconds"])),
                }
                for a, b in zip(report["generation"], ref["generation"], strict=True)
            ],
        }
        print(json.dumps(report["comparison"], indent=2))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
