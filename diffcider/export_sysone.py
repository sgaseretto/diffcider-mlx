"""Build a standalone MLX sysone bundle without merging the decision adapter."""

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .model import checkpoint_path, load_model
from .sysone import BROWSER_MODEL, BROWSER_REVISION, bundle_path, validate_metadata


def sha256(path):
    """Hash a file incrementally, including large model shards."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(path):
    """Verify every manifest checksum, rejecting paths outside the bundle."""
    path = Path(path).resolve()
    manifest = json.loads((path / "mlx_bundle.json").read_text())
    if manifest.get("format") != "diffcider-sysone-v1" or not manifest.get("sha256"):
        raise ValueError("Expected a diffcider bundle with file checksums.")
    for name, expected in manifest["sha256"].items():
        target = (path / name).resolve()
        if not target.is_relative_to(path) or not target.is_file() or sha256(target) != expected:
            raise ValueError(f"Bundle checksum failed: {name}")
    return manifest


def base_license():
    """Find the distributed Apache license in an editable or wheel installation."""
    for path in importlib.metadata.files("diffcider-mlx") or []:
        if str(path).endswith("licenses/Apache-2.0.txt"):
            return Path(path.locate()).read_text()
    raise FileNotFoundError("The installation is missing its Apache-2.0 license file.")


def export_bundle(destination, *, source=BROWSER_MODEL, revision=None, dtype="float32"):
    """Write base weights, unmerged LoRA, trained head, tokenizer and provenance.

    Args:
        destination: New output directory; an existing directory is never overwritten.
        source: Local sysone export or Hub model ID.
        revision: Pinned source revision; defaults to the tested browser revision.
        dtype: Stored base precision; float32 preserves the reference runtime values.

    Returns:
        The output path, ready for local inference or a later Hub upload.
    """
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Choose a new export directory: {destination}")
    source_path = bundle_path(source, revision)
    meta = json.loads((source_path / "sysone.json").read_text())
    validate_metadata(meta)
    ref = meta["encoder"]
    base_path = checkpoint_path(ref["id"], revision=ref["revision"])
    network, tokenizer = load_model(base_path, dtype)
    destination.mkdir(parents=True)
    try:
        encoder = destination / "encoder"
        encoder.mkdir()
        mx.save_safetensors(
            str(encoder / "model.safetensors"), dict(tree_flatten(network.parameters()))
        )
        config = json.loads((base_path / "config.json").read_text())
        config.pop("auto_map", None)  # the diffcider architecture is installed, not remote Python
        config["dtype"] = dtype
        (encoder / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        tokenizer.save_pretrained(encoder)
        for folder in ("adapter", "tokenizer"):
            (destination / folder).mkdir()
            for path in (source_path / folder).iterdir():
                if path.suffix in {".json", ".safetensors", ".jinja"}:
                    shutil.copyfile(path, destination / folder / path.name)
        for name in ("head.safetensors", "sysone.json"):
            shutil.copyfile(source_path / name, destination / name)
        (destination / "LICENSE").write_text(base_license())
        manifest = {
            "format": "diffcider-sysone-v1",
            "dtype": dtype,
            "source": {
                "id": str(source),
                "revision": source_path.name
                if source_path.parent.name == "snapshots"
                else revision,
            },
            "encoder": {"id": ref["id"], "revision": ref["revision"]},
            "adapter_merged": False,
            "sha256": {
                str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*"))
                if p.is_file()
            },
        }
        (destination / "mlx_bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (destination / "README.md").write_text(model_card(manifest))
        (destination / ".gitattributes").write_text(
            "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        )
    except BaseException:
        # Only this call's new, incomplete export is removed.
        shutil.rmtree(destination)
        raise
    return destination


def model_card(manifest):
    """Describe the executable format and retain source, precision and usage limits."""
    return f"""---
library_name: diffcider-mlx
license: apache-2.0
base_model:
- {manifest["encoder"]["id"]}
- {manifest["source"]["id"]}
datasets:
- osunlp/Multimodal-Mind2Web
tags:
- mlx
- diffusion
- lora
- browser-agent
- typed-decisions
---

# diffcider-browser-mlx

A standalone **{manifest["dtype"]}** MLX bundle of
[diffcider-browser](https://huggingface.co/{manifest["source"]["id"]}/tree/{manifest["source"]["revision"]}),
for Apple Silicon. The Qwen3 masked diffusion backbone is bidirectional.
Decisions use its unmerged LoRA, learned Yes/No head, question-type embeddings,
and saved calibration. Generation switches that adapter off on the same backbone.
There is no second model or hosted text-generation API.

## Install and use

Requires Apple Silicon, macOS and Python 3.13+.

```sh
uv pip install 'git+https://github.com/sgaseretto/diffcider-mlx.git'
```

```python
from diffcider import SysoneDiffcider
model = SysoneDiffcider.from_pretrained("sgaseretto/diffcider-browser-mlx")
answer = model.decide(
    "The Search field is empty.",
    {{"type": "choice", "instructions": "Search for Ada Lovelace. Choose the next operation.",
     "criteria": {{"TYPE_TEXT": "Enter text into Search", "CLICK": "Click Search"}}}},
)
text = model.generate(
    "The user's goal: Search for Ada Lovelace.\\nThe page: Reading room\\n"
    "The field: Search (textbox)\\nWhat should be typed into this field?",
    system="You fill in web forms for a user. Answer with the exact text to type, nothing else.",
    max_new_tokens=16, steps=8,
)
```

`predict(state, questions)` batches named questions; `decide_batch(requests)`
batches independent states/questions. Answers follow sysone's Jev schema.
`generate_tokens` retains the entire output budget; `generate` decodes only up
through the first EOS/pad. Sampling is greedy, following sysone's mask-suppressed
schedule. The bundle is loaded by **SysoneDiffcider**, not mlx-lm's autoregressive loader.

## Gradio example

```sh
uv pip install 'diffcider-mlx[browser-demo] @ git+https://github.com/sgaseretto/diffcider-mlx.git'
playwright install chromium
diffcider-browser --model sgaseretto/diffcider-browser-mlx
# Share with a password prompt:
diffcider-browser --model sgaseretto/diffcider-browser-mlx --share
```

See [the demo documentation](https://github.com/sgaseretto/diffcider-mlx/blob/main/docs/browser-demo.md)
for the tested examples, timing data and optional public-site mode.

## Contents and provenance

- `encoder/`: native MLX safetensors for the full base model; no quantization or LoRA merge.
- `adapter/`: original unmerged PEFT weights and configuration.
- `head.safetensors`, `sysone.json`, `tokenizer/`: original learned head, calibration and tokenizer.
- `mlx_bundle.json`: source revisions, precision and SHA-256 checksums.
- Base revision: `{manifest["encoder"]["revision"]}`.
- Fine-tuned source revision: `{manifest["source"]["revision"]}`.

The source model was trained with sysone on text representations of Mind2Web
browser steps. Consult the original card for its training metrics; those metrics
are not new MLX evaluations. The conversion preserves the original model's limits:
it does not consume screenshots, has never trained on DONE pages, and can choose
wrong actions or generate wrong field values. Use it for supervised demonstrations.
The Gradio fixture verifies completion independently; arbitrary websites do not.

The base model is Apache-2.0. The original fine-tune card does not declare a separate
license; this derivative retains the base license. Source training-data terms
remain applicable. The runtime is MIT/Apache-2.0; sysone and the adapted Laya browser
snapshot are MIT, attributed in the runtime repository's THIRD_PARTY.md.
"""


def main():
    """Export locally; this command never authenticates or uploads to the Hub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/diffcider-browser-mlx")
    parser.add_argument("--source", default=BROWSER_MODEL)
    parser.add_argument("--revision", default=BROWSER_REVISION)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--verify", metavar="DIRECTORY", help="Verify an existing bundle instead of exporting."
    )
    args = parser.parse_args()
    if args.verify:
        manifest = verify_bundle(args.verify)
        print(f"Verified {len(manifest['sha256'])} files in {args.verify}")
        return
    print(export_bundle(args.output, source=args.source, revision=args.revision, dtype=args.dtype))


if __name__ == "__main__":
    main()
