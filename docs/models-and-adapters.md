# Models, PEFT adapters, and both inference modes

`Diffcider` keeps one model in memory and exposes decision scoring and diffusion
generation. Named adapters load once; each request explicitly selects one or
uses the unchanged base. MLX inference does not require PyTorch or PEFT.

For **sysone exports with a learned decision head**, such as
`sgaseretto/diffcider-browser`, use `SysoneDiffcider.from_pretrained(...)`.
Its `decide`/`predict` methods preserve the trained head and calibration;
`generate` uses the same backbone with the adapter off. See the
[browser model guide](browser-demo.md). Loading only its PEFT subdirectory
through `Diffcider` would omit the trained head.

## Supported workflows

| Capability | API |
|---|---|
| Local fine-tuned full checkpoint | `Diffcider.from_pretrained("./checkpoint")` |
| Custom Hugging Face full checkpoint | `Diffcider.from_pretrained("org/model", revision="commit")` |
| Base and unmerged PEFT adapter in one call | `Diffcider.from_pretrained("base", adapters={"task": "org/adapter"})` |
| Add another adapter without reloading the base | `engine.load_adapter("./adapter", name="task")` |
| Enable an adapter for one call | `engine.decide(state, question, adapter="task")` |
| Use the unchanged base for the next call | `engine.generate(prompt, adapter=None)` |
| Alternate decision and generation modes | Call `decide()` and `generate()` on the same engine |
| Generate several prompts together | `engine.generate_batch(prompts, adapter="task")` |
| Score several independent decisions together | `engine.decide_batch(requests, adapter="task")` |

Install runtime dependencies with `uv sync`. The `base` and `s1` aliases retain
their pinned Hugging Face revisions. Custom repository IDs default to the Hub's
default revision; pass commit hashes for reproducibility. Local directories do
not accept a Hub `revision`. Private/gated repositories use normal Hugging Face
authentication. `local_files_only=True` uses local directories and the HF cache.

Full checkpoints must retain the supported dense, bidirectional `a2d-qwen3`
architecture, standard parameter names, a compatible tokenizer with a mask
token, and consistent tied/untied output weights. Both `model.safetensors` and
HF sharded exports with `model.safetensors.index.json` are supported. Model
Python files are not executed by the MLX loader. Quantized checkpoints, causal
Qwen3 models, scaled RoPE, and sliding-window attention are unsupported.

## One model, named adapters, explicit requests

```python
from diffcider import Diffcider

# These repositories must contain adapters trained on this exact common base.
engine = Diffcider.from_pretrained(
    "my-org/diffusion-base",
    revision="BASE_COMMIT_HASH",
    dtype="float32",
    adapters={
        "decisions": "my-org/decision-lora",
        "reasoning": "my-org/reasoning-lora",
    },
    adapter_revisions={
        "decisions": "DECISION_ADAPTER_COMMIT_HASH",
        "reasoning": "REASONING_ADAPTER_COMMIT_HASH",
    },
)

question = {
    "type": "choice",
    "instructions": "Choose the next action.",
    "criteria": ["Answer immediately", "Investigate further"],
}
decision = engine.decide("The available evidence is incomplete.", question, adapter="decisions")
answer = engine.generate("Explain the next steps.", adapter=None)
specialized_answer = engine.generate("Explain the next steps.", adapter="reasoning")

# Later: add an adapter; existing weights and adapters stay loaded.
engine.load_adapter("./another-adapter", name="domain")
print(engine.loaded_adapters)  # ('decisions', 'reasoning', 'domain')
```

Repository names and commit placeholders above are illustrative. Every inference
call defaults to `adapter=None`, even immediately after loading an adapter.
Unknown or duplicate adapter names raise errors. There is no implicit sticky
adapter state. One adapter is active per call; composition is unsupported.

`decide(state, question, *, adapter=None, max_length=4096)` returns a dictionary
with `index`, `option`, and `probabilities`. Ordinal `score` questions also return
`expected_score`. Binary `noul` probabilities are ordered **[No, Yes]**. The
prompt limit is bounded by the model's context length.

`generate(prompt, *, adapter=None, max_new_tokens=64, steps=64, block_size=32)`
returns decoded text with special tokens omitted. `generate_tokens()` takes the
same arguments and returns generated token IDs, excluding the prompt but
including special tokens and tokens after EOS. Sampling remains greedy,
low-confidence, without classifier-free guidance or EOS early stopping. Token
budgets must divide into whole blocks, and steps must divide across blocks.

Calls on one engine are serialized with a lock. Inference completes before the
adapter is reset, including when a request raises. Adapters do not merge into
or modify base weights. Disabling therefore restores the engine's original MLX
base computation. Accessing `engine.model` directly bypasses request management;
use the public methods for concurrent or alternating requests.

For several prompts in one GPU operation, use `generate_batch(prompts, ...)`
or `generate_batch_tokens(prompts, ...)`. They take the same generation settings
and one adapter for the entire batch, support different prompt lengths, and
return lists in input order. See [batch generation](batch-generation.md) for
details, CLI usage, and measured comparisons.

For single-pass S1 batches, use `decide_batch(requests, adapter=None, max_length=4096)`.
Each item contains `state` and `question`; types, prompt lengths, and candidate
counts can differ. One adapter applies to the whole batch, and results follow
input order. See [batched decisions](batch-decisions.md) for the six-request
example, CLI usage, and probability/throughput measurements.

## What S1 and generation mean

Inference mode and adapter selection are independent. Decision scoring reads
Yes/No probabilities in one forward pass; generation iteratively fills masked
tokens using the full vocabulary. Both use the same bidirectional backbone.

The published `s1` checkpoint has different weights from `base`. A domain LoRA
trained on S1 returns to **S1** when disabled. It does not turn into the original
diffusion checkpoint. For a shared-base S1/S2 setup, train each specialization
against the same frozen base, or train one model for both tasks. Selecting a
mode does not create the corresponding learned capability. Generated text from
an S1 checkpoint and reasoning quality from a new adapter need separate quality
evaluation.

## PEFT compatibility and export

The adapter loader reads `adapter_config.json` and
`adapter_model.safetensors` directly. It maps PEFT's `lora_A`/`lora_B` tensors to
the MLX projections without a conversion step. Standard LoRA uses
`lora_alpha / r`; rsLoRA uses `lora_alpha / sqrt(r)`. Adapter weights are promoted
to float32, matching PEFT's default adapter-loader behavior, and the result is
cast back to the base activation dtype. Dropout is disabled for inference.

Supported targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. Target lists, regular expressions, `all-linear`,
exclusions, layer selection, and per-layer `rank_pattern`/`alpha_pattern` are
supported. The loader validates target coverage, tensor shapes, and finite
weights before attaching an adapter.

Adapters modifying embeddings, `lm_head`, base biases, or `modules_to_save` are
rejected. Our selected-token output projection reads weight rows directly, so
silently accepting an adapted output head would be incorrect. DoRA, prompt or
prefix tuning, layer replication, activated/quantized LoRA variants, and
base-modifying initializations such as unconverted PiSSA/OLoRA are also
unsupported. Unknown enabled PEFT options fail explicitly. Standard initializers
(`True`, `False`, or `"gaussian"`) and `bias="none"` are supported. Export
safetensors; pickle-based `adapter_model.bin` files are not loaded.

For an existing PyTorch fine-tuning workflow:

```python
from peft import LoraConfig, get_peft_model

# base_model is your supported A2D-Qwen3 model loaded in PyTorch.
trained = get_peft_model(
    base_model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "gate_proj"],
        bias="none",
    ),
)
# Train using the intended masked-diffusion or Shared Yes/No objective.
# Keep the base weights, tokenizer, and embeddings frozen.
trained.peft_config["default"].base_model_name_or_path = BASE_REPO_ID
trained.peft_config["default"].revision = BASE_COMMIT_HASH
trained.save_pretrained(
    "./my-adapter",
    safe_serialization=True,
    save_embedding_layers=False,
)
```

These are export instructions, not a training implementation. Adapters trained
with a quantized base may export standard LoRA tensors, but this loader uses a
full-precision base and does not promise parity with the quantized training
runtime. See the official [PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint)
and [LoRA configuration](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora).

To produce a full fine-tuned checkpoint instead:

```python
merged = trained.merge_and_unload()
merged.save_pretrained("./merged-model", safe_serialization=True)
tokenizer.save_pretrained("./merged-model")
engine = Diffcider.from_pretrained("./merged-model")
```

Keep the `a2d-qwen3` config and consistent weight tying. A merged export is a new
base checkpoint and has no separately toggleable adapter.

## Base identity and tokenizer checks

The engine compares `base_model_name_or_path` with the loaded repo ID or local
directory, and checks the declared base `revision` against the loaded Hub
snapshot when present. An adapter's own revision is independent and is passed
through `adapter_revisions` or `load_adapter(..., revision=...)`.

Some training exports contain a stale local path or omit base metadata. By
default these fail. After independently verifying the exact original weights,
use `allow_base_mismatch=True` when loading. This bypasses only base identity
metadata checks; it cannot bypass architecture, tensor, or tokenizer checks.
Metadata without a pinned base revision cannot prove the historical base's
contents, so record the exact training revision when exporting.

The base tokenizer is used for every request. If the adapter includes a saved
tokenizer, its vocabulary, special-token mapping, and chat template must match.
If it contains no tokenizer, compatibility remains the exporter's responsibility.
Changing tokens or vocabulary requires a compatible full-checkpoint export.

## Command line

```sh
uv run main.py --model base --adapter ./my-adapter \
  --decision examples/sentiment.json

uv run main.py --model my-org/diffusion-base --revision BASE_COMMIT_HASH \
  --adapter my-org/adapter --adapter-revision ADAPTER_COMMIT_HASH \
  --prompt "Explain the next steps." --max-new-tokens 32 --steps 32 --block-size 16

# Omit --adapter to use base weights. Each CLI process loads its model anew.
uv run main.py --model base --prompt "Explain the next steps."
```

Use the persistent Python engine to switch without reloading. CLI options also
include `--local-files-only`, `--dtype`, and `--allow-base-mismatch`.

## Validation and reproducibility

`uv sync --extra benchmark` installs the optional PyTorch/PEFT reference tools.
The PEFT reference is pinned to 0.17.1 alongside Transformers 4.57.6. Runtime
inference uses neither package's PyTorch model/adapter implementation.

The original adapter implementation passed **50 tests**, including the existing
model/inference tests. Adapter
tests export nonzero adapters using PEFT and compare independent Torch logits,
decision probabilities, and sampler outputs with MLX. They cover adapter/base
alternation, multiple adapters, request failures, concurrent calls, float32 /
float16 / BF16 projection arithmetic, ranks/scaling, target selection, invalid
exports, tokenizer mismatches, base revision checks, merged and sharded model
exports, and mocked custom Hub downloads. The suite does not download models.

For real-model compatibility and timing measurements:

```sh
uv run --extra benchmark python -m diffcider.benchmark_adapters \
  --model base --dtype float32 --output reports/peft-base-float32.json
uv run --extra benchmark python -m diffcider.benchmark_adapters \
  --model base --dtype bfloat16 --output reports/peft-base-bfloat16.json
uv run --extra benchmark python -m diffcider.benchmark_adapters \
  --model s1 --dtype float32 --output reports/peft-s1-float32.json
```

The benchmark creates a deterministic **synthetic, nonzero** LoRA using PEFT,
then tests base → adapter → base while alternating decision and generation
workloads on one loaded checkpoint. It retains raw outputs, timings, errors,
adapter configuration/hash, and source hashes. Measurements synchronize GPU work
and exclude loading, tokenization, decoding, and adapter selection. Decision
timings cover model scoring; generation timings cover the sampler. These tests
measure compatibility and speed, not fine-tuning quality or general reasoning ability.

Unmerged LoRA adds matrix multiplications to every targeted layer; generation
pays that cost on every denoising step. Adapter switching bypasses or selects
these branches without reloading the base. Previous adapter-free performance
numbers should not be assumed to apply to adapter-enabled requests.

## Measured adapter results

Measured October 5, 2026 on an **Apple M1 Max, 32 GPU cores, 64 GiB memory**,
using MLX GPU versus PyTorch MPS. Each configuration tests six decisions (two
each of binary, choice, and ordinal scoring) and two generation prompts in
three states: adapter disabled, enabled, then disabled again. The adapter
tensors are already loaded in all three timed states.
Generation uses 16 new tokens, 16 denoising steps, and blocks of 8. Each case
has one warm-up and three timed repetitions, with alternating backend order.
Latency is the mean of each case's median. Versions: MLX 0.32.3, PyTorch 2.14.1,
Transformers 4.57.6, and PEFT 0.17.1.

**With the adapter enabled:**

| Checkpoint / dtype | Workload | PyTorch + PEFT (ms) | MLX (ms) | Speedup |
|---|---|---:|---:|---:|
| Base / FP32 | Decisions | 68.8 | 51.5 | 1.34× |
| Base / FP32 | Generation | 810.6 | 463.3 | 1.75× |
| Base / BF16 | Decisions | 89.7 | 50.2 | 1.79× |
| Base / BF16 | Generation | 922.8 | 462.0 | 2.00× |
| S1 / FP32 | Decisions | 67.0 | 51.6 | 1.30× |
| S1 / FP32 | Generation | 831.3 | 462.0 | 1.80× |

Every configuration matched PyTorch on **6/6 decisions and 2/2 complete token
sequences in each of the three states**. Disabling the adapter restored MLX
probe logits, decision probabilities, and generated token IDs **exactly**.
The active adapter changed the probe logits, confirming it was not a no-op.

Maximum adapter-enabled decision-probability differences were **1.49e-6** for
base FP32, **3.43e-7** for S1 FP32, and **0.0243** for base BF16. Matching selected
outputs does not imply bitwise-equal probabilities across backends.

Compared with the first base-off MLX measurement, enabling this adapter added
approximately **27–34% decision latency** and **55–61% generation latency**.
MLX remained faster than PyTorch with the same adapter. Base-off timings also
varied during the run: S1 generation measured 297.7 ms initially and 342.8 ms
after switching back. Exact restoration refers to outputs, not constant latency.

Raw outputs, every timing sample, and verified source hashes:

- [Base FP32](../reports/peft-base-float32.json)
- [Base BF16](../reports/peft-base-bfloat16.json)
- [S1 FP32](../reports/peft-s1-float32.json)

These short, synthetic-adapter workloads validate execution and switching;
they do not establish parity for every prompt or fine-tuned adapter. Float32
remains the parity-first default. Even when BF16 selects identical tokens and
decisions, its probability differences can be larger. The previously recorded
long-generation BF16 mismatch remains documented in
[the original optimization results](generation-optimization.md).
