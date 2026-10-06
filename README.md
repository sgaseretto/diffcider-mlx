# diffcider-mlx

MLX inference for two pinned Qwen3 masked language models:

- **S1:** `SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1` — single-pass candidate selection,
  binary judgments, and ordinal scoring.
- **Base:** `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` — iterative masked-token
  text generation.

Both use a small, shared bidirectional Qwen3 implementation. Original
safetensors load directly; no weight conversion, quantization, or retraining
is needed. The two aliases use fixed revisions and the standard Hugging Face cache.

The `Diffcider` Python API also loads compatible fine-tuned checkpoints from
local directories or custom Hugging Face repositories, with optional named PEFT
LoRA/rsLoRA adapters. One resident model can alternate between decisions and
generation, selecting an adapter or the unchanged base on each call:

```python
from diffcider import Diffcider

engine = Diffcider.from_pretrained("base", adapters={"task": "./my-adapter"})
question = {
    "type": "noul",
    "instructions": "Is the customer satisfied?",
    "criteria": "Customer satisfaction",
}
decision = engine.decide("The delivery was excellent.", question, adapter="task")
text = engine.generate("Explain what makes good customer service.", adapter=None)
```

The adapter must be trained against the loaded base checkpoint. See
[models, adapters, and both inference modes](docs/models-and-adapters.md) for
one-call Hub loading, revisions, adapter switching, PEFT export examples,
supported configurations, and validation results.

## Measured results

Measured October 5, 2026 on an **M1 Max, 32 GPU cores, 64 GiB memory**,
using MLX 0.32.3, PyTorch 2.14.1, and Transformers 4.57.6. These results use
the optimized generation loop. Both backends use the GPU, the same checkpoint
weights, and the precision shown below.

| Workload | Precision | Output agreement | PyTorch MPS | MLX Metal | Speedup |
|---|---|---|---:|---:|---:|
| S1 decisions | float32 | 60/60 decisions | 91.2 ms | 69.3 ms | 1.32× |
| S1 decisions | BF16 | 60/60 decisions | 109.6 ms | 58.3 ms | 1.88× |
| MDLM, 32 tokens / 32 steps | float32 | 4/4 exact token sequences | 1.502 s | 0.879 s | 1.71× |
| MDLM, 32 tokens / 32 steps | BF16 | 4/4 exact token sequences | 1.791 s | 0.778 s | 2.30× |
| MDLM, 128 tokens / 128 steps | float32 | 2/2 exact token sequences | 10.211 s | 7.741 s | 1.32× |
| MDLM, 128 tokens / 128 steps | BF16 | 1/2 exact token sequences | 12.329 s | 5.865 s | 2.10× |

A separate same-process comparison against the previous MLX sampler
(`2d49097`) measured **9–10% lower latency for 32-token generation** and
**17–18% lower latency for 128-token generation**, across float32 and BF16.
It used one prompt per length, one warmup, three measured repetitions, and
rotating variant order. All runs produced identical tokens to the previous
sampler. The [ablation report](reports/generation-optimization-ablation.json)
separates block projection from asynchronous dispatch and retains raw timings
and peak active MLX memory. These paired measurements isolate the optimization
more reliably than comparing timings from separate historical runs.
See the [optimization notes](docs/generation-optimization.md) for each change's
measured contribution, memory usage, implementation details, and validation.

The S1 sample contains 20 choice, 20 binary, and 20 ordinal decisions,
73–818 input tokens, and 1–120 masks. All 60 prompts, token IDs, and mask
positions were also checked against the original upstream preprocessing.
The PyTorch sampler's AST was checked against the pinned model card: only
docstrings and the explicit tokenizer argument differ. The S1 scoring path
and PyTorch reference algorithms are unchanged by these optimizations.

**Long BF16 generation is not always equivalent to PyTorch.** The 128-token
coding example diverges at output token index 23 (zero-based), with 43/128 token
positions matching; its visible text also differs. Rerunning the previous MLX
sampler reproduced exactly the optimized MLX output on both long examples, so
this mismatch predates the optimization. The
[baseline validation](reports/bfloat16-long-baseline-validation.json) retains
those token IDs. All six float32 generation cases match PyTorch exactly;
use float32 when reference agreement is the priority.

**Probabilities are close, not bit-identical.** Maximum absolute probability
error was **0.0000237 in float32** and **0.0278 in BF16**, despite identical
selected decisions. Float32 remains the default when close numerical agreement
matters. These samples do not guarantee identical outputs on all inputs,
especially near tied logits or confidence scores. BF16 performance here is
specific to this Mac and these backends.

The longer generation runs use the original model card's coding and arithmetic
prompts, block size 64, one warmup, and three measured repeats. Speedups depend
on sequence length and precision; short-generation results should not be
extrapolated to longer generations.

Full outputs and timings: [float32](reports/optimized-float32.json),
[BF16](reports/optimized-bfloat16.json),
[longer float32 generation](reports/optimized-float32-long.json), and
[longer BF16 generation](reports/optimized-bfloat16-long.json).
The earlier [float32](reports/float32.json), [BF16](reports/bfloat16.json), and
[longer float32](reports/float32-long.json) reports are retained for reference.
See the methodology below before interpreting the timings as application
latency or a model-quality benchmark.

### Adapter-enabled measurements

The persistent engine adds one-call model/adapter loading, per-request adapter
switching, and shared decision/generation methods. Compatible local and Hub
checkpoints can use single-file or sharded safetensors exports.

With a synthetic, nonzero PEFT LoRA enabled, the same M1 Max measured:

| Checkpoint / precision | Decision speedup vs PyTorch + PEFT | Generation speedup |
|---|---:|---:|
| Base / float32 | 1.34× | 1.75× |
| Base / BF16 | 1.79× | 2.00× |
| S1 / float32 | 1.30× | 1.80× |

Each configuration matched **6/6 decisions and 2/2 generated token sequences**
with the adapter disabled, enabled, and disabled again. Disabling restored MLX
base outputs exactly. Generation used 16 tokens / 16 steps; timings exclude
loading, tokenization, decoding, and adapter selection. Active LoRA adds compute
cost, and BF16 probability differences reached 0.0243. These are compatibility
and timing checks, not fine-tuning quality measurements.

The [adapter guide](docs/models-and-adapters.md#measured-adapter-results) records
absolute latencies, adapter overhead, precision differences, raw reports,
reproduction commands, and the supported PEFT configurations. All 50 tests pass.

## Install

Supported runtime: **macOS on Apple Silicon, Python 3.13+**, and a macOS version
supported by the installed MLX Metal runtime. The distribution name is
`diffcider-mlx`; the Python import and command are both `diffcider`.

From a local checkout, choose pip or uv:

```sh
# pip
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

```sh
# uv
uv venv --python 3.13
uv pip install .
source .venv/bin/activate
```

To add Diffcider directly from GitHub, run this **inside the uv project that will
use it**, after the packaging changes have been pushed:

```sh
uv add git+https://github.com/sgaseretto/diffcider-mlx.git
uv run diffcider --help
```

Use the `git+https://` prefix to identify a Git repository. This installs from
the remote's default branch and requires no PyPI publication. The consuming
project can then use `from diffcider import Diffcider`.

To select a specific source, add `--branch main`, `--tag <tag>`, or
`--rev <commit>` to the `uv add` command. The selected revision must contain the
packaging configuration and be available on the remote. See
[uv's Git dependency documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/#git).

With pip, replace `<revision>` below with the desired commit, tag, or branch:

```sh
python -m pip install "git+https://github.com/sgaseretto/diffcider-mlx.git@<revision>"
```

Installation does not download model weights. They are fetched on first model
load and stored in the Hugging Face cache. PyTorch and PEFT are optional benchmark
dependencies; add them with `python -m pip install '.[benchmark]'` or
`uv pip install '.[benchmark]'` from the checkout. The regular install supports
MLX inference and loading PEFT-exported adapters without either package.
A Transformers message about missing PyTorch/TensorFlow/Flax is expected in a
core-only install; tokenizers remain available, and Diffcider inference uses MLX.

## Run

After installation, these commands work outside the repository:

```sh
diffcider --help
python -m diffcider --help
diffcider --model base --prompt "What is the capital of France?" \
  --max-new-tokens 32 --steps 32 --block-size 16
```

Python applications can use `from diffcider import Diffcider`. Decision JSON
files are user inputs; pass their path to `diffcider --decision /path/to/input.json`.

When working from a checkout, `uv sync` installs the package in editable mode
alongside the development tools. The existing script also remains available:

```sh
uv sync
uv run main.py --model s1 --decision examples/sentiment.json
uv run main.py --model base --prompt "What is the capital of France?" \
  --max-new-tokens 32 --steps 32 --block-size 16
```

The default precision is **float32** for numerical comparison. Use
`--dtype bfloat16` or `--dtype float16` to explore reduced precision. Precision
can change probabilities and the iterative generation trajectory.

Decision inputs contain `state` and `question`. Questions use the upstream
`type`, `instructions`, and `criteria` fields:

- `choice`: a list of labels, or an ordered mapping from label to description.
- `noul`: a binary question. Output probabilities are in **[No, Yes]** order.
- `score`: an ordered list of level descriptions. The result also includes the
  probability-weighted expected level index.

Generation currently supports one prompt, greedy sampling (`temperature=0`),
low-confidence remasking, and no classifier-free guidance. It uses a fixed
token budget, including tokens after EOS, to match the reference sampler.
The token count must divide into whole blocks, and steps must divide across
those blocks. Ordinary autoregressive generation and persistent KV caching
are not used.

The MLX sampler runs the full bidirectional backbone on every step, then projects
only the current block onto the vocabulary. Softmax also runs only on that
block. Confidence precision and global selection indices are preserved. A queue
of at most two denoising steps overlaps CPU dispatch with GPU execution; generation
waits for the final result before returning. These changes do not reduce the
number of denoising steps or quantize weights.

## Compare with the original PyTorch implementation

```sh
uv sync --extra benchmark
uv run --extra benchmark python -m diffcider.benchmark \
  --dtype float32 --output reports/optimized-float32.json
uv run --extra benchmark python -m diffcider.benchmark \
  --dtype bfloat16 --output reports/optimized-bfloat16.json
```

For the longer coding and arithmetic examples from the original model card:

```sh
uv run --extra benchmark python -m diffcider.benchmark \
  --task generation --dtype float32 --prompts examples/generation_prompts.json \
  --generation-cases 2 --max-new-tokens 128 --steps 128 --block-size 64 \
  --warmup 1 --repeats 3 --output reports/optimized-float32-long.json
```

Repeat the longer command with `--dtype bfloat16` and
`--output reports/optimized-bfloat16-long.json` to compare BF16 generation.

The reference loads the checkpoint's **unmodified PyTorch model code** using
Transformers 4.57.6 and SDPA. The generation baseline is the original
model-card sampler; its only API change passes the tokenizer explicitly.
S1 uses the upstream mask-state gather and two-row Yes/No projection.

Both backends run on the same Mac GPU: PyTorch **MPS** and MLX **Metal**, with
matching precision, weights, inputs, and generation settings. MPS CPU fallback
is rejected. To measure a CPU baseline separately, explicitly pass
`--device cpu`; that result is not a GPU-to-GPU comparison.

Timing uses alternating backend order, two warmups, five measured repetitions,
and GPU synchronization. Per-case medians include completed inference, not
just dispatch. Downloads, loading, tokenization, input transfer, and output
decoding are excluded. Generation includes the complete sampling loop.
The reported aggregate speedup is the ratio of mean per-case median latencies.

The default suite uses 60 deterministic, type-balanced decisions from the
upstream held-out S0 test set and four fixed generation prompts. Data downloads
to `.cache/`. Use `--cases path/to/test.jsonl.gz` for a local canonical dataset,
`--decision-cases`, `--generation-cases`, or `--task decisions|generation` to
adjust the run. These are compatibility/performance samples, not a full model
quality evaluation.

JSON reports include package versions, hardware, checkpoint revisions, input
hashes, every timing sample, probability/logit errors, decisions, generated
token IDs, decoded text, and implementation source hashes. The initial logit
comparison covers the first block using the same selected-position readout as
generation. Reports retain mismatches: a faster result is not
automatically an equivalent result. Exact token agreement includes special
tokens; visible-text agreement is reported separately.

## Development

```sh
uv sync --extra benchmark
uv run --extra benchmark pytest
uv run ruff check .
uv run ruff format --check .
```

Build a source archive and wheel with `uv build`. This builds the wheel from the
source archive, checking that the archive contains the files needed to package
the project. Build output goes into the ignored `dist/` directory. To install
the wheel in a separate activated environment:

```sh
python -m pip install /path/to/dist/diffcider_mlx-0.1.0-py3-none-any.whl
# Or:
uv pip install /path/to/dist/diffcider_mlx-0.1.0-py3-none-any.whl
```

The wheel contains the `diffcider` package and existing project/upstream license
notices. It does not bundle model weights, downloaded datasets, or benchmark
reports. A build creates local artifacts; publishing to a package index is a
separate step. These instructions do not assume a PyPI release exists.

Packaging was verified with clean pip wheel and uv source-archive installations,
both CLI entry points, and real-model decision/generation runs outside the
checkout without PyTorch or PEFT. Dependency checks and `twine check` passed.

Unit tests compare a small MLX model against PyTorch Qwen3 with bidirectional
attention and padding, check future-token visibility, and validate the
selected-position projection with tied and untied output weights. Sampler tests
compare changing predictions against the original PyTorch loop across block
boundaries, multi-token transfers, and uneven schedules. They do not download models.

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream code and dataset attribution.
