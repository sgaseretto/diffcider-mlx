# diffcider-mlx

MLX inference for two pinned Qwen3 masked language models:

- **S1:** `SEU-ZZH/Shared-YesNo-Qwen3-0.6B-S1` — single-pass candidate selection,
  binary judgments, and ordinal scoring.
- **Base:** `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` — iterative masked-token
  text generation.

Both use a small, shared bidirectional Qwen3 implementation. Original
safetensors load directly; no weight conversion, quantization, or retraining
is needed. Downloads use fixed revisions and the standard Hugging Face cache.

## Measured results

Measured October 5, 2026 on an **M1 Max, 32 GPU cores, 64 GiB memory**,
using MLX 0.32.3, PyTorch 2.14.1, and Transformers 4.57.6. Both backends use
the GPU, the same checkpoint weights, and the precision shown below.

| Workload | Precision | Output agreement | PyTorch MPS | MLX Metal | Speedup |
|---|---|---|---:|---:|---:|
| S1 decisions | float32 | 60/60 decisions | 86.3 ms | 65.5 ms | 1.32× |
| S1 decisions | BF16 | 60/60 decisions | 104.8 ms | 55.2 ms | 1.90× |
| MDLM, 32 tokens / 32 steps | float32 | 4/4 exact token sequences | 1.479 s | 1.014 s | 1.46× |
| MDLM, 32 tokens / 32 steps | BF16 | 4/4 exact token sequences | 1.743 s | 0.888 s | 1.96× |
| MDLM, 128 tokens / 128 steps | float32 | 2/2 exact token sequences | 9.713 s | 8.846 s | 1.10× |

The S1 sample contains 20 choice, 20 binary, and 20 ordinal decisions,
73–818 input tokens, and 1–120 masks. All 60 prompts, token IDs, and mask
positions were also checked against the original upstream preprocessing.
The PyTorch sampler's AST was checked against the pinned model card: only
docstrings and the explicit tokenizer argument differ.

**Probabilities are close, not bit-identical.** Maximum absolute probability
error was **0.0000237 in float32** and **0.0278 in BF16**, despite identical
selected decisions. Float32 remains the default when close numerical agreement
matters. These samples do not guarantee identical outputs on all inputs,
especially near tied logits or confidence scores. BF16 performance here is
specific to this Mac and these backends.

The longer generation run uses the original model card's coding and arithmetic
prompts, block size 64, one warmup, and three measured repeats. The smaller
speedup shows why the short-prompt measurements should not be extrapolated to
longer generations.

Full outputs and timings: [float32](reports/float32.json),
[BF16](reports/bfloat16.json), and [longer float32 generation](reports/float32-long.json).
See the methodology below before interpreting the timings as application
latency or a model-quality benchmark.

## Run

Requires Apple Silicon and Python 3.13+.

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

## Compare with the original PyTorch implementation

```sh
uv sync --extra benchmark
uv run --extra benchmark python -m diffcider.benchmark \
  --dtype float32 --output reports/float32.json
uv run --extra benchmark python -m diffcider.benchmark \
  --dtype bfloat16 --output reports/bfloat16.json
```

For the longer coding and arithmetic examples from the original model card:

```sh
uv run --extra benchmark python -m diffcider.benchmark \
  --task generation --dtype float32 --prompts examples/generation_prompts.json \
  --generation-cases 2 --max-new-tokens 128 --steps 128 --block-size 64 \
  --warmup 1 --repeats 3 --output reports/float32-long.json
```

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
token IDs, and decoded text. They retain mismatches: a faster result is not
automatically an equivalent result. Exact token agreement includes special
tokens; visible-text agreement is reported separately.

## Development

```sh
uv sync --extra benchmark
uv run --extra benchmark pytest
uv run ruff check .
```

Unit tests compare a small MLX model against PyTorch Qwen3 with bidirectional
attention and padding, check future-token visibility, and validate the
selected-token projection and sampler contracts. They do not download models.

See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream code and dataset attribution.
