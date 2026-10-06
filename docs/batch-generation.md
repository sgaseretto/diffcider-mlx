# Batched diffusion generation

Diffcider can generate several independent prompts in one GPU batch. Model
weights are shared, and every row advances through the same denoising schedule.
The public API accepts different prompt lengths and returns results in input order.

## Python API

```python
from diffcider import Diffcider

engine = Diffcider.from_pretrained("base")
outputs = engine.generate_batch(
    [
        "Explain recursion in one sentence.",
        "Write a short poem about rain.",
        "What is the capital of France?",
        "Give two examples of renewable energy.",
    ],
    max_new_tokens=64,
    steps=32,
    block_size=32,
    adapter=None,
)
```

`generate_batch()` returns `list[str]`. `generate_batch_tokens()` accepts the
same arguments and returns `list[list[int]]`, excluding prompts and padding.
Each token list has exactly `max_new_tokens` entries, including special tokens
and tokens after EOS. Text output omits special tokens, as with `generate()`.

The input must be a nonempty list of strings. Every prompt must fit within the
context limit together with its generated token budget. A one-item batch uses
the original single-prompt sampler. Existing `generate()` and `generate_tokens()`
methods retain their original implementation and behavior.

Each call is one batch. There is no automatic chunking or length-based sorting.
Group similarly sized prompts to reduce padding overhead. All rows share
`max_new_tokens`, `steps`, `block_size`, and one selected adapter. Use separate
calls for different adapters or schedules. Adapter selection remains per-request;
`None` uses the unchanged base, and the engine resets adapter state after completion
or failure. Calls on one engine remain serialized, including batches.

Generation is still greedy: temperature sampling, early EOS stopping, and adaptive
step counts are not added. Steps are distributed over generation blocks; iterations
that would transfer zero tokens are skipped, matching the single-prompt MLX path.
A shared schedule reduces the number of model calls needed for a workload but
increases the work and temporary memory within each call. Speedup must be measured.

## Command line

Save a JSON list of prompts, for example:

```json
["Explain recursion.", "Write a poem about rain.", "What is photosynthesis?"]
```

Then run:

```sh
diffcider --model base --prompts prompts.json \
  --max-new-tokens 64 --steps 32 --block-size 32
```

The command prints a JSON list of texts in input order. It also accepts the
existing `--adapter`, `--dtype`, revision, and offline flags. `--prompts`,
`--prompt`, `--decision`, and `--decisions` are mutually exclusive. From a checkout, use
`uv run diffcider ...` with the same arguments.

## Implementation and correctness

Each row contains its prompt followed immediately by its generated region.
Right padding follows the entire generated region. Padding keys are excluded
from attention at every layer; real tokens keep the same rotary positions as
in an individual request. Changing the padding token therefore does not change
the meaning of the prompt.

At each step, the sampler gathers each row's current generation block and ranks
confidence independently within that row. Prompt tokens, completed blocks, and
padding cannot be selected for updates. The model still computes bidirectional
attention over the complete valid sequence, then projects only the selected
block onto the vocabulary. This retains the previous output-projection optimization.
The bounded asynchronous dispatch keeps at most two steps in flight.

Tests cover unequal lengths, independent predictions, prompt/padding preservation,
block boundaries, uneven schedules, skipped zero-transfer iterations, multiple
LoRA adapters, restoration of base outputs, absent/different padding IDs, invalid
inputs, singleton batches, selected-position logits, and the CLI route. They
compare against independent PyTorch/PEFT computations and the unchanged upstream
sampler. Batch-dependent GPU kernels can still introduce numerical differences,
so exact agreement for every prompt is not guaranteed, especially in BF16.

## Reproduce the performance comparison

```sh
uv sync --extra benchmark
uv run --extra benchmark pytest
uv run --extra benchmark python -m diffcider.benchmark_batch \
  --dtype float32 --output reports/batch-float32.json
uv run --extra benchmark python -m diffcider.benchmark_batch \
  --dtype bfloat16 --output reports/batch-bfloat16.json
```

Defaults: the same four existing benchmark prompts, batch sizes 1/2/4, 32 output
tokens per prompt, 32 denoising steps, blocks of 16, one warm-up and three timed
repetitions. `--batch-sizes`, `--prompts`, `--model`, and the generation settings
can be changed. The last batch may be smaller than the requested size.

Every variant processes the entire ordered workload: sequential MLX, batched
MLX, sequential PyTorch, and batched PyTorch. Variant order rotates, and GPU work
is synchronized. Timings exclude model loading, chat templating, and decoding.
MLX sampler input-array construction is included; PyTorch's padded prompt tensors
and attention masks are prepared before timing. Throughput counts the fixed
output-token budget, including tokens after EOS, rather than visible text tokens.

The upstream PyTorch sampler does not pass `attention_mask` to its model. The
benchmark wraps that model to supply the known valid-token mask for each batch,
so padding cannot influence its predictions. The upstream model and sampler
algorithms remain unchanged. This explicit mask is required for a meaningful
comparison with unequal prompt lengths.

Reports retain all timing samples, output tokens/text, agreement between batch
and sequential execution on both backends, repeated-run stability, configuration,
versions, checkpoint revisions, and source hashes. Peak **MLX active memory**
includes its resident weights and temporary arrays; it excludes the allocator
cache, PyTorch allocations, and other system memory. It is not total unified
memory usage. Run GPU benchmarks without other concurrent GPU workloads.

## Measured results

Measured October 5, 2026 on an **Apple M1 Max (32 GPU cores, 64 GiB memory)**,
using MLX 0.32.3, PyTorch 2.14.1, and Transformers 4.57.6. These measurements use
the pinned base checkpoint **without adapters**. The four prompts have 37, 36,
32, and 33 tokens after chat templating; every prompt generates 32 token IDs.

Every latency below is the median time to process **all four prompts**. Batch
size 2 runs two batches of two; batch size 4 runs one batch. Sequential MLX is
measured again alongside each batch configuration to provide a matched comparison.

| Precision | Batch size | MLX sequential → batch (s) | Speedup vs sequential MLX | PyTorch batch (s) | MLX vs PyTorch batch | MLX peak active (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1 | 3.826 → 3.837 | 1.00× | 5.876 | 1.53× | 2.411 |
| FP32 | 2 | 3.827 → 3.164 | 1.21× | 4.458 | 1.41× | 2.591 |
| FP32 | 4 | 3.859 → 2.881 | 1.34× | 3.501 | 1.22× | 2.945 |
| BF16 | 1 | 2.994 → 2.980 | 1.00× | 6.913 | 2.32× | 1.242 |
| BF16 | 2 | 2.995 → 2.224 | 1.35× | 5.180 | 2.33× | 1.365 |
| BF16 | 4 | 2.990 → 1.644 | 1.82× | 4.175 | 2.54× | 1.606 |

Batch size 4 reduced total MLX workload latency by **25.4% in FP32** and **45.0%
in BF16**. Fixed-budget throughput was **44.44 tokens/s in FP32** and **77.85
tokens/s in BF16**. It increased peak MLX active memory by about **0.534 GiB**
and **0.364 GiB**, respectively, compared with the singleton-batch runs. The
weights remain one resident copy; additional memory is used by batch activations
and temporary arrays.

All **4/4 complete token sequences** matched between sequential and batched MLX,
between sequential and batched PyTorch, and across the two backends, at every
tested batch size and within each precision. Outputs were stable across all
warm-up and measured repetitions. The generation-batching **57-test suite** also passed; a separate
three-prompt real-model smoke test checks the public API against sequential
inference and the installed CLI outside the repository.

These results cover short prompts with similar lengths. Longer or uneven
contexts, different block sizes, and active adapters can change the speedup and
memory cost. Numerical differences and tied confidence scores can change the
iterative trajectory on other inputs. The earlier long-generation BF16 mismatch
remains documented in [the optimization notes](generation-optimization.md);
FP32 remains the parity-first default.

Raw reports, including every output and timing sample:

- [FP32 batch comparison](../reports/batch-float32.json)
- [BF16 batch comparison](../reports/batch-bfloat16.json)
