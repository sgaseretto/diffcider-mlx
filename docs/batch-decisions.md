# Batched S1 decisions

`decide_batch()` evaluates independent decisions in one forward pass. It supports
mixed binary (`noul`), multiple-choice (`choice`), and ordinal (`score`) questions,
different prompt lengths, and different numbers of candidates. Results stay in
input order, with exactly the same fields as `decide()`.

## Python API

```python
import json
from pathlib import Path
from diffcider import Diffcider

engine = Diffcider.from_pretrained("s1")  # float32 is the default
requests = json.loads(Path("examples/decisions.json").read_text())
results = engine.decide_batch(requests, adapter=None, max_length=4096)
for result in results:
    print(result["option"], result["probabilities"])
```

The [example file](../examples/decisions.json) contains six requests, two per
question type. Every request is a dictionary with `state` and `question`:

```python
requests = [
    {
        "state": "The delivery was excellent.",
        "question": {
            "type": "noul",
            "instructions": "Is the customer satisfied?",
            "criteria": "Customer satisfaction",
        },
    },
    {
        "state": "The product arrived broken.",
        "question": {
            "type": "choice",
            "instructions": "Classify the sentiment.",
            "criteria": ["negative", "neutral", "positive"],
        },
    },
]
results = engine.decide_batch(requests)
```

Each result contains `index`, `option`, and `probabilities`. Binary probabilities
are **[No, Yes]**. Choice and ordinal probabilities follow candidate order;
ordinal results also contain `expected_score`, the probability-weighted mean of
zero-based level indices. Probabilities are normalized independently for each
request, never across the batch.

A call accepts a nonempty list and performs one forward pass. Six is a tested
batch size, not an API limit. There is no automatic chunking or sorting; split
larger workloads into batches appropriate for available memory. Grouping similar
prompt lengths reduces padding overhead. `max_length` applies separately to each
encoded prompt and is bounded by the model's context. Overlong prompts raise an
error before inference; candidates are never silently truncated. A singleton
batch uses the existing single-decision scoring path.

One model and one adapter serve the entire batch. `adapter=None` uses the loaded
checkpoint's unchanged weights; `adapter="name"` selects a compatible loaded
adapter for this call. The engine resets adapter state after completion or failure,
and serializes concurrent calls. You can alternate `decide_batch()`, `decide()`,
`generate_batch()`, and `generate()` on the same resident engine. Mode selection
and adapter selection remain independent; see the
[model and adapter guide](models-and-adapters.md#what-s1-and-generation-mean).

## Command line

```sh
uv run diffcider --model s1 --decisions examples/decisions.json
uv run diffcider --model s1 --decisions examples/decisions.json \
  --dtype bfloat16 --max-length 2048
```

The command returns a JSON list. Model, revision, adapter, and offline flags work
as on individual requests. `--decision`, `--decisions`, `--prompt`, and `--prompts`
are mutually exclusive. `--max-length` is the per-prompt limit for both decision
routes; generation uses its existing budget and model context validation.

## Implementation and validation

Rows are right-padded after their complete prompts. Attention excludes padding
keys by length, including when the tokenizer has no padding ID; a literal padding
token within a real prompt remains valid. Real tokens retain their original
rotary positions and fully bidirectional attention.

`Model.score_masks()` accepts per-row mask positions and projects gathered states
onto only the two Yes/No embedding rows. Unused candidate slots repeat a valid
position; their logits are discarded before the existing per-question readout.
The vocabulary projection and denoising loop used for generation are unnecessary.

Tests exercise sizes 1–6, mixed question types, unequal lengths and candidate
counts, exactly one forward pass per call, independent PyTorch full-vocabulary
logits, LoRA/rsLoRA switching, exact restoration of base results, singleton and
reordered inputs, missing/changed padding IDs, validation before inference, and
the CLI. GPU kernel changes can introduce numerical drift even when the selected
answer stays the same. Close probabilities in FP32 do not guarantee universal
answer agreement, and BF16 warrants separate probability checks.

## Reproduce the benchmark

```sh
uv sync --extra benchmark
uv run --extra benchmark pytest
uv run --extra benchmark python -m diffcider.benchmark_decisions_batch \
  --dtype float32 --output reports/decisions-batch-float32.json
uv run --extra benchmark python -m diffcider.benchmark_decisions_batch \
  --dtype bfloat16 --output reports/decisions-batch-bfloat16.json
```

Defaults: 60 deterministic held-out S0 decisions (20 per type), sizes 1–6, one
warm-up and three timed repetitions per variant and size. Each variant processes
the same ordered 60 requests: sequential MLX, batched MLX, sequential PyTorch,
and batched PyTorch. No sorting or filtering by result occurs. Use `--cases` for
a local canonical JSONL/gzip dataset, `--decision-cases` for a different count,
and `--batch-sizes` to change grouping. The final group may be smaller.

Timings synchronize GPU work and rotate variant order. They include input-array
construction, model scoring, host transfer, and probability/result readout on
both backends. They exclude model loading, tokenization, adapter selection, and
JSON serialization. Sequential requests complete individually before the next
starts. These timings differ in scope from the earlier model-scoring-only S1
benchmark and should not be treated as an optimization of that historical number.

The runner limits the MLX allocator cache to 1 GiB and clears both backends'
unused GPU caches before each timed variant, outside the timer. Compiled kernels
remain warm; first-use buffer allocation is timed. This policy prevents growing
caches across many input shapes from distorting comparisons on a shared-memory
Mac. The limit applies only inside this benchmark, not to the inference API.
Allocator-memory samples are retained alongside the latency measurements.
The `torch_driver` telemetry field comes from PyTorch's Metal-driver API and
describes allocations for the process, including memory outside Torch tensors;
it must not be added to MLX allocations as if they were disjoint totals.

The PyTorch batch wrapper supplies an explicit bidirectional padding mask to the
unchanged upstream backbone, gathers real candidate positions, and applies the
original two-row projection. Single requests use the existing reference scorer.
No upstream model or reference algorithm was edited.

Reports include every output and timing sample, probability and expected-score
drift, answer mismatches, repeated-run stability, pinned model/dataset revisions,
input hashes, versions, and source hashes. Peak MLX active allocation includes
resident MLX weights and temporary arrays, but excludes the allocator cache,
PyTorch, and other system memory. Run benchmarks without concurrent GPU workloads.

For the short six-request example, first convert its JSON to the benchmark's
canonical JSONL layout:

```sh
uv run python - <<'PY'
import json
from pathlib import Path

requests = json.loads(Path("examples/decisions.json").read_text())
path = Path(".cache/s1-example-cases.jsonl")
path.parent.mkdir(exist_ok=True)
path.write_text("".join(json.dumps({
    "case_id": f"example-{i + 1}", "state": item["state"],
    "questions": {"q": item["question"]},
}) + "\n" for i, item in enumerate(requests)))
PY
uv run --extra benchmark python -m diffcider.benchmark_decisions_batch \
  --cases .cache/s1-example-cases.jsonl --decision-cases 6 \
  --dtype float32 --output reports/decisions-batch-examples-float32.json
uv run --extra benchmark python -m diffcider.benchmark_decisions_batch \
  --cases .cache/s1-example-cases.jsonl --decision-cases 6 \
  --dtype bfloat16 --output reports/decisions-batch-examples-bfloat16.json
```

These use the same type-balanced selection order as the held-out comparison.

## Measured results

Measured October 5, 2026 on an **Apple M1 Max (32 GPU cores, 64 GiB)** with
MLX 0.32.3, PyTorch 2.14.1, and Transformers 4.57.6. Both backends use the pinned
S1 checkpoint, the same precision and inputs, and no adapter. All numbers below
use the bounded-cache methodology above.

### Six short example decisions

The six example prompts have 42–101 tokens and 1–5 answer masks. Every latency
is the median time for the **complete six-request workload**, not one decision.
Size 4 runs a group of four followed by two; size 5 runs five followed by one.

| Batch size | MLX FP32 (ms) | PyTorch FP32 (ms) | MLX BF16 (ms) | PyTorch BF16 (ms) |
|---|---:|---:|---:|---:|
| 1 | 135.0 | 252.5 | 131.8 | 309.9 |
| 2 | 120.2 | 187.0 | 110.3 | 240.6 |
| 3 | 112.5 | 159.8 | 99.6 | 211.1 |
| 4 | 107.4 | 158.1 | 100.8 | 215.8 |
| 5 | 101.9 | 152.9 | 96.2 | 211.8 |
| 6 | 100.0 | 137.5 | 89.3 | 198.7 |

At size 6, matched sequential MLX took **135.9 ms in FP32** and **132.7 ms in
BF16**. Batching improves throughput by **1.36×** and **1.49×**, respectively;
MLX is **1.37×** and **2.22×** faster than batched PyTorch. This corresponds to
about **60.0 decisions/s in FP32** and **67.2 decisions/s in BF16**.

All **6/6 answers** match sequential MLX and both PyTorch variants at every size.
Across all sizes, maximum MLX batch-versus-sequential probability drift is
**1.95e-6 in FP32** and **0.0135 in BF16**. Maximum MLX-versus-PyTorch batch
probability drift is **2.42e-6 in FP32** and **0.0262 in BF16**. At size 6 alone,
that cross-backend BF16 difference is **0.0157**. The selected answer can agree
while probabilities and ordinal expected scores differ.

Raw short-example measurements:

- [FP32](../reports/decisions-batch-examples-float32.json)
- [BF16](../reports/decisions-batch-examples-bfloat16.json)

### Sixty held-out decisions with uneven lengths

This workload has 20 questions per type, **73–818 prompt tokens**, and **1–120
answer masks** per request. All variants process the same ordered 60 requests;
no length sorting is applied. Every latency below covers the **whole workload**.

| Precision | Batch size | MLX sequential → batch (s) | Speedup vs sequential MLX | PyTorch batch (s) | MLX vs PyTorch batch | MLX peak active (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| FP32 | 1 | 4.288 → 4.317 | 0.99× | 4.983 | 1.15× | 3.307 |
| FP32 | 2 | 4.338 → 5.746 | 0.76× | 6.362 | 1.11× | 3.556 |
| FP32 | 3 | 4.219 → 6.583 | 0.64× | 7.628 | 1.16× | 3.738 |
| FP32 | 4 | 4.165 → 6.358 | 0.66× | 7.630 | 1.20× | 3.724 |
| FP32 | 5 | 4.190 → 7.487 | 0.56× | 8.943 | 1.19× | 3.890 |
| FP32 | 6 | 4.372 → 7.639 | 0.57× | 9.442 | 1.24× | 3.834 |
| BF16 | 1 | 3.362 → 3.670 | 0.92× | 6.509 | 1.77× | 1.780 |
| BF16 | 2 | 3.325 → 4.406 | 0.75× | 7.780 | 1.77× | 1.785 |
| BF16 | 3 | 3.530 → 5.231 | 0.67× | 8.906 | 1.70× | 1.869 |
| BF16 | 4 | 3.507 → 5.248 | 0.67× | 8.858 | 1.69× | 1.863 |
| BF16 | 5 | 3.570 → 6.059 | 0.59× | 10.175 | 1.68× | 1.946 |
| BF16 | 6 | 3.611 → 6.192 | 0.58× | 10.377 | 1.68× | 1.917 |

**Batching is slower than sequential MLX on this uneven workload.** Size 6 pads
to roughly **2.18×** the valid-token count and takes about **75% longer in FP32**
and **71% longer in BF16** than matched sequential MLX. It remains faster than
PyTorch with the same batch size, by **1.24×** and **1.68×**, respectively.
One forward pass still performs more work when its batch and padded sequence
length grow. The practical choice is to batch similarly sized short requests;
for long, uneven queues, compare length-based grouping with sequential execution.

All **60/60 selected answers** agree across sequential/batched MLX and
sequential/batched PyTorch at **every size from 1 through 6**, within both
precisions. Across all sizes:

| Difference | FP32 maximum | BF16 maximum |
|---|---:|---:|
| MLX batch vs sequential probability | 5.89e-6 | 0.0305 |
| MLX batch vs PyTorch batch probability | 2.37e-5 | 0.0455 |
| MLX batch vs PyTorch batch expected score | 5.66e-6 | 0.0604 |

All warmup and measured repetitions produce exactly stable results for each
configuration, including the short examples. Answer agreement is not bitwise
probability agreement; BF16 changes can matter to confidence thresholds. Keep
FP32 when numerical agreement matters, and benchmark the actual request lengths
and adapter configuration before choosing a batch size.

The **65-test suite** passes. A separate real S1 smoke test checks the public
six-request API against individual requests, exact result restoration after a
generation call on the same model, and the installed CLI from outside the checkout.

Raw held-out measurements:

- [FP32](../reports/decisions-batch-float32.json)
- [BF16](../reports/decisions-batch-bfloat16.json)

An [interrupted exploratory FP32 run](../reports/decisions-batch-float32-exploratory.json)
is retained for transparency, with its original benchmark source embedded. Without
allocator controls, sequential MLX latency drifted substantially while the Mac
was under memory pressure. Its partial measurements are excluded from these
tables; the contribution of allocator growth versus other system memory use is
unknown. The final runner uses the explicit cache policy documented above and
retains memory telemetry rather than attributing the slowdown to an unverified cause.
