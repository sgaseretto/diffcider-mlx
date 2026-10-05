# Generation optimization results

October 5, 2026. Baseline: `2d49097`, the previous MLX implementation.
Measurements use an M1 Max with 32 GPU cores and 64 GiB memory, MLX 0.32.3,
PyTorch 2.14.1, and Transformers 4.57.6. Checkpoint revisions remain pinned.

## Implementation changes

**Project only the current block.** Previously, every denoising step projected
all sequence positions into the 151,936-token vocabulary and computed their
probabilities. [Model.__call__](../diffcider/model.py) now accepts the optional
keyword `logit_positions`, a slice or one-dimensional position array. Selection
happens after the full bidirectional backbone, immediately before the output
projection. Omitting the keyword retains the full output. Tied and untied output
weights and padding masks are supported.

The [generation loop](../diffcider/inference.py) requests the current block and
computes its argmax and softmax. It preserves confidence precision and scatters
confidence back into the original sequence positions before selecting tokens,
retaining the existing selection indices and treatment of ties. This reduces
vocabulary projection, softmax work, and intermediate tensor sizes.

**Overlap dispatch with GPU execution.** Previously, `mx.eval(x)` blocked after
every step. The loop now submits each result with `mx.async_eval(x)` and waits
for the preceding result, keeping at most two denoising steps in flight. The CPU
can prepare subsequent work while the GPU executes it. Steps still depend on
the preceding token state; the final `mx.eval(x)` completes generation before
returning. The bounded queue also limits intermediate memory growth.

Both changes preserve the full attention context, greedy sampling, transfer
schedule, denoising-step count, confidence precision, and fixed output length
including tokens after EOS. S1 scoring, weights, and the PyTorch reference
algorithms are unchanged. No new runtime dependencies or CLI flags are needed.

## Contribution of each change

The [component comparison](../reports/generation-optimization-ablation.json)
uses the same loaded model for three variants: the previous sampler, block
projection with blocking evaluation, and block projection with asynchronous
dispatch. It rotates variant order, performs one warmup and three measured
repetitions, and synchronizes completed GPU work. Values below are median
complete-generation latency in milliseconds.

| Precision | Tokens / steps / block size | Previous MLX | Block projection only | Both changes | Total latency reduction |
|---|---|---:|---:|---:|---:|
| float32 | 32 / 32 / 16 | 675.32 | 661.19 | 607.50 | 10.0% |
| BF16 | 32 / 32 / 16 | 776.89 | 741.09 | 703.88 | 9.4% |
| float32 | 128 / 128 / 64 | 9,340.80 | 7,896.70 | 7,750.45 | 17.0% |
| BF16 | 128 / 128 / 64 | 7,086.84 | 5,982.71 | 5,795.55 | 18.2% |

Block projection alone reduced short-generation latency by 2.1–4.6% and long
generation by 15.5–15.6%. Asynchronous dispatch then reduced the block-only
latencies by another 5.0–8.1% and 1.9–3.1%, respectively. These percentages use
different baselines and should not be added.

Peak active MLX memory for the longer float32 case fell from 2,964.6 MB to
2,885.9 MB; BF16 fell from 1,647.9 MB to 1,552.0 MB. These are the maximum peaks
across measured repetitions, include model weights, exclude the allocator
cache, and use decimal MB. Asynchronous dispatch uses slightly more memory than
block projection alone, while remaining below the previous implementation.

This comparison uses one prompt per length: “What is the capital of France?”
and the DFS coding prompt in [generation_prompts.json](../examples/generation_prompts.json).
Every run matched the previous sampler's tokens. The larger PyTorch comparison
uses different short prompts and aggregates per-case medians, so its absolute
latencies should be interpreted separately.

## PyTorch comparison and parity

The [README results table](../README.md#measured-results) contains the full
comparison and [rerun commands](../README.md#compare-with-the-original-pytorch-implementation).
On the tested samples, optimized MLX is 1.71× faster for short float32 generation
and 1.32× faster for long float32 generation, with all six sequences matching
PyTorch exactly. BF16 is 2.30× and 2.10× faster, respectively, with five of six
sequences matching.

The long BF16 coding example differs from PyTorch starting at output token
index 23, with 43/128 positions matching and different visible text. The
[baseline rerun](../reports/bfloat16-long-baseline-validation.json) confirms that
both long BF16 outputs are identical to the previous MLX sampler. Together with
the earlier reports, all 12 optimized generation outputs match their previous
MLX counterparts. The BF16 discrepancy predates these changes. Float32 remains
the default for reference agreement; these samples cannot guarantee agreement
on every input.

S1 still selects the same decisions as PyTorch in 60/60 cases at each precision.
Maximum absolute probability error remains 0.0000237 in float32 and 0.0278 in
BF16. Its measured speedups are 1.32× and 1.88×; its scoring implementation was
not changed by this work.

## Benchmark and test changes

- [benchmark.py](../diffcider/benchmark.py) now checks initial logits through the
  same selected-position projection used by generation, records the first
  block's position range, and hashes the model, sampler, reference, and benchmark
  source files. Older reports checked all generated positions, so their initial
  logit-error maxima cover a different range. Full-sequence token comparisons
  remain unchanged.
- [Model tests](../tests/test_model.py) cover tied and untied heads, selected
  slices and position arrays, padding, and future-token influence after output
  selection.
- [Sampler tests](../tests/test_inference.py) compare changing predictions with
  the original PyTorch loop across block boundaries, multiple transfers per
  step, uneven schedules, and steps with zero transfers.

Validation completed: **27 tests passed**, Ruff lint and formatting checks
passed, report timing aggregates and source hashes were verified, and the
PyTorch reference files were confirmed unchanged. Raw reports, including the
BF16 mismatch and previous measurements, are retained under [reports/](../reports/).
