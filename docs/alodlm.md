# AloDiffcider: experimental ALoDLM generation and zero-shot decisions

**Status: experimental.** The `AloDiffcider` API and behavior may change.
Generation parity has been checked on a small sample; this does not establish
agreement on arbitrary prompts. Zero-shot decisions have low measured accuracy
and uncalibrated probabilities, so they are not ready for production decision
tasks. Use this backend for research and evaluation; see the results below.

`AloDiffcider` loads the pinned `amazon/ALoDLM-1.7B` checkpoint once and
provides `generate` and `decide`. It uses a separate causal recurrent backend;
the existing MDLM, S1 and browser APIs keep their own algorithms.

This implementation covers generation parity and experimental zero-shot reads.
It does not train or load ALoDLM decision adapters, calibrate probabilities,
batch requests, or implement adaptive stopping for decisions.

**Licensing:** ALoDLM weights and Amazon-authored code are CC BY-NC 4.0. The
upstream NOTICE retains additional WeDLM terms. See [THIRD_PARTY.md](../THIRD_PARTY.md)
before redistribution or product use; these additions are not simply MIT/Apache.

## Use one loaded model

```sh
uv sync --extra benchmark
```

The original backbone safetensors load directly. On the first upstream load,
PyTorch safely reads the small `exit_gate.pt` with `weights_only=True` and caches
its MLX-compatible conversion by content hash under
`~/.cache/diffcider/alodlm-gates/`. Subsequent loads use that safetensors cache.
A local checkpoint with `exit_gate.safetensors` needs no PyTorch conversion.
The source checkpoint and base weights are not modified.

```python
from diffcider import AloDiffcider
from diffcider.alodlm import DecodeConfig

engine = AloDiffcider.from_pretrained(dtype="float32")

generated = engine.generate(
    "Explain binary search in two short sentences.",
    config=DecodeConfig(mode="left1", q=0.4, max_new_tokens=64),
)
print(generated["text"])

decision = engine.decide(
    "I was charged twice for the same order.",
    [
        {
            "name": "route",
            "type": "choice",
            "question": "Which support team should handle the request?",
            "options": ["Billing", "Technical support", "Shipping"],
        }
    ],
    passes=1,
    readout="letters",
)
print(decision["answers"])
```

`from_pretrained` also accepts a local checkpoint directory or a custom Hub ID
and optional revision. Custom checkpoints must retain compatible Qwen3
architecture, recurrent metadata, tokenizer and learned gate. Quantized models,
scaled RoPE and sliding attention are rejected. BF16 is available via
`dtype="bfloat16"`; the primary comparison uses float32.

Generation supports upstream `left1` and `entropy` modes, `q`, `tau`, position
penalty, window size and output budget. Its default window is 16 tokens. There
are at most four recurrent passes per window; outer decoding steps are adaptive.
This is distinct from the MDLM demo's fixed denoising-step slider. Generation
is greedy, and `thinking=False` selects the non-thinking chat template.

## Decision semantics and limits

- `choice`: `options` is an ordered list of 2–26 strings.
- `noul`: a binary proposition in `question`; outputs probabilities in
  **[No, Yes]** order and a Boolean answer.
- `score`: `levels` is a list of distinct numeric values. The result includes
  their probabilities, the most likely level and `expected_value`.
- `descriptions` optionally supplies one rubric description per option/level
  (or No/Yes for a binary question).

`readout="letters"` uses one A–Z answer slot per question, including A=No/B=Yes
for binary questions. `readout="yesno"` uses one slot per choice/score option,
then normalizes the Yes-minus-No logits across those options. A binary question
uses one Yes/No slot. These are alternative zero-shot prompts, not trained heads.

All questions and state go in the observed prefix. Numbered answer anchors and
actual checkpoint mask IDs form a tail of at most 32 tokens. Labels are checked
for single-token stability at the anchor boundary. Oversized groups raise a
clear error; split them explicitly rather than silently truncating questions.
Scores from separate option groups should not be assumed calibrated/comparable.
The 32-token cap is conservative, not a guarantee of matching training blocks.

`passes=1..4` selects a fixed number of recurrent reads, without committing
answers between passes. The prefix is currently computed at all trained depths.
Returned gate values are diagnostics, and `calibrated` is always false. They
report raw learned hazards and their cumulative value; reaching the fixed pass
budget does not force that diagnostic value to one. A high normalized score or
halt value is not a validated probability of correctness.

For controlled experiments, `engine.decoder.prefill(prefix_ids)` and
`engine.decoder.read(tail_ids, cache, passes=..., label_ids=...)` expose prefix
reuse and per-depth logits. Reads preserve the cache and reject a cache from a
different model instance. Reuse requires identical prefix tokens and unchanged
weights; do not manually mutate weights while retaining a cache. The convenience
`decide` method creates a fresh request cache and cannot leak an earlier state.

## Reproduce comparisons

Run GPU workloads sequentially. The reference generation implementation is
vendored byte-for-byte at `1da9ddafeca02425480230b9bb7091fdef5672b7`, with a hash
manifest and test. The PyTorch decision reference separately implements the
explicit no-commit intervention; it does not call the MLX recurrence.

```sh
uv run --extra benchmark pytest tests/test_alodlm.py tests/test_alodlm_decisions.py
uv run --extra benchmark python -m diffcider.benchmark_alodlm \
  --backend mlx --output reports/alodlm-mlx-float32.json

# Match the upstream portable runtime in a separate environment.
uv venv .cache/alodlm-reference-env --python 3.13
uv pip install --python .cache/alodlm-reference-env/bin/python \
  torch==2.8.0 transformers==4.56.1 accelerate==1.10.1 \
  mlx==0.32.3 numpy jinja2 PyYAML
uv run --no-project --python .cache/alodlm-reference-env/bin/python \
  python -m diffcider.benchmark_alodlm --backend torch --device mps \
  --output reports/alodlm-torch-float32.json

uv run --extra benchmark python -m diffcider.benchmark_alodlm \
  --compare reports/alodlm-torch-float32.json reports/alodlm-mlx-float32.json \
  --output reports/alodlm-parity-float32.json

uv run --extra benchmark python -m diffcider.benchmark_alodlm_baselines \
  --baseline s1 --output reports/alodlm-baseline-s1.json
uv run --extra benchmark python -m diffcider.benchmark_alodlm_baselines \
  --baseline alodlm-ar --output reports/alodlm-baseline-ar.json
```

Defaults: three generation prompts in both modes, 48-token output budget,
one warmup and two measured repetitions. All measurements synchronize the
selected device before and after execution. Use external `samples` for PyTorch
generation: upstream internal timers synchronize CUDA, not MPS.

Decision evaluation uses 30 canonical held-out S0 records, 10 per type, at both
readouts and every depth: 240 decision configurations. Selection takes the first
eligible records per type, with at most 26 options; oversized questions and the
dataset hash are recorded. No case is filtered by prediction. The initial
unfiltered run stopped at an unsupported 32-option question and is retained as
an explicitly incomplete report.

Cold-prefix and cached-tail latency are recorded separately. Their sum estimates
a fresh request's model work; it excludes loading and prompt compilation.
Baselines use the same case IDs/gold labels. S1 uses its original trained prompt;
ALoDLM-AR uses ordinary first-pass next-token letter logits on the same backbone,
without masks or recurrence. It is not the original Qwen3 base model.

Accuracy, NLL, Brier, ECE15 and ordinal expected-value MAE are descriptive results
on a small public sample, not a general benchmark or calibration guarantee.
S1 has been trained on related decision data; ALoDLM uses zero-shot prompting.
No training, prompt tuning against these labels, or calibration is performed.

The earlier [feasibility assessment](alodlm-feasibility.md) is retained as the
design review preceding this implementation.

## Measured FP32 results

Measured on this M1 Max (64 GiB), with MLX 0.32.3 and the pinned reference
runtime PyTorch 2.8.0 / Transformers 4.56.1 on MPS. The MLX runner uses
Transformers 4.57.6 for tokenization; every compared input-token hash matches.
The main generation and decision benchmarks use one warmup and two measured
repetitions, with resident weights and no concurrent GPU inference workloads.

**Generation matched all 6/6 output sequences (168 returned tokens), including
exit-depth sequences.** MLX was 1.63–2.03× faster than the portable PyTorch MPS
decoder, averaging 1.78× across the six per-case ratios. This compares the
portable reference on this Mac, not the optimized NVIDIA engine.

| Mode / prompt | Output tokens | PyTorch | MLX | Speedup |
| --- | ---: | ---: | ---: | ---: |
| left1 / arithmetic | 7 | 1.091 s | 0.670 s | 1.63× |
| left1 / binary search | 48 | 5.087 s | 2.503 s | 2.03× |
| left1 / Python function | 36 | 4.073 s | 2.155 s | 1.89× |
| entropy / arithmetic | 2 | 1.080 s | 0.664 s | 1.63× |
| entropy / binary search | 48 | 7.241 s | 4.165 s | 1.74× |
| entropy / Python function | 27 | 4.493 s | 2.537 s | 1.77× |

The binary-search responses reached the 48-token budget. The other responses
stopped on EOS. Different generation modes can produce different text; parity
is assessed between backends using the same mode and settings.

**All 240/240 decision configurations selected the same answers**, with maximum
absolute probability drift **1.373e-5**. The numerical implementation agrees,
but decision quality is poor on this initial sample:

| Method | Accuracy / 30 | MLX cached-tail read | MLX fresh-prefix + read |
| --- | ---: | ---: | ---: |
| Letters, depth 1 | 23.3% | 43 ms | 737 ms |
| Letters, depth 2 | 20.0% | 87 ms | 781 ms |
| Letters, depth 3 | 26.7% | 131 ms | 825 ms |
| Letters, depth 4 | 30.0% | 169 ms | 863 ms |
| Yes/No, depth 1 | 26.7% | 47 ms | 789 ms |
| Yes/No, depth 2 | 30.0% | 94 ms | 836 ms |
| Yes/No, depth 3 | 23.3% | 140 ms | 883 ms |
| Yes/No, depth 4 | 33.3% | 181 ms | 924 ms |
| ALoDLM ordinary next-token letters, first pass | 26.7% | — | 142 ms |
| Existing trained 0.6B S1 | 90.0% | — | 39 ms |

At depth 4, Yes/No gets 7/10 binary questions, 0/10 multiple-choice questions,
and 3/10 ordinal-score questions correct. Its 33.3% aggregate therefore does
not establish useful multi-option ranking. This tests a zero-shot shared
Yes/No prompt; no decision-specific fine-tuning has been performed.

Latencies are means of per-case medians. Fresh-request values exclude model
loading and tokenization; the two ALoDLM decision components are measured
separately and summed. Uniform random selection has expected accuracy 28.7%
given these option counts. These small-sample results do not establish that
one zero-shot depth or format is reliably better than another.

ALoDLM's cached MLX reads are 1.41–1.52× faster than PyTorch by group. However,
MLX's all-depth prefix processing is slower here: **fresh-prefix decisions are
about 4–17% slower than PyTorch end to end**, using the ratio of group mean
latencies. Reusing an identical prefix changes that tradeoff; it does not apply
when a browser state or task prompt has changed. The model is not currently a
quality or latency replacement for the trained S1 model.

The evaluated prefixes span 85–340 tokens and tails 4–19 tokens. Sixteen
oversized candidate questions were excluded by the declared 26-option limit,
before predictions. Quality is therefore scoped to this eligible subset.

Full-vocabulary diagnostics on the first six eligible cases found two distinct
problems. Letter slots often put more than 79% of vocabulary probability on
allowed letters but choose the wrong answer. Multi-option Yes/No slots often
put only a few percent (sometimes below 1%) on the allowed labels; their top
tokens include letters, punctuation, newlines and EOS. Restricted normalization
can hide this format mismatch. Binary Yes/No slots have much higher allowed-label
mass. This supports further format research or decision-specific training,
without establishing that either will solve the task.

Raw results: [MLX](../reports/alodlm-mlx-float32.json),
[PyTorch](../reports/alodlm-torch-float32.json),
[parity comparison](../reports/alodlm-parity-float32.json),
[S1 baseline](../reports/alodlm-baseline-s1.json),
[ordinary next-token baseline](../reports/alodlm-baseline-ar.json), and
[label-mass diagnostics](../reports/alodlm-label-mass.json).
Run the diagnostic with `uv run --extra benchmark python scripts/alodlm_diagnostics.py`.

The initial S1 timing attempt synchronized without materializing its lazy MLX
output. It is retained with an explicit invalid-timing annotation; only the
corrected, evaluated-output measurements above are used. The source checkpoint,
upstream reference, incomplete first run, exclusions and raw timing samples are
retained for auditability.

Validation: 114 tests passed, including 20 ALoDLM tests covering every recurrent
depth, full-loop and prelude/coda layouts, exact cache preservation, no-commit
reads, non-monotonic rotary positions, early/late exits, forced commitments,
generation sequences, checkpoint/gate round trips, schema mapping and reference
file hashes. The wheel builds with the source, reference manifest and license
files, and contains no model weights.

## BF16 precision check

A smaller check repeats all six generation configurations and the first six
eligible decision cases (48 readout/depth configurations), using the same
warmup/repetition policy:

- **6/6 generated token sequences match** between PyTorch and MLX, but only
  **5/6 exit-depth sequences match**. Equal visible output does not imply equal
  adaptive execution in this precision.
- **47/48 decisions match.** Maximum probability drift is **0.06218**, and
  maximum selected-logit drift is 0.375.
- The mismatch is the first SNLI entailment case, Yes/No at depth 4. PyTorch
  produces `[0.5, 0.5]` and its argmax selects No; MLX produces approximately
  `[0.43782, 0.56218]` and selects Yes. Both outcomes are retained.
- Measured generation speedups range from 1.92–2.40×, but the differing adaptive
  execution prevents treating every run as an equal-computation comparison.

Use **float32 for reference agreement**. The limited BF16 sample does not
establish parity on other prompts, and its six-case accuracy should not be
compared directly with the 30-case FP32 aggregate.

Reports: [BF16 MLX](../reports/alodlm-mlx-bfloat16.json),
[BF16 PyTorch](../reports/alodlm-torch-bfloat16.json), and
[BF16 comparison](../reports/alodlm-parity-bfloat16.json).
Reproduce the main benchmark commands with `--dtype bfloat16 --decision-cases 6`
and these separate report paths.
