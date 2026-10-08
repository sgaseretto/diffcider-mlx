# ALoDLM on MLX: feasibility and decision-mode review

Reviewed on 2026-10-07 against the [decision proposal](../examples/alodlm-decider.md),
the released checkpoint, and upstream implementation. This is an assessment:
no ALoDLM inference port, accuracy evaluation, training, or latency benchmark has
been performed at the time of this assessment. Tokenizer/configuration checks
used the actual released files. For the subsequent implementation and measurements,
see [the ALoDLM guide](alodlm.md).

| Capability | Assessment | Main work |
| --- | --- | --- |
| Faithful MLX generation | High feasibility | New recurrent causal decoder and depth-aware cache |
| Zero-shot structured decisions | High implementation feasibility; accuracy unproven | Answer-slot compiler, read-only recurrence, restricted readouts |
| Trained decision adapter | Feasible; a separate training project | Recurrence-aware supervision, held-out evaluation and calibration |
| One model with `generate` and `decide` | Straightforward API design once both paths exist | Shared weights, request-local state, adapter-aware caches |
| Faster decisions than our 0.6B S1 model | Unknown | Measure accuracy and cold/warm latency; size and recurrence can outweigh gains |

## Porting requirements

The released 1.7B variant repeats all 28 Qwen3 layers, with a maximum of four
recurrent passes. Its recurrent metadata sets `mask_token_id=151665`. The
backbone uses causal attention and an **untied** output projection. The learned
gate is a separate artifact, `exit_gate.pt`; loading standard Qwen3 weights alone
does not implement ALoDLM. Sources: [checkpoint configuration][config],
[recurrent configuration][recurrence], [upstream model][model].

Our `diffcider/model.py` supplies reusable Qwen3 operations and selected-token
output projections. However, its current attention path has no persistent KV
cache and uses sequential rotary positions. The loader deliberately accepts
`a2d-qwen3`, not arbitrary `qwen3` checkpoints. Changing that validation alone
would produce incorrect inference.

Add a separate ALoDLM backend that preserves:

- Causal attention over physically reordered tokens, while RoPE uses their
  original logical positions.
- Normalized latent carry between recurrent passes and the trained exit gate.
- Cache entries indexed by layer **and recurrent depth**.
- Generation's token commitments, feedback embeddings, observed-token cache
  refresh, missing-depth completion, sliding window and stopping rules.

These requirements are visible in the [portable decoder][inference]. MLX's
ordinary array operations and attention primitives should suffice for an initial
implementation; custom Metal kernels are an optimization question. The CUDA
graphs and kernels in the optimized NVIDIA engine do not transfer directly.

Keep the existing MDLM/S1 implementations working unchanged. Expose ALoDLM's
`generate` / `decide` as user-facing operations, and retain `entropy` / `left1`
as generation decoder settings. Its maximum recurrent depth of four is not the
existing MDLM demo's 24-denoising-step setting: generation also has a variable
number of outer decoding steps. [Upstream inference guide][guide]

## Decision mode

The proposed prefix plus short masked tail is a practical experimental starting
point. Compile all state, questions and options into the observed prefix; run
the tail without committing predicted answers; read only the required LM-head
rows at answer slots. Start with one letter slot per question, then compare
shared Yes/No slots per option. Return logits, normalized scores, actual pass
count and gate diagnostics, with calibration status explicit.

Use fixed execution budgets of 1, 2, 3 and 4 passes initially. This is a separate
execution limit, not a change to the checkpoint's trained four-depth
configuration. Adaptive stopping should follow evidence that it preserves
decision quality. The paper's halting objective concerns generation trajectories;
it does not establish decision confidence for this new read-only path.
[Training method][method]

Corrections and qualifications to the proposal:

1. **A 32-token tail is a conservative starting limit, not an architectural hard
   limit or a guarantee of matching training.** Training blocks start at the
   beginning of each sequence segment, not automatically at the first answer
   mask. A short tail can straddle a block boundary. Test prefix alignment and
   answer formats; no-commit recurrence is itself an intervention.
   [Batch construction][batch]
2. **Softmax does not remove relative Yes/No or S/N bias.** It removes only a
   common additive offset. A bias identical across all option scores cancels
   when those scores are normalized together; binary label bias remains.
3. **Concatenating scores from separate tails does not establish calibration.**
   Chunk context, option order and execution depth can change their scales.
   Compare joint versus split reads and calibrate on held-out data.
4. **Slots are not computationally independent.** Under the proposed causal
   ordering, later masks can attend to earlier masks' hidden states even without
   committing tokens. Measure order sensitivity. Anchored runs need empirical
   validation before becoming the default format.
5. **Gate values are diagnostics, not probabilities of correctness.** A mean
   stopping rule can also hide one difficult question among easy questions.
   Fit review thresholds on decision outcomes, not on generation gate values
   alone. Restricted-label probabilities can conceal low total label mass;
   inspect full-vocabulary mass on evaluation samples.
6. Per-slot compute skipping and dependency branching should be later work.
   Freezing a returned score alone saves no backbone compute, and actually
   removing a slot requires preserving attention/cache semantics. The proposed
   independent fallback for unexplored dependency branches is an approximation,
   not an exact conditional marginal.

The actual tokenizer encodes `A`, `B`, `Yes` and `No` as individual tokens
(IDs 32, 33, 9454 and 2753). Literal `[M]` is two ordinary tokens. Insert the
configured mask ID directly and verify labels at their real template boundaries;
standalone tokenization is only a preliminary check.

## Training and switching modes

No fine-tuning is necessary to build and evaluate a zero-shot prototype. A useful
trained decision model would need labeled tasks representative of its intended
use. For browser use, include state/action pairs and ambiguous or unavailable
actions rather than relying on general generation benchmarks.

A suitable initial training experiment is a new LoRA adapter with supervised
answer-slot losses across selected depths. To target fast S1 behavior, emphasize
first-pass accuracy; optionally test distillation from a stronger/deeper teacher.
Keep the base weights frozen and evaluate adapter-off generation restoration.
Start with fixed-depth evaluation and revisit the gate after the decision path
works. Ordinary causal next-token LoRA training alone does not exercise this
masked recurrent decision path.

Reuse our adapter management approach, not the existing 0.6B adapter/head
weights: dimensions, backbone and training formats differ. An ALoDLM adapter must
apply to the same shared layer weights on every recurrence. Export its required
head/gate changes and calibration metadata explicitly.

One loaded backbone can serve base generation and adapter-enabled decisions
sequentially. Cache reuse requires matching token IDs, logical positions,
attention semantics, dtype, checkpoint and adapter state; switching prompts or
adapters cannot blindly reuse a prior cache. Keep request state isolated.

## Memory and latency

Hugging Face metadata reports **2,031,739,904 BF16 backbone parameters** despite
the 1.7B model name. The separate output head is material to storage: the
safetensors file is 4,063,515,640 bytes. Raw backbone weights therefore occupy
approximately **3.78 GiB BF16** or **7.57 GiB FP32**, excluding runtime memory
and the small gate. [Released weights][weights]

From the configuration, a fully populated four-depth cache in BF16 costs
`2 × 28 layers × 4 depths × 8 KV heads × 128 dimensions × 2 bytes` per token:
**448 KiB/token**, or **1.75 GiB at 4,096 cached tokens per request**. FP32 doubles
that cache estimate. This excludes temporary attention/activation buffers and
allocator overhead. Four passes share one set of weights; they do not require
four weight copies. [Configuration][config]

Upstream generation prefills the prompt at all four depths. Accordingly,
“one-pass decide” on an already prepared prefix is different from cold request
latency. A promising later optimization for a fixed-depth decision request is
to prefill only the depths it will use; recurrence has no dependency on future
depths. This needs parity tests and explicit handling when deeper caches are
subsequently requested. It is not implemented or benchmarked here.

The most useful initial optimizations are prefix reuse for genuinely identical
requests, selected-label projections, and batching compatible tails. These do
not establish an advantage over our smaller S1 model. The paper itself notes
prefill overhead and input-dependent speed; its NVIDIA generation results cannot
be used as Apple Silicon decision benchmarks. [Paper, limitations][paper]

## Recommended implementation order and acceptance checks

1. Port generation against a pinned, independent PyTorch reference. Begin with
   tiny random models in FP32, then the released weights. Compare hidden states,
   gate values, depth caches, token commitments and complete outputs in both
   generation modes, including early exits and noncontiguous commitments.
2. Implement the same explicit no-commit read specification in a separate
   PyTorch reference and in MLX. Keep upstream generation untouched. Validate
   fixed-depth results, label mapping and multiple tails before adaptive reads.
3. Evaluate letters versus Yes/No at depths 1–4 on held-out tasks. Report
   accuracy, NLL/Brier/ECE, order/chunk sensitivity and review coverage. Compare
   with current S1 and a suitable autoregressive baseline on the same tasks.
4. Compare PyTorch and MLX using matched precision, weights, inputs and
   settings. Report cold prefill, warm read, end-to-end latency, throughput and
   peak memory separately. Warm up and synchronize; do not time concurrent GPU
   workloads. Record numerical differences and changed outputs rather than
   promising bitwise equality, especially near adaptive thresholds in BF16.
5. Train a decision adapter only after the baseline exposes the relevant errors.
   Evaluate generation preservation and decision quality separately. Then
   consider BF16, lazy depth prefill, batching and adaptive stopping for speed.

## Licensing and provenance

The released weights and Amazon-authored code use **CC BY-NC 4.0**. The code's
NOTICE explicitly retains WeDLM terms, including a territorial restriction, for
derived material in both implementations. A redistribution must account for
those terms; this is not simply the original Qwen Apache license. Preserve
applicable notices and scope any integration accordingly. [Model NOTICE][notice-hf],
[code NOTICE][notice-code], [retained WeDLM terms][wedlm-license]

Reviewed revisions:

- Code: `amazon-science/ALoDLM@1da9ddafeca02425480230b9bb7091fdef5672b7`.
- Checkpoint: `amazon/ALoDLM-1.7B@588c7dc1946e3e09fd496da0b0808bd26f46a758`.

[config]: https://huggingface.co/amazon/ALoDLM-1.7B/blob/588c7dc1946e3e09fd496da0b0808bd26f46a758/config.json
[recurrence]: https://huggingface.co/amazon/ALoDLM-1.7B/blob/588c7dc1946e3e09fd496da0b0808bd26f46a758/alodlm_config.json
[weights]: https://huggingface.co/amazon/ALoDLM-1.7B/blob/588c7dc1946e3e09fd496da0b0808bd26f46a758/model.safetensors
[model]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/alodlm/model.py
[inference]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/alodlm/inference.py
[batch]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/alodlm/batch.py
[guide]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/docs/inference.md
[method]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/docs/method.md
[paper]: https://arxiv.org/html/2610.04198v1
[notice-hf]: https://huggingface.co/amazon/ALoDLM-1.7B/blob/588c7dc1946e3e09fd496da0b0808bd26f46a758/NOTICE
[notice-code]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/NOTICE
[wedlm-license]: https://github.com/amazon-science/ALoDLM/blob/1da9ddafeca02425480230b9bb7091fdef5672b7/optimized/licenses/WeDLM.txt
