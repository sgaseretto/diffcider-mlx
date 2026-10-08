# ALoDLM-1.7B as a decision model — implementation notes

Status: design draft, nothing implemented yet. Date: 2026-10-07.

This document turns the explanatory web artifact ("La función de lectura que falta") into an implementation guide. It describes how to make **ALoDLM-1.7B** answer typed questions with probabilities, the way **djev** does with DiffusionGemma, instead of generating text.

All probabilities, entropies and token counts in examples are **illustrative**. Facts about ALoDLM's code refer to `amazon-science/ALoDLM` at the commit cloned on 2026-10-07; re-check line numbers before relying on them.

---

## 0. Summary

- **Goal.** Given a *state* (for example a support ticket) and typed questions (`choice`, `noul` = yes/no, `score`), return a probability distribution per question in one prefill plus 1–4 recurrent passes, without generating text.
- **Approach.**
  - Use the djev zero-shot layout. Everything (instructions, questions, options, the state) goes in a **prefix** that is prefilled once into ALoDLM's prefix cache.
  - A short **tail** (≤ 32 tokens) holds the answer slots `[M]` plus minimal template text.
  - Run the recurrent passes on the tail **without committing tokens**, then read restricted softmaxes at the slots.
- **Most of it already exists** in `alodlm/inference.py`:
  - The prefix cache.
  - Prefill via `_window()` with no masks.
  - Observed tokens anywhere in the window, physically moved ahead of masks.

  What's missing is a variant of `_window()` that does not commit and that returns logits and exit-gate values.
- **Two readouts.**
  - **Letters (djev):** one `[M]` per question; read the logits of single-token labels A/B/C.
  - **Yes/No (jev-dllm):** one `[M]` per option; read z_Yes − z_No and take a softmax across options.

  Start with letters. Test Yes/No in parallel, preferably in the **anchored-run** format, which keeps the tail almost as short as letters.
- **Main hard constraint:** keep each tail ≤ 32 tokens, the training block size.
- **Licence:** ALoDLM weights and code are **CC BY-NC 4.0**. Some WeDLM-derived components carry a territorial restriction. Non-commercial use only.

---

## 1. Background

### 1.1 Jev and djev

- **Jev** (TypeSafe AI, launched 2026-09-15) is a closed "System One" decision model. It takes a state and typed questions and returns choices, scores or probabilities, never prose.
- **djev** (Matt Mastracci; vLLM PR #57250, merged 2026-09-22) reproduces this with DiffusionGemma:
  1. Seed the diffusion canvas with a response template.
  2. Leave only the answer slots noisy.
  3. Run **one** denoising step.
  4. Read the probability distribution over allowed single-token labels at each slot.
- Known djev limitations: each option must be exactly one token, and the probabilities are uncalibrated. Its JevBench sealed-set accuracy dropped sharply (29.9% sealed vs 84.0% public), which suggests overfitting to public examples.

### 1.2 ALoDLM-1.7B

Sources: the paper (arXiv 2610.04198), `README.md`, `docs/inference.md`, the HF model card `amazon/ALoDLM-1.7B`, and the code.

- **Base and training.** Initialised from **Qwen3-1.7B** (Qwen3 tokenizer, ~151k vocabulary). SFT on about 5B tokens of maths, code and instruction data, with no continued pretraining.
- **Recurrent core.** In 1.7B the core spans **all 28 layers** (`loop_start = 0`, `loop_end = 28`). The prelude is just the embedding table, and there is no coda. Maximum recurrent depth is **K = 4**.
- **Exit gate.** After each pass, a learned gate gives each position a halting probability λ. In the code: `hazard = sigmoid(exit_gate(features, depth))`, accumulated as `survival *= 1 − hazard`, so the cumulative halt is `a = 1 − survival`.
- **Commit rule (entropy mode, the default).** A position commits when `entropy + relative_position × penalty < τ`.
  - Defaults: `τ = 0.4`, `penalty = 0.05 · τ` per position, measured from the first masked position.
  - The `left1` mode instead commits the leftmost uncommitted position on every pass.
- **Inner-loop stop.** The loop over passes ends when no masks remain, when the mean cumulative halt over uncommitted positions reaches **q**, or at K.
  - Default in `DecodeConfig`: `q = 0.5`.
  - The paper's quality setting for 1.7B uses `q = 0.4`.
- **Attention.** Causal attention inherited from **WeDLM**. *Topological reordering* places observed tokens physically before masks while keeping their logical positions for RoPE. As a result, every mask sees all observed tokens in its window, including ones that are logically after it.
- **Licence:** CC BY-NC 4.0.

### 1.3 How `generate()` works today (`alodlm/inference.py`)

**`DecodeConfig`** holds `q = 0.5`, `tau = 0.4`, `window_size = 16`, `max_new_tokens`, `position_penalty = None` (meaning τ × 0.05) and `mode = "entropy" | "left1"`.

**`PrefixCache`** stores depth-specific K/V: `key(layer, depth)` uses the real depth for recurrent layers and depth 0 otherwise.
- `complete(window, completed)` copies the last computed depth into depths that were never run.
- `append(window, rows)` concatenates the selected rows into **new** tensors.

**`Decoder._window(ids, positions, cache, mask_rows, config, logical_mask_positions)`** (around lines 71–146):
1. Embeds the inputs, builds RoPE from the given positions, and builds attention: everything in the cache is visible, and the window is causal (`tril`).
2. Loops over `depth in range(max_depth)`. On each pass:
   - Newly committed tokens are re-injected as token embeddings.
   - The core layers run at that depth, followed by the readout norm.
   - At the mask rows it computes `logits`, `entropy`, `hazard`, and updates `survival`.
   - Positions are committed by the entropy rule, or by `left1` (lines ~123–135).
   - The stop condition is checked (line ~136). If the step ends with nothing committed, the position with the lowest adjusted entropy is force-committed (lines ~137–141).
3. Between depths: `hidden = base.norm(hidden)`.
4. **With zero masks**, as in prefill, the loop only stops at the last depth, so **prefill runs the prompt through all 4 depths**. For 1.7B that is roughly 4 forward passes over the prompt.

**`Decoder.generate()`** (around lines 149–242):
1. Prefill: `_window(prompt, arange(len(prompt)), cache, empty, …)`, then `cache.append(…)`.
2. Each step:
   - `prefix` = the number of **contiguous** committed tokens from the left of the window (line ~184).
   - Masks are added until the window holds `prefix + window_size` positions (lines ~196–199).
   - The order is observed tokens first, then masks; `positions = cache.length + order` (logical positions).
   - `_window()` runs once, then `cache.append(window, inverse[:prefix])`. The contiguous run is recomputed as observed tokens in this step, and its K/V enters the cache now: there is a one-step lag.
   - Newly committed tokens become observed. Uncommitted positions **lose their latent state** and start the next step as `[M]` again.

**Window movement.**
- The window advances by the contiguous committed run from its left edge: **anywhere from 0 to 16 positions per step**.
- A token committed out of order (say N+3 before N) stays committed and is observed, moved physically ahead of the masks. The window only jumps once the left side catches up.
- The position penalty and the forced commit make the left side resolve first in practice.
- In `left1` mode the slide equals the number of passes the step ran (1–4).

### 1.4 What's missing for decision reads

1. **No fixed template inside the generated region.** Everything after the prompt starts as `[M]` in a sliding window.
2. **It always commits.** It samples and commits on every pass, and force-commits if nothing passed the threshold.
3. **It returns text**, token IDs, exit depths, first-pass halt probabilities and timings, but **not the distributions**. The per-option probabilities are used for sampling and then discarded.

### 1.5 Training data layout (why the tail limit is 32)

`alodlm/batch.py` (`build_batch`, dual-stream masking from WeDLM) works like this, with `block_size: 32`:
- The sequence is cut into **32-token blocks**. Within each block, a fraction p ~ U(0, 1) of the **response** tokens (`labels != -100`; prompt tokens are never masked) is masked at random positions.
- Within a block, observed tokens are reordered ahead of the masks.
- Attention (`attention_mask`): a noisy-stream token sees tokens of its own block causally (after reordering) and **the clean stream of all preceding blocks**.

Consequences for reads:
- A mask has seen observed tokens *after* it only within its own 32-token block.
- It has never seen tokens from later blocks, and never coexisted with masks in another block.
- Masks only ever appeared in the **assistant/response** region, so put slots in the assistant turn, not the user turn.

---

## 2. Design: `structured_read`

### 2.1 Layout: prefix plus tail (djev zero-shot)

```
[PREFIX → cache, computed once]
  system/user: instructions
               questions, each with options and descriptions
               the state (ticket)
  assistant:   <template up to the first slot>, e.g. "categoria:"
[TAIL ≤ 32 tokens → processed on every pass]
  [M] ↵ urgente : [M]
```

- **Prefix:** everything before the first `[M]`. It is prefilled once into the `PrefixCache` (all 4 depths) and can be any length up to the backbone's context limit.
- **Tail:** from the first `[M]` to the end: the slots plus the template text between them. It is the only part recomputed on each pass.
- **Ordering for reuse:** put what doesn't change first. With many tickets and the same questions, the order is instructions + questions + options, then the state, then the template. The constant part's cache is computed once.
- **Chat template:** Qwen3 template with `enable_thinking=False`. Slots go in the assistant turn.
- **Labels** must each be a **single token**, unique and non-special. Check with `tokenizer.encode(label, add_special_tokens=False)`; jev-dllm's `token_mapping()` is a good reference check.

### 2.2 The four implementation steps

| Step | What | Status |
|---|---|---|
| 1 | Seed the template: build `prefix_ids` and `tail_ids` with `is_slot` flags | new, simple |
| 2 | Reorder the tail (observed first, masks last; logical positions kept) | **already done by `generate()`'s logic**; reuse the same construction |
| 3 | Disable commits: copy `_window()` as `read_window()` without the commit and forced-commit lines; stop after N passes or at q | new, about 40 lines |
| 4 | Restricted softmax over allowed label tokens per slot; also return exit-gate halts | new, simple |

### 2.3 Reference sketch

Not an existing API; written against `alodlm/inference.py`.

```python
def structured_read(dec, prefix_ids, tail_ids, is_slot, allowed, slot_names,
                    passes=4, q=None, cache=None):
    cfg = dec.model.config
    if cache is None:
        cache = PrefixCache(cfg)
        pre = dec._window(prefix_ids, arange(len(prefix_ids)), cache,
                          empty, decode_cfg, empty)[0]       # runs all 4 depths
        cache.append(pre, arange(len(prefix_ids)))
    observed = [i for i in range(len(tail_ids)) if not is_slot[i]]
    masked   = [i for i in range(len(tail_ids)) if is_slot[i]]
    order    = tensor(observed + masked)
    reads = read_window(dec, tail_ids[order], cache.length + order, cache,
                        mask_rows=arange(len(observed), len(order)),
                        passes=passes, q=q)
    logits, halt_cum, lam1 = reads.final_logits, reads.final_halt, reads.first_hazard
    probs = {name: softmax(logits[j, allowed[name]]) for j, name in enumerate(slot_names)}
    return probs, halt_cum, lam1, reads
```

```python
def read_window(dec, ids, positions, cache, mask_rows, passes=4, q=None):
    base, c = dec.model.backbone.model, dec.model.config
    hidden = base.embed_tokens(ids)
    cos, sin = rope(base, hidden, positions)
    attn = ones(L, cache.length + L); attn[:, cache.length:] = tril(ones(L, L))
    for layer in range(c.loop_start):                 # empty for 1.7B
        hidden = layer_fwd(layer, hidden, cache, depth=0)
    survival, per_pass = ones(len(mask_rows)), []
    for depth in range(min(passes, c.max_depth)):
        for layer in range(c.loop_start, c.loop_end):
            hidden = layer_fwd(layer, hidden, cache, depth)   # reads depth-specific K/V
        readout = hidden
        for layer in range(c.loop_end, n_layers):     # empty for 1.7B
            readout = layer_fwd(layer, readout, cache, depth)
        feats  = base.norm(readout)[mask_rows]
        logits = lm_head(feats).float()
        lam    = exit_gate(feats, depth).float().sigmoid()    # set λ = 1 at depth K−1
        survival = survival * (1 - lam)
        per_pass.append((logits, lam, 1 - survival))
        if q is not None and (1 - survival).mean() >= q:
            break
        hidden = base.norm(hidden)                    # as in _window between depths
    return per_pass                                   # no commits, no cache.append
```

Notes:
- **No `cache.append` in reads**, so one `PrefixCache` can serve many reads.
- **Branching off a cached prefix** (shared constant part, then a different ticket): `append` builds new tensors with `torch.cat`, so a shallow copy (`copy.copy(cache)` with `values = dict(cache.values)`) can be extended without changing the original.
- **Batching:** `_window()` handles one sequence. Version 1 can loop over tails. Later, batch tails as rows (shared prefix K/V, per-row padding and attention masks), or port the logic to the optimized engine under `optimized/` (a nano-vLLM fork with CUDA graphs).

### 2.4 Tail must fit the training block

- Keep the tail at **≤ 32 tokens** (§1.5).
- The prefix can be long; the tail cannot.
- Do **not** change `window_size`; it only affects `generate()`.
- Tails of 64, 128 or 256 tokens are **out of distribution**: masks would look more than 31 tokens ahead and span several blocks. Treat that as an experiment, not a default.
- If the questions don't fit, use **several tails over the same cache**, run as rows of a batch. Each read stays in distribution.

### 2.5 Tail optimization rules

1. Put everything possible in the prefix: instructions, options with descriptions, and the first question's label.
2. Use short labels between slots ("urg:" rather than "Is this case urgent?:"), ideally single tokens.
3. Put nothing after the last `[M]`. A trailing instruction is paid for on every pass.
4. If it doesn't fit in 32, split into several tails rather than making one long one.

Illustrative comparison (two questions, 4 passes):

| Template | Tail tokens | Tokens processed by passes | Fits 32 | vs optimized |
|---|---|---|---|---|
| Optimized `[M] ↵ urg : [M]` | ~5 | 20 | yes | ×1.0 |
| Long labels between slots | ~13 | 52 | yes | ×2.6 |
| Descriptions and instruction in the tail | ~55 | 220 | **no** | ×11 |

**Cost model.** A forward pass over a few dozen tokens is weight-bandwidth bound, so per-pass time barely changes between 5 and 30 tokens. What you actually optimize is:
1. Never exceeding 32 tokens per tail.
2. The number of tails, since each is an extra batch row.
3. Balanced tail lengths, since batch rows pad to the longest.

---

## 3. Readouts

### 3.1 Letters (djev style; start here)

- One `[M]` per question.
- Options are listed in the prefix with single-token labels (A, B, C… or S/N, or digits for scores).
- p = softmax of the LM-head logits restricted to the allowed label tokens.
- Tail cost: about 4 tokens per question (`label : [M] ↵`); the first label sits in the prefix.
- Limits:
  - About 26 options per question (single-token letters).
  - Each letter has its own output vector, so there may be an identifier bias (a preference for "A"). It is unmeasured.
  - A question can't be split across reads: each read normalizes to 100%, so the winners of separate reads aren't comparable.

### 3.2 Yes/No per option (jev-dllm style)

Source: `zhouzihao11/jev-dllm` (Apache-2.0 code, last commit 2026-10-02). Their model is a Qwen3-0.6B bidirectional MDLM, single forward pass.

- **Their template:** "For each option, mark Yes if it answers the question, otherwise No.\n" followed by `option: [MASK]` lines. For noul: "Answer Yes or No.\nAnswer:[MASK]".
- **Readout:** s_i = z_Yes(i) − z_No(i), using the same two vocabulary vectors for every option.
  - Choice: p = softmax(s).
  - Noul: p = σ(s).
  - Score: softmax over the level masks, plus the expected value.
- **Adaptation for ALoDLM:** put the `option: [M]` lines in the **assistant** turn, since only response tokens were masked in training, with the option descriptions in the prefix. Check that "Yes" and "No" are single tokens (jev-dllm uses English Yes/No).
- **Their reported numbers** (0.6B bidirectional; reference only, not predictive for ALoDLM):

| Set | Zero-shot base | Fine-tuned (S1, 40k decisions) |
|---|---|---|
| JevBench public (231) | 58.4%, ECE 13% | 67.1%, ECE 14% |
| JevBench "original" (72) | 55.6% | 86.1% |
| Banking77 (77 classes) | 24.1% | 60.5% |
| Emotion ECE | 8% | 23% (calibration worse) |

- There is **no published comparison** between Yes/No and a single-mask letter readout. Their code has both modes (`identifier` and `shared_yesno`), but only Yes/No results are reported.

### 3.3 Advantages and drawbacks of Yes/No versus letters

Each was demonstrated with toy numbers in the artifact's tabs.

**For Yes/No:**
1. **More than 26 options.** Letters run out of single-token labels; Yes/No just adds lines and tails.
2. **Combinable scores.** If a question is split across reads, letters give two winners with incomparable probabilities. Example: 0.58 and 0.51 from separate reads. With Yes/No, all s_i are concatenated before one softmax, and the weak "winner" (s = +0.2) drops to 0.07.
3. **Same weights for all options.** Moving an option from A to C doesn't change which vector scores it. Demo with a hypothetical letter bias of +0.4 for A and −0.3 for C: "facturación" goes from 0.86 as A to 0.67 as C with letters, and stays at 0.87 with Yes/No.
4. **A global "Yes" bias cancels** in the choice softmax (shift invariance).
5. **One mechanism for all three question types.**

**Against Yes/No:**
1. **Longer tail.** About 5 tokens per option in jev-dllm's format; mitigated by the anchored run (§3.4).
2. **Each option is judged alone.** A mask sees the other options' labels but not their judgments.
   - When two options fit, letters can prefer the more specific one (illustrative: 0.70 vs 0.25), while Yes/No nearly ties (s ≈ +2.0 and +1.9).
   - Upside: when none fits, all raw s values are negative, which signals "none", something letters can't express.
3. **Noul bias doesn't cancel.** With a single mask there is no softmax, so a "Yes" bias directly moves P(yes) (illustrative: 0.40 → above 0.5, flipping the answer). Calibrate it (§4).

### 3.4 Anchored run: the tail-efficient Yes/No format

- **Format.**
  - The prefix enumerates each question's options in a fixed order: `1 · categoria, in order: 1 facturación, 2 técnico, 3 cuenta, 4 otro`.
  - The tail has a short anchor per question followed by a run of k masks: `1:[M][M][M][M] 2:[M] 3:[M][M][M] 4:[M][M][M]`.
  - The i-th mask after an anchor answers option i of that question.
- **k is set by the schema, not the model:**
  - `choice`: k = number of options.
  - `noul`: k = 1.
  - `score`: k = number of levels.
- **Cost:** about 2 + k tokens per question. With k = 4, about 5 questions fit in one 32-token tail (letters: about 8).
- **Why it should work:** masks know their position (logical RoPE positions), and runs of consecutive masks are common in training (p up to 100% within a block).
- **Risk:** identification is by position, and long runs may miscount. Mitigations:
  - Re-anchor every 5–6 masks, e.g. `1:[M]×6 1·7:[M]×6`, which costs about 4 extra tokens per continuation anchor.
  - Or fall back to **short codes**: `↵a[M]b[M]c[M]`, about 1 + 2k tokens per question. Use letters or single digits; Qwen3 splits multi-digit numbers into separate tokens.
- **Order consistency** between prefix and tail is guaranteed by the compiler, not by the model.
- **Untested on ALoDLM.** Measure it against short codes.

Format comparison from the artifact's comparator (four questions: categoria with k ∈ {4, 12}, urgente k = 1, sentimiento k = 3, idioma k = 3; first-fit-decreasing packing into 32-token tails, Yes/No questions splittable):

| Format | Per-question tail cost | k = 4: tails / tokens | k = 12: tails / tokens |
|---|---|---|---|
| jev-dllm Yes/No `option: [M]↵` | ~5 per option | 2 / 55 | 4 / 95 |
| Short codes `↵a[M]b[M]…` | 1 + 2k | 1 / 26 | 2 / 42 |
| **Anchored run `1:[M]…`** | **2 + k** | **1 / 19** | **1 / 27** |
| Letters (djev) | ~4 | 1 / 16 | 1 / 16 |

### 3.5 Format per question type (recommended defaults for the compiler)

| Question type | Letters path | Yes/No path |
|---|---|---|
| Choice, ≤ 26 options | one letter slot | anchored run, k = options |
| Choice, > 26 options, or multi-label | tree readout (§6.3) or not supported | anchored run (re-anchored), splittable; σ per option for multi-label |
| Noul | letter slot reading S vs N (the softmax cancels the bias) | single mask, σ(s), **calibrated** |
| Score 0..k | one slot reading digits 0..k, plus the expected value | k + 1 masks, softmax, expected value |

---

## 4. Calibration

- **Never trust raw probabilities.** djev and jev-dllm both show calibration problems; fine-tuning can raise accuracy and worsen ECE.
- **Temperature scaling** per question type and per number-of-options bucket, as Laya does, fitted on held-out labelled data.
- **Contextual (null-state) calibration**, cheap with the cache:
  1. Run each question schema once with an empty state ("N/A").
  2. Record the bias per answer token (letters) or per option (Yes/No).
  3. Subtract it from every real read.

  This is computed once per schema and reused for every ticket. It counters letter preferences, generic options like "otro" that always score low, and the noul "Yes" bias.
- **Optional for letters:** average 2–3 cyclic re-letterings run as batch rows. Put the option list after the state in the prefix, so only that part is recomputed per permutation.
- **Metrics:** accuracy, ECE (15 equal-width bins), NLL, Brier, and MAE / RPS for scores.

---

## 5. Refinement passes, q and confidence

### 5.1 Passes

- Within the single read, each pass re-runs the recurrent core on the tail's latent states.
- Masks keep their latent state between passes, because nothing is committed.
- After each pass you can read logits and halting.
- In generation, easy tokens commit after 1 pass and hard ones after up to 4. In a read, the commit is replaced by reading.
- Whether distributions sharpen with more passes is **not guaranteed**. The model card states that extra passes don't guarantee better answers.
- The goal of refinement is accuracy, not "higher confidence". Confidence rising without accuracy rising is overconfidence.

### 5.2 q and adaptive passes

- **Per-pass halt:** λ = sigmoid(exit_gate(·)). Cumulative: a ← a + (1 − a)·λ, with λ = 1 at K.
- **q:** the loop stops when mean(a over the slots) ≥ q, or at K = 4.
- **Higher q** means more refinement on average.
  - Paper data (8B, GSM8K, τ = 0.2): raising q from 0.1 to 0.9 moved accuracy from 93.3% to 93.8% and throughput from 455.3 to 309.0 tok/s.
  - Defaults: q = 0.4 (1.7B), 0.5 (8B).
- Illustrative λ example (three slots, λ per pass):
  - A: .60, .70, .80
  - B: .30, .45, .60
  - C: .10, .20, .35

  Mean cumulative a per pass is .33, .59, .79, 1.0. So q = 0.3 stops after pass 1, q = 0.5 after pass 2, q = 0.7 after pass 3, and q = 0.9 after pass 4.
- **Adaptive passes are dynamic per request:** one request may stop after 2 passes and another use all 4.

### 5.3 Two different confidence signals

| Signal | What it says | Use | Caveat |
|---|---|---|---|
| Slot distribution | How much the model believes in each option | Pick the answer; minimum probability threshold | Uncalibrated; more passes may inflate it |
| Exit gate (halting) | Whether the position needs more compute | Number of passes (via q); doubt flag when first-pass halt **λ₁** is low | That λ₁ predicts errors is a **hypothesis**; validate on labelled data |

Example flag rule used in the artifact: review if λ₁ < 0.4 or p_max < 0.6.

Illustrative batch of 6 tickets at q = 0.5: mean 2.2 passes instead of 4, with 2 of 6 flagged (one for low p, one for low λ₁).

### 5.4 Optional: per-slot freezing

- **Group stop (current rule):** all slots stop when the *mean* halt reaches q. An easy slot can drag a hard one to stop early.
  - Example: categoria a = .62 / .86 and urgente a = .24 / .45. The group stops at pass 2, and urgente is read with a = .45 < q.
- **Per-slot:** freeze each slot when its own a ≥ q. Store its read at that pass, stop computing it, and keep its last latent visible to later slots. Continue with the rest.
  - Example: categoria freezes at pass 1, urgente at pass 3.
  - Same total compute (4 slot-passes), better allocated.
  - Latency equals that of the hardest slot (3 passes vs 2).
- This is consistent with training, where each token had its own exit depth, but it is an additional decoder change. Do it after the basic read works, and measure it.

---

## 6. Dependent questions

**Default:** all questions are independent and answered in a single read. This is the cheapest option and enough for most Jev-style schemas.

### 6.1 Declared dependencies (recommended when needed)

Whoever defines the questions declares them. These fields are a **proposal**; they are not in Jev's API:

```json
{
  "categoria": { "type": "choice", "criteria": { "facturación": "…", "técnico": "…", "cuenta": "…", "otro": "…" } },
  "idioma":    { "type": "choice", "criteria": { "es": "…", "en": "…", "pt": "…" } },
  "urgente":   { "type": "noul", "depends_on": "categoria" },
  "importe":   { "type": "score", "criteria": ["bajo", "medio", "alto"], "ask_if": { "categoria": "facturación" } }
}
```

Algorithm:
1. **Compile a dependency graph and sort it into levels.** Here, level 1 is {categoria, idioma} and level 2 is {urgente, importe}.
2. **Level 1:** a normal read.
3. **Branching:** for each parent, keep the most probable values until they cover a threshold (e.g. 90%).
   - Example: facturación .78 + técnico .14 = .92.
   - The leftover mass (.08) is covered by an independent read of the child.
4. **Level 2:** one batch row per kept parent value, with the parent written as observed text in the tail (`categoria=facturación ↵ urgente:[M]`). Only the tail changes; the prefix stays cached.
   - `ask_if` children are only read in matching branches; `importe` only in the facturación branch.
5. **Combine by total probability:**

   P(child) ≈ Σ over kept values v of P(parent = v) · P(child | parent = v), plus (1 − Σ kept) · P_independent(child).

   Example: 0.78 × 0.80 + 0.14 × 0.30 + 0.08 × 0.67 = **0.72**.

Cost: one extra step per dependency level. The number of rows is the number of kept branches (+1 for the independent read).

### 6.2 Joint mode (optional, model decides the order)

- **Procedure:**
  1. Read all questions.
  2. Commit those above a confidence threshold (e.g. ≥ 0.80) as observed text in the tail.
  3. Run another step for the rest, and repeat.
  4. If none passes, force the most confident one, as the decoder does.
- **Pros:** no schema work, and it can capture undeclared dependencies.
- **Cons:**
  1. **Error propagation.** A wrong early commit (e.g. «técnico» at 0.81 when it was billing) conditions every later answer.
  2. **Conditional probabilities.** You get P(urgente | committed answers), not the marginal P(urgente), which makes calibration harder.
  3. **Order and number of steps vary per request.** Latency varies and results are harder to reproduce.
- Treat it as an **accuracy experiment**, not a default.

### 6.3 Related (letters, more than 26 options): tree readout

Split the options into groups and read `group: [M]`. For the top groups (one batch row each, with the group committed), read `option: [M]`. Then P(option) = P(group) · P(option | group).

---

## 7. Latency and batching

- **Prefill** costs about 4 forward passes over the prefix for 1.7B (all depths). It is paid once per prefix and can be reused.
- **Each pass** costs about one decode-step forward of the 1.7B model over the tail tokens, which attend to the cached prefix.
- With **long prompts the prefill likely dominates**, and saving passes helps little. With short prompts, passes are a large share. **Measure prefill and passes separately before optimizing.**
- **p99:** with adaptive passes, plan capacity for K = 4.
- **Batching:** with "wait for the batch", every request waits for the slowest one (4 passes).
  - Illustrative: 6 tickets with stops at 1, 2, 1, 4, 3, 2 passes.
  - Waiting: everyone gets 4 passes of latency, and 11 of 24 cells do no useful work.
  - **Retire on stop:** each request returns at its own pass (mean 2.2), and freed slots take new requests.
- **Prefix cache reuse** for several questions about the same state, or the same questions about many states, is the biggest single saving.

---

## 8. Evaluation plan

1. **Trivial baseline.** Put the slot as the first generated token and read next-token logprobs, both for ALoDLM and for Qwen3-1.7B through djev's AR engine. If ALoDLM's diffusion read doesn't beat this, the approach isn't paying off.
2. **Harness: reuse jev-dllm's** (`benchmark/`):
   - Core17: 17 suites, 6,744 decisions, with an overlap audit.
   - An official JevBench public runner (231 decisions).
   - A Jev-compatible stdlib HTTP server (`/v1/systemone`).
   - **Add an `alodlm` backend:** an adapter plus a new `--backend` choice (it currently accepts `dllm`, `laya` and `kev`). This gives numbers comparable to their base, S0/S1, Laya and Kev.
3. **Sweeps:**
   - Passes ∈ {1, 2, 3, 4}, and q ∈ {0.1 … 0.9} for adaptive passes.
   - Readout: letters vs Yes/No (jev-dllm lines) vs anchored run vs short codes.
   - Re-anchor interval for long runs.
   - With and without null-state calibration; with and without temperature.
   - Tail length (≤ 32 vs experimental 64+).
   - For each: accuracy, ECE15, NLL, Brier, mean passes, prefill and pass latency.
4. **λ₁ as an error predictor:** AUROC of λ₁ (and of entropy) for detecting wrong answers.
5. **Held-out / sealed data:** djev's sealed-set collapse is a warning. Keep a private labelled set.

---

## 9. Implementation milestones

1. **Prompt builder and compiler** (pure Python):
   - Schema → prefix text, tail tokens, `is_slot` flags, allowed label ids per slot.
   - Single-token label validation.
   - Bin packing (first-fit decreasing into 32-token tails; Yes/No questions splittable with a continuation anchor; minimize tail count, then balance lengths).
   - Per-type format selection (§3.5).
2. **`read_window()` and `structured_read()`** on the portable decoder (§2.3). Test on CPU (fp32) with a tiny random model from the repo's tests, then on GPU.
3. **Prefix cache management:** constant-prefix caching and copy-on-extend per state.
4. **Readouts:** letters, then Yes/No (anchored run and short codes). Restricted softmax and sigmoid.
5. **Calibration:** null-state bias per schema; temperature per type and option-count bucket.
6. **Adaptive passes** (q) and confidence outputs (distribution, cumulative halt, λ₁).
7. **Evaluation:** jev-dllm harness adapter; baselines; sweeps (§8).
8. **Optional:** batching across tails; declared dependencies (§6.1); per-slot freezing (§5.4); joint mode (§6.2); tree readout (§6.3); porting to the optimized engine.

Suggested output schema per question:

```json
{
  "categoria": {
    "answer": "facturación",
    "probabilities": {"facturación": 0.78, "técnico": 0.14, "cuenta": 0.05, "otro": 0.03},
    "passes": 2,
    "halt_cumulative": 0.86,
    "first_pass_halt": 0.62,
    "flag_review": false
  }
}
```

---

## 10. Open questions and risks

- **Zero-shot quality** of ALoDLM-1.7B on classification. Its SFT data is maths, code and instructions; expect less than DiffusionGemma 26B.
- Whether **refinement passes** improve accuracy or only inflate confidence on decision reads.
- Whether **λ₁** predicts errors.
- Whether the **anchored run** identifies options reliably by position, and how long a run can be.
- **Letters vs Yes/No:** no published head-to-head; we need to run it.
- **Tails over 32 tokens:** out of distribution; behaviour unknown.
- **Exact token counts** depend on the Qwen3 tokenizer. The artifact's counts are estimates.
- **Licence:** CC BY-NC 4.0 limits use; check before any product use.

---

## 11. Alternatives considered and set aside (for now)

- **Laya-style layout** (`[MASK] option0 [MASK] option1 …` with a trained option-scorer head on a bidirectional encoder). Requires fine-tuning: Laya's own base checkpoints are near chance zero-shot (about 0.36) and reach 0.766 fine-tuned. ALoDLM's causal attention also only gives full visibility inside the 32-token block. Revisit if we fine-tune a scorer head on ALoDLM latents.
- **Larger `window_size` or long tails:** see §2.4.
- **Fine-tuning:** jev-dllm's S0/S1 data (40k decisions; each source keeps its own licence) and loss (cross-entropy plus 0.25 × RPS for scores) could be ported to ALoDLM's trainer (`alodlm-train`), still under CC BY-NC.

---

## 12. References

- ALoDLM: https://alo-dlm.github.io/ · https://github.com/amazon-science/ALoDLM · https://huggingface.co/amazon/ALoDLM-1.7B · paper arXiv 2610.04198
- WeDLM (topological reordering, streaming decoding): https://arxiv.org/abs/2512.22737
- djev / DiffusionGemma-as-Jev: vLLM PR #57250 (https://github.com/vllm-project/vllm/pull/57250), https://github.com/mmastrac/djev
- jev-dllm (shared Yes/No): https://github.com/zhouzihao11/jev-dllm
- Laya: https://github.com/NandhaKishorM/laya
- Explanatory web artifact (Spanish, animated): "La función de lectura que falta"