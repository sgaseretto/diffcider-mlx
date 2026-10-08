# Diffcider Browser: one model, two inference modes

The example runs [sgaseretto/diffcider-browser](https://huggingface.co/sgaseretto/diffcider-browser)
on Apple Silicon, with a Gradio UI inspired by
[laya-ultrafast](https://github.com/ipenywis/laya-ultrafast). A fresh Chromium
session observes the page, the trained model chooses an operation and target,
and the same backbone generates text when filling a field. There is no OpenAI
API, external planner, second model, or PyTorch dependency in the application.

## Run it

From this checkout, with Python 3.13+ on Apple Silicon:

```sh
uv sync --extra browser-demo
uv run playwright install chromium
uv run diffcider-browser
```

Open <http://127.0.0.1:7860>. The first launch downloads the pinned base and
fine-tune; subsequent launches use the Hugging Face cache. Choose a **Website**:
**Google Flights (mock)**, **Skyscanner (mock)**, or **Reading room (mock)**.
Changing it loads the matching suggested goal and clears the previous run,
screenshot, rankings and trace. Reading room also offers three **Suggested task**
choices. **Use suggested task** restores the selected example after editing it.
Flight dates start 30 days ahead; changing the date updates the goal and resets
the run. Merely focusing and leaving an unchanged date does not overwrite edits.

**Run automatically** starts a fresh browser, or resumes the current paused one.
For inspection, use **Inspect** to open a page without calling the
model, then **Choose next** to compute a decision and any field text without
executing it. Inspect the orange target and probabilities, then **Execute choice**
to perform exactly that action. **Pause** stops automatic progression after the
current action; choose/execute or resume from the same page afterwards. **Stop**
ends the run after any in-flight inference and closes its browser, keeping the
final view. Manual inspection releases the model between calls, so another viewer
can use it; the shared model's request lock still serializes inference.

**Restart** cancels the current/queued run, clears the screenshot and trace, and
resets that viewer's run identity. It keeps the edited goal, scenario, date and
settings. Start again to open the site's initial page with empty action history
and fresh cookies. Editing the goal or public URL now invalidates the old run
automatically; it cannot resume under a different prompt. Starting any new run
clears the previous output immediately. Other viewers' runs are unaffected.
An in-flight model call can finish in the background; its old output is discarded
after a restart, and the model's request lock serializes subsequent inference.
Unattended inspection times out after three minutes at a decision/action boundary.
Time spent inspecting does not consume the separate three-minute active run limit;
abandoned viewer state is expired after an hour.

**Generation diffusion steps** is visible in the task panel: **1–24**, default
**24**. The browser demo uses `max_new_tokens=max(16, steps)` in one block
(`block_size=32`), so 24 means **24 real forward passes over 24 output positions**.
Settings 1–16 retain the earlier 16-token budget; 17–24 increase the budget to
match the selected passes. This is necessary because the sampler reveals at least
one token per pass and stops when no masked positions remain. The core sampler
and direct Python API defaults are unchanged.

More steps increase generation time and can change the text; they do not guarantee
better values. The setting is captured when starting a run, and the actual token
budget and steps are recorded in each generation trace. Decision scoring remains
one batched forward pass and is unaffected by this slider.

The layout puts task setup in a compact side panel and the live browser at the
top of the main workspace. Automatic/inspection entry points are beside the goal;
choose/execute, pause, stop and restart sit just below the browser. **Decision**,
**Action trail**, **Model input**, and **Trace & export** tabs keep detailed output
from pushing the browser below the fold. Styling supports light and dark themes;
narrow windows stack the panels. The image scales with viewport height, retains
its aspect ratio, and still offers fullscreen viewing.

### Alignment with Laya's web inspector

The comparison used the original HTML/JavaScript frontend at
[Laya revision 571431b](https://github.com/ipenywis/laya-ultrafast/tree/571431b7d142d54f49ad962d9d083d9b7bb20040/laya_ultrafast/static),
including a local frontend preview; it did not run Laya's separate inference stack.

| Interaction | Gradio behavior |
|---|---|
| Website → task suggestion | Site and example selectors populate the goal; changing context clears the previous run. |
| Start / choose / execute | Inspect a fresh page, compute a pending choice, then explicitly execute it. |
| Automatic / pause | Resume the same browser from the current decision boundary. Restart opens a fresh run. |
| Targets / slow motion | Toggle boxes on the current frame; slow motion adds a one-second preview before actions. |
| Decision inspector | Current title/URL, chosen action, target probability, generation text/steps, timings, operation probabilities and ranked targets. |
| Decision trail | Only executed actions, their generated text, page-change result, and separate model-call timings. |
| Model input / export | Exact decision state/questions plus an export button that creates a downloadable JSON trace. |

The native Gradio controls preserve the shared 0.6B model and local fixtures.
Laya's hotel fixture, external planning model, existing-Chrome attachment and
hover-to-highlight target list are not included. The model policy and its recorded
flight-task limitations are unchanged by these inspector features.

For another computer to use this Mac's model:

```sh
uv run diffcider-browser --share
```

Enter a demo password at the terminal prompt. Gradio prints the share URL;
open it on the other computer and sign in as `demo` with that password. Keep this
process and Mac running. Inference and Chromium stay on this Mac. Alternatively,
set `DIFFCIDER_DEMO_PASSWORD` in your environment; avoid putting credentials in
command-line arguments or Git. `--username` and `--port` are configurable.
This uses Gradio's [share and authentication support](https://www.gradio.app/guides/sharing-your-app).
A public tunnel was not started during local testing.

The default app exposes only the bundled local mocks. To experiment with a
public website, enable the URL input explicitly:

```sh
uv run diffcider-browser --allow-public-web
```

Only give shared access to trusted viewers: this option lets them drive a browser
and make requests from this machine. The demo rejects private-address URLs,
credentials in URLs, non-HTTP(S) schemes and WebSockets, screens redirected and
subresource requests, and uses fresh cookies with no personal browser profile.
These checks are not an OS/network sandbox. Use the bundled fixture for a
predictable showcase. Public-site success was not benchmarked.

## Flight mocks and element boxes

The two flight pages are original, self-contained fixtures inspired by Laya's
[Google Flights](https://github.com/ipenywis/laya-ultrafast/blob/571431b7d142d54f49ad962d9d083d9b7bb20040/examples/flights.py)
and [Skyscanner](https://github.com/ipenywis/laya-ultrafast/blob/571431b7d142d54f49ad962d9d083d9b7bb20040/examples/skyscanner.py)
tasks: find one-way Zürich–London flights for one adult in economy, on the
requested date, then stop at results. Skyscanner also requires disabling
**Add a place to stay**, initially checked. These are functional approximations,
not copies of the live sites or their exact DOM.

Both include airport autocomplete, trip type, traveller/cabin selectors, a
calendar with month navigation and confirmation, and three fictional fares
(CHF 112, 148 and 174). Zürich starts selected, the destination starts empty,
and the trip starts as round-trip. Airport text must be confirmed by selecting
a suggestion. Search results reflect the settings actually submitted, including
incorrect ones. All assets and data are local; there are no bookings, external
requests, CAPTCHA, or live fare lookups.

The independent verifier checks nine conditions: a submitted search, origin,
destination, date, one-way, one adult, economy, no accommodation, and visible
matching results. Editing a form after submitting an incorrect search does not
pass: the corrected search must be submitted. Verification applies only to the
exact built-in goal; freely edited goals remain unverified.

**Show element boxes** is enabled by default. Blue numbered boxes
mark observed controls, and orange highlights the selected target before the
action executes. Numbers match the trace's `element` IDs; native dropdown options
use keys such as `9:1` and `9:2` for different values of element 9. The display
deduplicates nodes offering both click and fill, scales rectangles with the
screenshot, and excludes background controls obscured by a native modal dialog.

The earlier demo captured plain screenshots: Laya's frontend overlay renderer
had not been implemented in Gradio. Annotations are now drawn on a copy of each
screenshot, without injecting DOM elements or changing model inputs. The model
still reads DOM text, not pixels. A 250 ms preview pause makes the chosen box
visible before each action; **Slow motion** extends it to one second. These display
pauses are outside model-call timings. Both toggles apply during the current run;
boxes can also be hidden or restored on the final screenshot without new inference.

## Load and use the model directly

```python
from diffcider import SysoneDiffcider

model = SysoneDiffcider.from_pretrained("sgaseretto/diffcider-browser")
question = {
    "type": "choice",
    "instructions": "Search for Ada Lovelace. Choose the next operation.",
    "criteria": {"TYPE_TEXT": "Enter text into Search", "CLICK": "Click Search"},
}
answer = model.decide("The Search field is empty.", question)
text = model.generate(
    "The user's goal: Search for Ada Lovelace.\nThe page: Reading room\n"
    "The field: Search (textbox)\nWhat should be typed into this field?",
    system="You fill in web forms for a user. Answer with the exact text to type, nothing else.",
    max_new_tokens=16,
    steps=8,
)
```

`decide` always activates the loaded decision adapter and learned head.
`generate` defaults to the unchanged base, on the **same resident weights**.
Adapter selection is serialized, restored after every call, and never sticky.
`generate(..., adapter=True)` is available for explicit experiments, but the
browser model was evaluated with its adapter disabled for writing.

- `predict(state, {name: question, ...})` batches the operation and target questions
  for one page in a single forward pass.
- `decide_batch([{"state": state, "question": question}, ...])` batches independent
  requests. Group similar lengths to avoid excessive padding.
- Answers follow **sysone's Jev schema**, with full-precision probabilities:
  `choice` and a keyed `probabilities` mapping; `score`, `legend`, and probabilities;
  or `noul = p(true)`. Confidence and answer confidence are included.
  This differs from the original `Diffcider.decide` result schema.
- `generate_tokens` returns the fixed-budget token IDs; `generate` stops decoding
  at the first EOS/pad. Its greedy sampler follows sysone, suppressing the mask
  token and allowing partial final blocks. `steps // ceil(tokens / block_size)`
  determines steps per block (at least one), so a nondivisible request may execute
  fewer steps than requested. There is no early EOS stop in the denoising loop.

The loader supports this export's text-only Jev template, zero-layer anchor head,
LoRA, and choice/score/noul calibration. Other sysone architectures, side streams,
transforms and action heads fail explicitly. The exported tokenizer's vocabulary
and BOS/EOS/pad/mask IDs must match the base. Its own chat template is used for
writing. It does not execute checkpoint Python code.

The original `Diffcider` API and sampler remain unchanged. A plain PEFT adapter
loader cannot reproduce this fine-tune's decisions: the saved correction vector,
log-scale and question-type embeddings are also required.

## What was verified

Measured October 6, 2026 on an **M1 Max, 32 GPU cores, 64 GiB**, float32.
The independent reference runs unmodified sysone `55a8c9a` with PyTorch 2.14.1,
Transformers 5.19.0 and PEFT 0.21.2 in its own environment. MLX uses 0.32.3 and
the project's Transformers 4.57.6. Both load the same weights and saved head.

| Workload | PyTorch MPS median | MLX median | Speedup |
|---|---:|---:|---:|
| Model-card operation decision | 162.1 ms | 139.3 ms | 1.16× |
| Model-card operation + two target questions, batch 3 | 428.8 ms | 379.9 ms | 1.13× |
| Long-state truncation + binary + ordinal, batch 3 | 7,552.4 ms | 6,686.0 ms | 1.13× |
| Generate “Chicago”, 16 tokens / 8 steps | 399.2 ms | 298.0 ms | 1.34× |
| Generate “Ada Lovelace”, 16 tokens / 8 steps | 394.9 ms | 299.0 ms | 1.32× |
| Uneven schedule: “ Porto”, 13 tokens / 7 requested steps | 276.1 ms | 225.3 ms | 1.23× |

All **7 decision rows** matched the reference's token IDs, anchor positions and
selected outcomes, including a deliberately long state and 35 verbose options.
Maximum absolute probability difference: **4.71e-6**. All **45 generated token
IDs** across the three prompts matched, including EOS and the fixed-budget tail.
The model-card example also reproduces its published rounded probabilities.
These are small compatibility tests, not a new Mind2Web accuracy evaluation or
a guarantee of equality on all inputs. BF16 is optional but was not evaluated
for this fine-tune; float32 is the validated default.

Decision timings cover encoded-row scoring, head computation, adapter activation,
and transfer of logits to CPU; tokenization is outside the decision timer.
Generation timings include prompt tokenization and return of token IDs, but no
decoding. Each workload has one warmup and three measured runs. GPU work is
synchronized, backend runs are sequential, and allocator caches are cleared
outside timers; MLX's cache limit is 1 GiB. Loading is excluded. The long mixed
batch illustrates padding cost; do not extrapolate short-row throughput to it.

Raw results: [PyTorch](../reports/browser-torch-fp32.json),
[MLX and comparison](../reports/browser-mlx-fp32.json), and
[local bundle round-trip check](../reports/browser-bundle-fp32.json).
The bundle check repeats the parity cases with one measured repetition and is
not the three-repeat performance baseline above.

The real model + Chromium smoke test completed **3/3 reading-room goals**:
Ada Lovelace and Grace Hopper in three actions; Alan Turing in four (one repeated
Search click). Each generated the correct name, submitted the search, and opened
the article. Completion is checked against the resulting DOM independently of
model confidence. See [the full traces](../reports/browser-demo-smoke.json).
The [regression run with numbered overlays](../reports/browser-demo-overlays-smoke.json)
also completed all three goals in the same action counts, as did the
[inspector regression run](../reports/browser-inspector-smoke.json). The expanded suite
passes **94 tests**, including flight fixtures, annotation checks, generation-step
forwarding, context-switch isolation, manual decision/execution boundaries,
pause/resume, inspector output, trace export and cancellation cleanup. The manual
control test also simulates over three minutes of cumulative inspection to verify
that it does not consume the active execution budget.
Live UI checks covered website changes during inspection, matching goal/date
updates, restoring a suggestion after edits, clearing old outputs, and toggling
boxes. A [manually inspected Ada run](../reports/browser-inspector-manual.json)
chose and generated before execution, executed explicitly, paused automatic mode,
then resumed to verified completion in three actions. Its trace was exported using
the UI; timings from an interactive run are not performance benchmarks.
In the live Gradio UI, restarting an active flight run preserved an edited goal,
cleared its output without late frames reappearing, and allowed a fresh Ada run
to complete at four diffusion steps. Its generation trace recorded `steps: 4`.
The Gradio UI was also exercised with the local exported bundle. A
[clean wheel installation](../reports/browser-wheel-validation.json)
loaded the bundle offline, constructed the Gradio app, and preserved generation
tokens exactly across a decision call, with neither PyTorch nor PEFT installed.

The [initial failed traces](../reports/browser-demo-initial.json) are retained:
the model repeatedly clicked Search on an empty form. The fixture now declares
its search field required, and the observer excludes submits blocked by native
form validation. This general guard does not select the field, invent its value,
or hardcode the next action. The model still makes those decisions.

### Flight task outcomes

The expanded mocks were tested with the same float32 bundle, a fixed fixture
clock of **2026-10-06**, departure **2026-11-05**, and a 24-action budget. After an
initial exploratory run, both scenarios were repeated twice without changing
the agent or fixtures. Both repetitions followed the same action sequences:

| Mock | Complete goals | Actions per run | Final checks | Observed behavior |
|---|---:|---:|---:|---|
| Google Flights | 0/2 | 5 | 0/9 | Replaced the origin with `Z Zurich`, then repeatedly typed `Londonondon`; never submitted a valid airport search. |
| Skyscanner | 0/2 | 11 | 5/9 | Reached results for October 6, kept round-trip and accommodation enabled, then repeatedly clicked Search. |

These are **failed agent runs**, not successful flight demos. Zero of four runs
satisfied the full goal. Three consecutive actions without a page change stopped
each run. Skyscanner found London by clicking a suggestion and made no generation
calls; Google exercised both inference modes. The final checks describe submitted
results, so Google receives no credit for initially correct form defaults.

The fixture tests separately complete both workflows with scripted browser
actions, including changing months, confirming the date, switching trip type and
disabling accommodation. They also verify that incorrect searches fail. This
validates the test harness; it is not evidence that the model can solve the tasks.

To investigate a possible conversion error, five captured page states were
replayed against independent sysone/PyTorch and MLX: **18/18 decision outcomes**,
their input IDs and anchor positions matched; maximum probability difference was
**1.06e-5**. Both generated strings and **32/32 fixed-budget token IDs** also
matched, including the malformed values. This reproduces sampled failures in
PyTorch; it does not replay the entire browser session there. Each replay has one
warmup and one measured repetition, so use it for compatibility evidence, not a
new performance benchmark.

Full evidence: [initial runs](../reports/browser-flight-mocks-initial.json),
[four repeated runs with prompts and decisions](../reports/browser-flight-mocks.json),
[captured parity cases](../examples/browser_flight_cases.json),
[PyTorch replay](../reports/browser-flight-torch.json), and
[MLX replay and comparison](../reports/browser-flight-mlx.json).

A subsequent [16-step generation smoke test](../reports/browser-generation-steps16.json)
completed the Ada task in three actions, but Google Flights still failed in five.
Its generated values changed from `Z Zurich` / `Londonondon` at eight steps to
`Zur` / `London.` at sixteen. Changing the schedule affects text, but did not
solve the autocomplete or action-selection failures in this test. This is one
run per task, not an accuracy or speed benchmark.

### 24-step generation and compact layout

The browser demo now defaults to 24 steps and 24 output positions. A focused
independent PyTorch/MLX check matched **48/48 generated token IDs** and both decoded
strings (`Chicago`, `Ada Lovelace`) at those settings. With one warmup and one
measured repetition per prompt, PyTorch took 1.180 / 1.171 seconds and MLX took
0.729 / 0.694 seconds. These are compatibility smoke timings, not a replacement
for the three-repeat benchmark above. No backbone, adapter, precision, or sampling
algorithm was changed.

The [24-step browser regression](../reports/browser-generation-steps24.json) completed
all three reading-room tasks: Ada and Grace in three actions, Alan in four.

Layout checks at **1280×720** and **1024×768** kept the browser and primary
controls visible without scrolling. The 390-pixel layout stacked without horizontal
overflow; vertical scrolling still applies on narrow screens. See
[layout validation](../reports/browser-layout-validation.json).

Evidence: [24-step cases](../examples/browser_steps24_cases.json),
[PyTorch](../reports/browser-steps24-torch.json), and
[MLX and exact-token comparison](../reports/browser-steps24-mlx.json).

## Scope and limitations

This is a compact observe/decide/generate/act example, rather than all of Laya's
planning, benchmark, or Chrome attachment features. It uses Laya's DOM snapshot
approach, stable observed element IDs, freshness checks and Playwright's visible
element actions. It caps targets at 45 per operation and reports offered/omitted
counts; unsupported targets can be missed. Password/file/hidden inputs are
excluded. Iframes, closed shadow DOM, canvas controls, uploads, popup tabs and
persistent logins are not supported. Runs stop on budgets, cancellation, repeated
no-change actions, an empty generated value, or execution errors.

The model consumes **DOM text, not screenshots**. Screenshots are for the viewer.
Its source card reports incomplete field-value accuracy and no training on DONE
pages. A model's DONE answer on an arbitrary site is displayed as **unverified**;
it cannot prove that the entire goal was satisfied. The fixture success verifiers
apply only to the built-in goals. The simple direct operation/target policy has
no Laya-style planning stage; success on the reading-room tasks does not imply
reliable multi-control flight searches.

## Standalone MLX export and later publication

```sh
uv run diffcider-export-browser --output dist/diffcider-browser-mlx
uv run diffcider-export-browser --verify dist/diffcider-browser-mlx
uv run diffcider-browser --model dist/diffcider-browser-mlx
```

The exporter refuses to overwrite an existing directory. The prepared bundle is
about 2.5 GB and contains native float32 MLX base safetensors, the original
unmerged adapter, learned head, tokenizer, calibration, source revisions,
checksums, model card and Apache license. No quantization or training is needed.
`dist/` is ignored by Git. A local bundle loads with
`SysoneDiffcider.from_pretrained(path, local_files_only=True)` without Hub access.
It is a **Diffcider bundle**, not an autoregressive `mlx-lm` model.

The tested source is pinned to `f8059b1a532f42ce62179e13cf0a0d61a5c626ea`; its
base is pinned to `c8d24a3f4adaeef46881b450e1bf7d1005203bd7`.
After publishing, the same loader will accept `sgaseretto/diffcider-browser-mlx`.
The implementation must also be committed/pushed before the new model card's
Git installation instructions can be used outside this checkout.

To authenticate for the later upload, run these yourself in a terminal:

```sh
uv run hf auth login
uv run hf auth whoami
```

Create a [Hugging Face token](https://huggingface.co/settings/tokens) with write
access to your model repository, and paste it only into the terminal prompt.
Do not paste it in chat. Nothing has been uploaded by the export command or the
local validation. Once authenticated and publication is requested, upload the
prepared directory to `sgaseretto/diffcider-browser-mlx`.

## Reproduce validation

```sh
uv sync --extra benchmark --extra browser-demo
uv run playwright install chromium
uv run --extra benchmark --extra browser-demo pytest
uv run ruff check .
uv run ruff format --check .

# Independent reference dependencies are isolated from the MLX runtime.
uv venv .cache/sysone-reference-venv
uv pip install --python .cache/sysone-reference-venv/bin/python \
  -r reports/browser-reference-requirements.txt
.cache/sysone-reference-venv/bin/python scripts/compare_browser.py \
  --backend torch --output reports/browser-torch-fp32.json
uv run python scripts/compare_browser.py --backend mlx \
  --reference reports/browser-torch-fp32.json --output reports/browser-mlx-fp32.json
uv run --extra browser-demo python scripts/browser_smoke.py \
  --model dist/diffcider-browser-mlx

# Repeat the flight tasks with a fixed clock; failures remain in the JSON report.
uv run --extra browser-demo python scripts/browser_smoke.py \
  --model dist/diffcider-browser-mlx \
  --name 'Google Flights (mock)' --name 'Skyscanner (mock)' \
  --today 2026-10-06 --departure 2026-11-05 --repeats 2 \
  --output reports/browser-flight-mocks.json

# Try a different generation schedule; decision scoring is unchanged.
uv run --extra browser-demo python scripts/browser_smoke.py \
  --model dist/diffcider-browser-mlx \
  --name 'Google Flights (mock)' --name 'Ada Lovelace' \
  --today 2026-10-06 --departure 2026-11-05 --generation-steps 16 \
  --output reports/browser-generation-steps16.json

# Replay the captured failure states, sequentially, on both backends.
.cache/sysone-reference-venv/bin/python scripts/compare_browser.py \
  --backend torch --cases examples/browser_flight_cases.json --repeats 1 \
  --output reports/browser-flight-torch.json
uv run python scripts/compare_browser.py --backend mlx \
  --model dist/diffcider-browser-mlx --cases examples/browser_flight_cases.json \
  --repeats 1 --reference reports/browser-flight-torch.json \
  --output reports/browser-flight-mlx.json
```

The smoke script saves screenshots under `.cache/browser-demo/<report-name>/`,
including the selected orange boxes. These display frames are excluded from Git;
the JSON traces retain actual actions, prompts, outputs and verification failures.

Run GPU tests and benchmarks sequentially. The source model's training export
records an earlier sysone commit with a dirty working tree; the comparison pins
the available inference implementation above and matches the published example.
Source code, model and dataset attribution is retained in [THIRD_PARTY.md](../THIRD_PARTY.md).
