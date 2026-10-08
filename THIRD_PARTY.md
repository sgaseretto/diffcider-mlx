# Upstream sources

- `diffcider/alodlm.py` ports the recurrent causal decoder, learned gate and
  depth-aware cache from [ALoDLM revision 1da9dda](https://github.com/amazon-science/ALoDLM/tree/1da9ddafeca02425480230b9bb7091fdef5672b7).
  It changes tensor operations to MLX and adds fixed-depth decision reads.
  `diffcider/reference_alodlm.py` implements the corresponding no-commit PyTorch
  intervention separately. `diffcider/_vendor/alodlm/` retains the upstream
  reference files byte-for-byte, verified against `UPSTREAM.json`; upstream
  generation is unchanged. Amazon-authored material is **CC BY-NC 4.0**;
  WeDLM-derived material retains its additional terms, including the territorial
  restriction. See `licenses/alodlm-CC-BY-NC-4.0.txt`, `licenses/alodlm-NOTICE.txt`
  and `licenses/WeDLM.txt`. These additions are not covered solely by the root
  MIT license; package metadata includes their applicable licenses.
  The ALoDLM checkpoint is pinned to
  `amazon/ALoDLM-1.7B@588c7dc1946e3e09fd496da0b0808bd26f46a758` (CC BY-NC 4.0).
  Weights remain in the Hugging Face cache; the small gate conversion is cached
  under `~/.cache/diffcider/alodlm-gates/`. No weights or source datasets are bundled.

- `diffcider/model.py` adapts the Qwen3 architecture from
  [Apple MLX-LM](https://github.com/ml-explore/mlx-lm/blob/5cfec4cb39deba54210b3ff4d86f2337c7bc10b5/mlx_lm/models/qwen3.py)
  (Copyright © 2023 Apple Inc.; MIT, included in `licenses/mlx-lm-MIT.txt`).
  It uses bidirectional attention as in
  [dLLM](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py)
  (Apache-2.0, included in `licenses/Apache-2.0.txt`).
- `diffcider/reference_sampler.py` preserves the model-card PyTorch sampler from
  [Qwen3 MDLM revision c8d24a3](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1/blob/c8d24a3f4adaeef46881b450e1bf7d1005203bd7/README.md).
  The tokenizer is an explicit argument instead of a global; formatting and
  explanatory docstrings may differ. Sampling behavior is retained. Apache-2.0.
  `examples/generation_prompts.json` contains the two demonstration prompts
  from that same model card.
- Decision prompts and scoring reproduce
  [Jev-DLLM revision 92e681d](https://github.com/zhouzihao11/jev-dllm/blob/92e681da3cc01e6888a246378d418645c54ca0f0/research/scripts/bench_diff_yesno.py)
  and its Laya rendering helpers. Apache-2.0.
- Checkpoint Python code is downloaded at pinned revisions and loaded unchanged
  for PyTorch comparisons. Model weights remain in the Hugging Face cache.
- The optional benchmark downloads the upstream S0 held-out test set into
  `.cache/`. Source dataset licenses remain applicable; see
  [Jev-DLLM dataset licenses](https://github.com/zhouzihao11/jev-dllm/blob/92e681da3cc01e6888a246378d418645c54ca0f0/datasets/LICENSES.md).
  Data is not bundled with this project.
- `diffcider/sysone.py` ports text-only Jev prompt budgeting, the learned Yes/No
  head, calibration, answer schema and greedy MDLM sampling from
  [sysone revision 55a8c9a](https://github.com/sgaseretto/sysonelib/tree/55a8c9a38d1bdf8f42394295016b8e75a1277aff).
  `diffcider/browser_demo/prompts.py` retains its browser training instructions.
  Copyright (c) 2026 Sebastian Gonzalez Aseretto; MIT, included in
  `licenses/sysone-MIT.txt`. The independent comparison runs that source unchanged
  in a separate environment; sysone and PyTorch are not runtime dependencies.
- `diffcider/browser_demo/snapshot.js` adapts the DOM observation code from
  [laya-ultrafast revision 571431b](https://github.com/ipenywis/laya-ultrafast/tree/571431b7d142d54f49ad962d9d083d9b7bb20040).
  Copyright (c) 2026 Browser Use; MIT, included in `licenses/laya-ultrafast-MIT.txt`.
  The Gradio UI and Playwright executor are local implementations inspired by its
  observe/decide/act loop; they do not call an external text-generation model.
  The local Google Flights/Skyscanner task goals and numbered screenshot overlays
  are also inspired by its `examples/flights.py`, `examples/skyscanner.py` and
  frontend. The fixture HTML and fictional fares are original local test data,
  not copied live-site assets.
- The browser example pins
  [sgaseretto/diffcider-browser revision f8059b1](https://huggingface.co/sgaseretto/diffcider-browser/tree/f8059b1a532f42ce62179e13cf0a0d61a5c626ea).
  Its head, tokenizer, calibration and unmerged adapter are retained in the local
  export. Its base remains Qwen3 MDLM revision `c8d24a3` (Apache-2.0). Model weights
  are downloaded to the Hub cache and exported under ignored `dist/`, not Git.
