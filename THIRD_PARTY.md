# Upstream sources

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
