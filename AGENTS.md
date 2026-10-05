# Working on diffcider-mlx

MLX inference for Qwen3 masked diffusion and Shared Yes/No decisions on Apple Silicon.
See `README.md` for usage, supported behavior, and benchmark methodology.

## Development

- Use `uv` for dependencies and Python commands; keep `uv.lock` current.
- Keep changes small and practical. Prefer simple functions over new abstractions.
- Use Google-style docstrings for public APIs and meaningful correctness tests.
- Prefer codebase-memory-mcp (`search_graph`, `trace_path`, `get_code_snippet`) for
  code discovery; run `index_repository` first if needed. Use `rg` for literals,
  configuration, or when graph tools are unavailable or insufficient.

## Correctness

- `diffcider/model.py` owns the shared backbone; `inference.py` owns prompts and
  sampling. `reference*.py` provides the independent PyTorch baseline.
- Preserve bidirectional attention, padding semantics, tied embeddings, and exact
  tokenizer/mask IDs. Binary decision probabilities are ordered **[No, Yes]**.
- Keep the upstream reference algorithms unchanged when optimizing MLX. Preserve
  pinned checkpoint revisions and attribution in `THIRD_PARTY.md`.
- Float32 is the parity-first default. Validate probability drift and output
  agreement separately when changing precision or numerical operations.

## Validation

```sh
uv sync --extra benchmark
uv run --extra benchmark pytest
uv run ruff check .
uv run ruff format --check .
```

For model, scoring, or sampling changes, also run the relevant PyTorch–MLX
comparison from `README.md`. Match weights, precision, inputs, and settings;
warm up and synchronize GPU work. Avoid concurrent GPU workloads while timing.
Retain mismatches and raw measurements in `reports/`, and update documented
results when rerunning benchmarks. Keep downloaded weights and datasets out of Git.
