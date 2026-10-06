"""A persistent model with per-request adapter selection and both inference modes."""

import json
import re
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import mlx.core as mx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from .adapters import AdapterManager
from .inference import decide, decision_input, generate, generation_input
from .model import CHECKPOINTS, checkpoint_path, load_model


class Diffcider:
    """Keep one model resident and serialize requests with explicit adapter selection."""

    def __init__(self, model, tokenizer, *, source: str, path: Path):
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.path = path.resolve()
        self.revision = path.name if path.parent.name == "snapshots" else None
        self._adapters = AdapterManager(model)
        self._lock = RLock()

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = "base",
        *,
        revision: str | None = None,
        dtype: str = "float32",
        adapters: dict[str, str | Path] | None = None,
        adapter_revisions: dict[str, str] | None = None,
        local_files_only: bool = False,
        allow_base_mismatch: bool = False,
    ) -> "Diffcider":
        """Load a complete checkpoint and optional named PEFT adapters in one call.

        Args:
            model: Pinned alias (``base``/``s1``), local directory, or Hub repo ID.
            revision: Optional Hub base-model revision; aliases default to pinned commits.
            dtype: Base weight/activation precision; defaults to float32.
            adapters: Mapping from request-facing adapter names to paths or Hub repo IDs.
            adapter_revisions: Optional Hub revisions keyed by the same adapter names.
            local_files_only: Use only local files and the Hugging Face cache.
            allow_base_mismatch: Explicitly accept absent/stale adapter base metadata.
                Tensor, adapter-type, and tokenizer checks still apply.

        Returns:
            Engine ready for decision or generation calls. Every call defaults to
            the base model; pass ``adapter="name"`` to enable a loaded adapter.
        """
        adapters = adapters or {}
        adapter_revisions = adapter_revisions or {}
        if adapter_revisions.keys() - adapters.keys():
            raise ValueError("adapter_revisions contains names absent from adapters.")
        path = checkpoint_path(model, revision=revision, local_files_only=local_files_only)
        network, tokenizer = load_model(path, dtype)
        source = CHECKPOINTS.get(str(model), (str(model), None))[0]
        if Path(model).expanduser().is_dir():
            source = str(Path(model).expanduser().resolve())
        engine = cls(network, tokenizer, source=source, path=path)
        for name, adapter in adapters.items():
            engine.load_adapter(
                adapter,
                name=name,
                revision=adapter_revisions.get(name),
                local_files_only=local_files_only,
                allow_base_mismatch=allow_base_mismatch,
            )
        return engine

    @property
    def loaded_adapters(self) -> tuple[str, ...]:
        """Return the names available for per-request selection."""
        with self._lock:
            return tuple(self._adapters.names)

    def _check_base(self, config, local_files_only):
        declared = config.get("base_model_name_or_path")
        matches = declared == self.source
        if declared and Path(declared).expanduser().is_dir():
            matches |= Path(declared).expanduser().resolve() == self.path
        if not declared or not matches:
            raise ValueError(
                f"Adapter declares base {declared!r}, but the loaded base is {self.source!r}. "
                "Use the correct base, or allow_base_mismatch=True only after verifying stale metadata."
            )
        revision = config.get("revision")
        if revision:
            if self.revision is None:
                raise ValueError(
                    "Cannot verify the adapter's base revision against a local export; verify it before using allow_base_mismatch=True."
                )
            if revision != self.revision:
                if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                    raise ValueError(
                        "Adapter base revision does not match the loaded base snapshot."
                    )
                config_path = Path(
                    hf_hub_download(
                        self.source,
                        "config.json",
                        revision=revision,
                        local_files_only=local_files_only,
                    )
                )
                if config_path.parent.name != self.revision:
                    raise ValueError(
                        "Adapter base revision does not match the loaded base snapshot."
                    )

    def load_adapter(
        self,
        adapter: str | Path,
        *,
        name: str = "default",
        revision: str | None = None,
        local_files_only: bool = False,
        allow_base_mismatch: bool = False,
    ) -> None:
        """Add an adapter once; subsequent requests select it without reloading.

        Args:
            adapter: Local PEFT directory or Hugging Face adapter repo ID.
            name: Unique nonempty name to pass to inference methods.
            revision: Optional Hub revision of the adapter repository.
            local_files_only: Use only local files and cached Hub files.
            allow_base_mismatch: Bypass only base-name/revision metadata checks
                when independently verifying an adapter trained on this exact base.
        """
        with self._lock:
            if not isinstance(name, str) or not name or name in self._adapters.names:
                raise ValueError(f"Adapter name must be nonempty and unique: {name!r}")
            path = checkpoint_path(
                adapter,
                revision=revision,
                local_files_only=local_files_only,
                adapter=True,
            )
            config = json.loads((path / "adapter_config.json").read_text())
            if not allow_base_mismatch:
                self._check_base(config, local_files_only)
            if (path / "tokenizer_config.json").exists() or (path / "tokenizer.json").exists():
                tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    local_files_only=True,
                    trust_remote_code=False,
                )
                if (
                    tokenizer.get_vocab() != self.tokenizer.get_vocab()
                    or tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
                    or tokenizer.chat_template != self.tokenizer.chat_template
                ):
                    raise ValueError("Adapter tokenizer/template differs from the base tokenizer.")
            self._adapters.load(path, name)

    @contextmanager
    def _request(self, adapter):
        with self._lock:
            self._adapters.activate(adapter)
            try:
                yield
            finally:
                # Drain outstanding asynchronous work even when inference raises.
                try:
                    mx.synchronize()
                finally:
                    self._adapters.activate(None)

    def decide(
        self, state, question: dict, *, adapter: str | None = None, max_length: int = 4096
    ) -> dict:
        """Run single-pass Shared Yes/No scoring.

        Args:
            state: Text or JSON-serializable state.
            question: Question with type, instructions, and criteria fields.
            adapter: Loaded adapter name, or ``None`` for the unchanged base.
            max_length: Maximum prompt length, also bounded by the model context.

        Returns:
            Selected index/option and probabilities; ordinal scores also include
            expected_score. Binary probability order is [No, Yes].
        """
        with self._request(adapter):
            encoded = decision_input(
                self.tokenizer,
                state,
                question,
                max_length=min(max_length, self.model.args.max_position_embeddings),
            )
            return decide(self.model, encoded)

    def generate_tokens(
        self,
        prompt: str,
        *,
        adapter: str | None = None,
        max_new_tokens: int = 64,
        steps: int = 64,
        block_size: int = 32,
    ) -> list[int]:
        """Run greedy diffusion generation and return generated token IDs.

        Args:
            prompt: User prompt, encoded with the base tokenizer's chat template.
            adapter: Loaded adapter name, or ``None`` for the unchanged base.
            max_new_tokens: Fixed output length, including special tokens and tokens after EOS.
            steps: Denoising steps, distributed evenly across blocks.
            block_size: Generated positions in each block.

        Returns:
            Generated token IDs excluding the prompt, after GPU work completes.
        """
        with self._request(adapter):
            ids = generation_input(self.tokenizer, prompt)
            result = generate(
                self.model,
                ids,
                self.tokenizer.mask_token_id,
                max_new_tokens,
                steps,
                block_size,
            )
            return result[0, len(ids) :].tolist()

    def generate(
        self,
        prompt: str,
        *,
        adapter: str | None = None,
        max_new_tokens: int = 64,
        steps: int = 64,
        block_size: int = 32,
    ) -> str:
        """Generate text using the same options as ``generate_tokens``.

        Args:
            prompt: User prompt.
            adapter: Loaded adapter name, or ``None`` for the unchanged base.
            max_new_tokens: Fixed generated token budget.
            steps: Total denoising steps.
            block_size: Generated positions per block.

        Returns:
            Decoded generated text with special tokens omitted.
        """
        tokens = self.generate_tokens(
            prompt,
            adapter=adapter,
            max_new_tokens=max_new_tokens,
            steps=steps,
            block_size=block_size,
        )
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
