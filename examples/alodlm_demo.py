"""Run generation and a zero-shot decision using one loaded ALoDLM checkpoint.

From the repository: uv run --extra benchmark python examples/alodlm_demo.py
"""

import json

from diffcider import AloDiffcider
from diffcider.alodlm import DecodeConfig


def main():
    engine = AloDiffcider.from_pretrained(dtype="float32")
    print(
        engine.generate(
            "Explain binary search in two short sentences.",
            config=DecodeConfig(mode="left1", q=0.4, max_new_tokens=48),
        )["text"]
    )
    answer = engine.decide(
        "I was charged twice for the same order.",
        [
            {
                "name": "route",
                "type": "choice",
                "question": "Which support team should handle this request?",
                "options": ["Billing", "Technical support", "Shipping"],
            }
        ],
        readout="letters",
        passes=4,
    )
    print(json.dumps(answer, indent=2))


if __name__ == "__main__":
    main()
