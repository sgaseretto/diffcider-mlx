"""Exercise real MLX inference and Chromium, recording successes and failures."""

import argparse
import asyncio
import json
from pathlib import Path

import mlx.core as mx

from diffcider import SysoneDiffcider
from diffcider.browser_demo.agent import DEFAULT_GENERATION_STEPS, MAX_GENERATION_STEPS, run_agent
from diffcider.browser_demo.fixtures import GOALS, SCENARIOS, fixture_spec


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sgaseretto/diffcider-browser")
    parser.add_argument("--output", default="reports/browser-demo-smoke.json")
    parser.add_argument("--name", choices=SCENARIOS, action="append")
    parser.add_argument("--departure", help="ISO flight date; default 30 days ahead")
    parser.add_argument("--today", help="ISO fixture clock for repeatable calendar tests")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument(
        "--generation-steps",
        type=int,
        choices=range(1, MAX_GENERATION_STEPS + 1),
        default=DEFAULT_GENERATION_STEPS,
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    mx.set_cache_limit(1024**3)
    model = SysoneDiffcider.from_pretrained(args.model)
    runs = []
    for name in args.name or list(GOALS):
        spec = fixture_spec(name, args.departure, args.today)
        for repeat in range(args.repeats):
            folder = (
                Path(".cache/browser-demo")
                / Path(args.output).stem
                / name.lower().replace(" ", "-")
                / str(repeat + 1)
            )
            folder.mkdir(parents=True, exist_ok=True)
            frame = 0
            async for _image, status, _trace in run_agent(
                model,
                spec["goal"],
                scenario=name,
                departure=args.departure,
                today=args.today,
                max_steps=args.max_steps,
                generation_steps=args.generation_steps,
            ):
                print(name, status, flush=True)
                _image.save(folder / f"{frame:03d}.jpg")
                frame += 1
            runs.append(
                {
                    "scenario": name,
                    "repeat": repeat + 1,
                    "goal": spec["goal"],
                    "departure": spec.get("date"),
                    "fixture_today": args.today,
                    "generation_steps": args.generation_steps,
                    "status": status,
                    "trace": _trace,
                }
            )
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(runs, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
