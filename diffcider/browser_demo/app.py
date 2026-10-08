"""Launch a Gradio inspector served by this Mac's resident MLX model."""

import argparse
import getpass
import hashlib
import json
import os
import tempfile
import uuid
from contextlib import aclosing
from pathlib import Path

import gradio as gr
import mlx.core as mx

from diffcider import SysoneDiffcider
from diffcider.sysone import BROWSER_MODEL

from .agent import DEFAULT_GENERATION_STEPS, MAX_GENERATION_STEPS, run_agent
from .controls import RunControl
from .fixtures import FLIGHT_SCENARIOS, GOALS, default_departure, fixture_spec
from .inspector import inspect_trace, render_table

WEBSITES = [*FLIGHT_SCENARIOS, "Reading room (mock)"]
OPERATION_HEADERS = ["Operation", "Probability"]
TARGET_HEADERS = ["ID", "Target", "Probability"]
TRAIL_HEADERS = [
    "Step",
    "Operation",
    "Target",
    "Generated text",
    "Outcome",
    "Decision ms",
    "Generation ms",
]


def create_app(model, *, allow_public_web=False):
    """Build a per-viewer inspector with shared, serialized model inference.

    Args:
        model: A loaded SysoneDiffcider instance.
        allow_public_web: Expose a public URL override in addition to local mocks.

    Returns:
        Gradio Blocks app with automatic and manual browser controls.
    """
    runs = {}
    exports = tempfile.TemporaryDirectory(prefix="diffcider-traces-")

    def forget(session):
        previous = runs.pop(session, None)
        if previous is not None:
            previous.cancel()

    def view(control, status, events, image=None):
        snapshot = control.frame[1] if control and control.frame else None
        live = bool(
            control
            and not control.finished
            and control.phase != "finished"
            and not control.cancelled.is_set()
        )
        if control and control.frame:
            image = control.screenshot()
        summary, operations, targets, trail, model_input = inspect_trace(events, snapshot)
        return (
            image,
            status,
            events,
            summary,
            render_table(OPERATION_HEADERS, operations, max_height=210),
            render_table(TARGET_HEADERS, targets, max_height=240, widths=[12, 60, 28]),
            render_table(TRAIL_HEADERS, trail),
            model_input,
            gr.update(value=None, visible=False),
            gr.update(interactive=live and not control.automatic and control.phase == "choose"),
            gr.update(interactive=live and not control.automatic and control.phase == "execute"),
            gr.update(interactive=live and control.automatic),
        )

    async def run(
        goal,
        url,
        budget,
        session,
        scenario,
        departure,
        boxes,
        generation_steps,
        slow=False,
        inspect_mode=False,
    ):
        existing = runs.get(session)
        if (
            existing
            and not existing.finished
            and not existing.cancelled.is_set()
            and not inspect_mode
        ):
            existing.command("auto")
            yield view(existing, "Running automatically…", existing.events)
            return
        forget(session)
        control = RunControl(automatic=not inspect_mode, boxes=boxes, slow=slow)
        control.events = []
        control.goal, control.url = goal, url
        runs[session] = control
        image, events, status = None, [], "Starting a fresh browser session…"
        try:
            yield view(control, status, events)
            if runs.get(session) is not control:
                return
            stream = run_agent(
                model,
                goal,
                url=url.strip() if allow_public_web else None,
                scenario=scenario,
                departure=departure,
                show_boxes=boxes,
                max_steps=int(budget),
                generation_steps=generation_steps,
                cancelled=control.cancelled,
                control=control,
            )
            async with aclosing(stream):
                async for image, status, events in stream:
                    if runs.get(session) is not control:
                        return
                    control.events = events
                    yield view(control, status, events, image)
            if runs.get(session) is control:
                control.finished = True
                yield view(control, status, events, image)
        except Exception as error:
            if runs.get(session) is control:
                control.finished = True
                yield view(control, f"Run stopped: {type(error).__name__}: {error}", events, image)
        finally:
            control.finished = True

    async def inspect_start(*args):
        async with aclosing(run(*args, inspect_mode=True)) as stream:
            async for result in stream:
                yield result

    async def stop(session):
        if session in runs:
            control = runs[session]
            control.cancel()
            return view(
                control, "Stopping — the current model call may finish first", control.events
            )
        return (gr.skip(),) * 12

    async def restart(session):
        forget(session)
        return (uuid.uuid4().hex, *view(None, "Restarted — start a fresh demo", []))

    async def choose_next(session):
        if session in runs:
            control = runs[session]
            control.command("choose")
            return view(control, "Choosing the next action…", control.events)
        return (gr.skip(),) * 12

    async def execute_choice(session):
        if session in runs:
            control = runs[session]
            control.command("execute")
            return view(control, "Executing the inspected choice…", control.events)
        return (gr.skip(),) * 12

    async def pause(session):
        if session in runs:
            control = runs[session]
            control.command("pause")
            return view(
                control, "Paused — inspect the next choice or resume automatically", control.events
            )
        return (gr.skip(),) * 12

    async def edit_task(goal, url, session):
        control = runs.get(session)
        if control and (control.goal != goal or control.url != url):
            return await restart(session)
        return (gr.skip(),) * 13

    def describe(scenario):
        if scenario == "Skyscanner (mock)":
            return "**Skyscanner · local mock.** Try the suggested flight search, including turning off ‘Add a place to stay’."
        if scenario in FLIGHT_SCENARIOS:
            return "**Google Flights · local mock.** Try one-way Zürich–London flights on the selected date. All fares are fictional."
        return "**Reading room · local mock.** Choose an example person, search, and open their article."

    async def selection(scenario, departure, session):
        try:
            spec = fixture_spec(scenario, departure)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        return (
            spec["goal"],
            gr.update(visible=scenario in FLIGHT_SCENARIOS, value=spec.get("date", departure)),
            describe(scenario),
            (scenario, spec.get("date")),
            "",
            *(await restart(session)),
        )

    async def switch_website(website, departure, session):
        choices = list(GOALS) if website == "Reading room (mock)" else [website]
        selected = choices[0]
        if selected in FLIGHT_SCENARIOS:
            try:
                fixture_spec(selected, departure)
            except ValueError:
                departure = default_departure()
        return (
            gr.update(choices=choices, value=selected, visible=len(choices) > 1),
            *(await selection(selected, departure, session)),
        )

    async def change_date(scenario, departure, session, configured):
        if configured == (scenario, departure):
            return (gr.skip(),) * 18
        return await selection(scenario, departure, session)

    async def display(session, boxes, slow):
        control = runs.get(session)
        if control is None:
            return gr.skip()
        control.boxes, control.slow = boxes, slow
        return control.screenshot()

    async def export_trace(events, session):
        if not events:
            raise gr.Error("Run or inspect an action before exporting its trace.")
        path = Path(exports.name) / (hashlib.sha256(session.encode()).hexdigest()[:16] + ".json")
        path.write_text(json.dumps(events, indent=2, ensure_ascii=False) + "\n")
        return gr.update(value=str(path), visible=True)

    day = default_departure()
    first = "Google Flights (mock)"
    with gr.Blocks(
        title="Diffcider Browser",
        analytics_enabled=False,
        fill_width=True,
        delete_cache=(3600, 3600),
    ) as app:
        session = gr.State(
            value=lambda: uuid.uuid4().hex, time_to_live=3600, delete_callback=forget
        )
        configured = gr.State((first, day))
        gr.HTML(
            "<header class='app-header'><div><h1>Diffcider <span>Browser</span></h1>"
            "<p>Decide actions. Generate text. One shared model.</p></div>"
            "<span class='model-badge'>MLX · 0.6B · on this Mac</span></header>",
            elem_id="brand",
        )
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=0, min_width=280, elem_id="task-panel"):
                gr.Markdown("### Task setup", elem_classes="section-heading")
                with gr.Row(elem_classes="compact-buttons"):
                    auto = gr.Button("Run automatically", variant="primary", min_width=130)
                    start = gr.Button("Inspect", min_width=80)
                website = gr.Dropdown(WEBSITES, value=first, label="Website")
                example = gr.Dropdown([first], value=first, label="Suggested task", visible=False)
                departure = gr.Textbox(value=day, label="Departure date (YYYY-MM-DD)")
                goal = gr.Textbox(
                    value=fixture_spec(first, day)["goal"],
                    lines=4,
                    max_lines=6,
                    label="Browser goal",
                )
                restore = gr.Button("Use suggested task", size="sm", variant="tertiary")
                generation_steps = gr.Slider(
                    1,
                    MAX_GENERATION_STEPS,
                    value=DEFAULT_GENERATION_STEPS,
                    step=1,
                    label="Generation diffusion steps",
                    info="Field generation only. Decisions use one pass.",
                )
                with gr.Accordion("More settings & task info", open=False):
                    budget = gr.Slider(1, 30, value=24, step=1, label="Maximum actions")
                    url = gr.Textbox(
                        value="",
                        label="Public website URL override (empty = selected mock)",
                        visible=allow_public_web,
                    )
                    help_text = gr.Markdown(describe(first))
                    gr.Markdown(
                        "Changing the website, date or goal clears the previous run. "
                        "Inspect opens the page; Choose next proposes an action; Execute choice runs it."
                    )
                gr.Markdown(
                    "Local mocks · fictional data · no bookings. Edited goals are unverified.",
                    elem_classes="panel-note",
                )
            with gr.Column(scale=1, min_width=0, elem_id="browser-panel"):
                with gr.Row(elem_id="browser-toolbar"):
                    gr.Markdown("### Live browser", elem_classes="section-heading")
                    boxes = gr.Checkbox(value=True, label="Show element boxes", min_width=150)
                    slow = gr.Checkbox(value=False, label="Slow motion", min_width=110)
                status = gr.Textbox(
                    value="Ready — run a task, or inspect it step by step",
                    label="Run status",
                    show_label=False,
                    interactive=False,
                    elem_id="run-status",
                )
                screenshot = gr.Image(
                    label="Blue: observed elements · Orange: chosen target",
                    interactive=False,
                    height="clamp(280px, calc(100dvh - 320px), 640px)",
                    format="jpeg",
                    buttons=["fullscreen", "download"],
                    elem_id="browser-view",
                    placeholder="Choose a task and press Run automatically or Inspect",
                )
                with gr.Row(elem_id="action-toolbar", elem_classes="compact-buttons"):
                    choose = gr.Button("Choose next", interactive=False, min_width=100)
                    execute = gr.Button("Execute choice", interactive=False, min_width=110)
                    pause_button = gr.Button("Pause", interactive=False, min_width=60)
                    stop_button = gr.Button("Stop", min_width=60)
                    reset = gr.Button("Restart", min_width=65)
                with gr.Tabs(elem_id="inspector-tabs"):
                    with gr.Tab("Decision"):
                        summary = gr.HTML(inspect_trace([])[0])
                        with gr.Row():
                            with gr.Column(scale=1, min_width=220):
                                operations = gr.HTML(render_table(OPERATION_HEADERS, []))
                            with gr.Column(scale=2, min_width=260):
                                targets = gr.HTML(
                                    render_table(TARGET_HEADERS, [], widths=[12, 60, 28])
                                )
                    with gr.Tab("Action trail"):
                        trail = gr.HTML(render_table(TRAIL_HEADERS, []))
                    with gr.Tab("Model input"):
                        model_state = gr.JSON(label="Exact decision input")
                    with gr.Tab("Trace & export"):
                        with gr.Row(elem_classes="compact-buttons"):
                            export = gr.Button("Export trace JSON", interactive=False)
                            trace_file = gr.File(
                                label="Download trace", interactive=False, visible=False
                            )
                        trace = gr.JSON(label="Run trace")
        outputs = [
            screenshot,
            status,
            trace,
            summary,
            operations,
            targets,
            trail,
            model_state,
            trace_file,
            choose,
            execute,
            pause_button,
        ]
        inputs = [goal, url, budget, session, example, departure, boxes, generation_steps, slow]
        run_event = auto.click(
            run,
            inputs,
            outputs,
            concurrency_limit=None,
            trigger_mode="multiple",
            api_name="run_browser",
            show_progress="hidden",
        )
        inspect_event = start.click(
            inspect_start,
            inputs,
            outputs,
            concurrency_limit=None,
            trigger_mode="multiple",
            api_name="inspect_browser",
            show_progress="hidden",
            cancels=[run_event],
        )
        resets = [session, *outputs]
        selection_outputs = [goal, departure, help_text, configured, url, *resets]
        website.input(
            switch_website,
            [website, departure, session],
            [example, *selection_outputs],
            queue=False,
            cancels=[run_event, inspect_event],
        )
        example.input(
            selection,
            [example, departure, session],
            selection_outputs,
            queue=False,
            cancels=[run_event, inspect_event],
        )
        restore.click(
            selection,
            [example, departure, session],
            selection_outputs,
            queue=False,
            cancels=[run_event, inspect_event],
        )
        departure.blur(
            change_date, [example, departure, session, configured], selection_outputs, queue=False
        )
        departure.submit(
            change_date, [example, departure, session, configured], selection_outputs, queue=False
        )
        reset.click(
            restart,
            session,
            resets,
            queue=False,
            cancels=[run_event, inspect_event],
            api_name="restart_browser",
        )
        goal.input(edit_task, [goal, url, session], resets, queue=False)
        url.input(edit_task, [goal, url, session], resets, queue=False)
        stop_button.click(stop, session, outputs, queue=False, api_name=False)
        choose.click(choose_next, session, outputs, queue=False, api_name=False)
        execute.click(execute_choice, session, outputs, queue=False, api_name=False)
        pause_button.click(pause, session, outputs, queue=False, api_name=False)
        boxes.input(display, [session, boxes, slow], screenshot, queue=False)
        slow.input(display, [session, boxes, slow], screenshot, queue=False)
        export.click(export_trace, [trace, session], trace_file, queue=False)
        trace.change(lambda events: gr.update(interactive=bool(events)), trace, export, queue=False)
    return app.queue(max_size=8, default_concurrency_limit=1)


def main():
    """Load once and launch locally, or share with authenticated viewers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=BROWSER_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--allow-public-web",
        action="store_true",
        help="Trusted viewers may drive a fresh browser on public websites.",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--username", default="demo")
    args = parser.parse_args()
    password = os.environ.get("DIFFCIDER_DEMO_PASSWORD")
    if args.share and not password:
        password = getpass.getpass("Password for the shared demo: ")
        if not password:
            parser.error("Sharing requires a nonempty password.")
    mx.set_cache_limit(1024**3)
    model = SysoneDiffcider.from_pretrained(args.model, revision=args.revision, dtype=args.dtype)
    app = create_app(model, allow_public_web=args.allow_public_web)
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=args.share,
        auth=(args.username, password) if password else None,
        show_error=False,
        theme=gr.themes.Base(
            primary_hue="teal",
            neutral_hue="slate",
            spacing_size="sm",
            radius_size="lg",
            font=["ui-sans-serif", "system-ui", "sans-serif"],
        ),
        css_paths=Path(__file__).with_name("styles.css"),
        footer_links=[],
    )


if __name__ == "__main__":
    main()
