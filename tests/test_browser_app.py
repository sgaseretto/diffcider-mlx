"""Restart must isolate sessions, suppress old output, and close browser streams."""

import asyncio

import pytest

pytest.importorskip("gradio")
pytest.importorskip("playwright")

from diffcider.browser_demo import app as browser_app  # noqa: E402


def test_restart_discards_old_output_and_preserves_other_viewers(monkeypatch):
    calls, closed = {}, []

    async def agent(model, goal, **kwargs):
        calls[goal] = kwargs
        try:
            yield None, f"Observing {goal}", []
            yield None, f"Late output from {goal}", [{"stale": True}]
        finally:
            closed.append(goal)

    monkeypatch.setattr(browser_app, "run_agent", agent)
    app = browser_app.create_app(None)
    handlers = {f.name: f.fn for f in app.fns.values()}
    run_index = next(i for i, f in app.fns.items() if f.name == "run")
    assert any(run_index in f.cancels for f in app.fns.values())

    async def check():
        old = handlers["run"]("old goal", "", 24, "viewer-a", "Ada Lovelace", "", True, 4)
        other = handlers["run"]("other goal", "", 24, "viewer-b", "Grace Hopper", "", True, 8)
        fresh = None
        try:
            assert (await anext(old))[2] == []  # Starting clears the previous trace.
            await anext(old)
            await anext(other)
            await anext(other)
            session, image, status, trace = (await handlers["restart"]("viewer-a"))[:4]
            assert session != "viewer-a" and image is None and trace == []
            assert "Restarted" in status
            assert calls["old goal"]["cancelled"].is_set()
            assert not calls["other goal"]["cancelled"].is_set()
            with pytest.raises(StopAsyncIteration):
                await anext(old)  # An old frame must not replace the cleared screen.
            assert "old goal" in closed
            fresh = handlers["run"]("edited goal", "", 24, session, "Ada Lovelace", "", True, 16)
            await anext(fresh)
            assert (await anext(fresh))[1] == "Observing edited goal"
            assert calls["edited goal"]["generation_steps"] == 16
            assert not calls["edited goal"]["cancelled"].is_set()
        finally:
            await old.aclose()
            await other.aclose()
            if fresh is not None:
                await fresh.aclose()

    asyncio.run(check())


def test_cancelling_inflight_gradio_iteration_closes_browser_stream(monkeypatch):
    async def check():
        entered, closed = asyncio.Event(), asyncio.Event()

        async def agent(model, goal, **kwargs):
            try:
                entered.set()
                await asyncio.Event().wait()  # Simulate in-flight inference.
                yield None, "Unexpected output", []
            finally:
                closed.set()

        monkeypatch.setattr(browser_app, "run_agent", agent)
        app = browser_app.create_app(None)
        run = next(f.fn for f in app.fns.values() if f.name == "run")
        stream = run("goal", "", 24, "viewer", "Ada Lovelace", "", True, 8)
        await anext(stream)
        pending = asyncio.create_task(anext(stream))
        await asyncio.wait_for(entered.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert closed.is_set()
        await stream.aclose()

    asyncio.run(check())


def test_website_switch_updates_suggestion_and_invalidates_previous_run(monkeypatch):
    calls = []

    async def agent(model, goal, **kwargs):
        calls.append(kwargs)
        yield None, "Old page", []
        yield None, "Late old page", []

    monkeypatch.setattr(browser_app, "run_agent", agent)
    app = browser_app.create_app(None)
    handlers = {f.name: f.fn for f in app.fns.values()}

    async def check():
        stream = handlers["run"]("Custom goal", "", 24, "old", "Ada Lovelace", "", True, 8)
        await anext(stream)
        await anext(stream)
        day = browser_app.default_departure()
        changed = await handlers["switch_website"]("Skyscanner (mock)", day, "old")
        assert "without adding a place to stay" in changed[1]
        assert changed[2]["visible"] and changed[2]["value"] == day
        assert changed[4] == ("Skyscanner (mock)", day)
        assert changed[5] == ""  # A public URL override must not carry into another site.
        assert changed[7] is None and changed[9] == []
        assert calls[0]["cancelled"].is_set()
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        reading = await handlers["switch_website"]("Reading room (mock)", day, changed[6])
        assert reading[0]["choices"] == ["Ada Lovelace", "Grace Hopper", "Alan Turing"]
        assert reading[0]["visible"] and not reading[2]["visible"]
        assert reading[1] == "Search for Ada Lovelace and open the article."
        grace = await handlers["selection"]("Grace Hopper", day, reading[6])
        assert grace[0] == "Search for Grace Hopper and open the article."
        same = await handlers["change_date"](
            "Skyscanner (mock)", day, "untouched", ("Skyscanner (mock)", day)
        )
        assert len(same) == 18 and all(v == {"__type__": "update"} for v in same)

    asyncio.run(check())


def test_editing_goal_discards_run_and_export_retains_actual_events():
    import json
    from pathlib import Path

    app = browser_app.create_app(None)
    handlers = {f.name: f.fn for f in app.fns.values()}

    async def check():
        stream = handlers["run"]("old", "", 24, "viewer", "Ada Lovelace", "", True, 8)
        await anext(stream)
        result = await handlers["edit_task"]("edited", "", "viewer")
        assert result[0] != "viewer" and result[1] is None and result[3] == []
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        events = [{"mode": "test", "text": "Actual Zürich data"}]
        download = await handlers["export_trace"](events, "viewer")
        assert download["visible"]
        assert json.loads(Path(download["value"]).read_text()) == events

    asyncio.run(check())


def test_overlay_toggle_renders_existing_frame_without_new_agent_work(monkeypatch):
    from PIL import Image

    async def agent(model, goal, **kwargs):
        control = kwargs["control"]
        control.phase = "choose"
        control.frame = (
            Image.new("RGB", (100, 100), "white"),
            {
                "w": 100,
                "h": 100,
                "actions": [
                    {"node": 1, "label": "Search", "rect": {"x": 10, "y": 30, "w": 30, "h": 30}}
                ],
            },
            None,
        )
        yield control.screenshot(), "Observed", []

    monkeypatch.setattr(browser_app, "run_agent", agent)
    app = browser_app.create_app(None)
    handlers = {f.name: f.fn for f in app.fns.values()}

    async def check():
        stream = handlers["run"]("goal", "", 24, "viewer", "Ada Lovelace", "", True, 8)
        await anext(stream)
        annotated = (await anext(stream))[0]
        clean = await handlers["display"]("viewer", False, True)
        assert annotated.getpixel((10, 40)) == (37, 99, 235)
        assert clean.getpixel((10, 40)) == (255, 255, 255)
        await stream.aclose()

    asyncio.run(check())
