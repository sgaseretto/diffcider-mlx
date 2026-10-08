"""Exercise observed-element execution, validation, cancellation and networking bounds."""

import asyncio
import socket
import threading

import pytest

pytest.importorskip("playwright")
pytest.importorskip("PIL")
from playwright.async_api import async_playwright  # noqa: E402

from diffcider.browser_demo.agent import (  # noqa: E402
    ASSETS,
    SNAPSHOT,
    StalePage,
    decision_questions,
    execute,
    public_url,
    run_agent,
)
from diffcider.browser_demo.fixtures import GOALS  # noqa: E402


def test_private_urls_and_credentials_are_rejected(monkeypatch):
    def resolve(host, port):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    assert not public_url("http://localhost/admin")
    assert not public_url("https://user:password@example.com")
    assert not public_url("file:///etc/passwd")
    assert not public_url("http://169.254.169.254/")
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda host, port: [(2, 1, 6, "", ("93.184.215.14", port))]
    )
    assert public_url("https://example.com")


def test_observed_targets_form_validation_and_stale_guard():
    async def check():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1120, "height": 780})
                await page.set_content((ASSETS / "reading_room.html").read_text())
                snap = await page.evaluate(SNAPSHOT)
                _, qs, lookup = decision_questions(snap, GOALS["Ada Lovelace"], [])
                assert "CLICK" not in qs  # invalid required form cannot submit
                assert "TYPE_TEXT" in qs
                fill = lookup[("TYPE_TEXT", "1")]
                await page.locator("input").fill("Changed by page")
                with pytest.raises(StalePage):
                    await execute(page, snap, fill, "Ada Lovelace")
                snap = await page.evaluate(SNAPSHOT)
                await execute(page, snap, fill, "Ada Lovelace")
                snap = await page.evaluate(SNAPSHOT)
                _, qs, lookup = decision_questions(snap, GOALS["Ada Lovelace"], [])
                assert "CLICK" in qs
                await execute(page, snap, next(a for (op, _), a in lookup.items() if op == "CLICK"))
                assert await page.get_by_role("link", name="Ada Lovelace", exact=True).count() == 1
            finally:
                await browser.close()

    asyncio.run(check())


def test_cancel_before_inference_closes_run():
    async def check():
        event = threading.Event()
        event.set()
        statuses = [
            status async for _, status, _ in run_agent(None, GOALS["Ada Lovelace"], cancelled=event)
        ]
        assert statuses[-1].startswith("Stopped by user")

    asyncio.run(check())


@pytest.mark.parametrize("scenario", ["Google Flights (mock)", "Skyscanner (mock)"])
def test_flight_fixture_results_are_verified_independently(scenario):
    from diffcider.browser_demo.fixtures import fixture_spec, verify_fixture

    async def check():
        spec = fixture_spec(scenario, "2026-11-05", "2026-10-06")
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            try:
                page = await browser.new_page(viewport={"width": 1120, "height": 780})
                await page.route(
                    "**/*",
                    lambda route: route.fulfill(
                        path=ASSETS / spec["asset"], content_type="text/html"
                    ),
                )
                await page.goto(spec["url"])
                assert not (await verify_fixture(page, spec))["passed"]
                # Free text is insufficient: the observed airport suggestion must be selected.
                await page.get_by_role("combobox", name="Where to?").fill("London")
                await page.get_by_role(
                    "option", name="London (LON) — United Kingdom", exact=True
                ).click()
                await page.get_by_role(
                    "button", name="Departure date: Select date", exact=True
                ).click()
                snap = await page.evaluate(SNAPSHOT)
                nodes = {a["node"] for a in snap["actions"] if "node" in a}
                assert nodes
                for node in nodes:
                    assert await page.evaluate(
                        '(id)=>document.querySelector("dialog").contains(window.__jevFast.nodes.get(id))',
                        node,
                    )
                await page.get_by_role("button", name="Next month", exact=True).click()
                await page.get_by_role(
                    "button", name="Thursday, November 5, 2026", exact=True
                ).click()
                await page.get_by_role("button", name="Done", exact=True).click()
                await page.get_by_role("button", name="Search flights", exact=True).click()
                wrong = await verify_fixture(page, spec)
                assert wrong["checks"]["visible_results"]
                assert not wrong["passed"] and not wrong["checks"]["one_way"]
                if spec["site"] == "skyscanner":
                    assert not wrong["checks"]["no_accommodation"]
                await page.get_by_label("Trip type", exact=True).select_option("oneway")
                if spec["site"] == "skyscanner":
                    await page.get_by_label("Add a place to stay", exact=True).uncheck()
                # Merely fixing the form does not fix the previous submitted search.
                assert not (await verify_fixture(page, spec))["passed"]
                await page.get_by_role("button", name="Search flights", exact=True).click()
                correct = await verify_fixture(page, spec)
                assert correct["passed"], correct
                assert correct["search"]["date"] == "2026-11-05"
                # A different requested date must fail despite otherwise correct visible fares.
                other = dict(spec, date="2026-11-06")
                assert not (await verify_fixture(page, other))["passed"]
            finally:
                await browser.close()

    asyncio.run(check())


def test_flight_default_date_uses_fixture_clock_and_rejects_past_dates():
    from diffcider.browser_demo.fixtures import fixture_spec

    spec = fixture_spec("Skyscanner (mock)", today="2028-12-20")
    assert spec["date"] == "2029-01-19"
    assert "January 19, 2029" in spec["goal"]
    with pytest.raises(ValueError, match="today or later"):
        fixture_spec("Skyscanner (mock)", "2028-12-19", "2028-12-20")


def test_overlay_deduplicates_nodes_and_highlights_without_changing_source():
    import numpy as np
    from PIL import Image

    from diffcider.browser_demo.overlays import annotate

    image = Image.new("RGB", (200, 100), "white")
    original = np.array(image)
    snapshot = {
        "w": 100,
        "h": 50,
        "actions": [
            {"node": 1, "kind": "fill", "rect": {"x": 10, "y": 20, "w": 20, "h": 15}},
            {"node": 1, "kind": "click", "rect": {"x": 10, "y": 20, "w": 20, "h": 15}},
            {"node": 2, "kind": "click", "rect": {"x": 60, "y": 20, "w": 20, "h": 15}},
            {"kind": "wait"},
        ],
    }
    actual = annotate(image, snapshot, selected_node=2)
    np.testing.assert_array_equal(np.array(image), original)
    assert actual.getpixel((20, 50)) == (37, 99, 235)  # blue; scaled rectangle x=20
    assert actual.getpixel((120, 50)) == (249, 115, 22)  # orange selected target
    single = dict(snapshot, actions=[snapshot["actions"][0], snapshot["actions"][2]])
    np.testing.assert_array_equal(np.array(actual), np.array(annotate(image, single, 2)))


def test_select_targets_share_overlay_node_but_keep_distinct_option_keys():
    snapshot = {
        "url": "https://fixture.test",
        "title": "Flight",
        "text": "Flight form",
        "actions": [
            {
                "id": "e1",
                "node": 9,
                "kind": "select",
                "role": "combobox",
                "label": "Trip → One way",
                "value": "oneway",
            },
            {
                "id": "e2",
                "node": 9,
                "kind": "select",
                "role": "combobox",
                "label": "Trip → Round trip",
                "value": "roundtrip",
            },
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }
    _, questions, lookup = decision_questions(snapshot, "Find a flight", [])
    assert list(questions["SELECT"]["criteria"]) == ["9:1", "9:2"]
    assert lookup[("SELECT", "9:1")]["value"] == "oneway"
    assert lookup[("SELECT", "9:2")]["value"] == "roundtrip"


@pytest.mark.parametrize("steps", [1, 16, 17, 24])
def test_generation_step_setting_reaches_model_and_trace(steps):
    class Model:
        def predict(self, state, questions):
            return {
                "operation": {"choice": "TYPE_TEXT"},
                "TYPE_TEXT": {"choice": next(iter(questions["TYPE_TEXT"]["criteria"]))},
            }

        def generate(self, prompt, **settings):
            self.settings = settings
            return "Ada Lovelace"

    async def check():
        model = Model()
        async for _, _, _events in run_agent(
            model, GOALS["Ada Lovelace"], max_steps=1, generation_steps=steps
        ):
            pass
        generation = next(e for e in _events if e["mode"] == "generate")
        expected = {"max_new_tokens": max(16, steps), "steps": steps, "block_size": 32}
        assert generation["settings"] == expected
        assert {k: model.settings[k] for k in expected} == expected
        assert next(e for e in _events if e["mode"] == "decide")["executed"]

    asyncio.run(check())


def test_invalid_generation_steps_fail_before_starting_browser():
    async def check():
        for steps in (0, 25, 8.5, True):
            with pytest.raises(ValueError, match="diffusion steps"):
                await anext(run_agent(None, GOALS["Ada Lovelace"], generation_steps=steps))

    asyncio.run(check())


def test_manual_choose_does_not_execute_until_explicitly_permitted(monkeypatch):
    import time
    from types import SimpleNamespace

    from diffcider.browser_demo import agent
    from diffcider.browser_demo.controls import RunControl

    idle_seconds = 0
    monkeypatch.setattr(
        agent, "time", SimpleNamespace(perf_counter=lambda: time.perf_counter() + idle_seconds)
    )

    class Model:
        calls = 0

        def predict(self, state, questions):
            self.calls += 1
            return {"operation": {"choice": "TYPE_TEXT"}, "TYPE_TEXT": {"choice": "1"}}

        def generate(self, prompt, **kwargs):
            return "Ada Lovelace"

    async def check():
        nonlocal idle_seconds
        model = Model()
        control = RunControl(automatic=False)
        stream = run_agent(model, GOALS["Ada Lovelace"], control=control, max_steps=2)
        try:
            await anext(stream)
            decision = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.02)
            assert model.calls == 0 and not decision.done()
            control.command("execute")  # Cannot execute without first choosing.
            await asyncio.sleep(0.02)
            assert not decision.done()
            idle_seconds += 100  # Manual inspection must not exhaust the active run budget.
            control.command("choose")
            _, status, events = await asyncio.wait_for(decision, timeout=5)
            assert status.startswith("Decided") and model.calls == 1
            assert not any(e.get("executed") for e in events)
            assert (
                next(a for a in control.frame[1]["actions"] if a["kind"] == "fill")["value"] == ""
            )
            execution = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.02)
            control.command("pause")
            await asyncio.sleep(0.02)
            assert not execution.done()
            idle_seconds += 100
            control.command("execute")
            _, status, events = await asyncio.wait_for(execution, timeout=5)
            assert status.startswith("Step 1")
            assert next(e for e in events if e["mode"] == "decide")["executed"]
            assert (
                next(a for a in control.frame[1]["actions"] if a["kind"] == "fill")["value"]
                == "Ada Lovelace"
            )
            control.command("choose")
            _, status, _ = await asyncio.wait_for(anext(stream), timeout=5)
            assert status.startswith("Decided") and model.calls == 2
        finally:
            await stream.aclose()

    asyncio.run(check())


def test_paused_controller_resumes_and_cancellation_unblocks_wait():
    from diffcider.browser_demo.controls import RunControl

    async def check():
        control = RunControl(automatic=False)
        pending = asyncio.create_task(control.wait("choose"))
        await asyncio.sleep(0)
        control.command("auto")
        assert await pending
        control.command("pause")
        pending = asyncio.create_task(control.wait("execute"))
        await asyncio.sleep(0)
        control.cancel()
        assert not await pending
        expired = RunControl(automatic=False)
        assert not await expired.wait("choose", timeout=0.01)
        assert expired.cancelled.is_set()

    asyncio.run(check())
