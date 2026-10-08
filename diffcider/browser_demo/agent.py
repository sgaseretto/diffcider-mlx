"""Observe, decide, generate field text, act, and verify in an isolated browser."""

import asyncio
import io
import ipaddress
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image
from playwright.async_api import async_playwright

from .fixtures import fixture_spec, verify_fixture
from .overlays import annotate
from .prompts import CONTROLS, NEXT_ACTION, OPERATIONS, TARGET

ASSETS = Path(__file__).parent
SNAPSHOT = (ASSETS / "snapshot.js").read_text()
TEXT_SYSTEM = "You fill in web forms for a user. Answer with the exact text to type, nothing else."
DEFAULT_GENERATION_STEPS = 24
MAX_GENERATION_STEPS = 24


def public_url(url):
    """Allow ordinary public HTTP(S) hosts; reject credentials and private addresses."""
    try:
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.username
            or parts.password
        ):
            return False
        addresses = socket.getaddrinfo(
            parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        )
        return bool(addresses) and all(ipaddress.ip_address(a[4][0]).is_global for a in addresses)
    except (ValueError, OSError):
        return False


def decision_questions(snapshot, goal, history):
    """Offer only observed operations and up to 45 targets per operation."""
    grouped = {op: [] for op in OPERATIONS}
    for action in snapshot["actions"]:
        op = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}.get(action["kind"])
        if op is None or action.get("blocked_by_validation"):
            continue
        # Plain text fields need fill, not an extra click; dropdowns still need click.
        if op == "CLICK" and action.get("role") in {"textbox", "searchbox", "spinbutton"}:
            continue
        if len(grouped[op]) < 45:
            grouped[op].append(action)
    available = {k: OPERATIONS[k] for k, v in grouped.items() if v}
    controls = {
        k: v
        for k, v in CONTROLS.items()
        if not k.startswith("SCROLL") or any(a["id"] == k.lower() for a in snapshot["actions"])
    }
    questions = {
        "operation": {
            "type": "choice",
            "instructions": {"goal": goal, "rules": NEXT_ACTION},
            "criteria": available | controls,
        }
    }
    lookup = {}
    for op, actions in grouped.items():
        if not actions:
            continue
        criteria = {}
        option_counts = {}
        for a in actions:
            key = str(a["node"])
            if op == "SELECT":
                option_counts[key] = option_counts.get(key, 0) + 1
                key += f":{option_counts[key]}"
            label = f"[{key}] {a['label']} ({a.get('role', a['kind'])})"
            if a.get("value") is not None and a["kind"] == "fill":
                label += f" = {a['value']!r}"
            if "checked" in a:
                label += f" checked={a['checked']}"
            if a.get("hint"):
                label += f" [{a['hint']}]"
            criteria[key] = label
            lookup[(op, key)] = a
        questions[op] = {
            "type": "choice",
            "instructions": {"goal": goal, "operation": op, "rules": [NEXT_ACTION, TARGET]},
            "criteria": criteria,
        }
    state = {
        "page": {k: snapshot[k] for k in ("url", "title", "text")},
        "recent_actions": history[-6:],
    }
    return state, questions, lookup


class StalePage(RuntimeError):
    """The document or observed target changed during inference."""


async def execute(page, snapshot, action, text=None):
    """Act on a still-visible observed element, never on a model-generated selector."""
    fresh = await page.evaluate(
        """([key,node,guard]) => {
      const c=window.__jevFast;
      return !!c && JSON.stringify(c.pageKey())===JSON.stringify(key) &&
        (node==null || JSON.stringify(c.guard(c.nodes.get(node)))===JSON.stringify(guard));
    }""",
        [snapshot["page_key"], action.get("node"), snapshot["guards"].get(str(action.get("node")))],
    )
    if not fresh:
        raise StalePage("Page changed during inference; observing again.")
    if action["kind"] == "wait":
        await page.wait_for_timeout(300)
    elif action["kind"] == "scroll":
        await page.mouse.wheel(0, action["delta"])
    else:
        handle = await page.evaluate_handle("id => window.__jevFast.nodes.get(id)", action["node"])
        element = handle.as_element()
        if element is None:
            raise StalePage("Observed element disappeared.")
        try:
            if action["kind"] == "fill":
                await element.fill(text, timeout=3000)
            elif action["kind"] == "select":
                await element.select_option(action["value"], timeout=3000)
            else:
                await element.click(timeout=3000)
        finally:
            await handle.dispose()
    await page.wait_for_timeout(150)


async def run_agent(
    model,
    goal,
    *,
    url=None,
    scenario="Ada Lovelace",
    departure=None,
    today=None,
    max_steps=12,
    generation_steps=DEFAULT_GENERATION_STEPS,
    cancelled=None,
    show_boxes=True,
    control=None,
):
    """Stream screenshots, status and a trace from one bounded browser run.

    Args:
        model: One loaded SysoneDiffcider shared by the Gradio queue.
        goal: User's browser task; never replaced by a hardcoded action sequence.
        url: Public HTTP(S) URL, or None for the selected local fixture.
        scenario: Named reading-room or flight mock scenario.
        departure: ISO flight departure date; defaults to 30 days ahead.
        today: Optional fixed fixture clock for repeatable tests.
        show_boxes: Draw observed elements and the selected target on screenshots.
        max_steps: Maximum executed browser actions, between 1 and 30.
        generation_steps: Denoising passes per generated field value, from 1 to 24.
        cancelled: Optional threading.Event checked between bounded actions.
        control: Optional per-viewer RunControl for manual choose/execute and display.

    Yields:
        Screenshot PIL image, status string, and JSON-serializable event trace.
    """
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 2000:
        raise ValueError("Enter a goal of 1–2000 characters.")
    if not 1 <= max_steps <= 30:
        raise ValueError("The action budget must be between 1 and 30.")
    if type(generation_steps) is not int or not 1 <= generation_steps <= MAX_GENERATION_STEPS:
        raise ValueError("Generation diffusion steps must be an integer between 1 and 24.")
    # A pass must unmask at least one position. Preserve the old budget for 1–16 steps.
    generation_tokens = max(16, generation_steps)
    if url and not await asyncio.to_thread(public_url, url):
        raise ValueError("Use a public HTTP(S) URL without credentials or a private address.")
    fixture = not url
    spec = fixture_spec(scenario, departure, today) if fixture else None
    verify = bool(spec and goal.strip() == spec["goal"])
    history, events = [], []
    if control is not None:
        cancelled = control.cancelled
    started = time.perf_counter()
    inspection_seconds = 0.0

    async def permitted(phase):
        nonlocal inspection_seconds
        if control is None:
            return True
        before = time.perf_counter()
        try:
            return await control.wait(phase)
        finally:
            inspection_seconds += time.perf_counter() - before

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1120, "height": 780}, accept_downloads=False, service_workers="block"
        )

        # No personal profile/cookies; every navigation and subresource is screened.
        async def route(request_route):
            target = request_route.request.url
            if fixture:
                if target == spec["url"]:
                    await request_route.fulfill(
                        path=ASSETS / spec["asset"], content_type="text/html"
                    )
                else:
                    await request_route.abort()
            elif await asyncio.to_thread(public_url, target):
                await request_route.continue_()
            else:
                await request_route.abort()

        await context.route("**/*", route)
        await context.route_web_socket("**/*", lambda ws: ws.close())
        page = await context.new_page()
        page.set_default_navigation_timeout(15000)
        page.on("dialog", lambda dialog: dialog.dismiss())
        context.on("page", lambda popup: popup.close() if popup != page else None)

        async def frame(status, snapshot=None, selected_node=None):
            snapshot = snapshot or await page.evaluate(SNAPSHOT)
            image = Image.open(io.BytesIO(await page.screenshot(type="jpeg", quality=85))).copy()
            if control is not None:
                control.frame = image, snapshot, selected_node
                image = control.screenshot()
            elif show_boxes and snapshot:
                image = annotate(image, snapshot, selected_node)
            return image, status, list(events)

        async def verification():
            if not verify:
                return None
            return await verify_fixture(page, spec)

        async def finish(status):
            if control is not None:
                control.phase = "finished"
            result = await verification()
            if result:
                events.append(
                    {
                        "mode": "verify",
                        **result,
                        "verified": result["passed"],
                        "seconds": time.perf_counter() - started,
                    }
                )
                if result["passed"]:
                    status = "Verified — " + result["description"]
                else:
                    status += (
                        f" · {sum(result['checks'].values())}/{len(result['checks'])} checks passed"
                    )
            return await frame(status)

        try:
            await page.goto(url or spec["url"], wait_until="domcontentloaded")
            if control is not None:
                control.phase = "choose"
            yield await frame("Ready — observing the page")
            attempts = 0
            while len(history) < max_steps and attempts < max_steps * 2:
                if cancelled is not None and cancelled.is_set():
                    yield await finish("Stopped by user")
                    return
                if time.perf_counter() - started - inspection_seconds > 180:
                    yield await finish("Stopped — three-minute active run limit")
                    return
                if not await permitted("choose"):
                    yield await finish("Stopped — cancelled or inspection timed out")
                    return
                attempts += 1
                snapshot = await page.evaluate(SNAPSHOT)
                if snapshot is None:
                    raise StalePage("The page has no document body.")
                result = await verification()
                if result and result["passed"]:
                    yield await finish("Verified")
                    return
                state, questions, lookup = decision_questions(snapshot, goal, history)
                before = time.perf_counter()
                answers = await asyncio.to_thread(model.predict, state, questions)
                decision_ms = (time.perf_counter() - before) * 1000
                op = answers["operation"]["choice"]
                event = {
                    "step": len(history) + 1,
                    "mode": "decide",
                    "adapter": "decision",
                    "operation": op,
                    "milliseconds": round(decision_ms, 2),
                    "answers": answers,
                    "state": state,
                    "questions": questions,
                    "url": snapshot["url"],
                    "omitted_actions": snapshot["omitted_actions"],
                    "offered_targets": len(lookup),
                }
                events.append(event)
                if cancelled is not None and cancelled.is_set():
                    yield await finish("Stopped by user")
                    return
                if op in {"DONE", "BLOCKED"}:
                    yield await finish(
                        "Model claims completion — unverified"
                        if op == "DONE"
                        else "Model could not find a next action"
                    )
                    return
                if op in OPERATIONS:
                    action = lookup[(op, answers[op]["choice"])]
                else:
                    action = next(a for a in snapshot["actions"] if a["id"] == op.lower())
                event["target"] = action["label"]
                event["element"] = action.get("node")
                event["action_id"] = action["id"]
                text = None
                if op == "TYPE_TEXT":
                    prompt = (
                        f"The user's goal: {goal}\nThe page: {snapshot['title']}\n"
                        f"The field: {action['label']} ({action.get('role', 'textbox')})\n"
                        "What should be typed into this field?"
                    )
                    before = time.perf_counter()
                    text = await asyncio.to_thread(
                        model.generate,
                        prompt,
                        system=TEXT_SYSTEM,
                        max_new_tokens=generation_tokens,
                        steps=generation_steps,
                        block_size=32,
                    )
                    events.append(
                        {
                            "step": len(history) + 1,
                            "mode": "generate",
                            "adapter": None,
                            "text": text,
                            "prompt": prompt,
                            "system": TEXT_SYSTEM,
                            "settings": {
                                "max_new_tokens": generation_tokens,
                                "steps": generation_steps,
                                "block_size": 32,
                            },
                            "milliseconds": round((time.perf_counter() - before) * 1000, 2),
                        }
                    )
                    if not text.strip():
                        yield await finish("Stopped — model generated an empty field value")
                        return
                if cancelled is not None and cancelled.is_set():
                    yield await finish("Stopped by user")
                    return
                if control is not None:
                    control.phase = "execute"
                yield await frame(
                    f"Decided · {op} · {action['label']}", snapshot, action.get("node")
                )
                if not await permitted("execute"):
                    yield await finish("Stopped — cancelled or inspection timed out")
                    return
                await asyncio.sleep(1.0 if control is not None and control.slow else 0.25)
                if cancelled is not None and cancelled.is_set():
                    yield await finish("Stopped by user")
                    return
                try:
                    await execute(page, snapshot, action, text)
                except StalePage as error:
                    events.append({"mode": "observe", "reason": str(error)})
                    if control is not None:
                        control.phase = "choose"
                    yield await frame(str(error))
                    continue
                after = await page.evaluate(SNAPSHOT)
                history.append(
                    {
                        "action": f"[{action.get('role', action['kind'])}]  {action['label']}",
                        "kind": action["kind"],
                        "text": text,
                        "page_changed": after["marker"] != snapshot["marker"],
                    }
                )
                event["executed"] = True
                event["page_changed"] = history[-1]["page_changed"]
                event["elapsed_seconds"] = time.perf_counter() - started
                if control is not None:
                    control.phase = "choose"
                yield await frame(f"Step {len(history)} · {op} · {action['label']}")
                result = await verification()
                if result and result["passed"]:
                    yield await finish("Verified")
                    return
                if len(history) >= 3 and all(not h["page_changed"] for h in history[-3:]):
                    yield await finish("Stopped — three actions made no page change")
                    return
            yield await finish("Stopped — action budget reached; completion unverified")
        finally:
            await context.close()
            await browser.close()
