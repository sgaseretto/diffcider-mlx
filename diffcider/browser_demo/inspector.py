"""Human-readable views derived from the actual browser trace."""

from html import escape


def render_table(headers, rows, *, max_height=280, widths=None):
    """Render escaped, read-only rows without hiding updates behind a virtual grid."""
    columns = "".join(f"<col style='width:{width}%'>" for width in widths or [])
    html = (
        f"<div style='max-height:{max_height}px;overflow:auto'>"
        "<table class='inspector-table'>"
        f"<colgroup>{columns}</colgroup><thead><tr>"
    )
    html += "".join(f"<th scope='col'>{escape(h)}</th>" for h in headers)
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>"
    return html + "</tbody></table></div>"


def inspect_trace(events, snapshot=None):
    """Return summary HTML, operation/target ranks, action trail and model input.

    Args:
        events: Actual decision, generation and verification events from a run.
        snapshot: Latest observed page, if available.

    Returns:
        Summary, two probability tables, executed-action table, and model input.
    """
    decisions = [e for e in events if e.get("mode") == "decide"]
    d = decisions[-1] if decisions else {}
    generation = {e["step"]: e for e in events if e.get("mode") == "generate"}
    g = generation.get(d.get("step"), {})
    op = d.get("operation", "—")
    answers = d.get("answers", {})
    target = answers.get(op, {})
    probability = target.get("probabilities", {}).get(target.get("choice"))
    operations = [
        [name, f"{value:.2%}"]
        for name, value in sorted(
            answers.get("operation", {}).get("probabilities", {}).items(),
            key=lambda item: -item[1],
        )
    ]
    criteria = d.get("questions", {}).get(op, {}).get("criteria", {})
    targets = [
        [key, criteria.get(key, key), f"{value:.2%}"]
        for key, value in sorted(target.get("probabilities", {}).items(), key=lambda p: -p[1])
    ]
    if not d and snapshot:
        nodes = {a["node"]: a for a in reversed(snapshot["actions"]) if "node" in a}
        targets = [[str(key), a["label"], "—"] for key, a in sorted(nodes.items())]
    trail = []
    for event in decisions:
        if not event.get("executed"):
            continue
        text = generation.get(event["step"], {})
        trail.append(
            [
                event["step"],
                event["operation"],
                event.get("target", ""),
                text.get("text", ""),
                "Page changed" if event.get("page_changed") else "No change",
                event["milliseconds"],
                text.get("milliseconds", ""),
            ]
        )
    page = snapshot or d.get("state", {}).get("page", {})
    model_input = (
        {"state": d["state"], "questions": d["questions"]}
        if "state" in d
        else {"page": {key: page.get(key) for key in ("url", "title", "text")}}
    )
    summary = [
        ("Page", page.get("title", "Waiting for a page")),
        ("URL", page.get("url", "—")),
        ("Choice", f"{op} · {d.get('target', 'Choose next to inspect')}"),
        ("Decision", f"{d['milliseconds']:.1f} ms" if "milliseconds" in d else "—"),
        ("Target probability", f"{probability:.2%}" if probability is not None else "—"),
        ("Generated text", g.get("text", "—")),
        (
            "Generation",
            f"{g['milliseconds']:.1f} ms · {g['settings']['steps']} steps" if g else "—",
        ),
        ("Executed actions", str(len(trail))),
    ]
    html = "<dl class='decision-summary'>"
    html += "".join(
        f"<div><dt>{escape(key)}</dt><dd>{escape(str(value))}</dd></div>" for key, value in summary
    )
    return html + "</dl>", operations, targets, trail, model_input
