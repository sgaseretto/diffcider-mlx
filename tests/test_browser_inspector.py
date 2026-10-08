"""The inspector displays actual probabilities and executed actions, with safe HTML."""

from diffcider.browser_demo.inspector import inspect_trace, render_table


def test_pending_choices_are_ranked_but_not_counted_as_executed():
    decision = {
        "mode": "decide",
        "step": 1,
        "operation": "CLICK",
        "milliseconds": 10,
        "target": "<script>alert(1)</script>",
        "answers": {
            "operation": {"probabilities": {"CLICK": 0.8, "DONE": 0.2}},
            "CLICK": {"choice": "2", "probabilities": {"1": 0.1, "2": 0.9}},
        },
        "state": {"page": {"title": "<img src=x onerror=alert(1)>", "url": "https://fixture.test"}},
        "questions": {"CLICK": {"criteria": {"1": "Other", "2": "Search"}}},
    }
    html, operations, targets, trail, model_input = inspect_trace([decision])
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html
    assert operations[0] == ["CLICK", "80.00%"]
    assert targets[0] == ["2", "Search", "90.00%"]
    assert trail == [] and model_input["state"] == decision["state"]
    decision.update(executed=True, page_changed=False)
    trail = inspect_trace([decision])[3]
    assert len(trail) == 1 and trail[0][4] == "No change"
    table = render_table(["<unsafe heading>"], [["<script>one</script>"], ["second"], ["third"]])
    assert "<script>" not in table and "&lt;script&gt;" in table
    assert "&lt;unsafe heading&gt;" in table and table.count("<td>") == 3
