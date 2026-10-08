"""Local showcase scenarios and independent final-page checks."""

import datetime as dt
from urllib.parse import urlencode

GOALS = {
    name: f"Search for {name} and open the article."
    for name in ("Ada Lovelace", "Grace Hopper", "Alan Turing")
}
FLIGHT_SCENARIOS = {"Google Flights (mock)": "google", "Skyscanner (mock)": "skyscanner"}
SCENARIOS = [*FLIGHT_SCENARIOS, *GOALS]


def default_departure(today=None):
    """Default to a future date, as the upstream live examples do."""
    day = dt.date.fromisoformat(today) if today else dt.date.today()
    return (day + dt.timedelta(days=30)).isoformat()


def flight_goal(scenario, departure):
    """Return the live example's route/date goal for a local fixture."""
    day = dt.date.fromisoformat(departure)
    accommodation = ", without adding a place to stay" if scenario == "Skyscanner (mock)" else ""
    return (
        f"Find one-way flights from Zurich to London on {day:%B} {day.day}, {day.year}, "
        f"for one adult in economy{accommodation}. Stop when matching flight options are visible. "
        "Do not select or book a flight."
    )


def fixture_spec(scenario, departure=None, today=None):
    """Resolve the mock asset, start URL and goal without accessing real websites."""
    if scenario in GOALS:
        return {
            "asset": "reading_room.html",
            "url": "https://reading-room.diffcider.test/",
            "goal": GOALS[scenario],
            "article": scenario,
        }
    if scenario not in FLIGHT_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    today = today or dt.date.today().isoformat()
    departure = departure or default_departure(today)
    if dt.date.fromisoformat(departure) < dt.date.fromisoformat(today):
        raise ValueError("The departure date must be today or later.")
    site = FLIGHT_SCENARIOS[scenario]
    return {
        "asset": "flights.html",
        "url": f"https://{site}-flights.diffcider.test/?"
        + urlencode({"site": site, "today": today}),
        "goal": flight_goal(scenario, departure),
        "date": departure,
        "site": site,
    }


async def verify_fixture(page, spec):
    """Check the submitted search and visible result cards, independent of model claims."""
    if "article" in spec:
        locator = page.locator("article[data-article]")
        passed = (
            bool(await locator.count())
            and await locator.get_attribute("data-article") == spec["article"]
        )
        return {
            "passed": passed,
            "checks": {"article_open": passed},
            "description": f"opened {spec['article']}",
        }
    result = await page.evaluate("""() => {
      const root=document.querySelector('#results');
      return {search:root?.dataset.search ? JSON.parse(root.dataset.search) : null,
              cards:[...document.querySelectorAll('[data-flight]')].filter(e=>{
                const r=e.getBoundingClientRect();return r.width&&r.height&&r.top<innerHeight&&r.bottom>0;
              }).map(e=>({date:e.dataset.date,text:e.innerText}))};
    }""")
    search, cards = result["search"] or {}, result["cards"]
    checks = {
        "submitted": bool(search),
        "origin": search.get("origin") == "ZRH",
        "destination": search.get("destination") == "LON",
        "date": search.get("date") == spec["date"],
        "one_way": search.get("trip") == "oneway",
        "one_adult": search.get("adults") == "1",
        "economy": search.get("cabin") == "Economy",
        "no_accommodation": search.get("hotel") is False,
        "visible_results": len(cards) >= 2
        and all(c["date"] == spec["date"] and "CHF" in c["text"] for c in cards),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "search": search,
        "description": f"Zürich → London on {spec['date']}",
    }
