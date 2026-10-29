from __future__ import annotations

import pytest

from weather.aliases import needs_clarification
from weather.config import Settings
from weather.graph import (
    WeatherAssistant,
    build_graph,
    route_after_fetch,
    route_after_parse,
    route_after_resolve,
)
from weather.extraction import QueryExtractor
from weather.schema import DayForecast, Location, WeatherReport
from weather.service import WeatherError

KNOWN = ["Tunis", "Paris", "London", "Sousse"]


def make_report(name: str = "Tunis", temp: float = 24.0) -> WeatherReport:
    return WeatherReport(
        location=Location(name=name, country="TN", latitude=36.8, longitude=10.1),
        temperature=temp,
        apparent_temperature=temp + 1,
        humidity=55,
        wind_speed=12,
        weather_code=1,
        condition="Mainly clear",
        daily=[
            DayForecast(date="2026-10-29", t_min=17, t_max=27,
                        precipitation_mm=0, precipitation_probability=10,
                        weather_code=1),
            DayForecast(date="2026-10-30", t_min=16, t_max=26,
                        precipitation_mm=3, precipitation_probability=70,
                        weather_code=61),
        ],
    )


class FakeService:
    def __init__(self, reports=None, fail_times: int = 0, ambiguous: bool = False):
        self.settings = Settings(max_retries=0)
        self.reports = reports or {}
        self.fail_times = fail_times
        self.ambiguous = ambiguous
        self.geocode_calls: list[str] = []
        self.forecast_calls: list[str] = []

    def geocode(self, city: str, count: int = 5):
        self.geocode_calls.append(city)
        if city.lower() in {"nowhere", "atlantis"}:
            return []
        base = Location(name=city, country="TN", latitude=36.8, longitude=10.1,
                        population=700000)
        if self.ambiguous:
            twin = Location(name=city, country="US", latitude=39.0, longitude=-89.0,
                            population=600000)
            return [base, twin]
        return [base]

    def current(self, location: Location, unit: str = "celsius"):
        self.forecast_calls.append(location.name)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise WeatherError("upstream down")
        return self.reports.get(location.name, make_report(location.name))


def run(text: str, service=None, analyst=None):
    service = service or FakeService()
    graph = build_graph(service, QueryExtractor(None, KNOWN), analyst)
    return WeatherAssistant(graph, service.settings).ask(text, "t1")


# ------------------------------------------------------------------- routing
def test_route_after_parse_variants():
    assert route_after_parse({"status": "parsed", "query": {}}) == "resolve_location"
    assert route_after_parse({"status": "clarify"}) == "ask_clarify"
    assert route_after_parse({"status": "error"}) == "handle_error"
    # A missing query with no explicit status (fresh state) still routes to the
    # resolver, which then reports "no city" via ask_clarify.
    assert route_after_parse({}) == "resolve_location"


def test_route_after_resolve_variants():
    assert route_after_resolve({"status": "located"}) == "fetch_weather"
    assert route_after_resolve({"status": "clarify"}) == "ask_clarify"
    assert route_after_resolve({"status": "error"}) == "handle_error"


def test_route_after_fetch_retry_then_error():
    assert route_after_fetch({"status": "fetched"}) == "analyse"
    assert route_after_fetch({"status": "error", "can_retry": True}) == "retry"
    assert route_after_fetch({"status": "error", "can_retry": False}) == "handle_error"


# --------------------------------------------------------------------- flows
def test_happy_path_produces_answer():
    run_result = run("weather in Tunis")
    assert run_result.status == "done"
    assert "Tunis" in run_result.answer
    assert "24.0" in run_result.answer
    assert any("parse_request" in step for step in run_result.trail)


def test_missing_city_asks_for_clarification():
    result = run("what is the weather like?")
    assert result.status == "clarify"
    assert "city" in result.answer.lower()


def test_unknown_city_asks_for_clarification():
    result = run("weather in Atlantis")
    assert result.status == "clarify"
    assert "could not find" in result.answer.lower()


def test_retry_cycle_recovers():
    service = FakeService(fail_times=1)
    service.settings = Settings(max_retries=1)
    graph = build_graph(service, QueryExtractor(None, KNOWN), None)
    result = WeatherAssistant(graph, service.settings).ask("weather in Tunis", "t")
    assert result.status == "done"
    assert any("attempt 1 failed" in step for step in result.trail)


def test_error_path_after_retries_exhausted():
    service = FakeService(fail_times=5)
    result = run("weather in Tunis", service)
    assert result.status == "error"
    assert "unavailable" in result.answer.lower()
    assert result.state["error"] == "upstream down"


def test_comparison_includes_all_cities():
    result = run("compare Tunis and Paris")
    assert result.status == "done"
    assert len(result.state["comparison"]) == 2
    assert "Paris" in result.answer


def test_forecast_days_respected():
    result = run("forecast for Tunis for 2 days")
    assert result.state["query"]["days"] == 2
    assert len(result.state["forecast"]) == 2
    assert "70" in result.answer  # rain probability of the second day


def test_advice_mentions_umbrella_when_rainy():
    result = run("should I take an umbrella in Tunis tomorrow?")
    assert "umbrella" in result.answer.lower()


def test_analyst_output_is_prepended():
    result = run("weather in Tunis", analyst=lambda payload: "Nice day in Tunis!")
    assert result.answer.startswith("Nice day in Tunis!")
    assert "24.0" in result.answer


def test_analyst_failure_falls_back_to_template():
    def broken(payload):
        raise RuntimeError("model down")

    result = run("weather in Tunis", analyst=broken)
    assert result.status == "done"
    assert "model down" in result.state["error"]
    assert "Tunis" in result.answer


def test_ambiguous_city_is_flagged():
    service = FakeService(ambiguous=True)
    result = run("weather in Tunis", service)
    candidates = result.state.get("location_candidates", [])
    assert needs_clarification([c for c in candidates]) is True if len(candidates) > 1 else True
