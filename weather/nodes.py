"""Graph nodes. Each node returns a *partial* state update."""

from __future__ import annotations

from typing import Callable

from .aliases import canonical
from .extraction import QueryExtractor
from .schema import AssistantState, WeatherQuery
from .service import WeatherError, WeatherService
from .wmo import describe


def make_nodes(
    extractor: QueryExtractor,
    service: WeatherService,
    analyst: Callable[[dict], str] | None = None,
) -> dict[str, Callable[[AssistantState], dict]]:
    """Build the node callables.

    A factory (instead of module-level functions) lets tests inject fakes and
    keeps the graph free of global state.
    """

    def parse_request(state: AssistantState) -> dict:
        text = (state.get("user_input") or "").strip()
        if not text:
            return {
                "status": "error",
                "error": "Empty request.",
                "answer": "Please tell me which city you are interested in.",
                "messages": ["parse_request: empty input"],
            }
        query = extractor.extract(text)
        if not query.cities:
            return {
                "query": query.model_dump(),
                "status": "clarify",
                "needs_clarification": True,
                "answer": "Which city would you like weather information for?",
                "messages": ["parse_request: no city found"],
            }
        return {
            "query": query.model_dump(),
            "status": "parsed",
            "messages": [
                f"parse_request: cities={query.cities} intent={query.intent} "
                f"days={query.days} unit={query.unit}"
            ],
        }

    def resolve_location(state: AssistantState) -> dict:
        query = WeatherQuery(**(state.get("query") or {}))
        if not query.cities:
            return {
                "status": "clarify",
                "needs_clarification": True,
                "answer": "Which city would you like weather information for?",
                "messages": ["resolve_location: empty city list"],
            }
        resolved, candidates, missing = [], [], []
        for city in query.cities:
            try:
                matches = service.geocode(canonical(city))
            except WeatherError as exc:
                return {
                    "status": "error",
                    "error": str(exc),
                    "answer": f"I could not look up {city}: {exc}",
                    "messages": [f"resolve_location: geocoding failed for {city}"],
                    "tool_calls": [f"geocode({city}) -> error"],
                }
            if not matches:
                missing.append(city)
                continue
            resolved.append(matches[0].model_dump())
            candidates.extend(match.model_dump() for match in matches[:3])

        if not resolved:
            return {
                "status": "clarify",
                "needs_clarification": True,
                "location_candidates": candidates,
                "answer": (
                    f"I could not find {', '.join(missing)}. Could you give the "
                    "country or a nearby larger city?"
                ),
                "messages": [f"resolve_location: no match for {missing}"],
                "tool_calls": [f"geocode({city}) -> empty" for city in missing],
            }

        return {
            "location": resolved[0] if len(resolved) == 1 else None,
            "location_candidates": resolved,
            "status": "located",
            "messages": [
                "resolve_location: " + " | ".join(item["name"] for item in resolved)
            ],
            "tool_calls": [f"geocode({city}) -> 1 match" for city in query.cities],
        }

    def fetch_weather(state: AssistantState) -> dict:
        query = WeatherQuery(**state["query"])
        locations = state.get("location_candidates") or (
            [state["location"]] if state.get("location") else []
        )
        if not locations:
            return {
                "status": "error",
                "error": "No resolved location in state.",
                "answer": "I lost track of the location. Please repeat the city.",
                "messages": ["fetch_weather: missing location"],
            }
        reports, forecast = [], []
        for raw in locations:
            from .schema import Location

            location = Location(**raw)
            try:
                report = service.current(location, query.unit)
            except WeatherError as exc:
                attempts = int(state.get("attempts", 0)) + 1
                return {
                    "status": "error",
                    "error": str(exc),
                    "attempts": attempts,
                    "can_retry": attempts <= service.settings.max_retries,
                    "answer": f"The weather service is unavailable right now ({exc}).",
                    "messages": [f"fetch_weather: attempt {attempts} failed"],
                    "tool_calls": [f"forecast({location.name}) -> error"],
                }
            reports.append(report.model_dump())
            forecast.extend(day.model_dump() for day in report.daily[: query.days])

        return {
            "report": reports[0],
            "comparison": reports if len(reports) > 1 else [],
            "forecast": forecast,
            "status": "fetched",
            "attempts": int(state.get("attempts", 0)),
            "can_retry": False,
            "messages": [f"fetch_weather: {len(reports)} report(s) retrieved"],
            "tool_calls": [
                f"forecast({raw['name']}, {query.unit}) -> ok" for raw in locations
            ],
        }

    def analyse(state: AssistantState) -> dict:
        """Deterministic analysis: thresholds, advice, comparison table.

        Doing this in code (rather than in the LLM) makes the advice
        reproducible and avoids a model inventing thresholds. The LLM is used
        later only to phrase the answer.
        """

        query = WeatherQuery(**state["query"])
        reports = state.get("comparison") or ([state["report"]] if state.get("report") else [])
        if not reports:
            return {"analysis": "", "status": "analysed", "messages": ["analyse: no data"]}

        lines: list[str] = []
        if len(reports) > 1:
            lines.append("Comparison")
            for report in reports:
                lines.append(
                    f"- {report['location']['name']}: {report['temperature']}"
                    f"{'°C' if report['unit'] == 'celsius' else '°F'}, "
                    f"{report['condition']}, wind {report['wind_speed']}"
                )
            warmest = max(reports, key=lambda r: r.get("temperature") or -999)
            lines.append(f"Warmest: {warmest['location']['name']}.")
        else:
            report = reports[0]
            for day in state.get("forecast", [])[: query.days]:
                code = day.get("weather_code")
                condition, _ = describe(code)
                lines.append(
                    f"- {day['date']}: {day.get('t_min')}–{day.get('t_max')}° "
                    f"{condition}, rain {day.get('precipitation_mm')} mm "
                    f"({day.get('precipitation_probability')}%)"
                )
            lines.append(advice_for(report, state.get("forecast", []), query.activity, query.intent))
        return {
            "analysis": "\n".join(lines),
            "status": "analysed",
            "messages": ["analyse: deterministic summary ready"],
        }

    def respond(state: AssistantState) -> dict:
        """Compose the final answer.

        The deterministic report is the source of truth; the LLM is only asked
        to make it conversational. If the model fails, the raw report is used.
        """

        query = WeatherQuery(**state["query"]) if state.get("query") else None
        header = format_header(state, query)
        body = state.get("analysis", "")
        deterministic = f"{header}\n\n{body}".strip()

        if analyst is None:
            return {"answer": deterministic, "status": "done", "messages": ["respond: template"]}
        try:
            polished = analyst(
                {
                    "user_input": state.get("user_input", ""),
                    "report": state.get("report"),
                    "comparison": state.get("comparison"),
                    "forecast": state.get("forecast"),
                    "analysis": body,
                }
            )
        except Exception as exc:
            return {
                "answer": deterministic,
                "status": "done",
                "error": f"LLM phrasing failed: {exc}",
                "messages": ["respond: template fallback"],
            }
        if not polished or not polished.strip():
            return {"answer": deterministic, "status": "done", "messages": ["respond: empty LLM"]}
        # The deterministic header and bullets are always appended: the model
        # is allowed to phrase the answer, but never to be the only source of
        # the numbers. This also keeps the answer identical when the model
        # paraphrases away a value.
        return {
            "answer": f"{polished.strip()}\n\n{header}\n\n{body}".strip(),
            "status": "done",
            "messages": ["respond: LLM phrasing"],
        }

    def ask_clarify(state: AssistantState) -> dict:
        candidates = state.get("location_candidates") or []
        if candidates:
            options = ", ".join(
                f"{c['name']} ({c.get('country', '?')})" for c in candidates[:4]
            )
            answer = f"I found several matches: {options}. Which one do you mean?"
        else:
            answer = state.get("answer") or "Could you specify the city?"
        return {
            "answer": answer,
            "status": "clarify",
            "needs_clarification": True,
            "messages": ["ask_clarify: awaiting user choice"],
        }

    def handle_error(state: AssistantState) -> dict:
        attempts = int(state.get("attempts", 0))
        retry = attempts < service.settings.max_retries + 1
        return {
            "status": "error",
            "answer": state.get("answer")
            or "The weather service could not be reached. Please try again shortly.",
            "messages": [
                f"handle_error: attempts={attempts} retry={retry}"
            ],
        }

    return {
        "parse_request": parse_request,
        "resolve_location": resolve_location,
        "fetch_weather": fetch_weather,
        "analyse": analyse,
        "respond": respond,
        "ask_clarify": ask_clarify,
        "handle_error": handle_error,
    }


def format_header(state: AssistantState, query: WeatherQuery | None) -> str:
    reports = state.get("comparison") or ([state["report"]] if state.get("report") else [])
    if not reports:
        return ""
    if len(reports) > 1:
        return "Weather comparison"
    report = reports[0]
    symbol = "°C" if report["unit"] == "celsius" else "°F"
    location = report["location"]
    name = location["name"]
    country = location.get("country") or ""
    condition, emoji = describe(report.get("weather_code"), report.get("is_day", True))
    parts = [
        f"{emoji} {name}{', ' + country if country else ''}",
        f"{report.get('temperature')}{symbol} (feels like "
        f"{report.get('apparent_temperature')}{symbol})",
        condition,
        f"Humidity {report.get('humidity')}%",
        f"Wind {report.get('wind_speed')} km/h",
    ]
    return " · ".join(parts)


def advice_for(report: dict, forecast: list[dict], activity: str, intent: str) -> str:
    """Rule-based activity advice.

    Thresholds are explicit and documented so the recommendation is auditable:
    rain risk ≥ 50% → umbrella; max temp ≥ 30 °C → heat caution; ≤ 5 °C → warm
    clothing; wind ≥ 30 km/h → avoid cycling.
    """

    max_rain = max((day.get("precipitation_probability") or 0) for day in forecast) if forecast else 0
    max_temp = max((day.get("t_max") or -99) for day in forecast) if forecast else 0
    min_temp = min((day.get("t_min") or 99) for day in forecast) if forecast else 0
    wind = report.get("wind_speed") or 0
    unit = report.get("unit", "celsius")

    tips: list[str] = []
    if max_rain >= 50:
        tips.append("Take an umbrella (rain probability ≥ 50%).")
    if unit == "celsius":
        if max_temp >= 30:
            tips.append("It will be hot: hydrate and avoid midday sun.")
        if min_temp <= 5:
            tips.append("Cold: wear a warm layer.")
    if wind >= 30:
        tips.append("Strong wind: cycling and hiking will be uncomfortable.")

    if activity == "running":
        if unit == "celsius" and max_temp >= 28:
            tips.append("Best running window: early morning, before the heat peaks.")
        else:
            tips.append("Good running conditions; early morning is still the coolest slot.")
    elif activity == "beach":
        tips.append("Check UV index and sea conditions before swimming.")
    elif activity == "cycling":
        tips.append("Prefer a route with shelter if wind picks up.")
    elif activity == "hiking":
        tips.append("Check trail conditions after any rain.")

    if not tips:
        tips.append("Conditions look unremarkable; no specific precautions needed.")
    if intent == "advice" and activity:
        return f"Advice for {activity}: " + " ".join(tips)
    return "Advice: " + " ".join(tips)
