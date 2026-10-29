# Weather Assistant with LangGraph

A weather assistant built as an explicit **state machine**: LangGraph nodes for
parsing, geocoding, fetching, analysis and phrasing, conditional edges for
clarification and error handling, a bounded retry cycle, and a checkpointer for
per-conversation memory.

## Graph

```
START → parse_request ─┬─► resolve_location ─┬─► fetch_weather ─┬─► analyse → respond → END
                       │                     │                  │
                       │ (no city)           │ (unknown city)   │ (API error)
                       ▼                     ▼                  ▼
                  ask_clarify           ask_clarify        fetch_weather (retry, once)
                                                                   │ (still failing)
                                                                   ▼
                                                              handle_error → END
```

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
ollama pull mistral
copy .env.example .env
```

No weather API key is needed: the project uses **Open-Meteo** (geocoding +
forecast).

### CLI

```powershell
python -m weather.cli "What is the weather in Tunis?"
python -m weather.cli "forecast for Sousse for 3 days" --trace
python -m weather.cli "Should I go running in Sfax tomorrow?"
python -m weather.cli "compare Paris and London" --offline
python -m weather.cli                       # interactive, one thread
```

`--offline` runs the graph with deterministic templates instead of the LLM, so
the workflow can be demonstrated without a model server.

### Web UI

```powershell
streamlit run weather/app.py
```

Each answer exposes a "Graph trace" panel listing the nodes that ran and the
tool calls made.

### Tests

```powershell
pytest
```

39 tests, all offline: the weather service is faked, the LLM is absent or
scripted, and the graph is exercised end to end.

## Layout

```
weather/
  schema.py      state TypedDict, WeatherQuery, Location, WeatherReport
  service.py     Open-Meteo client: geocoding, forecast, retries, cache
  extraction.py  NL → WeatherQuery (deterministic + LLM fallback)
  aliases.py     city aliases, ambiguity heuristic
  nodes.py       the seven graph nodes
  graph.py       StateGraph wiring, routers, checkpointer, facade
  cli.py         CLI with --trace
  app.py         Streamlit UI
tests/
```

## What is demonstrated

| Assignment requirement | Where |
| --- | --- |
| Workflow orchestration | `graph.py` — seven nodes, explicit edges |
| State management | `AssistantState` TypedDict + `messages` reducer |
| Conditional execution | `route_after_parse`, `route_after_resolve`, `route_after_fetch` |
| Cycle / retry | `fetch_weather → fetch_weather` bounded by `can_retry` |
| External data | Open-Meteo geocoding + forecast, with cache and retry |
| Persistence | `InMemorySaver` keyed by `thread_id` |
| Human clarification | `ask_clarify` when no or ambiguous city |

## How LangGraph contributes

LangChain provides the model and prompt abstractions; LangGraph provides the
*control flow*. The weather assistant needs conditional branching (ask vs act),
a cycle (retry), and persisted per-conversation state. Expressing that as a
graph makes every path explicit, testable and traceable — `--trace` and the UI
panel read the actual execution path, not a reconstruction.

## Limitations

- Ambiguity detection is heuristic (population ratio + country), not a real
  disambiguation model.
- The checkpointer is in-memory: threads vanish on restart. Swap in
  `SqliteSaver` for durability.
- No user location, no alerts/severe-weather feed, no hourly forecast.
- City extraction relies on capitalisation or the known-city list; arbitrary
  lower-case names need the LLM path.
- Advice thresholds are fixed in code and not tunable by the user.
- Open-Meteo's free tier has no SLA.

## Improvements

- Add `SqliteSaver` and a session picker in the UI for durable threads.
- Add an hourly chart and a 7-day trend view.
- Use `interrupt()` for a true human-in-the-loop node instead of ending the
  graph at clarification, so the same thread resumes with the chosen city.
- Add a weather-alert node using a second data source and a conditional edge on
  severity.
- Replace the capitalisation heuristic with a small NER model.
- Add unit tests against recorded API fixtures for faster CI.

Full write-up: [`docs/REPORT.md`](docs/REPORT.md).
