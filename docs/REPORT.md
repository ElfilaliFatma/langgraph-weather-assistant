# Technical Report — Weather Assistant with LangGraph

Author: Fatma Ezzahra Elfilali

## 1. Problem and objectives

A weather assistant looks simple until the happy path stops being enough: the
user names no city, names a city that exists in five countries, the upstream
API times out, or the question is advice ("should I run tomorrow?") rather than
a lookup. Each of these is a different control-flow decision, which is exactly
what a graph is for.

Objectives:

1. model the workflow explicitly with LangGraph nodes and conditional edges;
2. keep a typed state that every node reads and partially updates;
3. handle clarification, ambiguity and failures without crashing;
4. integrate a real external data source;
5. keep the numbers deterministic while still producing natural language;
6. persist per-conversation state through a checkpointer.

## 2. Architecture

Seven nodes, three conditional routers, one cycle. Full state table and router
semantics are in `docs/architecture.md`.

| Stage | Implementation choice | Rationale |
| --- | --- | --- |
| Request parsing | regex + known-city list, LLM only as a fallback | instant, offline-testable, deterministic |
| Geocoding | Open-Meteo geocoding API | no API key, so the project runs anywhere |
| Forecast | Open-Meteo forecast API | same |
| Analysis | pure Python thresholds | auditable advice, no hallucinated values |
| Phrasing | optional LLM node | natural language without controlling facts |
| Memory | `InMemorySaver` + `thread_id` | native LangGraph persistence |

### 2.1 Why a graph instead of an if-chain

The assistant has three genuinely different outcomes (answer, ask, fail) and
one cycle. Implementing that as nested `if`s hides the paths and makes them
untestable in isolation. With `StateGraph`:

- each path is a function with a typed input and a partial output;
- routers are pure functions from state to edge name, so they can be unit
  tested without a graph;
- `graph.stream()` exposes which nodes ran, which the CLI prints with
  `--trace`;
- retries and future human-in-the-loop steps are additions to the graph rather
  than rewrites of the controller.

### 2.2 Determinism where it matters

The advice rules are explicit: rain probability ≥ 50% → umbrella; max ≥ 30 °C →
heat caution; min ≤ 5 °C → warm layer; wind ≥ 30 km/h → avoid exposed
activities; running advice mentions heat only when the maximum actually
exceeds 28 °C. During testing, an early version said "before the heat peaks" at
25 °C — a real defect found by reading the output, then fixed by conditioning
the sentence on the temperature.

### 2.3 Bounded retry

`fetch_weather` increments `attempts` and sets `can_retry` while the budget
(`MAX_WEATHER_RETRIES`) lasts. `route_after_fetch` sends the flow back to the
same node, forming a cycle that terminates. The HTTP client also retries
transient failures, but *not* 4xx responses other than 429, because retrying a
bad request only wastes time.

### 2.4 Clarification

`ask_clarify` is reached for "what is the weather like?" (no city) and for
unknown names. Ambiguity between cities in different countries is detected by a
population-ratio heuristic (`aliases.needs_clarification`) and the candidates
are listed. The node currently ends the graph; the documented improvement is
`interrupt()` so the same thread resumes.

## 3. Environment

Python 3.12, LangGraph 1.0.x, langchain-ollama, requests, Streamlit, Pydantic.
Model: `mistral` 7B Q4_K_M on Ollama. Data: Open-Meteo (geocoding + forecast),
which needs no key.

## 4. Testing and results

### 4.1 Unit tests (39, fully offline)

**Extraction** — city detection with and without capitalisation, Arabic alias
table, intent (current/forecast/compare/advice), day counts including the
weekend and "5 days", activity detection, Fahrenheit, and the regression that
"for 5 days" must not produce a city called *Days*.

**Service** — geocoding parse and cache, empty city rejection, network failure,
report construction, Fahrenheit forwarding, 4xx not retried, 5xx retried then
failing, retry then success, and short daily arrays tolerated.

**Routers** — all four branches, including the "fresh state" case.

**Flows** — happy path, missing city → clarify, unknown city → clarify, retry
recovers, retries exhausted → error, comparison of two cities, forecast length
honours `days`, umbrella advice when the rain probability is high, analyst
output prepended while facts are preserved, and analyst failure falling back to
the template.

**Checkpointing** — thread isolation, trace accumulation in one thread, and
`stream()` emitting node-level updates.

The fake service and the optional scripted analyst mean no network and no model
server are required, which is what makes the suite runnable in CI.

### 4.2 Live verification

Real API calls, offline LLM mode:

```
> python -m weather.cli "forecast for Sousse for 3 days" --offline
☁️ Sousse, Tunisia · 20.3°C (feels like 21.9°C) · Overcast · Humidity 78% · Wind 3.9 km/h
- 2026-10-29: 17.2–23.2° Overcast, rain 0.3 mm (25%)
- 2026-10-30: 15.2–22.2° Overcast, rain 0.2 mm (48%)
- 2026-10-31: 14.3–21.3° Slight rain showers, rain 0.9 mm (60%)
Advice: Take an umbrella (rain probability ≥ 50%).
```

With the LLM phrasing node enabled:

```
> python -m weather.cli "Should I go running in Sfax tomorrow?" --trace
  · parse_request: cities=['Sfax'] intent=advice days=2 unit=celsius
  · resolve_location: Sfax
  · fetch_weather: 1 report(s) retrieved
  · analyse: deterministic summary ready
  · respond: LLM phrasing
  ↳ geocode(Sfax) -> 1 match
  ↳ forecast(Sfax, celsius) -> ok
```

The boundary cases were also executed: "What is the weather like?" ends in
`ask_clarify`, "compare Paris and London" returns a two-row comparison, and a
forced network failure reaches `handle_error` after the retry cycle.

## 5. How LangGraph contributes

LangChain supplies the model, the prompt template and the provider
abstraction. LangGraph supplies what a single chain cannot express cleanly:

- **Orchestration** — nodes are units of work; edges declare order.
- **Conditional execution** — routers encode "ask, act or fail" as data.
- **Cycles** — the retry loop is a graph edge, not a `while` with hidden state.
- **State management** — a typed state with a reducer for append-only fields.
- **Persistence** — checkpointing keyed by thread, the foundation for
  resumable clarification and human-in-the-loop.
- **Observability** — `stream()` reveals the path that actually ran.

## 6. Limitations

1. Clarification ends the graph instead of suspending the thread.
2. In-memory checkpointer loses threads on restart.
3. Ambiguity detection uses population ratios, not a disambiguation model.
4. City extraction depends on capitalisation or a fixed city list.
5. Advice thresholds are hard-coded and not user-tunable.
6. No hourly data, alerts or historical trends.
7. The LLM extraction fallback is one call; a misparse costs latency.

## 7. Improvements

1. Use `interrupt()` + `Command(resume=...)` so clarification continues the
   same thread instead of starting over.
2. Add `SqliteSaver` for durable conversations and a session picker.
3. Add a severe-weather alert node with a conditional edge on severity.
4. Add hourly and 7-day charts, plus a "trend" node comparing today to the
   weekly average.
5. Replace the capitalisation heuristic with a NER model or a
   `with_structured_output()` extraction chain.
6. Add recorded HTTP fixtures so the suite also covers real payload shapes in
   CI.
7. Add `RetryPolicy` on the network nodes instead of the manual cycle, keeping
   the graph smaller.
