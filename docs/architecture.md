# Architecture

## State

`AssistantState` is a `TypedDict` with `total=False`: every key is optional, so
each node writes only what it knows. `messages` and `tool_calls` use a
`merge_lists` reducer, which is what lets every node append a trace line
without reading the current list.

| Key | Written by | Consumed by |
| --- | --- | --- |
| `user_input` | caller | `parse_request`, `respond` |
| `query` | `parse_request` | resolve, fetch, analyse |
| `location` / `location_candidates` | `resolve_location` | `fetch_weather`, `analyse` |
| `report` / `forecast` / `comparison` | `fetch_weather` | `analyse`, `respond` |
| `analysis` | `analyse` | `respond` |
| `answer`, `status` | several | `WeatherAssistant` |
| `attempts`, `can_retry` | `fetch_weather` | `route_after_fetch` |
| `messages`, `tool_calls` | all | UI, CLI trace |

## Nodes

| Node | Pure? | Network? | LLM? |
| --- | --- | --- | --- |
| `parse_request` | no (LLM fallback) | no | only if rules fail |
| `resolve_location` | no | geocoding | no |
| `fetch_weather` | no | forecast | no |
| `analyse` | yes | no | no |
| `respond` | no | no | optional |
| `ask_clarify` | yes | no | no |
| `handle_error` | yes | no | no |

Separating "gather numbers" from "phrase numbers" is deliberate: the analysis
is deterministic and auditable, and the LLM can only *reword* it. `respond`
always appends the factual header and bullets, so a model that drops a value
cannot remove it from the final answer.

## Routers

```python
route_after_parse:    error → handle_error | clarify → ask_clarify | else resolve
route_after_resolve:  error → handle_error | clarify → ask_clarify | else fetch
route_after_fetch:    fetched → analyse  | can_retry → fetch_weather | else handle_error
```

`route_after_fetch` is the only cycle in the graph. The budget lives in the
node (`can_retry`), the router only reads the flag, which keeps policy in one
place and makes the loop trivially testable.

## Checkpointing

Every invocation passes `{"configurable": {"thread_id": ...}}`. The
`InMemorySaver` stores a checkpoint per super-step per thread, which is how
`WeatherAssistant.history()` can return the accumulated trace and how a future
`interrupt()`-based clarification can resume the same conversation.
