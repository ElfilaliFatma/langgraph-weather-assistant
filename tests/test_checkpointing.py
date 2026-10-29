from __future__ import annotations

from weather.config import Settings
from weather.extraction import QueryExtractor
from weather.graph import WeatherAssistant, build_graph
from weather.schema import DayForecast, Location, WeatherReport


class FakeService:
    settings = Settings(max_retries=0)

    def geocode(self, city, count=5):
        return [Location(name=city, country="TN", latitude=36.8, longitude=10.1)]

    def current(self, location, unit="celsius"):
        return WeatherReport(
            location=location,
            temperature=20.0,
            humidity=50,
            wind_speed=5,
            weather_code=0,
            condition="Clear sky",
            daily=[DayForecast(date="2026-10-29", t_min=15, t_max=25,
                               precipitation_mm=0, precipitation_probability=0)],
        )


def test_thread_history_is_isolated():
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        FakeService(), QueryExtractor(None, ["Tunis", "Paris"]),
        checkpointer=InMemorySaver(),
    )
    assistant = WeatherAssistant(graph, Settings())
    assistant.ask("weather in Tunis", "thread-a")
    assistant.ask("weather in Paris", "thread-b")
    history_a = assistant.history("thread-a")
    history_b = assistant.history("thread-b")
    assert any("Tunis" in step for step in history_a)
    assert any("Paris" in step for step in history_b)
    assert history_a != history_b


def test_same_thread_accumulates_trace():
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        FakeService(), QueryExtractor(None, ["Tunis"]), checkpointer=InMemorySaver()
    )
    assistant = WeatherAssistant(graph, Settings())
    assistant.ask("weather in Tunis", "same")
    first = len(assistant.history("same"))
    assistant.ask("weather in Tunis", "same")
    assert len(assistant.history("same")) > first


def test_stream_yields_node_updates():
    from langgraph.checkpoint.memory import InMemorySaver

    graph = build_graph(
        FakeService(), QueryExtractor(None, ["Tunis"]), checkpointer=InMemorySaver()
    )
    assistant = WeatherAssistant(graph, Settings())
    updates = list(assistant.stream("weather in Tunis", "stream"))
    node_names = [name for update in updates for name in update]
    assert "parse_request" in node_names
    assert "respond" in node_names
