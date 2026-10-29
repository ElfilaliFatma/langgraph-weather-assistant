"""LangGraph weather assistant.

Graph shape (see ``docs/architecture.md``):

    START → parse_request → resolve_location → fetch_weather
                              │                    │
                              │ (ambiguous city)   │ (api error)
                              ▼                    ▼
                          ask_clarify          handle_error
                                                  │
                                                  ▼ (retry once)
                                              fetch_weather
    fetch_weather → analyse → respond → END
"""

from .graph import WeatherAssistant, build_compiled_graph, build_graph
from .schema import AssistantState, WeatherQuery, WeatherReport
from .service import WeatherService, build_weather_service

__all__ = [
    "AssistantState",
    "WeatherAssistant",
    "WeatherQuery",
    "WeatherReport",
    "WeatherService",
    "build_compiled_graph",
    "build_graph",
    "build_weather_service",
]

__version__ = "1.0.0"
