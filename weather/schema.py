"""State, query and report models.

``AssistantState`` is the LangGraph state: a ``TypedDict`` whose keys are
reduced by the graph on every node return. LangGraph merges a partial dict
returned by a node into the running state, so each node only declares what it
produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

Intent = Literal["current", "forecast", "advice", "compare", "unknown"]
Unit = Literal["celsius", "fahrenheit"]


class WeatherQuery(BaseModel):
    """What the user wants, extracted from free text."""

    cities: list[str] = Field(default_factory=list)
    intent: Intent = "current"
    days: int = Field(default=1, ge=1, le=7)
    unit: Unit = "celsius"
    activity: str = ""


class DayForecast(BaseModel):
    date: str
    t_min: float | None = None
    t_max: float | None = None
    precipitation_mm: float | None = None
    precipitation_probability: int | None = None
    weather_code: int | None = None


class Location(BaseModel):
    name: str
    country: str = ""
    admin1: str = ""
    latitude: float
    longitude: float
    timezone: str = "auto"
    population: int | None = None

    @property
    def label(self) -> str:
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


class WeatherReport(BaseModel):
    """Normalised weather data for one location."""

    location: Location
    unit: Unit = "celsius"
    temperature: float | None = None
    apparent_temperature: float | None = None
    humidity: int | None = None
    wind_speed: float | None = None
    weather_code: int | None = None
    condition: str = ""
    is_day: bool = True
    observed_at: str = ""
    daily: list[DayForecast] = Field(default_factory=list)

    @property
    def unit_symbol(self) -> str:
        return "\u00b0C" if self.unit == "celsius" else "\u00b0F"


def merge_lists(left: list, right: list) -> list:
    """Reducer used by the state for append-only collections."""

    return (left or []) + (right or [])


class AssistantState(TypedDict, total=False):
    """Graph state.

    ``messages`` uses a reducer so nodes can append log lines without knowing
    the current content. Everything else is last-write-wins.
    """

    user_input: str
    query: dict
    location: dict
    location_candidates: list[dict]
    report: dict
    forecast: list[dict]
    comparison: list[dict]
    analysis: str
    answer: str
    status: str
    error: str
    attempts: int
    can_retry: bool
    needs_clarification: bool
    messages: Annotated[list[str], merge_lists]
    tool_calls: Annotated[list[str], merge_lists]


@dataclass
class GraphRun:
    """Result of one graph invocation, useful for tests and the UI."""

    answer: str
    status: str
    state: dict = field(default_factory=dict)
    trail: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.status == "clarify"
