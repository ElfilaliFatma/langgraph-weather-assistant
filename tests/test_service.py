"""Tests for the Open-Meteo client.

The HTTP session is faked, so these tests never touch the network. Degree
symbols are written with escapes to stay encoding-proof on Windows consoles.
"""

from __future__ import annotations

import pytest
import requests

from weather.config import Settings
from weather.schema import Location
from weather.service import WeatherError, WeatherService

DEG_C = "\u00b0C"
DEG_F = "\u00b0F"


class FakeResponse:
    def __init__(self, payload: dict | None = None, status: int = 200):
        self._payload = payload or {}
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error


class FakeSession:
    """Records calls and returns queued responses (or raises them)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_geocode_parses_results():
    session = FakeSession([
        FakeResponse({"results": [
            {"name": "Tunis", "country": "Tunisia", "admin1": "Tunis",
             "latitude": 36.8, "longitude": 10.1, "timezone": "Africa/Tunis",
             "population": 693210}
        ]})
    ])
    service = WeatherService(Settings(), session=session)
    results = service.geocode("Tunis")
    assert results[0].name == "Tunis"
    assert results[0].label == "Tunis, Tunisia"
    assert results[0].population == 693210


def test_geocode_caches_repeated_lookups():
    session = FakeSession([FakeResponse({"results": []})])
    service = WeatherService(Settings(), session=session)
    service.geocode("Nowhere")
    service.geocode("Nowhere")
    assert len(session.calls) == 1


def test_empty_city_rejected():
    service = WeatherService(Settings(), session=FakeSession([]))
    with pytest.raises(WeatherError):
        service.geocode("   ")


def test_geocode_network_failure():
    session = FakeSession([requests.ConnectionError("dns")])
    service = WeatherService(Settings(), session=session)
    with pytest.raises(WeatherError, match="Geocoding request failed"):
        service.geocode("Tunis")


def _forecast_payload() -> dict:
    return {
        "current": {
            "temperature_2m": 24.5,
            "apparent_temperature": 25.1,
            "relative_humidity_2m": 60,
            "weather_code": 1,
            "wind_speed_10m": 12,
            "is_day": 1,
            "time": "2026-10-29T10:00",
        },
        "daily": {
            "time": ["2026-10-29", "2026-10-30"],
            "temperature_2m_max": [27, 26],
            "temperature_2m_min": [17, 16],
            "precipitation_sum": [0, 2.4],
            "precipitation_probability_max": [5, 70],
            "weather_code": [1, 61],
        },
    }


def test_current_builds_report():
    session = FakeSession([FakeResponse(_forecast_payload())])
    service = WeatherService(Settings(), session=session)
    report = service.current(Location(name="Tunis", latitude=36.8, longitude=10.1))
    assert report.temperature == 24.5
    assert report.condition == "Mainly clear"
    assert len(report.daily) == 2
    assert report.daily[1].precipitation_probability == 70
    assert report.unit_symbol == DEG_C
    assert report.daily[1].t_min == 16


def test_fahrenheit_unit_is_forwarded():
    payload = {"current": {"temperature_2m": 75, "is_day": 1}, "daily": {"time": []}}
    session = FakeSession([FakeResponse(payload)])
    service = WeatherService(Settings(), session=session)
    report = service.current(
        Location(name="X", latitude=1, longitude=2), "fahrenheit"
    )
    assert session.calls[0][1]["temperature_unit"] == "fahrenheit"
    assert report.unit_symbol == DEG_F


def test_4xx_is_not_retried():
    session = FakeSession([FakeResponse(status=400)])
    service = WeatherService(Settings(max_retries=2), session=session)
    with pytest.raises(WeatherError, match="rejected"):
        service.current(Location(name="X", latitude=1, longitude=2))
    assert len(session.calls) == 1


def test_5xx_is_retried_then_fails():
    session = FakeSession([FakeResponse(status=500), FakeResponse(status=500)])
    service = WeatherService(Settings(max_retries=1), session=session)
    with pytest.raises(WeatherError, match="unavailable"):
        service.current(Location(name="X", latitude=1, longitude=2))
    assert len(session.calls) == 2


def test_retry_then_success():
    payload = {"current": {"temperature_2m": 20, "is_day": 1}, "daily": {"time": []}}
    session = FakeSession([requests.Timeout("slow"), FakeResponse(payload)])
    service = WeatherService(Settings(max_retries=1), session=session)
    report = service.current(Location(name="X", latitude=1, longitude=2))
    assert report.temperature == 20


def test_conditional_fields_tolerate_short_arrays():
    """Open-Meteo occasionally omits a trailing daily value."""

    payload = {
        "current": {"temperature_2m": 20, "is_day": 1},
        "daily": {
            "time": ["2026-10-29", "2026-10-30"],
            "temperature_2m_max": [25],
            "temperature_2m_min": [15, 14],
        },
    }
    session = FakeSession([FakeResponse(payload)])
    service = WeatherService(Settings(), session=session)
    report = service.current(Location(name="X", latitude=1, longitude=2))
    assert report.daily[0].t_max == 25
    assert report.daily[1].t_max is None
