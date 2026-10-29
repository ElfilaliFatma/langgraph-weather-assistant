"""External weather data source (Open-Meteo) with retries and caching.

Open-Meteo is used because it needs no API key: the project must run for any
student without registration. The service is a plain class, not a LangChain
tool, because the graph calls it directly; it is wrapped as a tool only where
an LLM has to choose it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from .config import Settings
from .schema import DayForecast, Location, WeatherReport
from .wmo import describe


class WeatherError(RuntimeError):
    """Raised when the upstream API cannot be used."""


@dataclass
class WeatherService:
    settings: Settings = field(default_factory=Settings)
    session: requests.Session = field(default_factory=requests.Session)
    _cache: dict[tuple, tuple[float, list[Location]]] = field(default_factory=dict)
    cache_ttl: int = 3600

    # ------------------------------------------------------------- geocoding
    def geocode(self, city: str, count: int = 5) -> list[Location]:
        city = (city or "").strip()
        if not city:
            raise WeatherError("No city name was provided.")
        key = (city.lower(), count)
        cached = self._cache.get(key)
        if cached and time.time() - cached[0] < self.cache_ttl:
            return cached[1]

        try:
            response = self.session.get(
                self.settings.geocoding_url,
                params={"name": city, "count": count, "language": "en", "format": "json"},
                timeout=self.settings.http_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise WeatherError(f"Geocoding request failed: {exc}") from exc
        except ValueError as exc:
            raise WeatherError("Geocoding returned invalid JSON.") from exc

        results = [
            Location(
                name=item.get("name", city),
                country=item.get("country", ""),
                admin1=item.get("admin1", ""),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                timezone=item.get("timezone", "auto"),
                population=item.get("population"),
            )
            for item in payload.get("results") or []
        ]
        self._cache[key] = (time.time(), results)
        return results

    # --------------------------------------------------------------- forecast
    def current(self, location: Location, unit: str = "celsius") -> WeatherReport:
        params = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "current": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "apparent_temperature",
                    "is_day",
                    "weather_code",
                    "wind_speed_10m",
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                ]
            ),
            "forecast_days": self.settings.forecast_days,
            "timezone": "auto",
        }
        if unit == "fahrenheit":
            params["temperature_unit"] = "fahrenheit"
            params["wind_speed_unit"] = "mph"

        payload = self._get(self.settings.forecast_url, params)
        current = payload.get("current") or {}
        code = current.get("weather_code")
        condition, _ = describe(code, bool(current.get("is_day", 1)))
        daily = payload.get("daily") or {}
        days: list[DayForecast] = []
        for index, date in enumerate(daily.get("time") or []):
            days.append(
                DayForecast(
                    date=date,
                    t_min=_at(daily.get("temperature_2m_min"), index),
                    t_max=_at(daily.get("temperature_2m_max"), index),
                    precipitation_mm=_at(daily.get("precipitation_sum"), index),
                    precipitation_probability=_at(
                        daily.get("precipitation_probability_max"), index
                    ),
                    weather_code=_at(daily.get("weather_code"), index),
                )
            )
        return WeatherReport(
            location=location,
            unit="fahrenheit" if unit == "fahrenheit" else "celsius",
            temperature=current.get("temperature_2m"),
            apparent_temperature=current.get("apparent_temperature"),
            humidity=current.get("relative_humidity_2m"),
            wind_speed=current.get("wind_speed_10m"),
            weather_code=code,
            condition=condition,
            is_day=bool(current.get("is_day", 1)),
            observed_at=str(current.get("time", "")),
            daily=days,
        )

    def _get(self, url: str, params: dict) -> dict:
        """GET with a bounded retry, since a single transient failure should
        not end the conversation."""

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self.session.get(
                    url, params=params, timeout=self.settings.http_timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                # 4xx other than 429 will not improve on retry
                if 400 <= status < 500 and status != 429:
                    raise WeatherError(f"Weather API rejected the request ({status}).") from exc
                last_error = exc
            except requests.RequestException as exc:
                last_error = exc
            except ValueError as exc:
                raise WeatherError("Weather API returned invalid JSON.") from exc
            if attempt < self.settings.max_retries:
                time.sleep(0.5 * (attempt + 1))
        raise WeatherError(f"Weather API unavailable: {last_error}")

    # ------------------------------------------------------------ convenience
    def resolve(self, city: str) -> list[Location]:
        return self.geocode(city)


def _at(values: list | None, index: int):
    if not values or index >= len(values):
        return None
    value = values[index]
    return None if value is None else value


def build_weather_service(settings: Settings | None = None) -> WeatherService:
    return WeatherService(settings or Settings.from_env())
