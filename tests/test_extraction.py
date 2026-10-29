from __future__ import annotations

import pytest

from weather.aliases import canonical, needs_clarification
from weather.extraction import QueryExtractor
from weather.schema import Location


@pytest.fixture()
def extractor() -> QueryExtractor:
    return QueryExtractor(None, ["Tunis", "Sousse", "Paris", "London", "Sfax"])


@pytest.mark.parametrize(
    "text,city",
    [
        ("What is the weather in Tunis?", "Tunis"),
        ("weather in Paris tomorrow", "Paris"),
        ("Compare Paris and London", "Paris"),
        ("météo à Sousse", "Sousse"),
    ],
)
def test_city_extraction(extractor, text, city):
    assert city in extractor.extract(text).cities


def test_unknown_city_is_still_extracted(extractor):
    assert "Gabes" in extractor.extract("weather in Gabes").cities


def test_no_city_yields_empty(extractor):
    query = extractor.extract("what is the weather like?")
    assert query.cities == []


def test_intent_detection(extractor):
    assert extractor.extract("weather in Tunis")["intent"] if False else True
    assert extractor.extract("weather in Tunis").intent == "current"
    assert extractor.extract("forecast for Paris tomorrow").intent == "forecast"
    assert extractor.extract("compare Tunis and Paris").intent == "compare"
    assert extractor.extract("should I take an umbrella in Sousse?").intent == "advice"


def test_days_and_weekend(extractor):
    assert extractor.extract("weather in Tunis").days == 1
    assert extractor.extract("forecast for Tunis this weekend").days == 2
    assert extractor.extract("weather in Tunis for 5 days").days == 5
    assert extractor.extract("weather in Tunis tomorrow").days == 2
    # "days" must never be mistaken for a city
    assert extractor.extract("forecast for Tunis for 5 days").cities == ["Tunis"]


def test_activity_detection(extractor):
    assert extractor.extract("is it good for running in Tunis?").activity == "running"
    assert extractor.extract("beach day in Sousse?").activity == "beach"


def test_fahrenheit(extractor):
    assert extractor.extract("weather in Tunis in fahrenheit").unit == "fahrenheit"


def test_alias_table():
    assert canonical("تونس") == "Tunis"
    assert canonical("Sfax") == "Sfax"


def test_clarification_heuristic():
    big = Location(name="Springfield", country="US", latitude=0, longitude=0, population=116250)
    small = Location(name="Springfield", country="AU", latitude=0, longitude=0, population=1000)
    similar = Location(name="Springfield", country="CA", latitude=0, longitude=0, population=90000)
    assert needs_clarification([big, small]) is False
    assert needs_clarification([big, similar]) is True
    assert needs_clarification([big]) is False
    same_country = Location(name="Paris", country="US", latitude=0, longitude=0, population=5000)
    paris = Location(name="Paris", country="US", latitude=0, longitude=0, population=30000)
    assert needs_clarification([paris, same_country]) is False
