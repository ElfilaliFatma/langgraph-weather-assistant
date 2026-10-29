"""City aliases and disambiguation support.

A free geocoding API returns many "Springfield"-style matches. Two rules keep
the graph honest:

1. alias table for common non-English names (helps Tunisia/MENA demos);
2. if the top matches are in different countries with no dominant population,
   the graph asks the user instead of guessing.
"""

from __future__ import annotations

ALIASES: dict[str, str] = {
    "تونس": "Tunis",
    "تونس العاصمة": "Tunis",
    "صفاقس": "Sfax",
    "سوسة": "Sousse",
    "الجزائر": "Algiers",
    "الدار البيضاء": "Casablanca",
    "الرباط": "Rabat",
    "القاهرة": "Cairo",
    "دبي": "Dubai",
    "باريس": "Paris",
    "لندن": "London",
    "مرسيليا": "Marseille",
    "باردو": "Bardo",
    "نابل": "Nabeul",
    "بنزرت": "Bizerte",
    "القيروان": "Kairouan",
    "gabès": "Gabes",
    "kairouan": "Kairouan",
    "marseille": "Marseille",
}


def canonical(city: str) -> str:
    cleaned = (city or "").strip()
    return ALIASES.get(cleaned, cleaned)


def needs_clarification(candidates: list) -> bool:
    """Ask when the top results are genuinely different places.

    Heuristic: if the first two results are in different countries and the
    smaller one has at least 25% of the population of the larger, both are
    plausible and the user should choose. Otherwise the largest match wins.
    """

    if len(candidates) < 2:
        return False
    first, second = candidates[0], candidates[1]
    if (first.country or "").lower() == (second.country or "").lower():
        return False
    biggest = max(first.population or 0, second.population or 0)
    if biggest == 0:
        return True
    smaller = min(first.population or 0, second.population or 0)
    return smaller / biggest >= 0.25
