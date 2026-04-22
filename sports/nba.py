"""
sports/nba.py — NBA game-winner market filter.

Ported directly from phase1_nba_informed_flow.py with no logic changes.

NBA market identification:
  - Keywords in question/title/slug/eventSlug: "nba", " nba ", "national basketball"
  - Outcomes: exactly 2 team names (not Yes/No, not Over/Under)
  - Skip terms: spread, O/U, player props, MVP, draft, etc.

API note: The Gamma API ignores q= and date parameters for /markets.
All pages are fetched and filtered client-side.
"""

from core.api import parse_json_field
from sports.base import AbstractSport

_NBA_KEYWORDS = ("nba", " nba ", "national basketball")

_NON_TEAM_OUTCOMES = {"yes", "no", "over", "under"}

_SKIP_QUESTION_TERMS = (
    "spread",   # catches "Spread: X (±Y)" at start AND " spread" mid-question
    "o/u ", "over/under", " pts", "points", "assists",
    "rebounds", "steals", "blocks", "threes", "field goals", "turnovers",
    "minutes", "double-double", "triple-double", "first basket",
    "first team", "mvp", "draft", "trade",
)


class NBASport(AbstractSport):
    name = "nba"

    def is_sport_market(self, market: dict) -> bool:
        text = (
            (market.get("question") or "")
            + " " + (market.get("title") or "")
            + " " + (market.get("slug") or "")
            + " " + (market.get("eventSlug") or "")
        ).lower()
        return any(kw in text for kw in _NBA_KEYWORDS)

    def is_game_winner_market(self, market: dict) -> bool:
        """
        True if both outcomes are team names (not Yes/No, Over/Under, or player props).
        """
        outcomes = parse_json_field(market.get("outcomes", []))
        if len(outcomes) != 2:
            return False

        for o in outcomes:
            s = o.lower().strip()
            if s in _NON_TEAM_OUTCOMES:
                return False
            if any(c.isdigit() for c in s):
                return False

        q = (market.get("question") or "").lower()
        for term in _SKIP_QUESTION_TERMS:
            if term in q:
                return False

        return True
