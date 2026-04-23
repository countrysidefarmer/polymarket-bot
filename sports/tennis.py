"""
sports/tennis.py — Tennis (ATP/WTA) game-winner market filter.

Market format on Polymarket (verified April 2026):
  - Outcomes: Player names (NOT Yes/No), e.g. "Maxime Chazal", "Marin Cilic"
  - Question format: "City: Player1 vs Player2" or "Tournament: Player1 vs Player2"
  - Slug format: [atp|wta]-[p1abbrev]-[p2abbrev]-[date]
  - Prop slugs append more terms: [atp|wta]-...-[date]-first-set-total-9pt5
                                                              -handicap
                                                              -total-games-...

Sport identification: event slug prefix atp- or wta-

Estimated volume: ~9,500 game-winner markets per year (largest sport on Polymarket).
"""

from core.api import parse_json_field
from sports.base import AbstractSport

_TENNIS_SLUG_PREFIXES = ("atp-", "wta-")

# Prop market indicators in the slug (appear after the date portion)
_TENNIS_PROP_SLUG_TERMS = (
    "-first-set",
    "-handicap",
    "-total-",
    "-games-",
    "-set-",
    "-tiebreak",
    "-retirement",
    "-walkover",
)

# Prop/skip terms in question text
_TENNIS_SKIP_QUESTION_TERMS = (
    " set ", "total games", "handicap", "over/under",
    "o/u ", "retirement", "walkover", "correct score",
    "first set winner", "double fault", "ace",
    "will there be",
)

_NON_PLAYER_OUTCOMES = {"yes", "no", "over", "under", "-other-"}


def _get_event_slug(market: dict) -> str:
    es = market.get("eventSlug")
    if es:
        return es.lower()
    events = market.get("events", [])
    if events and isinstance(events, list):
        for ev in events:
            slug = ev.get("slug") or ev.get("ticker") or ""
            if slug:
                return slug.lower()
    return ""


class TennisSport(AbstractSport):
    name = "tennis"

    def is_sport_market(self, market: dict) -> bool:
        es = _get_event_slug(market)
        if not es:
            return False
        return any(es.startswith(pfx) for pfx in _TENNIS_SLUG_PREFIXES)

    def is_game_winner_market(self, market: dict) -> bool:
        """
        True for match-winner markets only (not set/game props).
        Requires exactly 2 player-name outcomes and no prop indicators.
        """
        outcomes = parse_json_field(market.get("outcomes", []))
        if len(outcomes) != 2:
            return False

        # Reject Yes/No and prop outcomes
        outcome_set = {o.lower().strip() for o in outcomes}
        if outcome_set & _NON_PLAYER_OUTCOMES:
            return False

        # Each outcome must look like a player name (no digits, no slashes)
        for o in outcomes:
            s = o.lower().strip()
            if any(c.isdigit() for c in s) or "/" in s:
                return False

        # Reject prop slugs
        es = _get_event_slug(market)
        for term in _TENNIS_PROP_SLUG_TERMS:
            if term in es:
                return False

        # Reject prop question text
        q = (market.get("question") or "").lower()
        for term in _TENNIS_SKIP_QUESTION_TERMS:
            if term in q:
                return False

        # Must contain "vs" — confirms it's a head-to-head match
        return "vs" in q
