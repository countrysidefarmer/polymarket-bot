"""
sports/esports.py — Esports (CS2, LoL, Dota 2, Valorant) series-winner market filter.

Market format on Polymarket (verified April 2026):
  - Series winner (keep): team-name outcomes, question "LoL: Team1 vs Team2 (BO3) - Tournament"
    Slug: [game]-[t1abbrev]-[t2abbrev]-[date]
  - Individual game winner (skip): same team-name outcomes but slug ends in -game1, -game2 etc.
    Question: "LoL: Team1 vs Team2 - Game 1 Winner"
  - Props (skip): Over/Under outcomes (total maps/rounds), slug contains -total-, -handicap-, -map-
    Question: "Total Games Over/Under", "Map Handicap"

Sport identification: event slug prefix from the known esports games.

Estimated volume: ~400-2,000 series-winner markets per year combined.
Note: individual game markets are excluded so we measure series-level informed flow.
"""

from core.api import parse_json_field
from sports.base import AbstractSport

_ESPORTS_SLUG_PREFIXES = (
    "cs2-",     # Counter-Strike 2
    "csgo-",    # Counter-Strike: Global Offensive (legacy)
    "lol-",     # League of Legends
    "dota2-",   # Dota 2
    "val-",     # Valorant
)

# Slug fragments that indicate individual-game or prop markets (appear after the match portion)
_ESPORTS_PROP_SLUG_TERMS = (
    "-game1", "-game2", "-game3", "-game4", "-game5",
    "-map1", "-map2", "-map3", "-map4", "-map5",
    "-total-",
    "-handicap",
    "-map-",
    "-first-",
    "-round",
    "-ace",
    "-knife",
)

# Skip terms in question text
_ESPORTS_SKIP_QUESTION_TERMS = (
    "game 1", "game 2", "game 3", "game 4", "game 5",
    "map 1", "map 2", "map 3", "map 4", "map 5",
    "total games", "total maps", "total rounds",
    "handicap",
    "first blood",
    "first tower",
    "first baron",
    "first dragon",
    "first roshan",
    "pistol round",
    "over/under", "o/u ",
    "correct score",
    "will there be",
    "mvp",
    "most kills",
)

_NON_TEAM_OUTCOMES = {"yes", "no", "over", "under", "-other-"}


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


class EsportsSport(AbstractSport):
    name = "esports"

    def is_sport_market(self, market: dict) -> bool:
        es = _get_event_slug(market)
        if not es:
            return False
        return any(es.startswith(pfx) for pfx in _ESPORTS_SLUG_PREFIXES)

    def is_game_winner_market(self, market: dict) -> bool:
        """
        True for series/match-winner markets only (not individual game or prop markets).
        Requires exactly 2 team-name outcomes and no prop/game-specific indicators.
        """
        outcomes = parse_json_field(market.get("outcomes", []))
        if len(outcomes) != 2:
            return False

        # Must be team names, not Yes/No or Over/Under
        outcome_set = {o.lower().strip() for o in outcomes}
        if outcome_set & _NON_TEAM_OUTCOMES:
            return False

        # Team names: no slashes (not a composite like "Team A/Draw")
        for o in outcomes:
            s = o.lower().strip()
            if "/" in s:
                return False

        # Reject individual-game and prop slugs
        es = _get_event_slug(market)
        for term in _ESPORTS_PROP_SLUG_TERMS:
            if term in es:
                return False

        q = (market.get("question") or "").lower()

        # Reject prop and individual-game questions
        for term in _ESPORTS_SKIP_QUESTION_TERMS:
            if term in q:
                return False

        # Must contain "vs" — a series matchup
        return "vs" in q
