"""
sports/soccer.py — Soccer game-winner market filter.

Soccer market format on Polymarket (verified April 2026):
  - Markets use Yes/No outcomes: "Will [Team] win on YYYY-MM-DD?"
  - NOT team-vs-team format (unlike NBA). The only team-name-outcome
    soccer markets are major cup finals (rare; ~3 per year).
  - Both formats are handled: Yes/No win markets AND team-name finals.

Sport identification:
  - Primary: event slug prefix (reliable, sport-specific 3-char codes)
    epl- (EPL), lal- (La Liga), ucl- (Champions League), fl1- (Ligue 1),
    uel- (Europa League), bun- (Bundesliga), sea- (Serie A),
    elc- (Europa Conference), efa- (EFL Championship/FA Cup lower rounds),
    ere- (Eredivisie), carabao- (Carabao Cup)
  - Fallback: competition name in question text (for cup finals with full slugs)

Leagues covered in 365-day window (verified from cache):
  EPL: 518, EFL/FA Cup: 262, Serie A: 366, La Liga: 366, Ligue 1: 308,
  Eredivisie: 274, Europa League: 274, Bundesliga: 270,
  Europa Conference: 458, Champions League: 254, Carabao Cup: 81
  Total: 3,431 resolved game-winner markets

Skip terms exclude: spread, O/U, BTTS, draw markets, first goalscorer,
  booking, halftime props, season winner, relegation, qualification progression.
"""

from core.api import parse_json_field
from sports.base import AbstractSport

# Event slug prefixes confirmed to be soccer competitions
_SOCCER_EVENT_SLUG_PREFIXES = (
    "epl-",       # English Premier League
    "lal-",       # La Liga (Spain)
    "ucl-",       # UEFA Champions League
    "fl1-",       # Ligue 1 (France)
    "uel-",       # UEFA Europa League
    "bun-",       # Bundesliga (Germany)
    "sea-",       # Serie A (Italy)
    "elc-",       # UEFA Europa Conference League
    "efa-",       # EFL Championship / FA Cup lower rounds
    "ere-",       # Eredivisie (Netherlands)
    "carabao-",   # Carabao Cup (England)
)

# Full-slug prefixes for cup finals and international tournaments
_SOCCER_FULL_SLUG_PREFIXES = (
    "champions-league-",
    "europa-league-",
    "premier-league-",
    "bundesliga-",
    "serie-a-",
    "la-liga-",
    "ligue-1-",
    "copa-del-rey-",
    "copa-america-",
    "world-cup-",
    "euro-2024-", "euro-2025-", "euro-2026-",
    "fa-cup-",
    "carabao-cup-",
    "supercopa-",
    "dfb-pokal-",
    "coupe-de-france-",
    "europa-conference-",
    "uefa-europa-conference-",
)

# Skip terms for non-win markets (applied to question text)
_SOCCER_WIN_SKIP_TERMS = (
    "spread",
    "o/u ", "over/under",
    "both teams to score",
    "end in a draw",
    "draw?",
    "first goal", "first scorer", "anytime scorer",
    "yellow card", "corner",
    "booking",
    "halftime", "half time", "half-time",
    "win the premier league", "win the champions league",
    "win the bundesliga", "win the serie a", "win the la liga",
    "win the europa", "win the fa cup", "win the carabao",
    "win the 2024", "win the 2025", "win the 2026",
    "be relegated", "relegated",
    "finish first", "finish second", "finish top",
    "qualify to the",
    "advance to the",
    "to advance",
    "line:", "line :",
    "favorite", "underdog",
    "esports",
    "1st half", "2nd half",
    "clean sheet",
    "penalty shootout",
    "most goals", "fewest goals",
)

_NON_TEAM_OUTCOMES = {"yes", "no", "over", "under", "-other-"}


def _get_event_slug(market: dict) -> str:
    """Return the event slug from eventSlug field or events list, lowercased."""
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


class SoccerSport(AbstractSport):
    name = "soccer"

    def is_sport_market(self, market: dict) -> bool:
        es = _get_event_slug(market)
        if not es:
            return False
        return any(es.startswith(pfx) for pfx in _SOCCER_EVENT_SLUG_PREFIXES + _SOCCER_FULL_SLUG_PREFIXES)

    def is_game_winner_market(self, market: dict) -> bool:
        """
        True for soccer game-winner markets in either format:
          - Yes/No: "Will [Team] win on YYYY-MM-DD?" (the common format)
          - Team names: "Barcelona vs. Real Madrid" (cup finals only)

        Excludes spread, O/U, BTTS, draw, prop, and season markets.
        """
        outcomes = parse_json_field(market.get("outcomes", []))
        if len(outcomes) != 2:
            return False

        q = (market.get("question") or "").lower()

        # Reject all skip-term markets regardless of outcome format
        for term in _SOCCER_WIN_SKIP_TERMS:
            if term in q:
                return False

        outcome_set = {o.lower().strip() for o in outcomes}

        if outcome_set == {"yes", "no"}:
            # Yes/No format: must be a win market (question contains "win" or "beat")
            return "win" in q or "beat" in q

        else:
            # Team-name format (cup finals, international tournaments)
            for o in outcomes:
                s = o.lower().strip()
                if s in _NON_TEAM_OUTCOMES:
                    return False
                if any(c.isdigit() for c in s):
                    return False
                if "draw" in s or "/" in s:
                    return False
            return True
