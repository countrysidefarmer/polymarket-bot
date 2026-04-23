"""
sports/ufc.py — UFC / MMA fight-winner market filter.

Market format on Polymarket (verified April 2026):
  - Fight winner: outcomes are fighter names (e.g. "Khamzat Chimaev", "Lucas Rocha")
    Slug: ufc-[f1abbrev]-[f2abbrev]-[date]
    Question: "UFC [event]: Fighter1 vs. Fighter2 (Weightclass, Card)"
  - Non-fight markets: Yes/No outcomes
    Slug: will-[fighter]-be-the-ufc-...-champion-...
    Question: "Will [Fighter] be the UFC [division] Champion on [date]?"

Sport identification: "ufc" in question text (catches both fight and non-fight markets).
Game winner identification: exactly 2 fighter-name (non Yes/No) outcomes + "vs" in question.

Estimated volume: ~150-180 fight-winner markets per year.
"""

from core.api import parse_json_field
from sports.base import AbstractSport

_NON_FIGHTER_OUTCOMES = {"yes", "no", "over", "under", "-other-", "draw", "nc"}

# Skip terms for non-fight/prop questions
_UFC_SKIP_QUESTION_TERMS = (
    "champion on ",
    "title on ",
    "be the ufc",
    "method of victory",
    "round betting",
    "goes the distance",
    "total rounds",
    "will there be",
    "correct",
    "ko/tko",
    "submission",
    "decision",
    "finish in round",
    "fighter of the night",
    "performance of the night",
)


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


class UFCSport(AbstractSport):
    name = "ufc"

    def is_sport_market(self, market: dict) -> bool:
        q = (market.get("question") or "").lower()
        es = _get_event_slug(market)
        return "ufc" in q or es.startswith("ufc-")

    def is_game_winner_market(self, market: dict) -> bool:
        """
        True for straight fight-winner markets only.
        Requires exactly 2 fighter-name outcomes, "vs" in question,
        and no method-of-victory or prop terms.
        """
        outcomes = parse_json_field(market.get("outcomes", []))
        if len(outcomes) != 2:
            return False

        # Must be fighter names, not Yes/No or props
        outcome_set = {o.lower().strip() for o in outcomes}
        if outcome_set & _NON_FIGHTER_OUTCOMES:
            return False

        # Fighter names: no digits, no slashes
        for o in outcomes:
            s = o.lower().strip()
            if any(c.isdigit() for c in s) or "/" in s:
                return False

        q = (market.get("question") or "").lower()

        # Must be a matchup question
        if "vs" not in q and "vs." not in q:
            return False

        # Reject props and non-fight markets
        for term in _UFC_SKIP_QUESTION_TERMS:
            if term in q:
                return False

        return True
