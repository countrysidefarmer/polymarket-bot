"""
sports/base.py — Abstract sport interface.

Each sport implementation filters the full closed-market stream to its
own game-winner markets and determines the winning outcome.
"""

import datetime
from abc import ABC, abstractmethod
from typing import Optional

from core.api import determine_winner, parse_json_field


class AbstractSport(ABC):
    """
    Base class for sport-specific market filters.

    Subclasses must implement:
      - name          : str identifier used in logging and output
      - is_sport_market(market)       : True if the market belongs to this sport
      - is_game_winner_market(market) : True if it's a direct game-winner bet

    The default determine_winner() reads outcomePrices; override if needed.
    """

    name: str

    @abstractmethod
    def is_sport_market(self, market: dict) -> bool:
        """Return True if this market belongs to the sport."""
        ...

    @abstractmethod
    def is_game_winner_market(self, market: dict) -> bool:
        """
        Return True if this is a game-winner market (not a spread, O/U, or prop).
        Must also return False for any non-binary or ambiguous markets.
        """
        ...

    def get_winner(self, market: dict, win_price_threshold: float = 0.95) -> Optional[str]:
        """Return the winning outcome label, or None if not clearly resolved."""
        return determine_winner(market, win_price_threshold)

    def parse_end_date(self, market: dict) -> Optional[datetime.datetime]:
        """Return market end date as a naive UTC datetime, or None."""
        for key in ("endDateIso", "endDate", "closedTime", "updatedAt"):
            raw = market.get(key)
            if raw:
                s = str(raw)[:19].rstrip("Z")
                try:
                    return datetime.datetime.fromisoformat(s)
                except Exception:
                    continue
        return None

    def filter_markets(
        self,
        all_closed_markets: list[dict],
        lookback_days: int,
    ) -> list[dict]:
        """
        Filter all_closed_markets to sport game-winner markets within the lookback window.

        Returns a list of market dicts sorted by end date (oldest first), each with
        a '_winner' key set to the winning outcome label.
        """
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)
        result = []

        for m in all_closed_markets:
            if not self.is_sport_market(m):
                continue
            if not self.is_game_winner_market(m):
                continue

            end_dt = self.parse_end_date(m)
            # Include if end_dt is unknown (can't exclude) or within lookback
            if end_dt is not None and end_dt < cutoff:
                continue

            winner = self.get_winner(m)
            if winner is None:
                continue

            m = dict(m)  # shallow copy to avoid mutating the original
            m["_winner"] = winner
            result.append(m)

        result.sort(key=lambda m: m.get("endDate") or m.get("endDateIso") or "")
        return result
