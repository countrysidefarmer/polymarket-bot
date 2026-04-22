"""
core/api.py — Polymarket API client with disk-level JSON caching.

API facts (verified live, April 2026):
  - Gamma API:  GET https://gamma-api.polymarket.com/markets?closed=true
                Paginate ALL closed markets, filter client-side.
                q= parameter is ignored.
  - Data API:   GET https://data-api.polymarket.com/trades?market=<conditionId>
                Param is 'market', NOT 'conditionId'. Public, no auth.
                Supports: limit, offset, takerOnly, side.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

log = logging.getLogger(__name__)

# Defaults — callers can override via PolymarketClient constructor
_DEFAULT_CACHE_DIR = Path("./cache")
_DEFAULT_REQUEST_DELAY = 0.1
_DEFAULT_MARKETS_PER_PAGE = 100
_DEFAULT_TRADES_PER_PAGE = 500
_DEFAULT_MAX_TRADE_OFFSET = 9500


class PolymarketClient:
    """Thread-unsafe but simple API client with disk caching and backoff."""

    def __init__(
        self,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        request_delay: float = _DEFAULT_REQUEST_DELAY,
        markets_per_page: int = _DEFAULT_MARKETS_PER_PAGE,
        trades_per_page: int = _DEFAULT_TRADES_PER_PAGE,
        max_trade_offset: int = _DEFAULT_MAX_TRADE_OFFSET,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_delay = request_delay
        self.markets_per_page = markets_per_page
        self.trades_per_page = trades_per_page
        self.max_trade_offset = max_trade_offset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, prefix: str, url: str, params: dict) -> Path:
        raw = url + json.dumps(params, sort_keys=True)
        h = hashlib.md5(raw.encode()).hexdigest()[:16]
        return self.cache_dir / f"{prefix}_{h}.json"

    def _backoff(self):
        self._request_delay = min(self._request_delay * 2, 30.0)
        log.warning(f"Rate-limited; backing off to {self._request_delay:.1f}s delay")

    def _reset_delay(self):
        self._request_delay = _DEFAULT_REQUEST_DELAY

    def api_get(
        self, url: str, params: Optional[dict] = None, cache_prefix: Optional[str] = None
    ):
        """
        GET with disk-level JSON caching and polite rate limiting.
        Returns parsed JSON (list or dict) or None on failure.
        """
        params = params or {}

        if cache_prefix:
            cp = self._cache_path(cache_prefix, url, params)
            if cp.exists():
                try:
                    with open(cp) as f:
                        return json.load(f)
                except Exception:
                    cp.unlink(missing_ok=True)

        time.sleep(self._request_delay)

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                log.error(f"Network error ({url}): {exc}")
                time.sleep(5)
                continue

            if resp.status_code == 429:
                self._backoff()
                time.sleep(self._request_delay)
                continue

            if not resp.ok:
                log.debug(f"HTTP {resp.status_code} for {url} {params}")
                return None

            self._reset_delay()

            try:
                data = resp.json()
            except Exception:
                log.error(f"JSON decode error for {url}")
                return None

            if cache_prefix:
                cp = self._cache_path(cache_prefix, url, params)
                try:
                    with open(cp, "w") as f:
                        json.dump(data, f)
                except Exception:
                    pass

            return data

        log.error(f"All retries failed for {url}")
        return None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def fetch_all_closed_markets(self, log_prefix: str = "mkt") -> list[dict]:
        """
        Paginate through all closed markets from the Gamma API.
        Returns raw list of market dicts (unfiltered by sport or date).
        Cached page-by-page; subsequent runs are fast.
        """
        all_markets: list[dict] = []
        offset = 0

        log.info("Fetching all closed markets from Gamma API (cached after first run)...")

        while True:
            params = {
                "closed": "true",
                "limit": self.markets_per_page,
                "offset": offset,
            }
            raw = self.api_get(
                f"{GAMMA_API}/markets",
                params,
                cache_prefix=f"{log_prefix}_{offset}",
            )
            items = _unwrap(raw)

            if not items:
                break

            all_markets.extend(items)

            if offset % 10000 == 0:
                log.info(f"  scanned offset={offset:,}  total markets so far: {len(all_markets):,}")

            if len(items) < self.markets_per_page:
                break

            offset += self.markets_per_page

        log.info(f"Total closed markets fetched: {len(all_markets):,}")
        return all_markets

    def fetch_market_trades(self, condition_id: str) -> list[dict]:
        """
        Fetch all BUY trades for a market by conditionId.

        Uses the 'market' query parameter (confirmed working, public, no auth).
        Sets takerOnly=false to capture both limit-order makers and market takers.
        Paginates until exhausted or the API's offset cap (10,000) is reached.
        """
        all_trades: list[dict] = []
        offset = 0

        while True:
            params = {
                "market": condition_id,
                "limit": self.trades_per_page,
                "offset": offset,
                "takerOnly": "false",
                "side": "BUY",
            }
            # Use 'mt2' prefix to separate from old broken 'mktrade_' cache files
            raw = self.api_get(
                f"{DATA_API}/trades",
                params,
                cache_prefix=f"mt2_{condition_id[:12]}_{offset}",
            )
            trades = _unwrap(raw)
            if not trades:
                break
            all_trades.extend(trades)
            if len(trades) < self.trades_per_page:
                break
            offset += self.trades_per_page
            if offset > self.max_trade_offset:
                log.debug(f"  Hit offset cap for market {condition_id[:16]}")
                break

        return all_trades


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _unwrap(data, list_keys=("data", "markets", "results")) -> list:
    """Normalise API response to a list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in list_keys:
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


def parse_json_field(val) -> list:
    """Parse a field that may be a JSON-encoded string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def determine_winner(market: dict, win_price_threshold: float = 0.95) -> Optional[str]:
    """
    Return the winning outcome label, or None if not clearly resolved.

    Checks outcomePrices: the outcome whose price >= win_price_threshold is the winner.
    Returns None if zero or multiple outcomes cross the threshold.
    """
    outcomes = parse_json_field(market.get("outcomes", []))
    prices_raw = parse_json_field(market.get("outcomePrices", []))

    if not outcomes or not prices_raw or len(outcomes) != len(prices_raw):
        return None

    try:
        prices = [float(p) for p in prices_raw]
    except (TypeError, ValueError):
        return None

    winners = [outcomes[i] for i, p in enumerate(prices) if p >= win_price_threshold]
    return winners[0] if len(winners) == 1 else None
