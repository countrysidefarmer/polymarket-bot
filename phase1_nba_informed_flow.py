#!/usr/bin/env python3
"""
Phase 1: NBA Informed-Flow Analysis
====================================
Tests whether wallets with positive historical NBA PnL produce flow that predicts
the winner of new NBA markets out-of-sample.

Go/No-Go threshold: informed hit rate >= 58% AND (informed − retail) >= 8pp.

API facts (verified live, April 2026):
  - NBA markets:    GET https://gamma-api.polymarket.com/markets?closed=true
                    Paginate ALL closed markets, filter client-side for NBA
                    moneyline markets — the q= param is ignored by the API.
  - Market trades:  GET https://data-api.polymarket.com/trades?market=<conditionId>
                    The param is named 'market', NOT 'conditionId'. Public, no auth.
                    Supports: limit (0–10000), offset (0–10000), takerOnly, side.
  - Wallet trades:  GET https://data-api.polymarket.com/trades?user=<addr>
                    Also works — full wallet history. Fields: conditionId, outcome,
                    price, size, side, proxyWallet, slug.
  - /leaderboard, /positions, /activity — all return 404/400.

Architecture (conditionId-centric):
  1. Fetch all closed NBA moneyline markets from Gamma; split 50/50 training/test.
  2. For each training market: fetch all BUY trades via market=conditionId.
     Count BUY trades per wallet. Pre-filter to wallets with >= MIN_NBA_TRADES.
     Store training trades in memory — no separate wallet-history fetches needed.
  3. Classify wallets: top 10% PnL/$ = informed, bottom 50% = retail.
  4. For each test market: fetch BUY trades, aggregate informed/retail volume
     by outcome. Informed signal = outcome with highest informed BUY volume.
  5. Report hit rates, edge, and go/no-go decision.
"""

import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

CACHE_DIR = Path("./cache")
OUTPUT_DIR = Path("./output")

LOOKBACK_DAYS = 180
REQUEST_DELAY = 0.1        # seconds between requests; raised on 429
MIN_NBA_TRADES = 20        # minimum qualifying BUY trades for classification
INFORMED_TOP_PCT = 0.10    # top 10% by PnL/$ -> informed
RETAIL_BOTTOM_PCT = 0.50   # bottom 50% by PnL/$ -> retail
WIN_PRICE_THRESHOLD = 0.95 # outcome price >= this -> winner
MARKETS_PER_PAGE = 100
TRADES_PER_PAGE = 500      # trades per paginated request (max: 10,000)
MAX_TRADE_OFFSET = 9500    # API caps offset at 10,000; stop before that

# Go/No-Go thresholds
GO_HIT_RATE = 0.58
GO_EDGE_PP = 8.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_request_delay = REQUEST_DELAY


def _backoff():
    global _request_delay
    _request_delay = min(_request_delay * 2, 30.0)
    log.warning(f"Backing off; new delay = {_request_delay:.1f}s")


def _reset_delay():
    global _request_delay
    _request_delay = REQUEST_DELAY


# ---------------------------------------------------------------------------
# Caching + HTTP
# ---------------------------------------------------------------------------

def _cache_path(prefix: str, url: str, params: dict) -> Path:
    raw = url + json.dumps(params, sort_keys=True)
    h = hashlib.md5(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{prefix}_{h}.json"


def api_get(url: str, params: Optional[dict] = None, cache_prefix: Optional[str] = None):
    """
    GET with disk-level JSON caching and polite rate limiting.
    Returns parsed JSON (list or dict) or None on failure.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = params or {}

    if cache_prefix:
        cp = _cache_path(cache_prefix, url, params)
        if cp.exists():
            try:
                with open(cp) as f:
                    return json.load(f)
            except Exception:
                cp.unlink(missing_ok=True)

    time.sleep(_request_delay)

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            log.error(f"Network error ({url}): {exc}")
            time.sleep(5)
            continue

        if resp.status_code == 429:
            _backoff()
            time.sleep(_request_delay)
            continue

        if not resp.ok:
            log.debug(f"HTTP {resp.status_code} for {url} {params}")
            return None

        _reset_delay()

        try:
            data = resp.json()
        except Exception:
            log.error(f"JSON decode error for {url}")
            return None

        if cache_prefix:
            cp = _cache_path(cache_prefix, url, params)
            try:
                with open(cp, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass

        return data

    log.error(f"All retries failed for {url}")
    return None


def _unwrap(data, list_keys=("data", "markets", "results")) -> list:
    """Normalise API response to a list."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in list_keys:
            if k in data and isinstance(data[k], list):
                return data[k]
    return []


# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------

def _parse_json_field(val):
    """Parse a field that may be a JSON-encoded string or already a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


_NON_TEAM_OUTCOMES = {"yes", "no", "over", "under"}

_SKIP_QUESTION_TERMS = (
    " spread", "o/u ", "over/under", " pts", "points", "assists",
    "rebounds", "steals", "blocks", "threes", "field goals", "turnovers",
    "minutes", "double-double", "triple-double", "first basket",
    "first team", "mvp", "draft", "trade",
)


def _is_game_winner_market(market: dict) -> bool:
    """
    Return True if this looks like a full-game moneyline market where both
    outcomes are team names (not Yes/No, Over/Under, or player props).
    """
    outcomes = _parse_json_field(market.get("outcomes", []))
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


def determine_winner(market: dict) -> Optional[str]:
    """
    Return the winning outcome label, or None if not clearly resolved.
    """
    outcomes = _parse_json_field(market.get("outcomes", []))
    prices_raw = _parse_json_field(market.get("outcomePrices", []))

    if not outcomes or not prices_raw or len(outcomes) != len(prices_raw):
        return None

    try:
        prices = [float(p) for p in prices_raw]
    except (TypeError, ValueError):
        return None

    winners = [outcomes[i] for i, p in enumerate(prices) if p >= WIN_PRICE_THRESHOLD]
    return winners[0] if len(winners) == 1 else None


def _parse_end_date(market: dict) -> Optional[datetime.datetime]:
    for key in ("endDate", "end_date_iso", "endDateIso"):
        raw = market.get(key)
        if raw:
            try:
                s = raw.replace("Z", "+00:00")
                return datetime.datetime.fromisoformat(s).replace(tzinfo=None)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

_NBA_KEYWORDS = ("nba", " nba ", "national basketball")


def _is_nba_market(market: dict) -> bool:
    """Return True if the market question/title appears to be NBA-related."""
    text = (
        (market.get("question") or "")
        + " " + (market.get("title") or "")
        + " " + (market.get("slug") or "")
        + " " + (market.get("eventSlug") or "")
    ).lower()
    return any(kw in text for kw in _NBA_KEYWORDS)


def fetch_nba_markets(lookback_days: int = LOOKBACK_DAYS) -> list[dict]:
    """
    Fetch closed/resolved NBA markets from the last `lookback_days` days.

    Note: The Gamma API ignores q= and date-range parameters; it returns all
    closed markets regardless. We paginate all pages (cached after first run)
    and filter client-side.
    """
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)
    all_markets: list[dict] = []
    offset = 0
    exhausted = False

    log.info(f"Fetching NBA markets closed since {cutoff.date()} ...")
    log.info("(cached pages load fast; new pages cost ~0.1s each)")

    while not exhausted:
        params = {
            "closed": "true",
            "limit": MARKETS_PER_PAGE,
            "offset": offset,
        }
        raw = api_get(f"{GAMMA_API}/markets", params, cache_prefix=f"nba_mkt_{offset}")
        items = _unwrap(raw)

        if not items:
            break

        for m in items:
            if not _is_nba_market(m):
                continue
            end_dt = _parse_end_date(m)
            if end_dt is None or end_dt >= cutoff:
                all_markets.append(m)

        if offset % 10000 == 0:
            log.info(f"  scanned offset={offset:,}  NBA markets so far: {len(all_markets)}")

        if len(items) < MARKETS_PER_PAGE:
            exhausted = True
        else:
            offset += MARKETS_PER_PAGE

    log.info(f"Total NBA markets in lookback window: {len(all_markets)}")
    return all_markets


def fetch_market_trades(condition_id: str) -> list[dict]:
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
            "limit": TRADES_PER_PAGE,
            "offset": offset,
            "takerOnly": "false",
            "side": "BUY",
        }
        # Use 'mt2' prefix to keep separate from old broken 'mktrade_' cache files
        raw = api_get(
            f"{DATA_API}/trades",
            params,
            cache_prefix=f"mt2_{condition_id[:12]}_{offset}",
        )
        trades = _unwrap(raw)
        if not trades:
            break
        all_trades.extend(trades)
        if len(trades) < TRADES_PER_PAGE:
            break
        offset += TRADES_PER_PAGE
        if offset > MAX_TRADE_OFFSET:
            log.debug(f"  Hit offset cap for market {condition_id[:16]}")
            break

    return all_trades


# ---------------------------------------------------------------------------
# Wallet classification
# ---------------------------------------------------------------------------

def compute_wallet_nba_pnl(
    wallet_trades: list[dict],
    allowed_condition_ids: set,
    market_winners: dict,
) -> Optional[dict]:
    """
    Approximate risk-adjusted PnL for a wallet's training-period NBA BUY trades.

    Method (Phase 1 approximation):
      invested = price * size
      PnL = size * (1 - price)  if outcome == winner
            -invested             otherwise

    Returns None if fewer than MIN_NBA_TRADES qualifying trades.
    """
    total_pnl = 0.0
    total_invested = 0.0
    trade_count = 0

    for trade in wallet_trades:
        if trade.get("side") != "BUY":
            continue

        cid = trade.get("conditionId")
        if not cid or cid not in allowed_condition_ids:
            continue

        winner = market_winners.get(cid)
        if winner is None:
            continue  # market not cleanly resolved; skip

        try:
            price = float(trade["price"])
            size = float(trade["size"])
        except (KeyError, ValueError, TypeError):
            continue

        if price <= 0 or price >= 1:
            continue  # data artifact; skip

        invested = price * size
        outcome = trade.get("outcome", "")

        if outcome == winner:
            pnl = size * (1.0 - price)
        else:
            pnl = -invested

        total_pnl += pnl
        total_invested += invested
        trade_count += 1

    if trade_count < MIN_NBA_TRADES or total_invested <= 0:
        return None

    return {
        "total_pnl": total_pnl,
        "total_invested": total_invested,
        "pnl_per_dollar": total_pnl / total_invested,
        "trade_count": trade_count,
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Fetch and filter NBA markets
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1  Fetching NBA markets")
    log.info("=" * 60)

    raw_markets = fetch_nba_markets()

    if not raw_markets:
        log.error("No NBA markets found. Check the Gamma API.")
        return

    # Keep only full-game moneyline markets; drop spreads, O/U, player props
    winner_markets = [m for m in raw_markets if _is_game_winner_market(m)]
    log.info(
        f"Game-winner (moneyline) markets: {len(winner_markets)} / {len(raw_markets)} NBA total"
    )

    resolved_markets = []
    for m in winner_markets:
        winner = determine_winner(m)
        if winner:
            m["_winner"] = winner
            resolved_markets.append(m)

    log.info(
        f"Markets with clear winner: {len(resolved_markets)} / {len(winner_markets)}"
        f" ({len(winner_markets) - len(resolved_markets)} unresolved or ambiguous)"
    )

    if len(resolved_markets) < 10:
        log.warning(
            f"Only {len(resolved_markets)} resolved markets found. "
            "Results will not be statistically meaningful."
        )

    # Sort by end date; split 50/50 (walk-forward: classify on earlier, test on later)
    resolved_markets.sort(key=lambda m: m.get("endDate") or m.get("end_date_iso") or "")
    split_idx = len(resolved_markets) // 2
    training_markets = resolved_markets[:split_idx]
    test_markets = resolved_markets[split_idx:]

    log.info(f"Training: {len(training_markets)} markets | Test: {len(test_markets)} markets")

    train_cids = {m["conditionId"] for m in training_markets if m.get("conditionId")}
    train_winners = {
        m["conditionId"]: m["_winner"]
        for m in training_markets
        if m.get("conditionId")
    }

    # ------------------------------------------------------------------
    # Step 2: Fetch training-market BUY trades; build wallet pool in-memory
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2  Fetching training-market trades & building wallet pool")
    log.info("=" * 60)

    # Per-wallet: count of BUY trades in training markets (for pre-filtering)
    wallet_buy_counts: dict[str, int] = {}
    # Per-wallet: list of trade dicts from training markets (for in-memory classification)
    wallet_training_trades: dict[str, list] = {}

    for market in tqdm(training_markets, desc="Training market trades"):
        cid = market.get("conditionId")
        if not cid:
            continue
        trades = fetch_market_trades(cid)
        for t in trades:
            pw = t.get("proxyWallet")
            if not pw:
                continue
            # fetch_market_trades already filters to side=BUY, but double-check
            if t.get("side") == "BUY":
                wallet_buy_counts[pw] = wallet_buy_counts.get(pw, 0) + 1
            wallet_training_trades.setdefault(pw, []).append(t)

    # Pre-filter: only wallets with enough training BUY trades
    candidate_wallets = {
        w for w, cnt in wallet_buy_counts.items() if cnt >= MIN_NBA_TRADES
    }

    log.info(
        f"Unique wallets seen: {len(wallet_buy_counts):,} total, "
        f"{len(candidate_wallets):,} with >= {MIN_NBA_TRADES} BUY trades in training"
    )

    if not candidate_wallets:
        log.error(
            f"No wallets with >= {MIN_NBA_TRADES} training BUY trades. "
            "Consider lowering MIN_NBA_TRADES or extending LOOKBACK_DAYS."
        )
        return

    # ------------------------------------------------------------------
    # Step 3: Classify wallets using in-memory training trades
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3  Classifying wallets")
    log.info("=" * 60)

    wallet_stats: dict[str, dict] = {}

    for wallet in tqdm(candidate_wallets, desc="Classifying wallets"):
        training_trades = wallet_training_trades.get(wallet, [])
        stats = compute_wallet_nba_pnl(training_trades, train_cids, train_winners)
        if stats:
            wallet_stats[wallet] = stats

    log.info(
        f"Wallets with >= {MIN_NBA_TRADES} qualifying NBA BUY trades: {len(wallet_stats)}"
    )

    if len(wallet_stats) < 5:
        log.warning(
            f"Only {len(wallet_stats)} wallets qualify. "
            f"Consider lowering MIN_NBA_TRADES (currently {MIN_NBA_TRADES}) "
            "or extending LOOKBACK_DAYS."
        )

    if not wallet_stats:
        log.error("No wallets qualify for classification.")
        return

    # Percentile cutoffs
    pnl_series = pd.Series(
        {w: s["pnl_per_dollar"] for w, s in wallet_stats.items()}
    )
    informed_threshold = pnl_series.quantile(1 - INFORMED_TOP_PCT)
    retail_threshold = pnl_series.quantile(RETAIL_BOTTOM_PCT)

    wallet_class: dict[str, str] = {}
    for wallet, stats in wallet_stats.items():
        p = stats["pnl_per_dollar"]
        if p >= informed_threshold:
            wallet_class[wallet] = "informed"
        elif p <= retail_threshold:
            wallet_class[wallet] = "retail"
        else:
            wallet_class[wallet] = "middle"

    n_informed = sum(1 for v in wallet_class.values() if v == "informed")
    n_retail = sum(1 for v in wallet_class.values() if v == "retail")
    log.info(
        f"Classification: {n_informed} informed, {n_retail} retail, "
        f"{len(wallet_class) - n_informed - n_retail} middle"
    )

    # Diagnostic: PnL distribution
    log.info(
        f"PnL/$ distribution: "
        f"min={pnl_series.min():.3f} "
        f"p10={pnl_series.quantile(0.10):.3f} "
        f"median={pnl_series.median():.3f} "
        f"p90={pnl_series.quantile(0.90):.3f} "
        f"max={pnl_series.max():.3f}"
    )

    if n_informed == 0 or n_retail == 0:
        log.error("Zero informed or retail wallets. Cannot compute signal.")
        return

    # ------------------------------------------------------------------
    # Step 4: Compute informed-flow signal on test markets
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4  Signal on test markets")
    log.info("=" * 60)

    informed_hits = 0
    informed_total = 0
    retail_hits = 0
    retail_total = 0
    market_detail = []

    for market in tqdm(test_markets, desc="Test market signal"):
        cid = market.get("conditionId")
        winner = market.get("_winner")
        if not cid or not winner:
            continue

        trades = fetch_market_trades(cid)

        informed_flow: dict[str, float] = {}
        retail_flow: dict[str, float] = {}

        for trade in trades:
            if trade.get("side") != "BUY":
                continue
            pw = trade.get("proxyWallet")
            outcome = trade.get("outcome", "")
            if not outcome:
                continue
            try:
                size = float(trade["size"])
            except (KeyError, ValueError, TypeError):
                continue

            cls = wallet_class.get(pw)
            if cls == "informed":
                informed_flow[outcome] = informed_flow.get(outcome, 0.0) + size
            elif cls == "retail":
                retail_flow[outcome] = retail_flow.get(outcome, 0.0) + size

        row = {
            "conditionId": cid,
            "question": (market.get("question") or "")[:80],
            "winner": winner,
            "informed_signal": None,
            "retail_signal": None,
            "informed_hit": None,
            "retail_hit": None,
            "informed_volume": sum(informed_flow.values()),
            "retail_volume": sum(retail_flow.values()),
        }

        if informed_flow:
            sig = max(informed_flow, key=informed_flow.get)
            hit = sig == winner
            row["informed_signal"] = sig
            row["informed_hit"] = hit
            informed_total += 1
            if hit:
                informed_hits += 1

        if retail_flow:
            sig = max(retail_flow, key=retail_flow.get)
            hit = sig == winner
            row["retail_signal"] = sig
            row["retail_hit"] = hit
            retail_total += 1
            if hit:
                retail_hits += 1

        market_detail.append(row)

    # ------------------------------------------------------------------
    # Step 5: Results
    # ------------------------------------------------------------------
    informed_hr = informed_hits / informed_total if informed_total > 0 else 0.0
    retail_hr = retail_hits / retail_total if retail_total > 0 else 0.0
    edge_pp = (informed_hr - retail_hr) * 100

    goes = informed_hr >= GO_HIT_RATE and edge_pp >= GO_EDGE_PP

    print()
    print("=" * 60)
    print("  PHASE 1 RESULTS: NBA Informed-Flow Analysis")
    print("=" * 60)
    print(f"  Lookback window   : {LOOKBACK_DAYS} days")
    print(f"  Total NBA markets : {len(raw_markets)} raw, {len(resolved_markets)} resolved")
    print(f"  Training markets  : {len(training_markets)}")
    print(f"  Test markets      : {len(test_markets)}")
    print(f"  Qualifying wallets: {len(wallet_stats)}")
    print(f"    Informed        : {n_informed} (top {INFORMED_TOP_PCT*100:.0f}%)")
    print(f"    Retail          : {n_retail} (bottom {RETAIL_BOTTOM_PCT*100:.0f}%)")
    print(f"  Test mkts w/signal: informed={informed_total}, retail={retail_total}")
    print()
    print(f"  Informed hit rate : {informed_hr:.1%}  ({informed_hits}/{informed_total})")
    print(f"  Retail hit rate   : {retail_hr:.1%}  ({retail_hits}/{retail_total})")
    print(f"  Edge (inf - ret)  : {edge_pp:+.1f}pp")
    print()

    if informed_total < 50:
        print(
            f"  WARNING: only {informed_total} test markets with informed signal. "
            "Need >= 50 for statistical significance."
        )
        print()

    if goes:
        print("  GO — informed >= 58% AND edge >= 8pp. Proceed to Phase 2.")
    else:
        reasons = []
        if informed_hr < GO_HIT_RATE:
            reasons.append(f"hit rate {informed_hr:.1%} < 58%")
        if edge_pp < GO_EDGE_PP:
            reasons.append(f"edge {edge_pp:.1f}pp < 8pp")
        print(f"  NO-GO — {'; '.join(reasons)}.")
        print("     Do not massage parameters. This is the honest result.")

    print("=" * 60)
    print()

    # Save outputs
    results = {
        "run_date": datetime.datetime.utcnow().isoformat(),
        "lookback_days": LOOKBACK_DAYS,
        "raw_markets": len(raw_markets),
        "resolved_markets": len(resolved_markets),
        "training_markets": len(training_markets),
        "test_markets": len(test_markets),
        "qualifying_wallets": len(wallet_stats),
        "informed_wallets": n_informed,
        "retail_wallets": n_retail,
        "test_markets_informed": informed_total,
        "test_markets_retail": retail_total,
        "informed_hit_rate": round(informed_hr, 4),
        "retail_hit_rate": round(retail_hr, 4),
        "edge_pp": round(edge_pp, 2),
        "go_no_go": "GO" if goes else "NO-GO",
        "go_threshold_hit_rate": GO_HIT_RATE,
        "go_threshold_edge_pp": GO_EDGE_PP,
    }

    out_json = OUTPUT_DIR / "phase1_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_json}")

    if market_detail:
        df = pd.DataFrame(market_detail)
        out_csv = OUTPUT_DIR / "phase1_market_detail.csv"
        df.to_csv(out_csv, index=False)
        log.info(f"Market detail saved to {out_csv}")


if __name__ == "__main__":
    main()
