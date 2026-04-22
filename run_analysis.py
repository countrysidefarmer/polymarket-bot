#!/usr/bin/env python3
"""
run_analysis.py — Entry point for the multi-sport informed-flow pipeline.

Usage:
  python run_analysis.py [--config config.yaml] [--sports nba soccer]

Reads config.yaml, runs enabled sports, writes results/YYYY-MM-DD.json.
GitHub issue update is a separate step (called from CI via core/report.py).
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from core.api import PolymarketClient
from core.classify import build_wallet_pool, evaluate_test_markets
from sports.nba import NBASport
from sports.soccer import SoccerSport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPORT_REGISTRY = {
    "nba": NBASport,
    "soccer": SoccerSport,
}


def run_sport(
    sport_obj,
    all_closed_markets: list[dict],
    client: PolymarketClient,
    lookback_days: int,
    min_trades: int,
    informed_top_pct: float,
    retail_bottom_pct: float,
    go_cfg: dict,
) -> dict:
    """Run the full informed-flow analysis for one sport. Returns result dict."""
    name = sport_obj.name

    log.info("=" * 60)
    log.info(f"SPORT: {name.upper()}")
    log.info("=" * 60)

    # Step 1: Filter markets
    log.info(f"  Step 1: Filtering {name} game-winner markets (last {lookback_days} days)...")
    markets = sport_obj.filter_markets(all_closed_markets, lookback_days)
    log.info(f"  Resolved game-winner markets: {len(markets)}")

    if len(markets) < 20:
        log.warning(f"  Only {len(markets)} {name} markets — skipping (need >= 20).")
        return {
            "sport": name,
            "error": f"insufficient markets: {len(markets)} < 20",
            "go_no_go": "NO-GO",
        }

    # Step 2: Walk-forward split
    split_idx = len(markets) // 2
    training_markets = markets[:split_idx]
    test_markets = markets[split_idx:]
    log.info(f"  Training: {len(training_markets)} | Test: {len(test_markets)}")

    train_winners = {
        m["conditionId"]: m["_winner"]
        for m in training_markets
        if m.get("conditionId")
    }

    # Step 3: Fetch training trades and build wallet pool
    log.info(f"  Step 2/3: Fetching training trades and classifying wallets...")
    training_trades_by_market: dict[str, list] = {}
    for market in tqdm(training_markets, desc=f"  [{name}] Training trades"):
        cid = market.get("conditionId")
        if not cid:
            continue
        trades = client.fetch_market_trades(cid)
        training_trades_by_market[cid] = trades

    wallet_stats, wallet_class = build_wallet_pool(
        training_trades_by_market=training_trades_by_market,
        train_winners=train_winners,
        min_trades=min_trades,
        informed_top_pct=informed_top_pct,
        retail_bottom_pct=retail_bottom_pct,
    )

    n_informed = sum(1 for v in wallet_class.values() if v == "informed")
    n_retail = sum(1 for v in wallet_class.values() if v == "retail")
    log.info(f"  Qualifying wallets: {len(wallet_stats)} total, {n_informed} informed, {n_retail} retail")

    if n_informed == 0 or n_retail == 0:
        log.warning(f"  Zero informed or retail wallets for {name}. Cannot compute signal.")
        return {
            "sport": name,
            "error": "zero informed or retail wallets",
            "go_no_go": "NO-GO",
        }

    # Step 4: Evaluate signal on test markets
    log.info(f"  Step 4: Evaluating signal on {len(test_markets)} test markets...")
    signal = evaluate_test_markets(
        test_markets=test_markets,
        wallet_class=wallet_class,
        fetch_trades_fn=lambda cid: client.fetch_market_trades(cid),
    )

    # Step 5: Go/No-Go decision
    informed_ppd = signal["informed_test_pnl_per_dollar"]
    pnl_edge = signal["pnl_edge_per_dollar"]
    go_pnl = go_cfg.get("min_informed_pnl_per_dollar", 0.0)
    go_edge = go_cfg.get("min_pnl_edge_per_dollar", 0.03)

    import math
    goes = (
        informed_ppd is not None
        and not math.isnan(informed_ppd)
        and informed_ppd >= go_pnl
        and pnl_edge is not None
        and not math.isnan(pnl_edge)
        and pnl_edge >= go_edge
    )

    def _safe(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return round(v, 4)

    result = {
        "sport": name,
        "lookback_days": lookback_days,
        "resolved_markets": len(markets),
        "training_markets": len(training_markets),
        "test_markets": len(test_markets),
        "qualifying_wallets": len(wallet_stats),
        "informed_wallets": n_informed,
        "retail_wallets": n_retail,
        "test_markets_informed": signal["test_markets_informed"],
        "test_markets_retail": signal["test_markets_retail"],
        "informed_test_pnl_per_dollar": _safe(signal["informed_test_pnl_per_dollar"]),
        "retail_test_pnl_per_dollar": _safe(signal["retail_test_pnl_per_dollar"]),
        "pnl_edge_per_dollar": _safe(signal["pnl_edge_per_dollar"]),
        "informed_hit_rate": _safe(signal["informed_hit_rate"]),
        "retail_hit_rate": _safe(signal["retail_hit_rate"]),
        "direction_edge_pp": _safe(signal["direction_edge_pp"]),
        "go_no_go": "GO" if goes else "NO-GO",
        "go_threshold_pnl_per_dollar": go_pnl,
        "go_threshold_pnl_edge": go_edge,
    }

    # Print summary
    print()
    print(f"  {'='*50}")
    print(f"  {name.upper()} RESULTS")
    print(f"  {'='*50}")
    print(f"  Resolved markets   : {len(markets)}")
    print(f"  Training / Test    : {len(training_markets)} / {len(test_markets)}")
    print(f"  Informed wallets   : {n_informed}")
    print(f"  Retail wallets     : {n_retail}")
    print(f"  Informed PnL/$     : {_safe(signal['informed_test_pnl_per_dollar'])}")
    print(f"  Retail PnL/$       : {_safe(signal['retail_test_pnl_per_dollar'])}")
    print(f"  PnL edge           : {_safe(signal['pnl_edge_per_dollar'])}")
    print(f"  GO/NO-GO           : {'GO ✓' if goes else 'NO-GO'}")
    print(f"  {'='*50}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(description="Polymarket informed-flow analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--sports", nargs="+", help="Override enabled sports (e.g. nba soccer)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    lookback_days = config.get("lookback_days", 365)
    clf = config.get("classification", {})
    min_trades = clf.get("min_trades", 20)
    informed_top_pct = clf.get("informed_top_pct", 0.10)
    retail_bottom_pct = clf.get("retail_bottom_pct", 0.50)
    go_cfg = config.get("go_no_go", {})

    # Determine which sports to run
    sports_config = config.get("sports", {})
    if args.sports:
        enabled_sports = args.sports
    else:
        enabled_sports = [s for s, cfg in sports_config.items() if cfg.get("enabled", True)]

    log.info(f"Sports to run: {enabled_sports}")

    # Initialise API client (shared cache)
    client = PolymarketClient(cache_dir=Path("./cache"))

    # Fetch all closed markets once (shared across sports, cached)
    log.info("=" * 60)
    log.info("Fetching all closed markets (shared across sports)...")
    log.info("=" * 60)
    all_closed_markets = client.fetch_all_closed_markets(log_prefix="nba_mkt")
    # Note: log_prefix matches the existing phase1 cache keys so existing cache is reused

    # Run each sport
    results_by_sport = []
    for sport_name in enabled_sports:
        if sport_name not in SPORT_REGISTRY:
            log.warning(f"Unknown sport: {sport_name}. Available: {list(SPORT_REGISTRY)}")
            continue
        sport_obj = SPORT_REGISTRY[sport_name]()
        result = run_sport(
            sport_obj=sport_obj,
            all_closed_markets=all_closed_markets,
            client=client,
            lookback_days=lookback_days,
            min_trades=min_trades,
            informed_top_pct=informed_top_pct,
            retail_bottom_pct=retail_bottom_pct,
            go_cfg=go_cfg,
        )
        results_by_sport.append(result)

    # Save results
    run_date = datetime.datetime.utcnow().isoformat()
    output = {
        "run_date": run_date,
        "results_by_sport": results_by_sport,
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{run_date[:10]}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in results_by_sport:
        sport = r.get("sport", "?").upper()
        go = r.get("go_no_go", "?")
        inf_ppd = r.get("informed_test_pnl_per_dollar")
        edge = r.get("pnl_edge_per_dollar")
        print(f"  {sport:10s} : {go:6s}  | informed PnL/$ = {inf_ppd}  | edge = {edge}")
    print("=" * 60)


if __name__ == "__main__":
    main()
