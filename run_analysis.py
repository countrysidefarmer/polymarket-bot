#!/usr/bin/env python3
"""
run_analysis.py — Entry point for the multi-sport informed-flow pipeline.

Usage:
  python run_analysis.py [--config config.yaml] [--sports nba soccer]

Reads config.yaml, runs enabled sports with rolling walk-forward validation,
writes results/YYYY-MM-DD.json.
GitHub issue update is a separate step (called from CI via core/report.py).
"""

import argparse
import datetime
import json
import logging
import math
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from core.api import PolymarketClient
from core.classify import build_wallet_pool, evaluate_test_markets
from sports.nba import NBASport
from sports.soccer import SoccerSport
from sports.tennis import TennisSport
from sports.ufc import UFCSport
from sports.esports import EsportsSport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SPORT_REGISTRY = {
    "nba": NBASport,
    "soccer": SoccerSport,
    "tennis": TennisSport,
    "ufc": UFCSport,
    "esports": EsportsSport,
}


def _parse_market_date(market: dict) -> "datetime.datetime | None":
    """Parse the end/close date from a market dict. Returns naive UTC datetime or None."""
    for key in ("endDateIso", "endDate", "closedTime", "updatedAt"):
        raw = market.get(key)
        if raw:
            s = str(raw)[:19].rstrip("Z")
            try:
                return datetime.datetime.fromisoformat(s)
            except Exception:
                continue
    return None


def create_walk_forward_folds(
    markets: list[dict],
    train_min_months: int = 6,
    test_window_months: int = 1,
    min_train_markets: int = 20,
    min_test_markets: int = 10,
) -> list[tuple[list[dict], list[dict], dict]]:
    """
    Create expanding-window walk-forward folds.

    - Markets are assumed already sorted oldest-first.
    - Fold 1: train on first train_min_months, test on next test_window_months.
    - Each subsequent fold expands the training window by test_window_months.

    Returns list of (train_markets, test_markets, fold_meta) tuples.
    fold_meta: {"fold": n, "train_end": datetime, "test_start": datetime, "test_end": datetime}
    """
    if not markets:
        return []

    # Pair each market with its parsed date (None entries skipped)
    dated = [(m, _parse_market_date(m)) for m in markets]
    dated_valid = [(m, d) for m, d in dated if d is not None]
    if not dated_valid:
        return []

    min_date = dated_valid[0][1]
    max_date = dated_valid[-1][1]

    # Approximate months using 30-day periods
    day_step = datetime.timedelta(days=30)

    train_end = min_date + day_step * train_min_months
    if train_end >= max_date:
        return []

    folds = []
    fold_n = 0
    current = train_end

    while current < max_date:
        test_end = current + day_step * test_window_months

        train_mkts = [m for m, d in dated_valid if d < current]
        test_mkts = [m for m, d in dated_valid if current <= d < test_end]

        if len(train_mkts) >= min_train_markets and len(test_mkts) >= min_test_markets:
            fold_n += 1
            folds.append((
                train_mkts,
                test_mkts,
                {
                    "fold": fold_n,
                    "train_end": current.date().isoformat(),
                    "test_start": current.date().isoformat(),
                    "test_end": test_end.date().isoformat(),
                },
            ))

        current = test_end

    return folds


def _safe(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, 4)


def run_sport(
    sport_obj,
    all_closed_markets: list[dict],
    client: PolymarketClient,
    lookback_days: int,
    min_trades: int,
    informed_top_pct: float,
    retail_bottom_pct: float,
    go_cfg: dict,
    wf_cfg: dict,
) -> dict:
    """Run the full informed-flow analysis for one sport using rolling walk-forward."""
    name = sport_obj.name

    log.info("=" * 60)
    log.info(f"SPORT: {name.upper()}")
    log.info("=" * 60)

    # Step 1: Filter markets
    log.info(f"  Filtering {name} game-winner markets (last {lookback_days} days)...")
    markets = sport_obj.filter_markets(all_closed_markets, lookback_days)
    log.info(f"  Resolved game-winner markets: {len(markets)}")

    if len(markets) < 20:
        log.warning(f"  Only {len(markets)} {name} markets — skipping (need >= 20).")
        return {"sport": name, "error": f"insufficient markets: {len(markets)} < 20", "go_no_go": "NO-GO"}

    # Step 2: Pre-fetch ALL trades upfront (cached; shared across folds)
    log.info(f"  Pre-fetching trades for all {len(markets)} markets (cached)...")
    all_trades: dict[str, list] = {}
    for market in tqdm(markets, desc=f"  [{name}] Trade fetch"):
        cid = market.get("conditionId")
        if cid:
            all_trades[cid] = client.fetch_market_trades(cid)

    fetch_fn = lambda cid: all_trades.get(cid, [])

    # Step 3: Create walk-forward folds
    train_min_months = wf_cfg.get("train_min_months", 6)
    test_window_months = wf_cfg.get("test_window_months", 1)
    min_test_per_fold = wf_cfg.get("min_test_markets_per_fold", 10)

    folds = create_walk_forward_folds(
        markets=markets,
        train_min_months=train_min_months,
        test_window_months=test_window_months,
        min_train_markets=20,
        min_test_markets=min_test_per_fold,
    )

    if not folds:
        log.warning(f"  No valid walk-forward folds for {name} — falling back to single 50/50 split.")
        split = len(markets) // 2
        folds = [(
            markets[:split],
            markets[split:],
            {"fold": 1, "train_end": "n/a", "test_start": "n/a", "test_end": "n/a"},
        )]

    log.info(f"  Walk-forward: {len(folds)} folds (train_min={train_min_months}mo, test_window={test_window_months}mo)")

    # Step 4: Run each fold
    go_pnl = go_cfg.get("min_informed_pnl_per_dollar", 0.0)
    go_edge = go_cfg.get("min_pnl_edge_per_dollar", 0.03)

    fold_results = []
    for train_mkts, test_mkts, meta in folds:
        train_winners = {m["conditionId"]: m["_winner"] for m in train_mkts if m.get("conditionId")}
        training_trades_by_market = {cid: all_trades.get(cid, []) for cid in train_winners}

        wallet_stats, wallet_class = build_wallet_pool(
            training_trades_by_market=training_trades_by_market,
            train_winners=train_winners,
            min_trades=min_trades,
            informed_top_pct=informed_top_pct,
            retail_bottom_pct=retail_bottom_pct,
        )

        n_informed = sum(1 for v in wallet_class.values() if v == "informed")
        n_retail = sum(1 for v in wallet_class.values() if v == "retail")

        if n_informed == 0 or n_retail == 0:
            log.warning(f"  Fold {meta['fold']}: zero informed or retail wallets — skipping.")
            continue

        signal = evaluate_test_markets(
            test_markets=test_mkts,
            wallet_class=wallet_class,
            fetch_trades_fn=fetch_fn,
        )

        inf_ppd = signal["informed_test_pnl_per_dollar"]
        edge = signal["pnl_edge_per_dollar"]
        fold_go = (
            inf_ppd is not None and not math.isnan(inf_ppd) and inf_ppd >= go_pnl
            and edge is not None and not math.isnan(edge) and edge >= go_edge
        )

        fold_results.append({
            **meta,
            "training_markets": len(train_mkts),
            "test_markets": len(test_mkts),
            "qualifying_wallets": len(wallet_stats),
            "informed_wallets": n_informed,
            "retail_wallets": n_retail,
            "test_markets_informed": signal["test_markets_informed"],
            "test_markets_retail": signal["test_markets_retail"],
            "informed_pnl_per_dollar": _safe(inf_ppd),
            "retail_pnl_per_dollar": _safe(signal["retail_test_pnl_per_dollar"]),
            "pnl_edge_per_dollar": _safe(edge),
            "informed_hit_rate": _safe(signal["informed_hit_rate"]),
            "retail_hit_rate": _safe(signal["retail_hit_rate"]),
            "direction_edge_pp": _safe(signal["direction_edge_pp"]),
            "go_no_go": "GO" if fold_go else "NO-GO",
        })

    if not fold_results:
        return {"sport": name, "error": "all folds skipped (zero informed/retail)", "go_no_go": "NO-GO"}

    # Step 5: Aggregate across folds
    def _nanmean(vals):
        valid = [v for v in vals if v is not None and not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")

    mean_inf_ppd = _nanmean([f["informed_pnl_per_dollar"] for f in fold_results])
    mean_retail_ppd = _nanmean([f["retail_pnl_per_dollar"] for f in fold_results])
    mean_edge = _nanmean([f["pnl_edge_per_dollar"] for f in fold_results])
    mean_inf_hr = _nanmean([f["informed_hit_rate"] for f in fold_results])
    mean_retail_hr = _nanmean([f["retail_hit_rate"] for f in fold_results])
    go_rate = sum(1 for f in fold_results if f["go_no_go"] == "GO") / len(fold_results)

    overall_go = (
        not math.isnan(mean_inf_ppd) and mean_inf_ppd >= go_pnl
        and not math.isnan(mean_edge) and mean_edge >= go_edge
        and go_rate >= 0.6
    )

    result = {
        "sport": name,
        "lookback_days": lookback_days,
        "resolved_markets": len(markets),
        "validation": "walk_forward",
        "n_folds": len(fold_results),
        "mean_informed_pnl_per_dollar": _safe(mean_inf_ppd),
        "mean_retail_pnl_per_dollar": _safe(mean_retail_ppd),
        "mean_pnl_edge_per_dollar": _safe(mean_edge),
        "mean_informed_hit_rate": _safe(mean_inf_hr),
        "mean_retail_hit_rate": _safe(mean_retail_hr),
        "go_rate": round(go_rate, 3),
        "go_no_go": "GO" if overall_go else "NO-GO",
        "go_threshold_pnl_per_dollar": go_pnl,
        "go_threshold_pnl_edge": go_edge,
        "go_threshold_go_rate": 0.6,
        "folds": fold_results,
    }

    # Print summary
    print()
    print(f"  {'='*55}")
    print(f"  {name.upper()} RESULTS  ({len(fold_results)} walk-forward folds)")
    print(f"  {'='*55}")
    print(f"  Total markets      : {len(markets)}")
    print(f"  Mean informed PnL/$: {_safe(mean_inf_ppd)}")
    print(f"  Mean retail PnL/$  : {_safe(mean_retail_ppd)}")
    print(f"  Mean edge          : {_safe(mean_edge)}")
    print(f"  GO rate            : {go_rate:.0%} of folds")
    print(f"  GO/NO-GO           : {'GO ✓' if overall_go else 'NO-GO'}")
    print(f"  {'='*55}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(description="Polymarket informed-flow analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--sports", nargs="+", help="Override enabled sports (e.g. nba soccer)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    lookback_days = config.get("lookback_days", 365)
    clf = config.get("classification", {})
    min_trades = clf.get("min_trades", 20)
    informed_top_pct = clf.get("informed_top_pct", 0.10)
    retail_bottom_pct = clf.get("retail_bottom_pct", 0.50)
    go_cfg = config.get("go_no_go", {})
    wf_cfg = config.get("walk_forward", {})

    sports_config = config.get("sports", {})
    if args.sports:
        enabled_sports = args.sports
    else:
        enabled_sports = [s for s, cfg in sports_config.items() if cfg.get("enabled", True)]

    log.info(f"Sports to run: {enabled_sports}")

    client = PolymarketClient(cache_dir=Path("./cache"))

    log.info("=" * 60)
    log.info("Fetching all closed markets (shared across sports)...")
    log.info("=" * 60)
    all_closed_markets = client.fetch_all_closed_markets(log_prefix="nba_mkt")

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
            wf_cfg=wf_cfg,
        )
        results_by_sport.append(result)

    run_date = datetime.datetime.utcnow().isoformat()
    output = {"run_date": run_date, "results_by_sport": results_by_sport}

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"{run_date[:10]}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved to {out_path}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for r in results_by_sport:
        sport = r.get("sport", "?").upper()
        go = r.get("go_no_go", "?")
        inf_ppd = r.get("mean_informed_pnl_per_dollar")
        edge = r.get("mean_pnl_edge_per_dollar")
        go_rate = r.get("go_rate")
        n_folds = r.get("n_folds", "?")
        print(f"  {sport:10s} : {go:6s}  | informed PnL/$ = {inf_ppd}  | edge = {edge}  | go_rate = {go_rate} ({n_folds} folds)")
    print("=" * 60)


if __name__ == "__main__":
    main()
