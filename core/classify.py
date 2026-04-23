"""
core/classify.py — Sport-agnostic wallet classification and PnL tracking.

Classifies wallets into informed (top 10% PnL/$) and retail (bottom 50% PnL/$)
using BUY trades from training markets, then evaluates signal on test markets.

PnL formula (fee-adjusted):
  invested = price * size
  PnL = size * (0.98 - price)  if outcome == winner   (2% Polymarket taker fee on payout)
        -invested                otherwise
  PnL per dollar = total_pnl / total_invested

All thresholds in config.yaml are therefore after-fee breakeven values.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_wallet_pool(
    training_trades_by_market: dict[str, list[dict]],
    train_winners: dict[str, str],
    min_trades: int = 20,
    informed_top_pct: float = 0.10,
    retail_bottom_pct: float = 0.50,
) -> tuple[dict[str, dict], dict[str, str]]:
    """
    Classify wallets using in-memory BUY trades from training markets.

    Parameters
    ----------
    training_trades_by_market : {conditionId: [trade_dict, ...]}
    train_winners              : {conditionId: winner_outcome_label}
    min_trades                 : minimum qualifying BUY trades per wallet
    informed_top_pct           : top fraction by PnL/$ → informed
    retail_bottom_pct          : bottom fraction by PnL/$ → retail

    Returns
    -------
    wallet_stats  : {wallet_addr: {total_pnl, total_invested, pnl_per_dollar, trade_count}}
    wallet_class  : {wallet_addr: "informed" | "retail" | "middle"}
    """
    train_cids = set(train_winners.keys())

    # Accumulate per-wallet PnL across all training markets
    wallet_pnl: dict[str, float] = {}
    wallet_invested: dict[str, float] = {}
    wallet_count: dict[str, int] = {}

    for cid, trades in training_trades_by_market.items():
        if cid not in train_cids:
            continue
        winner = train_winners[cid]

        for trade in trades:
            if trade.get("side") != "BUY":
                continue
            pw = trade.get("proxyWallet")
            if not pw:
                continue

            try:
                price = float(trade["price"])
                size = float(trade["size"])
            except (KeyError, ValueError, TypeError):
                continue

            if price <= 0 or price >= 1:
                continue

            invested = price * size
            outcome = trade.get("outcome", "")
            # 2% Polymarket taker fee deducted from resolution payout
            pnl = size * (0.98 - price) if outcome == winner else -invested

            wallet_pnl[pw] = wallet_pnl.get(pw, 0.0) + pnl
            wallet_invested[pw] = wallet_invested.get(pw, 0.0) + invested
            wallet_count[pw] = wallet_count.get(pw, 0) + 1

    # Filter to wallets with enough qualifying trades
    wallet_stats: dict[str, dict] = {}
    for pw in wallet_count:
        if wallet_count[pw] < min_trades:
            continue
        inv = wallet_invested.get(pw, 0.0)
        if inv <= 0:
            continue
        pnl = wallet_pnl.get(pw, 0.0)
        wallet_stats[pw] = {
            "total_pnl": pnl,
            "total_invested": inv,
            "pnl_per_dollar": pnl / inv,
            "trade_count": wallet_count[pw],
        }

    if not wallet_stats:
        return {}, {}

    # Percentile cutoffs
    pnl_series = pd.Series({w: s["pnl_per_dollar"] for w, s in wallet_stats.items()})
    informed_threshold = pnl_series.quantile(1 - informed_top_pct)
    retail_threshold = pnl_series.quantile(retail_bottom_pct)

    wallet_class: dict[str, str] = {}
    for pw, stats in wallet_stats.items():
        p = stats["pnl_per_dollar"]
        if p >= informed_threshold:
            wallet_class[pw] = "informed"
        elif p <= retail_threshold:
            wallet_class[pw] = "retail"
        else:
            wallet_class[pw] = "middle"

    n_informed = sum(1 for v in wallet_class.values() if v == "informed")
    n_retail = sum(1 for v in wallet_class.values() if v == "retail")
    log.info(
        f"  Wallets classified: {n_informed} informed, {n_retail} retail, "
        f"{len(wallet_class) - n_informed - n_retail} middle"
    )
    log.info(
        f"  PnL/$ distribution: "
        f"min={pnl_series.min():.3f}  "
        f"p10={pnl_series.quantile(0.10):.3f}  "
        f"median={pnl_series.median():.3f}  "
        f"p90={pnl_series.quantile(0.90):.3f}  "
        f"max={pnl_series.max():.3f}"
    )

    return wallet_stats, wallet_class


def evaluate_test_markets(
    test_markets: list[dict],
    wallet_class: dict[str, str],
    fetch_trades_fn,
) -> dict:
    """
    Evaluate informed/retail flow signal on out-of-sample test markets.

    Parameters
    ----------
    test_markets    : list of market dicts, each with conditionId and _winner set
    wallet_class    : {wallet_addr: "informed"|"retail"|"middle"}
    fetch_trades_fn : callable(conditionId) -> list[trade_dict]

    Returns
    -------
    dict with keys:
      informed_test_pnl_per_dollar, retail_test_pnl_per_dollar, pnl_edge_per_dollar,
      informed_hit_rate, retail_hit_rate, direction_edge_pp,
      test_markets_informed, test_markets_retail, market_detail
    """
    informed_test_pnl = 0.0
    informed_test_invested = 0.0
    retail_test_pnl = 0.0
    retail_test_invested = 0.0

    informed_hits = 0
    informed_total = 0
    retail_hits = 0
    retail_total = 0

    market_detail = []

    for market in test_markets:
        cid = market.get("conditionId")
        winner = market.get("_winner")
        if not cid or not winner:
            continue

        trades = fetch_trades_fn(cid)

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
                price = float(trade["price"])
                size = float(trade["size"])
            except (KeyError, ValueError, TypeError):
                continue

            if price <= 0 or price >= 1:
                continue

            invested = price * size
            # 2% Polymarket taker fee deducted from resolution payout
            pnl = size * (0.98 - price) if outcome == winner else -invested

            cls = wallet_class.get(pw)
            if cls == "informed":
                informed_flow[outcome] = informed_flow.get(outcome, 0.0) + size
                informed_test_pnl += pnl
                informed_test_invested += invested
            elif cls == "retail":
                retail_flow[outcome] = retail_flow.get(outcome, 0.0) + size
                retail_test_pnl += pnl
                retail_test_invested += invested

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

    # --- Compute metrics ---
    informed_ppd = (
        informed_test_pnl / informed_test_invested if informed_test_invested > 0 else float("nan")
    )
    retail_ppd = (
        retail_test_pnl / retail_test_invested if retail_test_invested > 0 else float("nan")
    )
    pnl_edge = (
        informed_ppd - retail_ppd
        if not (np.isnan(informed_ppd) or np.isnan(retail_ppd))
        else float("nan")
    )

    informed_hr = informed_hits / informed_total if informed_total > 0 else float("nan")
    retail_hr = retail_hits / retail_total if retail_total > 0 else float("nan")
    dir_edge_pp = (
        (informed_hr - retail_hr) * 100
        if not (np.isnan(informed_hr) or np.isnan(retail_hr))
        else float("nan")
    )

    return {
        "informed_test_pnl_per_dollar": informed_ppd,
        "retail_test_pnl_per_dollar": retail_ppd,
        "pnl_edge_per_dollar": pnl_edge,
        "informed_hit_rate": informed_hr,
        "retail_hit_rate": retail_hr,
        "direction_edge_pp": dir_edge_pp,
        "test_markets_informed": informed_total,
        "test_markets_retail": retail_total,
        "market_detail": market_detail,
    }
