# Polymarket Informed-Flow Strategy

## Thesis

Every trade on Polymarket is on-chain and tied to a wallet address. By analysing each wallet's historical PnL, we can classify wallets as:

- **Informed** — consistent positive risk-adjusted PnL (sharp money)
- **Retail** — noise traders (negative or zero-sum PnL)
- **Market maker** — two-sided, high cancel/flip rate (future work)

The trading signal is the **imbalance of informed vs. retail flow** in a given market. When informed wallets are net-buying YES while retail is net-buying NO, we want to be on the informed side.

Inspiration: options market microstructure, where order flow classification (customer / pro customer / market maker) has well-documented predictive power for short-term price direction.

Target AUM: up to $100k USD.

---

## Why NBA First

- **Daily resolved events** during regular season + playoffs → fast PnL history accumulation
- **Volume**: $500k–$2M per marquee game → $2–5k positions fillable without meaningful slippage
- **Short resolution horizon** (hours) → walk-forward classification works in weeks, not years
- **Sharp bettors** exist in this space — sports modellers bring genuine edge

Rejected: politics (long horizons, whale manipulation, too few comparable events).

Next categories after NBA: soccer, NFL, crypto.

---

## Architectural Decisions

| Decision | Choice | Reason |
|---|---|---|
| APIs | Gamma + Data + CLOB (all free, no auth) | Data API pre-computes per-position PnL fields |
| Validation | Walk-forward only (no in-sample fit) | In-sample fit is the overfit failure mode |
| PnL metric | PnL per dollar traded | Avoids rewarding lucky one-shot whales |
| Min trade count | ≥ 20 NBA trades | No classification on noise |
| Caching | `./cache/` as JSON, keyed by MD5(url+params) | Wallet pulls take 10–30 min; preserve across reruns |

---

## Phase 1 — Go/No-Go Test (current)

**Script:** `phase1_nba_informed_flow.py`

**Question:** Does the cohort of wallets with positive historical NBA PnL produce flow that predicts the winner of *new* NBA markets, out-of-sample?

**Method:**
1. Pull all resolved NBA markets from the last 180 days (Gamma API `?q=NBA&closed=true`)
2. Discover wallet pool from per-market trade feeds (Data API `/trades?conditionId=`)
3. Split markets 50/50 by date. Classify wallets on the **earlier half** (informed = top 10% PnL/$, retail = bottom 50%)
4. For each **later-half** market: check whether informed net flow pointed at the eventual winner

**Headline metric:** hit rate of informed-flow direction vs. actual winner  
**Go threshold:** informed ≥ 58% AND (informed − retail) ≥ 8pp  
**Coin flip baseline:** 50%

**PnL approximation (Phase 1 only):** BUY trades only, treat independently. Phase 2 will use exact PnL from `/positions` and `/activity`.

---

## Phase 2 — Full Classification Pipeline (planned)

- Real-time wallet classification updated rolling every N days
- Exact PnL via Polymarket `/positions` and `/activity` endpoints
- Market-maker filtering (flag wallets with high two-sided volume ratio)
- Signal: continuous imbalance score, not binary direction
- Position sizing: Kelly fraction on imbalance score

---

## Phase 3 — Live Trading (planned)

- CLOB API for order placement
- Risk limits: max position size, max market concentration, daily stop-loss
- Monitoring: PnL dashboard, wallet classification drift alerts
- Review gate: 30-day live paper trading before real capital

---

## API Reference

| API | Base URL | Auth |
|---|---|---|
| Gamma (markets/metadata) | `https://gamma-api.polymarket.com` | None |
| Data (trades/positions) | `https://data-api.polymarket.com` | None |
| CLOB (order book/trading) | `https://clob.polymarket.com` | API key (Phase 3) |

Docs: https://docs.polymarket.com/api-reference/introduction
