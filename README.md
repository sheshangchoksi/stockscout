# Indian Stock Scout

NSE & BSE **Positional Scanner** — long-term value investing, built on
Streamlit. Every threshold that decides the score — market cap bands,
revenue/profit growth %, cash-to-revenue quality, RSI, MACD, Bollinger
Bands, volume, price range, operator/pump-detection sensitivity — is
adjustable from the sidebar. Nothing about the scoring bands is hardcoded.

## Files

| File | Purpose |
|---|---|
| `sheshscout.py` | Entry point — page config, renders the scanner |
| `scanner_common.py` | Shared infra: rate limiting, checkpoint/resume, dead-symbol skip-list, sidebar widgets, scan orchestration, CSV export |
| `tickers.py` | Loads `nse_tickers.csv` / `bse_bhavcopy.csv` (symbol **and** company name) |
| `indicators.py` | Technical-indicator math (RSI, MACD, Bollinger, volume multiple, operator/pump detection) — every period/threshold is a keyword argument |
| `mode_positional.py` | Scanner core logic + UI: fetch, scoring engine, and the full adjustable-parameters sidebar (`_params_ui`) |
| `bhavcopy.py` | NSE/BSE official EOD bhavcopy — sole source of daily OHLCV, plus a live-quote lookup for the auto-refresh price. No Yahoo/yfinance fallback. |
| `screener.py` | screener.in — sole source of fundamentals (market cap, P/E, annual + quarterly P&L, cash). No Yahoo/yfinance fallback. |
| `nse_tickers.csv` | NSE ticker universe: `NSE Ticker, Name`. Suffix `.NS` for internal symbol handling. |
| `bse_bhavcopy.csv` | BSE ticker universe — a BSE bhavcopy CSV (same format BSE publishes daily). Only `TckrSymb`, `FinInstrmId`, `FinInstrmNm` are used; the rest of the file (OHLC, volume, etc.) is ignored. Suffix `.BO`, addressed by ticker symbol (e.g. `ABB.BO`), not the numeric scrip code — refresh by dropping in any day's downloaded bhavcopy file, no code changes needed. |

To refresh either universe, replace the CSV — no code changes needed as
long as the same columns are present.

## Adjustable scoring parameters

Every band cutoff in the scoring engine lives in `mode_positional.py`'s
`_PARAM_GROUPS` and renders as a sidebar expander: qualification score
thresholds, market cap bands, revenue/profit growth tiers, profit margin,
cash & revenue quality, FII/DII activity, weekly consolidation range, RSI
(period + bounds), MACD (periods + bounds), Bollinger Bands (period,
std-dev multiplier, bounds), volume multiple (window + bounds), today's
price change, monthly trend, 3-month performance, upside-potential price
target, operator/pump-detection sensitivity, price-range filter, and the
daily-history lookback window / fundamentals cache TTL. The point value
*awarded* per band is fixed; the cutoffs that decide which band a stock
falls into are all adjustable.

## Session state / state isolation

Every session_state key and widget key is namespaced
(`scanner_common.sskey`) by mode, so nothing collides across reruns.

## Rate limiting & resiliency

- No Yahoo/yfinance calls exist anywhere in this app. `bhavcopy.py` is the
  sole source of daily OHLCV and live quotes; `screener.py` is the sole
  source of fundamentals. Neither has a fallback — a symbol either source
  can't cover is skipped for that scan rather than routed to Yahoo. There
  is no intraday-scanning mode, since 1-minute bars have no free NSE/BSE
  source and this app doesn't fall back to paid/Yahoo data to fake one.
- `bhavcopy.py`: NSE/BSE only publish one trading day's file at a time, so
  a lookback window means walking backward day by day; each day's file is
  disk-cached under `.bhavcopy_cache/` so later symbols/scans reuse it
  with zero new requests. `get_live_quote()` hits NSE/BSE's own quote
  endpoint for the auto-refresh price.
- `screener.py`: fundamentals (market cap, P/E, annual + quarterly P&L,
  cash) come from one screener.in company page per stock, tried
  consolidated-first then standalone if consolidated has no usable
  Sales/Revenue row. Results are disk-cached (default 6h TTL, adjustable)
  under `.screener_cache/` so a same-day re-scan makes zero new requests.
  See `screener.py`'s module docstring for its known limitation
  (balance-sheet cash isn't cleanly exposed on screener.in for most
  companies) and its terms-of-use note — screener.in has no free public
  API, so this reads public pages at a deliberately gentle, configurable
  rate; run `python screener.py RELIANCE TCS` once after deploying to
  sanity-check parsing against a live page before trusting it at
  full-universe scale.
- Scan concurrency (parallel workers / batch size / batch pause) is
  adjustable from the sidebar and applies to both bhavcopy and screener.in
  requests.
- Scans checkpoint to disk periodically so a killed process never loses
  more than the last few seconds of progress; a resumable scan offers
  ▶️ RESUME or 🔄 START FRESH on the next run.
- A disk-backed dead-symbol skip list avoids re-hitting symbols that have
  come back empty twice, an hour+ apart (avoids blacklisting on a one-off
  rate-limit burst).

⚠ Educational purposes only. Not financial advice.
