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
| `bhavcopy.py` | NSE/BSE official EOD bhavcopy as a free, non-Yahoo substitute for daily OHLCV, plus a free live-quote lookup for the auto-refresh price |
| `screener.py` | screener.in as a free, non-Yahoo substitute for fundamentals (market cap, P/E, annual + quarterly P&L, cash) |
| `yf_ratelimit.py` | Chrome-impersonation session + shared cooldown + retry ladder around every remaining Yahoo call |
| `nse_tickers.csv` | NSE ticker universe: `NSE Ticker, Name`. Suffix `.NS` for Yahoo. |
| `bse_bhavcopy.csv` | BSE ticker universe — a BSE bhavcopy CSV (same format BSE publishes daily). Only `TckrSymb`, `FinInstrmId`, `FinInstrmNm` are used; the rest of the file (OHLC, volume, etc.) is ignored. Suffix `.BO` for Yahoo, addressed by ticker symbol (e.g. `ABB.BO`), not the numeric scrip code — refresh by dropping in any day's downloaded bhavcopy file, no code changes needed. |

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

- `yf_ratelimit.py`: Chrome-impersonation (curl_cffi) session per worker
  thread, a shared "wait at least N seconds between requests" gate, a
  shared cooldown triggered by any 429 or silent empty-response block, and
  exponential backoff with jitter on retry. Worker count, batch
  size/pause, retry count/backoff, min delay, and cooldown length are all
  sidebar-adjustable (`scanner_common.render_rate_limit_controls`) and
  take effect immediately via `yf_ratelimit.configure()`.
- `bhavcopy.py`: NSE/BSE's own EOD files are used for daily OHLCV first —
  free, no Yahoo rate limit attached — falling back to yfinance only when
  bhavcopy has no usable data for a symbol. Its `get_live_quote()` also
  replaces the 1-minute Yahoo call the auto-refresh price used to make,
  again falling back to yfinance if NSE/BSE's quote endpoint is unreachable.
- `screener.py`: fundamentals (market cap, P/E, annual + quarterly P&L,
  cash) come from a screener.in company page first — one request instead
  of four separate Yahoo calls (`fast_info`, `income_stmt`, `balance_sheet`,
  `quarterly_income_stmt`), and this is the traffic that made a full
  2,995-stock scan so slow even with bhavcopy already in place. Falls back
  to yfinance automatically for any stock screener.in has no page for, or
  if disabled from the sidebar's "🌐 Fundamentals Source" panel. Results
  are disk-cached (default 6h TTL, adjustable) under `.screener_cache/` so
  a same-day re-scan makes zero new requests. When both bhavcopy and
  screener succeed for a stock, the scan makes **no Yahoo calls at all**
  for it — Yahoo is only still touched as a fallback, or for the intraday
  modes' 1-minute bars, which have no free NSE/BSE equivalent (see
  `intraday_data.py`'s docstring). See `screener.py`'s module docstring for
  its known limitation (balance-sheet cash isn't cleanly exposed on
  screener.in for most companies) and its terms-of-use note — screener.in
  has no free public API, so this reads public pages at a deliberately
  gentle, configurable rate; run `python screener.py RELIANCE TCS` once
  after deploying to sanity-check parsing against a live page before
  trusting it at full-universe scale.
- Scans checkpoint to disk periodically so a killed process never loses
  more than the last few seconds of progress; a resumable scan offers
  ▶️ RESUME or 🔄 START FRESH on the next run.
- A disk-backed dead-symbol skip list avoids re-hitting symbols that have
  come back empty twice, an hour+ apart (avoids blacklisting on a one-off
  rate-limit burst).

⚠ Educational purposes only. Not financial advice.
