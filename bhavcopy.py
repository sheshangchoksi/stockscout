"""
bhavcopy.py — NSE/BSE official EOD bhavcopy as a free, non-Yahoo substitute
for the *daily* OHLCV calls in mode_positional.py and intraday_data.py.
=============================================================================

WHY THIS EXISTS
----------------
Every yfinance/Yahoo call this app makes goes through the same shared
rate-limit gate (yf_ratelimit.py) and the same Yahoo IP-level throttling.
NSE and BSE separately publish their own end-of-day data files for free,
with no rate limit that matters to a single scanner instance -- using
those for plain daily OHLCV removes that traffic from Yahoo entirely
instead of just backing it off more aggressively.

WHAT THIS DOES AND DOES NOT REPLACE
-------------------------------------
Bhavcopy is EOD data only -- one row per symbol per trading day. It can
stand in for:
  • mode_positional.py's ticker.history(period="3mo", interval="1d")
    (the technical-indicator block: RSI/MACD/Bollinger/volume-multiple)
  • intraday_data.py's ticker.history(period="5d", interval="1d")
    (the "5-day daily reference" bars)

It does NOT and CANNOT replace:
  • intraday_data.py's ticker.history(period="1d", interval="1m") --
    1-minute intraday bars have no free public source for NSE/BSE; only
    paid broker APIs (Kite, Upstox, Fyers, TrueData, ...) carry that.
    This is also the call that produced almost all of the 429s in the
    2026-09-03 log, so this module does not fix that flood -- it only
    stops adding to Yahoo's load on top of it.
  • fast_info (market cap, PE) or the financial statements in
    mode_positional.py -- bhavcopy has no such data, those stay on
    yfinance regardless.

NOT LIVE-TESTED
-----------------
Written without network access to nseindia.com / bseindia.com from the
environment this was built in, so the exact URL/header requirements
below could not be exercised end-to-end. NSE in particular has changed
its bhavcopy path and anti-bot requirements more than once historically;
BSE's has moved even more often. Every entry point here is wrapped so a
failure of any kind (wrong URL, changed format, blocked request, a
holiday with no file) returns None rather than raising -- callers fall
back to their existing yfinance path automatically, so a broken or
stale fetch here never breaks a scan, it just fails to save any Yahoo
calls that day. Run `python bhavcopy.py` once right after deploying and
check the Streamlit Cloud log for the smoke-test output before assuming
this is actually reducing Yahoo traffic.

CACHING / REQUEST SHAPE
-------------------------
NSE/BSE only publish one trading day's file at a time (no bulk-history
endpoint), so a "65 trading day" series means walking backward day by
day. Each day's file covers the WHOLE exchange and is disk-cached once
per (exchange, date) under .bhavcopy_cache/ -- the first symbol looked
up on a given process triggers the real downloads, every symbol after
that (same day, same or later scan) reads cached files with zero new
HTTP requests. A missing date (weekend/holiday/not-yet-published) is
also cached as a "no data" marker so it isn't re-requested per symbol.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from curl_cffi import requests as _requests
    _HAS_CURL = True
except ImportError:
    import requests as _requests  # type: ignore[assignment]
    _HAS_CURL = False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE_DIR, ".bhavcopy_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Enough calendar days of backward search to find ~65+ trading days even
# with weekends/holidays mixed in.
_MAX_LOOKBACK_CALENDAR_DAYS = 130
_MIN_USABLE_ROWS = 5  # fewer than this isn't worth using over yfinance
_REQUEST_TIMEOUT_S = 15.0

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_session_lock = threading.Lock()
_session = None


def _get_session():
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        sess = _requests.Session(impersonate="chrome124") if _HAS_CURL else _requests.Session()
        sess.headers.update({
            "User-Agent": _CHROME_UA,
            "Accept-Language": "en-US,en;q=0.9",
        })
        _session = sess
        return sess


def _cache_path(exchange: str, day: date) -> str:
    return os.path.join(_CACHE_DIR, f"{exchange}_{day.isoformat()}.parquet")


def _load_cached_day(exchange: str, day: date) -> "pd.DataFrame | None":
    path = _cache_path(exchange, day)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def _save_cached_day(exchange: str, day: date, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(exchange, day), index=False)
    except Exception as e:
        logger.info("bhavcopy: failed to cache %s %s (non-fatal): %s", exchange, day, e)


def _mark_no_data(exchange: str, day: date) -> None:
    try:
        open(_cache_path(exchange, day) + ".nodata", "w").close()
    except Exception:
        pass


def _has_no_data(exchange: str, day: date) -> bool:
    return os.path.exists(_cache_path(exchange, day) + ".nodata")


def _fetch_nse_day(day: date) -> "pd.DataFrame | None":
    """NSE's full bhavcopy (all series) for one trading day. Verify against
    https://www.nseindia.com/all-reports if this URL stops resolving --
    NSE has moved this before (bhavcopy.nse -> archives.nse -> nsearchives)."""
    ddmmyyyy = day.strftime("%d%m%Y")
    referer = "https://www.nseindia.com/all-reports"
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
    try:
        sess = _get_session()
        try:
            # NSE's anti-bot layer generally wants a prior hit to the site
            # root to pick up cookies before serving archive files.
            sess.get(referer, timeout=_REQUEST_TIMEOUT_S)
        except Exception:
            pass
        resp = sess.get(url, timeout=_REQUEST_TIMEOUT_S, headers={"Referer": referer})
        if resp.status_code != 200 or not resp.content:
            return None
        df = pd.read_csv(io.BytesIO(resp.content))
        df.columns = [c.strip() for c in df.columns]
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip().isin(["EQ", "BE"])]
        rename = {
            "SYMBOL": "symbol", "OPEN_PRICE": "open", "HIGH_PRICE": "high",
            "LOW_PRICE": "low", "CLOSE_PRICE": "close", "TTL_TRD_QNTY": "volume",
            "PREV_CLOSE": "prev_close",
        }
        df = df.rename(columns=rename)
        needed = ["symbol", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        keep = needed + (["prev_close"] if "prev_close" in df.columns else [])
        df = df[keep]
        df["symbol"] = df["symbol"].astype(str).str.strip()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = day.isoformat()
        return df.dropna(subset=["close"])
    except Exception as e:
        logger.info("bhavcopy: NSE fetch failed for %s: %s", day, e)
        return None


def _fetch_bse_day(day: date) -> "pd.DataFrame | None":
    """BSE's daily bhavcopy, current (2026) endpoint and format — plain CSV
    in the SEBI-mandated UDiFF schema shared with NSE, confirmed against a
    real downloaded file (2026-09-02).

    Join key is 'TckrSymb' (the BSE ticker symbol, e.g. "ABB") — this
    matches tickers.py's yf_symbol, which is now built as
    f"{TckrSymb}.BO" (see tickers.py's load_bse_universe). The numeric
    scrip code ('FinInstrmId') is also kept as a 'code' column in case a
    caller needs to match on it instead.

    'SctySrs' holds BSE's trading *group* (A/B/T/X/Z/...), not an EQ/BE
    series code — filtering it the way the NSE fetch filters SERIES would
    silently drop every row. This file (Sgmt=CM, FinInstrmTp=STK only) is
    already restricted to cash-market equity with no other instrument
    types mixed in, so no series filter is needed here at all.
    """
    yyyymmdd = day.strftime("%Y%m%d")
    referer = "https://www.bseindia.com/markets/marketinfo/bhavcopy"
    url = f"https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{yyyymmdd}_F_0000.CSV"
    try:
        sess = _get_session()
        try:
            sess.get(referer, timeout=_REQUEST_TIMEOUT_S)
        except Exception:
            pass
        resp = sess.get(url, timeout=_REQUEST_TIMEOUT_S, headers={"Referer": referer})
        if resp.status_code != 200 or not resp.content:
            return None
        df = pd.read_csv(io.BytesIO(resp.content))
        df.columns = [c.strip() for c in df.columns]
        rename = {
            "TckrSymb": "symbol", "FinInstrmId": "code", "OpnPric": "open",
            "HghPric": "high", "LwPric": "low", "ClsPric": "close",
            "TtlTradgVol": "volume", "PrvsClsgPric": "prev_close",
        }
        df = df.rename(columns=rename)
        needed = ["symbol", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        keep = needed + [c for c in ("prev_close", "code") if c in df.columns]
        df = df[keep]
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = day.isoformat()
        return df.dropna(subset=["close"])
    except Exception as e:
        logger.info("bhavcopy: BSE fetch failed for %s: %s", day, e)
        return None


_FETCHERS = {"NSE": _fetch_nse_day, "BSE": _fetch_bse_day}


def _get_day(exchange: str, day: date) -> "pd.DataFrame | None":
    if _has_no_data(exchange, day):
        return None
    cached = _load_cached_day(exchange, day)
    if cached is not None:
        return cached
    fetcher = _FETCHERS.get(exchange)
    if fetcher is None:
        return None
    df = fetcher(day)
    if df is None or df.empty:
        _mark_no_data(exchange, day)
        return None
    _save_cached_day(exchange, day, df)
    return df


def _split_symbol(yf_symbol: str) -> "tuple[str, str] | tuple[None, None]":
    """('RELIANCE.NS' -> ('RELIANCE', 'NSE'), '500325.BO' -> ('500325', 'BSE'))."""
    if yf_symbol.endswith(".NS"):
        return yf_symbol[:-3], "NSE"
    if yf_symbol.endswith(".BO"):
        return yf_symbol[:-3], "BSE"
    return None, None


def debug_fetch_raw(exchange: str, day: "date | None" = None) -> "pd.DataFrame | None":
    """Diagnostic helper, not used by the normal fetch path: the raw,
    unmatched bhavcopy for one exchange/day (defaults to today), bypassing
    per-symbol lookup. Lets a caller tell apart 'the fetch itself failed'
    from 'the fetch worked but the symbol join-key assumption is wrong' --
    see the Bhavcopy Diagnostic panel in scanner_common.py's sidebar."""
    return _get_day(exchange, day or date.today())


def get_daily_series(yf_symbol: str, trading_days: int = 65) -> "pd.DataFrame | None":
    """Up to `trading_days` most recent trading days of open/high/low/close/
    volume for one symbol, oldest first, columns matching what
    mode_positional.py / intraday_data.py already build from yfinance's
    .history(). Returns None (never raises) on any failure -- including an
    unrecognized exchange suffix -- so callers fall back to yfinance."""
    symbol, exchange = _split_symbol(yf_symbol)
    if symbol is None:
        return None
    rows = []
    day = date.today()
    calendar_days_tried = 0
    while len(rows) < trading_days and calendar_days_tried < _MAX_LOOKBACK_CALENDAR_DAYS:
        calendar_days_tried += 1
        if day.weekday() >= 5:  # Sat/Sun -- no file published, skip without a request
            day -= timedelta(days=1)
            continue
        day_df = _get_day(exchange, day)
        if day_df is not None:
            match = day_df[day_df["symbol"] == symbol]
            if not match.empty:
                rows.append(match.iloc[0])
        day -= timedelta(days=1)

    if len(rows) < _MIN_USABLE_ROWS:
        return None
    out = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)  # oldest first
    return out


def get_latest_daily(yf_symbol: str, n: int = 5) -> "pd.DataFrame | None":
    """Convenience wrapper for intraday_data.py's short daily-reference
    window (replaces ticker.history(period=f'{n}d', interval='1d'))."""
    return get_daily_series(yf_symbol, trading_days=n)


# ── live last-traded-price, non-Yahoo ───────────────────────────────────────
# mode_positional.py's auto-refresh only needs a single current price per
# stock (not a full 1-minute bar series), so it doesn't need Yahoo at all —
# NSE and BSE each publish an unofficial JSON "quote" endpoint carrying the
# live last-traded price for free. Like the bhavcopy fetchers above, this is
# NOT LIVE-TESTED from this environment; every failure mode returns None so
# callers (fetch_live_price in mode_positional.py) fall back to yfinance's
# 1-minute-bar call automatically.
def _get_nse_live_price(symbol: str) -> "float | None":
    referer = f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}"
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    try:
        sess = _get_session()
        try:
            sess.get(referer, timeout=_REQUEST_TIMEOUT_S)
        except Exception:
            pass
        resp = sess.get(url, timeout=_REQUEST_TIMEOUT_S,
                         headers={"Referer": referer, "Accept": "application/json"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = data.get("priceInfo", {}).get("lastPrice")
        return float(price) if price is not None else None
    except Exception as e:
        logger.info("bhavcopy: NSE live quote failed for %s: %s", symbol, e)
        return None


def _get_bse_live_price(code: str) -> "float | None":
    url = (f"https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w"
           f"?Debtflag=&scripcode={code}&seriesid=")
    try:
        sess = _get_session()
        resp = sess.get(url, timeout=_REQUEST_TIMEOUT_S,
                         headers={"Referer": "https://www.bseindia.com/", "Accept": "application/json"})
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = data.get("LTP") or data.get("CurrRate")
        return float(price) if price is not None else None
    except Exception as e:
        logger.info("bhavcopy: BSE live quote failed for %s: %s", code, e)
        return None


def get_live_quote(yf_symbol: str, bse_code: "str | None" = None) -> "float | None":
    """Free, non-Yahoo substitute for mode_positional.fetch_live_price()'s
    1-minute-bar call. Returns None on any failure — caller falls back to
    yfinance."""
    symbol, exchange = _split_symbol(yf_symbol)
    if symbol is None:
        return None
    if exchange == "NSE":
        return _get_nse_live_price(symbol)
    if exchange == "BSE":
        return _get_bse_live_price(bse_code or symbol)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Smoke test -- run once after deploying to confirm NSE/BSE still")
    print("serve bhavcopy at these URLs before relying on it in a scan.\n")
    for sym in ["RELIANCE.NS", "500325.BO"]:
        s = get_daily_series(sym, trading_days=10)
        if s is None:
            print(f"{sym}: no bhavcopy data (fetch failed, or too little "
                  f"history found) -- scans will keep using yfinance for this exchange")
        else:
            print(f"{sym}: got {len(s)} days, latest close={s.iloc[-1]['close']}")
