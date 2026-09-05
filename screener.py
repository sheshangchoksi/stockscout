"""
screener.py — screener.in as the sole source of *fundamentals* (market
cap, P/E, income statement, balance sheet, quarterly income statement)
for mode_positional.py. No Yahoo/yfinance anywhere in this app.
================================================================================

WHY THIS EXISTS
----------------
bhavcopy.py covers the plain daily-OHLCV path (price, volume, RSI/MACD/BB).
The other thing a positional scanner needs per stock is fundamentals —
market cap, annual + quarterly P&L, cash position — and screener.in
renders all of that on one HTML page per company:
    current market cap, quarterly results (last ~12 quarters — enough for
    QoQ *and* YoY), annual profit & loss (up to ~12 years, often with a
    trailing-twelve-months column labelled "TTM"), and the annual balance
    sheet. One HTTP GET covers everything this app needs.

WHAT THIS DOES AND DOES NOT COVER
-------------------------------------
Supplies (when parsing succeeds):
  • market cap, P/E
  • annual Total Revenue, Net Income, multi-year history
  • annual cash position (best-effort — see below)
  • QoQ/YoY revenue & profit growth
Does NOT cover:
  • Daily OHLCV — bhavcopy.py owns that.
  • 1-minute intraday bars — no free source exists for those on NSE/BSE
    (only paid broker APIs), which is why this app has no intraday-
    scanning mode at all rather than a half-working one on Yahoo.

There is no yfinance fallback anywhere in this module: if this file
returns None for a symbol, mode_positional.py skips that symbol for the
scan rather than fetching its fundamentals from Yahoo.

KNOWN LIMITATION — CASH ON BALANCE SHEET
------------------------------------------
screener.in's standard balance-sheet template does not carry a standalone
"Cash & Cash Equivalents" line for most non-financial companies (it's
folded into "Other Assets"); only some companies expose it explicitly.
get_fundamentals() returns total_cash=None whenever it can't find an
unambiguous cash row rather than guessing — callers already treat a
missing/zero cash figure as "unknown, score neutrally on this factor".

NOT LIVE-TESTED
-----------------
Built without network access to screener.in from this environment, so the
exact section ids / markup below could not be exercised end-to-end against
a live page. screener.in's page structure (section ids `top-ratios`,
`quarters`, `profit-loss`, `balance-sheet`) has been stable for a long
time, but every extraction step is wrapped so a parsing failure of any
kind (id renamed, table restructured, page blocked) returns None rather
than raising — mode_positional.py's caller simply skips that stock for
the scan (see above), so a broken or stale parser here never breaks a
scan, it just means fewer stocks get scored that day. Run
`python screener.py RELIANCE TCS INFY` once after deploying and read the
printed output before assuming this is actually working — see the
`__main__` block at the bottom.

TERMS OF USE
-------------
screener.in has no free public API; this reads the same public company
pages a signed-out browser would. There is no login/session/paywall
bypass here — only public data, at a deliberately gentle rate (see
MIN_DELAY_S below). If you have a screener.in premium account, its
"export to Excel" feature is a friendlier, ToS-sanctioned alternative
worth checking against their current terms before relying on this at
scale in production.

CACHING
--------
Unlike scanner_common.py's in-memory TTL cache (wiped every process
restart), fundamentals barely change day to day, so results are cached to
disk under .screener_cache/ with a long default TTL (SCREENER_CACHE_TTL_S)
— configurable via configure(). A confirmed "page not found" is cached
separately and briefly (NOT_FOUND_TTL_S) so a genuinely uncovered small-cap
isn't re-requested on every scan, without permanently blacklisting it the
way the shared dead-symbol list would (screener coverage gaps don't mean
the symbol itself is dead on NSE/BSE).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import date

logger = logging.getLogger(__name__)

try:
    from curl_cffi import requests as _requests
    _HAS_CURL = True
except ImportError:
    import requests as _requests  # type: ignore[assignment]
    _HAS_CURL = False

try:
    from bs4 import BeautifulSoup
    _HAS_BS4 = True
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment]
    _HAS_BS4 = False

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE_DIR, ".screener_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
_NOT_FOUND_PATH = os.path.join(_CACHE_DIR, "_not_found.json")

# ── tunable config (overridable at runtime via configure()) ────────────────
MIN_DELAY_S = 2.0            # floor gap between outgoing screener.in requests
MAX_RETRIES = 2               # keep this low — a 404 is a real "no page", not a throttle
BASE_BACKOFF_S = 3.0
COOLDOWN_S = 20.0
REQUEST_TIMEOUT_S = 12.0
SCREENER_CACHE_TTL_S = 6 * 3600     # fundamentals disk-cache lifetime
NOT_FOUND_TTL_S = 24 * 3600         # how long a "no screener page" result is trusted


def configure(min_delay: float | None = None, max_retries: int | None = None,
              base_backoff: float | None = None, cooldown: float | None = None,
              cache_ttl: float | None = None) -> None:
    global MIN_DELAY_S, MAX_RETRIES, BASE_BACKOFF_S, COOLDOWN_S, SCREENER_CACHE_TTL_S
    if min_delay is not None:
        MIN_DELAY_S = max(0.2, float(min_delay))
    if max_retries is not None:
        MAX_RETRIES = max(1, int(max_retries))
    if base_backoff is not None:
        BASE_BACKOFF_S = max(0.1, float(base_backoff))
    if cooldown is not None:
        COOLDOWN_S = max(1.0, float(cooldown))
    if cache_ttl is not None:
        SCREENER_CACHE_TTL_S = max(60.0, float(cache_ttl))


_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_session_lock = threading.Lock()
_session = None

_gate_lock = threading.Lock()
_last_request_ts = 0.0
_cooldown_until = 0.0

# ── live status, mirrors yf_ratelimit's shape so the sidebar can show both ──
_stats_lock = threading.Lock()
_stats = {"requests": 0, "cache_hits": 0, "failures": 0, "not_found": 0}


def get_status() -> dict:
    with _stats_lock:
        return dict(_stats)


def _bump(key: str) -> None:
    with _stats_lock:
        _stats[key] = _stats.get(key, 0) + 1


def _get_session():
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        sess = _requests.Session(impersonate="chrome124") if _HAS_CURL else _requests.Session()
        sess.headers.update({
            "User-Agent": _CHROME_UA,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml",
        })
        _session = sess
        return sess


def _throttle() -> None:
    global _last_request_ts
    with _gate_lock:
        now = time.monotonic()
        if now < _cooldown_until:
            time.sleep(_cooldown_until - now)
            now = time.monotonic()
        wait = MIN_DELAY_S - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def _trigger_cooldown(seconds: float | None = None) -> None:
    global _cooldown_until
    with _gate_lock:
        _cooldown_until = max(_cooldown_until, time.monotonic() + (seconds or COOLDOWN_S))


# ── not-found cache (separate from scanner_common's global dead-symbol list:
# "no screener page" says nothing about whether the symbol is alive on
# NSE/BSE, so it must never feed the shared skip-list) ──────────────────────
_not_found_lock = threading.Lock()
_not_found_cache: dict | None = None


def _load_not_found() -> dict:
    try:
        with open(_NOT_FOUND_PATH, "r") as f:
            data = json.load(f)
        now = time.time()
        return {k: v for k, v in data.items() if now - v < NOT_FOUND_TTL_S}
    except Exception:
        return {}


def _is_known_not_found(key: str) -> bool:
    global _not_found_cache
    with _not_found_lock:
        if _not_found_cache is None:
            _not_found_cache = _load_not_found()
        return key in _not_found_cache


def _mark_not_found(key: str) -> None:
    global _not_found_cache
    with _not_found_lock:
        if _not_found_cache is None:
            _not_found_cache = _load_not_found()
        _not_found_cache[key] = time.time()
        try:
            with open(_NOT_FOUND_PATH, "w") as f:
                json.dump(_not_found_cache, f)
        except Exception:
            pass


# ── disk cache for parsed fundamentals ──────────────────────────────────────
def _cache_path(key: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)
    return os.path.join(_CACHE_DIR, f"{safe}.json")


def _cache_get(key: str) -> "dict | None":
    path = _cache_path(key)
    try:
        with open(path, "r") as f:
            entry = json.load(f)
        if time.time() - entry.get("ts", 0) < SCREENER_CACHE_TTL_S:
            _bump("cache_hits")
            return entry.get("data")
    except Exception:
        pass
    return None


def _cache_set(key: str, data: dict) -> None:
    try:
        with open(_cache_path(key), "w") as f:
            json.dump({"ts": time.time(), "data": data}, f)
    except Exception as e:
        logger.info("screener: failed to cache %s (non-fatal): %s", key, e)


# ── HTTP fetch ───────────────────────────────────────────────────────────────
def _fetch_url(url: str) -> "str | None":
    sess = _get_session()
    last_exc = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        if attempt:
            time.sleep(BASE_BACKOFF_S * (2 ** (attempt - 1)))
        try:
            resp = sess.get(url, timeout=REQUEST_TIMEOUT_S)
            _bump("requests")
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 404:
                return None  # real "no such page", not a throttle — don't retry/cooldown
            if resp.status_code in (429, 403):
                _trigger_cooldown()
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
                continue
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except Exception as e:
            last_exc = e
    if last_exc:
        logger.info("screener: fetch failed for %s: %s", url, last_exc)
        _bump("failures")
    return None


def _fetch_company_page(symbol: str, bse_code: "str | None") -> "tuple[str, str] | tuple[None, None]":
    """Returns (html, slug_used) trying, in order: consolidated view by
    symbol, standalone view by symbol (some companies — mostly banks/NBFCs —
    only publish standalone on screener), then the same two by BSE numeric
    code for BSE-only names that aren't indexed by ticker symbol.

    A page only "counts" if its annual Profit & Loss table has a real
    Sales/Revenue row — screener.in returns HTTP 200 with the
    /consolidated/ section present but empty for companies that file no
    consolidated financials (e.g. screener.in/company/DAVANGERE/consolidated/),
    so presence of the section alone is not enough; that case must fall
    through to the standalone URL for the same slug before moving on to
    the next slug. Mirrors the check in sheshvaluations' bulk_valuation.py
    (fetch_screener_bulk / _parse_screener_page)."""
    candidates = [symbol]
    if bse_code:
        candidates.append(bse_code)
    for slug in candidates:
        for suffix in ("consolidated/", ""):
            url = f"https://www.screener.in/company/{slug}/{suffix}"
            html = _fetch_url(url)
            if not html:
                continue
            if not ("company-ratios" in html or "top-ratios" in html or "profit-loss" in html):
                continue
            if _has_usable_revenue(html):
                return html, slug
    return None, None


def _has_usable_revenue(html: str) -> bool:
    """True if the page's annual Profit & Loss table has a Sales/Revenue
    row with at least one non-zero, non-None value. Without bs4 available
    we can't check, so don't block on it (get_fundamentals() already
    no-ops without bs4)."""
    if not _HAS_BS4:
        return True
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    parsed = _parse_table(soup, "profit-loss")
    if not parsed:
        return False
    _headers, rows = parsed
    sales = _find_row(rows, "sales", "revenue")
    return bool(sales) and any(v not in (None, 0) for v in sales)


# ── number parsing (Indian comma grouping, %, Cr., parenthesised negatives) ─
_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")


def _num(text: "str | None") -> "float | None":
    if not text:
        return None
    t = text.strip().replace("\u20b9", "").replace(",", "")
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    m = _NUM_RE.search(t.replace(",", ""))
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return -v if negative else v


# ── HTML section / table parsing ────────────────────────────────────────────
def _parse_table(soup, section_id: str) -> "tuple[list[str], dict[str, list[float | None]]] | None":
    """Finds the <table> inside the element with this id and returns
    (period_headers, {row_label: [values...]}). Both header and row-label
    text are lightly normalised (whitespace collapsed, trailing tooltip/
    "+"-toggle glyphs stripped)."""
    if not _HAS_BS4:
        return None
    container = soup.find(id=section_id)
    if container is None:
        return None
    table = container.find("table")
    if table is None:
        return None
    try:
        thead = table.find("thead")
        header_cells = thead.find_all("th") if thead else table.find("tr").find_all("th")
        headers = [" ".join(th.get_text(strip=True).split()) for th in header_cells[1:]]

        rows: dict[str, list[float | None]] = {}
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            label = " ".join(cells[0].get_text(strip=True).split())
            label = re.sub(r"[+\u2013\u2014]\s*$", "", label).strip()
            if not label:
                continue
            values = [_num(c.get_text()) for c in cells[1:]]
            rows[label] = values
        if not rows:
            return None
        return headers, rows
    except Exception:
        return None


def _find_row(rows: dict, *candidates: str) -> "list[float | None] | None":
    """Case-insensitive substring match against row labels — screener uses
    slightly different labels across company templates (e.g. 'Sales' vs
    'Revenue' for banks/NBFCs, 'Net Profit' vs 'Net profit')."""
    lowered = {k.lower(): v for k, v in rows.items()}
    for cand in candidates:
        cand_l = cand.lower()
        for label, values in lowered.items():
            if cand_l in label:
                return values
    return None


def _pct_change(new: "float | None", old: "float | None") -> "float | None":
    if new is None or old is None or old == 0:
        return None
    return ((new - old) / abs(old)) * 100


# screener.in's quarterly/annual/balance-sheet tables report every rupee
# figure in ₹ Crore, but this app's market-cap-relative scoring thresholds
# (rev_to_mcap_strong, etc.) are built around plain rupees. Growth
# percentages and margins are ratios of two same-table figures, so the
# Crore scaling cancels out and those need no conversion; only figures
# that get compared against market_cap (revenue, cash) need scaling up
# to match.
_CRORE = 1e7


# ── top ratios (market cap, P/E, ...) ───────────────────────────────────────
def _parse_top_ratios(soup) -> dict:
    out: dict = {}
    if not _HAS_BS4:
        return out
    container = soup.find(id="top-ratios")
    if container is None:
        return out
    for li in container.find_all("li"):
        spans = li.find_all("span")
        if len(spans) < 2:
            continue
        name = " ".join(spans[0].get_text(strip=True).split()).lower()
        value_text = " ".join(s.get_text(strip=True) for s in spans[1:])
        out[name] = value_text
    return out


def _market_cap_raw_rupees(ratios: dict) -> "float | None":
    for key in ("market cap", "market cap +"):
        if key in ratios:
            cr = _num(ratios[key])
            return cr * 1e7 if cr is not None else None
    return None


def _pe_ratio(ratios: dict) -> "float | None":
    for key in ("stock p/e", "p/e"):
        if key in ratios:
            return _num(ratios[key])
    return None


# ── public API ───────────────────────────────────────────────────────────────
def get_fundamentals(symbol: str, bse_code: "str | None" = None) -> "dict | None":
    """Best-effort screener.in fundamentals for one NSE/BSE symbol:
        market_cap, pe_ratio, total_cash, latest_fy_revenue, profit_margin,
        qoq_revenue_growth, yoy_revenue_growth,
        qoq_profit_growth, yoy_profit_growth, historical_data
    Returns None (never raises) if screener has no page for this symbol, or
    if parsing failed to recover even the minimum useful fields (market cap
    or latest revenue). There is no yfinance fallback — the caller
    (mode_positional.fetch_stock_data) skips the symbol for this scan when
    this returns None.
    """
    if not _HAS_BS4:
        logger.warning("screener: beautifulsoup4 not installed — fundamentals "
                        "unavailable until it's added to requirements.txt "
                        "(no yfinance fallback exists)")
        return None

    cache_key = f"fund:{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    if _is_known_not_found(symbol):
        return None

    html, _slug = _fetch_company_page(symbol, bse_code)
    if html is None:
        _bump("not_found")
        _mark_not_found(symbol)
        return None

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    ratios = _parse_top_ratios(soup)
    market_cap = _market_cap_raw_rupees(ratios)
    pe_ratio = _pe_ratio(ratios)

    # ── quarterly results: QoQ / YoY growth ──
    qoq_revenue_growth = yoy_revenue_growth = None
    qoq_profit_growth = yoy_profit_growth = None
    q_parsed = _parse_table(soup, "quarters")
    if q_parsed:
        _q_headers, q_rows = q_parsed
        q_sales = _find_row(q_rows, "sales", "revenue")
        q_profit = _find_row(q_rows, "net profit")
        if q_sales and len(q_sales) >= 2:
            qoq_revenue_growth = _pct_change(q_sales[-1], q_sales[-2])
        if q_sales and len(q_sales) >= 5:
            yoy_revenue_growth = _pct_change(q_sales[-1], q_sales[-5])
        if q_profit and len(q_profit) >= 2:
            qoq_profit_growth = _pct_change(q_profit[-1], q_profit[-2])
        if q_profit and len(q_profit) >= 5:
            yoy_profit_growth = _pct_change(q_profit[-1], q_profit[-5])

    # ── annual P&L: latest FY revenue/profit, margin, multi-year history ──
    latest_fy_revenue = 0.0
    profit_margin = None
    historical_data = {"years": [], "revenues": [], "cash_amounts": [], "sales_to_mcap": []}
    pl_parsed = _parse_table(soup, "profit-loss")
    if pl_parsed:
        pl_headers, pl_rows = pl_parsed
        pl_sales = _find_row(pl_rows, "sales", "revenue")
        pl_profit = _find_row(pl_rows, "net profit")
        # Last column is "TTM" for most (not all) companies — exclude it
        # from the "annual" series and treat it separately.
        has_ttm = bool(pl_headers) and pl_headers[-1].strip().upper() == "TTM"
        ann_headers = pl_headers[:-1] if has_ttm else pl_headers
        ann_sales = pl_sales[:-1] if (has_ttm and pl_sales) else pl_sales
        ann_profit = pl_profit[:-1] if (has_ttm and pl_profit) else pl_profit

        if ann_sales:
            latest_fy_revenue = (ann_sales[-1] or 0.0) * _CRORE
        if ann_sales and ann_profit and ann_sales[-1]:
            # ratio of two same-table (₹ Cr) figures — no scaling needed
            latest_net = ann_profit[-1]
            if latest_net is not None and ann_sales[-1]:
                profit_margin = latest_net / ann_sales[-1]

        n_years = min(3, len(ann_headers))
        for i in range(-n_years, 0):
            year_label = ann_headers[i] if ann_headers else ""
            rev = ((ann_sales[i] if ann_sales else None) or 0.0) * _CRORE
            historical_data["years"].append(year_label)
            historical_data["revenues"].append(rev)
            historical_data["cash_amounts"].append(0)  # see KNOWN LIMITATION above
        # most-recent-first, matching the chart/table rendering order in
        # mode_positional.py
        historical_data["years"].reverse()
        historical_data["revenues"].reverse()
        historical_data["cash_amounts"].reverse()
        if market_cap:
            historical_data["sales_to_mcap"] = [
                (r / market_cap) if market_cap > 0 and r > 0 else 0 for r in historical_data["revenues"]
            ]

    # ── balance sheet: cash, best-effort only (see KNOWN LIMITATION) ──
    total_cash = None
    bs_parsed = _parse_table(soup, "balance-sheet")
    if bs_parsed:
        _bs_headers, bs_rows = bs_parsed
        cash_row = _find_row(bs_rows, "cash equivalents", "cash & bank", "cash and bank")
        if cash_row and cash_row[-1] is not None:
            total_cash = cash_row[-1] * _CRORE

    if not market_cap and not latest_fy_revenue:
        return None  # nothing usable recovered — let the caller fall back

    result = {
        "market_cap": market_cap or 0,
        "pe_ratio": pe_ratio,
        "total_cash": total_cash if total_cash is not None else 0,
        "latest_fy_revenue": latest_fy_revenue,
        "profit_margin": profit_margin,
        "qoq_revenue_growth": qoq_revenue_growth,
        "yoy_revenue_growth": yoy_revenue_growth,
        "qoq_profit_growth": qoq_profit_growth,
        "yoy_profit_growth": yoy_profit_growth,
        "historical_data": historical_data,
        "source": "screener",
    }
    _cache_set(cache_key, result)
    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    symbols = sys.argv[1:] or ["RELIANCE", "TCS", "INFY"]
    print("Smoke test — run once after deploying to confirm screener.in still "
          "serves these pages/ids before relying on it in a scan.\n")
    print(f"beautifulsoup4 available: {_HAS_BS4}  |  curl_cffi available: {_HAS_CURL}\n")
    for sym in symbols:
        data = get_fundamentals(sym)
        if data is None:
            print(f"{sym}: no usable data (page missing, blocked, or parser found nothing "
                  f"— this stock would be skipped in a scan, no yfinance fallback)")
        else:
            print(f"{sym}: market_cap={data['market_cap']:.0f}  pe={data['pe_ratio']}  "
                  f"latest_fy_revenue={data['latest_fy_revenue']:.0f}  "
                  f"yoy_rev_growth={data['yoy_revenue_growth']}  "
                  f"yoy_profit_growth={data['yoy_profit_growth']}  "
                  f"years={data['historical_data']['years']}")
