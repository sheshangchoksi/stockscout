"""
yf_ratelimit.py — Yahoo Finance rate-limit shield.
====================================================
Drop-in replacement for `yfinance.Ticker` / `yfinance.download` used by
every mode. All tuning happens through module-level config + configure(),
which scanner_common.py's sidebar wires up to live UI controls — nothing
about retry/backoff/cooldown timing is fixed at import time.

Mechanics, briefly:
  1. curl_cffi Chrome-impersonation session (one per worker thread, not
     per symbol — bounds memory no matter how large a scan is)
  2. A single global "wait at least MIN_DELAY_S between requests" gate,
     shared across every thread
  3. A shared cooldown: any thread that hits a 429 (or an empty
     DataFrame, which on free-tier shared IPs is usually a silent
     throttle) pushes a cooldown that every other thread also waits out,
     instead of N threads independently retrying into the same block
  4. Exponential backoff with jitter on retry
  5. A bounded in-process LRU cache (ticker objects + fetched properties)
     so a long scan can't grow memory without limit
  6. A hard per-request timeout, since yfinance/curl_cffi never set one
     and a single stalled socket can otherwise wedge a worker forever

If 429s persist with sensible tuning, the remaining lever is routing
through a different outbound IP: set YF_HTTP_PROXY (or HTTPS_PROXY /
HTTP_PROXY) or call configure(proxy="http://user:pass@host:port"). Unset
by default.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from collections import OrderedDict, deque
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from curl_cffi import requests as _http
    _HAS_CURL = True
except ImportError:
    import requests as _http  # type: ignore[assignment]
    _HAS_CURL = False

import yfinance as _yf

logger.warning("yf_ratelimit: curl_cffi available = %s (install curl_cffi if False — "
                "biggest single fix for Streamlit Cloud / Render free-tier 429s)", _HAS_CURL)

# ── tunable config (all overridable at runtime via configure()) ─────────────
MIN_DELAY_S = 1.5          # minimum gap between outgoing Yahoo requests
MAX_DELAY_S = 4.0          # upper bound of random jitter added to that gap
MAX_RETRIES = 3            # retry attempts per call before giving up
BASE_BACKOFF_S = 4.0       # base for exponential backoff after a 429
COOLDOWN_S = 35.0          # shared pause for ALL threads after a 429/empty-response
REQUEST_TIMEOUT_S = 15.0   # hard ceiling per HTTP request
CACHE_TTL_S = 3600         # in-process property/history cache TTL
TICKER_REGISTRY_MAX = 300  # bounded LRU: live _CachedTicker objects
MEM_CACHE_MAX = 150        # bounded LRU: cached property/history values
EMPTY_BURST_WINDOW_S = 10.0  # window used to tell "one dead symbol" from "we're throttled"
EMPTY_BURST_THRESHOLD = 4    # this many empty responses across ALL threads within the window
                             # looks like a real block; fewer than that is just ordinary
                             # delisted/renamed tickers in a large universe (see below)

_PROXY_URL: str | None = (
    os.environ.get("YF_HTTP_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
)
_session_version = 0  # bumped on proxy change so live threads rebuild their session


def configure(max_retries: int | None = None, base_backoff: float | None = None,
              min_delay: float | None = None, cooldown: float | None = None,
              proxy: str | None = None, empty_burst_threshold: int | None = None,
              empty_burst_window: float | None = None) -> None:
    """Called from the sidebar's Retry/Backoff controls (and safe to call at
    the start of every scan) to change retry behaviour without a restart."""
    global MAX_RETRIES, BASE_BACKOFF_S, MIN_DELAY_S, COOLDOWN_S, _PROXY_URL, _session_version
    global EMPTY_BURST_THRESHOLD, EMPTY_BURST_WINDOW_S
    if max_retries is not None:
        MAX_RETRIES = max(1, int(max_retries))
    if base_backoff is not None:
        BASE_BACKOFF_S = max(0.1, float(base_backoff))
    if min_delay is not None:
        MIN_DELAY_S = max(0.05, float(min_delay))
    if cooldown is not None:
        COOLDOWN_S = max(1.0, float(cooldown))
    if proxy is not None:
        _PROXY_URL = proxy.strip() or None
        _session_version += 1  # existing per-thread sessions rebuild on next use
    if empty_burst_threshold is not None:
        EMPTY_BURST_THRESHOLD = max(1, int(empty_burst_threshold))
    if empty_burst_window is not None:
        EMPTY_BURST_WINDOW_S = max(0.5, float(empty_burst_window))


_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── shared throttle / cooldown gate ─────────────────────────────────────────
_gate_lock = threading.Lock()
_last_request_ts = 0.0
_cooldown_until = 0.0  # monotonic timestamp; every thread's _throttle() waits for this

# ── live status (surfaced in the app's live-status panel, not just logs) ───
_EVENT_LOG_MAX = 200
_event_log: "deque[dict]" = deque(maxlen=_EVENT_LOG_MAX)
_event_lock = threading.Lock()
_inflight = 0
_inflight_lock = threading.Lock()


def _log_event(message: str, level: str = "info") -> None:
    with _event_lock:
        _event_log.append({"ts": time.time(), "level": level, "message": message})


def get_recent_events(n: int = 15) -> list[dict]:
    with _event_lock:
        return list(_event_log)[-n:]


# ── empty-response burst detector ───────────────────────────────────────
# A single empty DataFrame from Yahoo is ambiguous: it's what a genuinely
# delisted/renamed/wrong-suffix symbol looks like, and it's ALSO what a
# silent throttle looks like on a shared/free-tier IP. Treating every
# empty response as "we're throttled" was the actual root cause behind
# scans stalling on nothing more than one bad ticker: it burned the full
# retry ladder (with the shared COOLDOWN_S applied on every attempt)
# against a symbol that was never going to return data no matter how long
# every worker thread waited. A large NSE/BSE universe routinely has
# hundreds of these, so that mistake compounds fast.
# Only treat it as a real block once several independent empty responses
# land close together — that pattern doesn't happen from scattered dead
# tickers, only from Yahoo actually blocking this process.
_empty_event_times: "deque[float]" = deque()
_empty_event_lock = threading.Lock()


def _empty_response_looks_like_throttle() -> bool:
    now = time.time()
    with _empty_event_lock:
        _empty_event_times.append(now)
        while _empty_event_times and (now - _empty_event_times[0]) > EMPTY_BURST_WINDOW_S:
            _empty_event_times.popleft()
        return len(_empty_event_times) >= EMPTY_BURST_THRESHOLD


def get_status() -> dict:
    """Snapshot for the live-status panel: is a cooldown active (and for how
    long), and how many requests are in flight across all worker threads."""
    now = time.monotonic()
    with _gate_lock:
        cooldown_remaining = max(0.0, _cooldown_until - now)
    with _inflight_lock:
        inflight = _inflight
    return {
        "cooldown_active": cooldown_remaining > 0,
        "cooldown_remaining": cooldown_remaining,
        "inflight": inflight,
        "min_delay_s": MIN_DELAY_S,
    }


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
    """Every other thread's next _throttle() call also pauses, instead of
    each thread independently backing off and retrying into each other."""
    global _cooldown_until
    seconds = COOLDOWN_S if seconds is None else seconds
    with _gate_lock:
        target = time.monotonic() + seconds
        if target > _cooldown_until:
            _cooldown_until = target
    logger.warning("yf_ratelimit: cooling down all threads for %.0fs", seconds)
    _log_event(f"🧊 Cooling down all workers for {seconds:.0f}s", "cooldown")


# ── one HTTP session per worker thread (not per symbol) ────────────────────
_thread_local = threading.local()


def _get_thread_session():
    sess = getattr(_thread_local, "session", None)
    if sess is not None and getattr(_thread_local, "session_version", None) == _session_version:
        return sess

    if _HAS_CURL:
        sess = _http.Session(impersonate="chrome124")
    else:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        sess = _http.Session()
        retry = Retry(total=3, backoff_factor=1.5, status_forcelist=[429, 500, 502, 503, 504],
                       allowed_methods=["GET", "HEAD"], raise_on_status=False)
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        sess.mount("http://", HTTPAdapter(max_retries=retry))

    if _PROXY_URL:
        sess.proxies = {"http": _PROXY_URL, "https": _PROXY_URL}
    sess.headers.update({
        "User-Agent": _CHROME_UA,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })

    # yfinance never passes a timeout, so a stalled socket hangs its worker
    # thread forever unless every request through this session gets one.
    _orig_request = sess.request

    def _request_with_timeout(method, url, *args, **kwargs):
        kwargs.setdefault("timeout", REQUEST_TIMEOUT_S)
        return _orig_request(method, url, *args, **kwargs)

    sess.request = _request_with_timeout
    _thread_local.session = sess
    _thread_local.session_version = _session_version
    return sess


# ── bounded in-process cache (property/history values, TTL + LRU) ──────────
_mem_cache: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
_cache_lock = threading.Lock()


def _mem_get(key: str) -> Any | None:
    with _cache_lock:
        entry = _mem_cache.get(key)
        if entry and (time.time() - entry[0]) < CACHE_TTL_S:
            _mem_cache.move_to_end(key)
            return entry[1]
    return None


def _mem_set(key: str, value: Any) -> None:
    with _cache_lock:
        _mem_cache[key] = (time.time(), value)
        _mem_cache.move_to_end(key)
        while len(_mem_cache) > MEM_CACHE_MAX:
            _mem_cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _mem_cache.clear()


# ── retry wrapper: throttle + exponential backoff + shared cooldown ────────
def _with_retry(fn, *args, **kwargs):
    global _inflight
    last_exc = None
    for attempt in range(MAX_RETRIES):
        _throttle()
        if attempt:
            backoff = BASE_BACKOFF_S * (2 ** (attempt - 1)) + random.uniform(0, MAX_DELAY_S - MIN_DELAY_S)
            logger.warning("yf_ratelimit: retry %d/%d — waiting %.1fs", attempt, MAX_RETRIES, backoff)
            _log_event(f"⏳ Retry {attempt}/{MAX_RETRIES} — waiting {backoff:.1f}s", "retry")
            time.sleep(backoff)
        with _inflight_lock:
            _inflight += 1
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and result.empty:
                # Only escalate to a shared cooldown + retry if several
                # empty responses have landed close together process-wide —
                # that's the actual throttle signature. A single empty
                # response is returned immediately as-is: it's very likely
                # just this one symbol having no data, and retrying it
                # against a manufactured cooldown would only slow down
                # every other worker for a result that was never coming.
                if attempt < MAX_RETRIES - 1 and _empty_response_looks_like_throttle():
                    logger.warning("yf_ratelimit: empty-response burst detected on attempt %d — retrying", attempt + 1)
                    _log_event(f"⚠️ Empty-response burst (attempt {attempt + 1}) — likely throttle", "warning")
                    last_exc = RuntimeError("Empty DataFrame returned (possible silent 429 burst)")
                    _trigger_cooldown()
                    continue
                return result
            return result
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if not any(x in msg for x in ("429", "rate", "too many", "forbidden", "403")):
                raise  # not a rate-limit error — no point retrying
            logger.warning("yf_ratelimit: rate-limit hit on attempt %d: %s", attempt + 1, exc)
            _log_event(f"🚫 429 rate limit (attempt {attempt + 1}/{MAX_RETRIES})", "warning")
            _trigger_cooldown()
        finally:
            with _inflight_lock:
                _inflight -= 1
    raise last_exc or RuntimeError("yf_ratelimit: all retries exhausted")


# ── public API ───────────────────────────────────────────────────────────────
class _CachedTicker:
    """Lazy, cached, retrying wrapper around yf.Ticker."""

    _PROPS = ("info", "financials", "income_stmt", "balance_sheet", "cashflow",
              "quarterly_financials", "quarterly_income_stmt", "quarterly_balance_sheet",
              "quarterly_cashflow", "fast_info", "dividends", "splits", "actions",
              "recommendations", "calendar", "earnings_dates", "options")

    def __init__(self, symbol: str):
        self._symbol = symbol
        self._yf_obj = None
        self._yf_lock = threading.Lock()

    def _get_yf(self) -> _yf.Ticker:
        with self._yf_lock:
            if self._yf_obj is None:
                self._yf_obj = _yf.Ticker(self._symbol, session=_get_thread_session())
        return self._yf_obj

    def _fetch_prop(self, prop: str) -> Any:
        key = f"{self._symbol}:prop:{prop}"
        cached = _mem_get(key)
        if cached is not None:
            return cached
        result = _with_retry(lambda: getattr(self._get_yf(), prop))
        _mem_set(key, result)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._PROPS:
            return self._fetch_prop(name)
        return getattr(self._get_yf(), name)

    def history(self, period="1mo", interval="1d", **kwargs) -> pd.DataFrame:
        key = f"{self._symbol}:history:{period}:{interval}:{sorted(kwargs.items())}"
        cached = _mem_get(key)
        if cached is not None:
            return cached
        result = _with_retry(lambda: self._get_yf().history(period=period, interval=interval, **kwargs))
        _mem_set(key, result)
        return result

    def option_chain(self, date: str | None = None) -> Any:
        key = f"{self._symbol}:option_chain:{date}"
        cached = _mem_get(key)
        if cached is not None:
            return cached
        result = _with_retry(lambda: self._get_yf().option_chain(date) if date else self._get_yf().option_chain())
        _mem_set(key, result)
        return result

    def __repr__(self):
        return f"<CachedTicker '{self._symbol}'>"


# Bounded LRU: a full-universe scan never revisits the same symbol twice in
# one run, so this only needs to help short-window repeats (dashboard
# refreshes, resume, retry-failed) — not hold the whole universe at once.
_ticker_registry: "OrderedDict[str, _CachedTicker]" = OrderedDict()
_registry_lock = threading.Lock()

logger.warning(
    "yf_ratelimit CONFIG: MIN_DELAY_S=%.1f MAX_DELAY_S=%.1f COOLDOWN_S=%.1f BASE_BACKOFF_S=%.1f "
    "REQUEST_TIMEOUT_S=%.1f TICKER_REGISTRY_MAX=%d MEM_CACHE_MAX=%d PROXY=%s",
    MIN_DELAY_S, MAX_DELAY_S, COOLDOWN_S, BASE_BACKOFF_S, REQUEST_TIMEOUT_S,
    TICKER_REGISTRY_MAX, MEM_CACHE_MAX, "configured" if _PROXY_URL else "none",
)


def safe_ticker(symbol: str) -> _CachedTicker:
    """Drop-in for yf.Ticker(symbol) — cached, rate-limit-aware."""
    with _registry_lock:
        existing = _ticker_registry.get(symbol)
        if existing is not None:
            _ticker_registry.move_to_end(symbol)
            return existing
        ticker = _CachedTicker(symbol)
        _ticker_registry[symbol] = ticker
        while len(_ticker_registry) > TICKER_REGISTRY_MAX:
            _ticker_registry.popitem(last=False)
        return ticker


def safe_download(tickers, period: str = "1mo", interval: str = "1d", flatten: bool = True, **kwargs) -> pd.DataFrame:
    """Drop-in for yf.download(tickers, ...). flatten=True (default) collapses
    the MultiIndex columns yfinance>=0.2 returns even for a single ticker."""
    ticker_key = tickers if isinstance(tickers, str) else "|".join(sorted(tickers))
    key = f"download:{ticker_key}:{period}:{interval}:{sorted(kwargs.items())}"
    cached = _mem_get(key)
    if cached is not None:
        return cached

    def _do():
        return _yf.download(tickers, period=period, interval=interval,
                             session=_get_thread_session(), progress=False, **kwargs)

    df = _with_retry(_do)
    if flatten and isinstance(df.columns, pd.MultiIndex):
        if isinstance(tickers, str) or (isinstance(tickers, (list, tuple)) and len(tickers) == 1):
            df.columns = df.columns.get_level_values(0)
    _mem_set(key, df)
    return df
