"""
yfinance_compat.py  –  Compatibility shim for yfinance >= 0.2.x
================================================================
Addresses all breaking changes between yfinance 0.1.x and 0.2.x/1.x:

1.  Attribute renames
    - Ticker.financials       → Ticker.income_stmt  (annual)
    - Ticker.cashflow         → Ticker.cash_flow    (annual)
    - Ticker.quarterly_financials → Ticker.quarterly_income_stmt
    - Ticker.quarterly_cashflow   → Ticker.quarterly_cash_flow

2.  yf.download() MultiIndex columns
    - In 0.2.x, downloading a single ticker still returns MultiIndex columns
      e.g.  ('Close', 'RELIANCE.NS')  instead of  'Close'
    - Fix: flatten with  df.columns = df.columns.get_level_values(0)

3.  News item structure
    - Old: {title, link, publisher, providerPublishTime}
    - New: {content: {title, canonicalUrl: {url}, provider: {displayName}, pubDate}}

4.  Timezone-aware DatetimeIndex
    - history() now returns tz-aware index (IST / Asia/Kolkata for Indian tickers)
    - Plotly handles this fine; comparison with tz-naive datetimes needs .tz_localize(None)

Usage
-----
    from yfinance_compat import safe_ticker, safe_download, flatten_df, safe_history

HOW TO USE IN EACH MODULE
--------------------------
Replace every occurrence of:
    ticker.financials          →  safe_financials(ticker)
    ticker.cashflow            →  safe_cashflow(ticker)
    ticker.quarterly_financials → safe_quarterly_financials(ticker)
    yf.download(...)           →  safe_download(...)
"""

import yfinance as yf
import pandas as pd

# ------------------------------------------------------------------ #
#  Financial statement accessors (with fallback for old/new yfinance) #
# ------------------------------------------------------------------ #

def safe_financials(ticker_obj):
    """Annual income statement – works on both yfinance 0.1.x and 0.2.x+"""
    try:
        stmt = ticker_obj.income_stmt
        if stmt is not None and not (hasattr(stmt, 'empty') and stmt.empty):
            return stmt
    except AttributeError:
        pass
    try:
        return ticker_obj.financials
    except Exception:
        return pd.DataFrame()


def safe_cashflow(ticker_obj):
    """Annual cash flow statement – works on both yfinance 0.1.x and 0.2.x+"""
    try:
        cf = ticker_obj.cash_flow
        if cf is not None and not (hasattr(cf, 'empty') and cf.empty):
            return cf
    except AttributeError:
        pass
    try:
        return ticker_obj.cashflow
    except Exception:
        return pd.DataFrame()


def safe_quarterly_financials(ticker_obj):
    """Quarterly income statement – works on both yfinance 0.1.x and 0.2.x+"""
    try:
        stmt = ticker_obj.quarterly_income_stmt
        if stmt is not None and not (hasattr(stmt, 'empty') and stmt.empty):
            return stmt
    except AttributeError:
        pass
    try:
        return ticker_obj.quarterly_financials
    except Exception:
        return pd.DataFrame()


def safe_quarterly_cashflow(ticker_obj):
    """Quarterly cash flow – works on both yfinance 0.1.x and 0.2.x+"""
    try:
        cf = ticker_obj.quarterly_cash_flow
        if cf is not None and not (hasattr(cf, 'empty') and cf.empty):
            return cf
    except AttributeError:
        pass
    try:
        return ticker_obj.quarterly_cashflow
    except Exception:
        return pd.DataFrame()


# ------------------------------------------------------------------ #
#  yf.download wrapper – flattens MultiIndex columns automatically    #
# ------------------------------------------------------------------ #

def safe_download(ticker, **kwargs):
    """
    Wrapper around yf.download() that always returns flat (single-level) columns.

    In yfinance >= 0.2.x, download() returns MultiIndex columns like:
        ('Close', 'RELIANCE.NS'), ('Open', 'RELIANCE.NS'), ...
    This wrapper collapses them to:
        'Close', 'Open', ...

    All kwargs are forwarded to yf.download().
    """
    kwargs.setdefault('auto_adjust', True)
    kwargs.setdefault('progress', False)
    df = yf.download(ticker, **kwargs)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ------------------------------------------------------------------ #
#  history() helper – strips tz for plotting if needed               #
# ------------------------------------------------------------------ #

def safe_history(ticker_obj, strip_tz=False, **kwargs):
    """
    Wrapper around Ticker.history() with optional timezone stripping.

    strip_tz=True  →  removes timezone info from the DatetimeIndex
                      (needed when comparing with tz-naive datetimes)
    """
    df = ticker_obj.history(**kwargs)
    if strip_tz and not df.empty and hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ------------------------------------------------------------------ #
#  News normaliser                                                    #
# ------------------------------------------------------------------ #

def parse_news_item(item):
    """
    Normalise a yfinance news item dict into a consistent structure:
        {title, link, publisher, date}

    Handles both old (0.1.x) and new (0.2.x) response shapes.
    """
    from datetime import datetime

    # New shape wraps payload under 'content'
    if isinstance(item, dict) and 'content' in item:
        item = item['content']

    title = item.get('title', 'No title')

    # Link: new shape uses canonicalUrl.url; old uses 'link'
    canonical = item.get('canonicalUrl') or {}
    link = canonical.get('url') if isinstance(canonical, dict) else None
    link = link or item.get('link', '#')

    # Publisher: new shape uses provider.displayName; old uses 'publisher'
    provider = item.get('provider') or {}
    publisher = provider.get('displayName') if isinstance(provider, dict) else None
    publisher = publisher or item.get('publisher', 'Yahoo Finance')

    # Date
    pub_time = item.get('pubDate') or item.get('providerPublishTime', 0)
    try:
        if isinstance(pub_time, (int, float)) and pub_time > 0:
            date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
        elif isinstance(pub_time, str) and pub_time:
            date_str = pub_time[:10]
        else:
            date_str = 'N/A'
    except Exception:
        date_str = 'N/A'

    return {'title': title, 'link': link, 'publisher': publisher, 'date': date_str}
