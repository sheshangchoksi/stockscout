"""
tickers.py — Shared NSE/BSE ticker universe, loaded once from CSV.
====================================================================
    nse_tickers.csv   columns: "NSE Ticker", "Name"
    bse_bhavcopy.csv  NSE/BSE UDiFF bhavcopy format (e.g. downloaded from
                       bseindia.com/download/BhavCopy/Equity/...). Only 3
                       columns are used: TckrSymb (col H), FinInstrmId,
                       FinInstrmNm — everything else in the file (OHLC,
                       volume, etc.) is ignored, so any day's bhavcopy file
                       works as a drop-in universe refresh.

BSE stocks on Yahoo Finance can be addressed either by their numeric BSE
scrip code ("500002.BO") or by the BSE ticker symbol ("ABB.BO"). This app
uses the ticker symbol (TckrSymb) as the primary yf_symbol, matching NSE's
SYMBOL.NS shape; the numeric code is still kept on the record (bse_code)
since bhavcopy.py's own EOD lookup joins on it.

Every ticker is returned as a small dict:
    {"symbol": "ABB", "name": "ABB India Limited", "yf_symbol": "ABB.BO",
     "exchange": "BSE", "bse_code": "500002"}
(NSE records omit "bse_code".)

Loaded once per process via st.cache_data (falls back to a plain
in-memory cache if Streamlit isn't importable, e.g. unit testing).
"""

from __future__ import annotations

import os
from typing import TypedDict

import pandas as pd

try:
    import streamlit as st
    _cache_data = st.cache_data
except Exception:  # pragma: no cover - non-Streamlit contexts (tests)
    def _cache_data(func=None, **_kwargs):
        if func is None:
            return lambda f: f
        return func

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CSV_PATH = os.path.join(_BASE_DIR, "nse_tickers.csv")
BSE_CSV_PATH = os.path.join(_BASE_DIR, "bse_bhavcopy.csv")


class TickerRecord(TypedDict, total=False):
    symbol: str
    name: str
    yf_symbol: str
    exchange: str
    bse_code: str  # BSE only — numeric scrip code, used by bhavcopy.py


def _clean_name(raw) -> str:
    if raw is None:
        return ""
    return " ".join(str(raw).split())  # collapse whitespace, strip trailing pad


@_cache_data(show_spinner=False)
def load_nse_universe() -> list[TickerRecord]:
    try:
        df = pd.read_csv(NSE_CSV_PATH, encoding="utf-8-sig", dtype=str)
    except Exception:
        return []
    df = df.dropna(subset=[df.columns[0]])
    records: list[TickerRecord] = []
    seen = set()
    for _, row in df.iterrows():
        sym = str(row.iloc[0]).strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        name = _clean_name(row.iloc[1]) if len(row) > 1 else ""
        records.append({
            "symbol": sym,
            "name": name,
            "yf_symbol": f"{sym}.NS",
            "exchange": "NSE",
        })
    return records


@_cache_data(show_spinner=False)
def load_bse_universe() -> list[TickerRecord]:
    """Loads BSE tickers from a bhavcopy CSV (bse_bhavcopy.csv). Column
    lookup is by name (TckrSymb / FinInstrmId / FinInstrmNm), not position,
    so it survives extra/reordered columns across different bhavcopy
    downloads — only requires those 3 to be present."""
    try:
        df = pd.read_csv(BSE_CSV_PATH, encoding="utf-8-sig", dtype=str)
    except Exception:
        return []
    required = {"TckrSymb", "FinInstrmId"}
    if not required.issubset(df.columns):
        return []
    df = df.dropna(subset=["TckrSymb"])
    records: list[TickerRecord] = []
    seen = set()
    for _, row in df.iterrows():
        sym = str(row["TckrSymb"]).strip().upper()
        code = str(row["FinInstrmId"]).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        name = _clean_name(row.get("FinInstrmNm"))
        records.append({
            "symbol": sym,
            "name": name,
            "yf_symbol": f"{sym}.BO",
            "exchange": "BSE",
            "bse_code": code,
        })
    return records


@_cache_data(show_spinner=False)
def load_universe(include_nse: bool, include_bse: bool) -> list[TickerRecord]:
    """Combined universe in a stable order: NSE first, then BSE, de-duplicated
    on yf_symbol (defensive — the two source files shouldn't overlap)."""
    records: list[TickerRecord] = []
    if include_nse:
        records.extend(load_nse_universe())
    if include_bse:
        records.extend(load_bse_universe())
    seen = set()
    deduped = []
    for r in records:
        if r["yf_symbol"] in seen:
            continue
        seen.add(r["yf_symbol"])
        deduped.append(r)
    return deduped


def name_lookup_map(records: list[TickerRecord]) -> dict[str, str]:
    """yf_symbol -> company name, for quick lookup when only the symbol is on hand."""
    return {r["yf_symbol"]: r["name"] for r in records}


def record_for_custom_symbol(raw_symbol: str, universe_by_yf: dict[str, "TickerRecord"]) -> TickerRecord:
    """Build a TickerRecord for a symbol typed into the Custom List box.
    Looks up the name (and BSE code, if any) from the known universe when
    recognised; otherwise ships with an empty name rather than guessing.
    """
    raw = raw_symbol.strip().upper()
    if raw.endswith(".NS") or raw.endswith(".BO"):
        yf_symbol = raw
        symbol = raw.rsplit(".", 1)[0]
        exchange = "NSE" if raw.endswith(".NS") else "BSE"
    elif raw.isdigit():
        # Bare numeric input is unambiguous: a BSE scrip code. Resolve to
        # the matching TckrSymb-based yf_symbol if we recognise the code.
        by_code = {r.get("bse_code"): r for r in universe_by_yf.values() if r.get("bse_code")}
        known = by_code.get(raw)
        if known:
            return dict(known)
        symbol, exchange, yf_symbol = raw, "BSE", f"{raw}.BO"
    else:
        symbol, exchange, yf_symbol = raw, "NSE", f"{raw}.NS"
    known = universe_by_yf.get(yf_symbol)
    if known:
        return dict(known)
    rec: TickerRecord = {"symbol": symbol, "name": "", "yf_symbol": yf_symbol, "exchange": exchange}
    return rec
