"""
mode_positional.py — Positional / Ultra-Strict long-term value scanner.

Every threshold that decides a scoring band — price range, growth %,
cash/revenue ratios, RSI/MACD/BB/volume bounds, operator-detection
sensitivity, everything — is a sidebar-adjustable parameter in PARAMS,
built by _params_ui() below. The point value *awarded* for each band is
still a fixed constant (that's the "score" the app assigns once a stock
falls in a band); it's the cutoffs that decide which band a stock falls
into that are fully adjustable.

fetch_stock_data() (history + financials) and analyze_stock() (the
scoring engine) are the CORE LOGIC that makes this mode distinct.
Everything else — exchange/scan-mode selection, rate limiting,
checkpointing, results table, filters, CSV export — comes from
scanner_common.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import bhavcopy
import indicators
import scanner_common as sc
import screener
import tickers as _tickers
from scanner_common import sskey, get_state, set_state, yf

MODE_KEY = sc.MODE_POSITIONAL

SECTOR_MAP = {
    'RELIANCE': 'Energy', 'TCS': 'IT', 'HDFCBANK': 'Banking', 'INFY': 'IT', 'ICICIBANK': 'Banking',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'SBIN': 'Banking', 'BHARTIARTL': 'Telecom', 'KOTAKBANK': 'Banking',
    'LT': 'Infrastructure', 'AXISBANK': 'Banking', 'ASIANPAINT': 'Paints', 'MARUTI': 'Auto', 'HCLTECH': 'IT',
    'BAJFINANCE': 'NBFC', 'WIPRO': 'IT', 'SUNPHARMA': 'Pharma', 'TITAN': 'Consumer', 'ULTRACEMCO': 'Cement',
    'NESTLEIND': 'FMCG', 'ONGC': 'Energy', 'TATAMOTORS': 'Auto', 'NTPC': 'Power', 'POWERGRID': 'Power',
    'JSWSTEEL': 'Metals', 'M&M': 'Auto', 'TECHM': 'IT', 'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Infrastructure',
}


# ── Adjustable parameter schema ──────────────────────────────────────────
# (key, label, default, min, max, step, kind)  kind: 'i' int / 'f' float
_PARAM_GROUPS: list[tuple[str, list[tuple], bool]] = [
    ("🎯 Qualification Score Thresholds", [
        ("th_exceptional", "Exceptional (≥)", 180, 100, 250, 10, 'i'),
        ("th_prime", "Prime (≥)", 160, 100, 250, 10, 'i'),
        ("th_excellent", "Excellent (≥)", 140, 100, 250, 10, 'i'),
        ("th_strong", "Strong (≥)", 120, 50, 200, 10, 'i'),
        ("th_good", "Good (≥)", 100, 50, 200, 10, 'i'),
        ("th_watchlist", "Watchlist (≥)", 80, 0, 150, 10, 'i'),
    ], True),
    ("📊 Market Cap Scoring (₹ Cr)", [
        ("mcap_large", "Large Cap cutoff (≥)", 20000, 1000, 200000, 1000, 'i'),
        ("mcap_mid", "Small-Mid Cap cutoff (≥)", 5000, 500, 100000, 500, 'i'),
    ], False),
    ("📈 Revenue Growth (%)", [
        ("rev_exceptional_yoy", "Exceptional YoY (≥)", 25.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_exceptional_qoq", "Exceptional QoQ (≥)", 15.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_excellent_yoy", "Excellent YoY (≥)", 20.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_excellent_qoq", "Excellent QoQ (≥)", 10.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_strong_yoy", "Strong YoY (≥)", 15.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_strong_qoq", "Strong QoQ (≥)", 8.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_good_yoy", "Good YoY (≥)", 10.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_good_qoq", "Good QoQ (≥)", 5.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_moderate_yoy", "Moderate YoY, no QoQ data (≥)", 5.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_solo_strong", "YoY-only: Strong (≥)", 20.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_solo_good", "YoY-only: Good (≥)", 12.0, -50.0, 200.0, 1.0, 'f'),
        ("rev_solo_moderate", "YoY-only: Moderate (≥)", 5.0, -50.0, 200.0, 1.0, 'f'),
    ], False),
    ("💹 Profit Growth (%)", [
        ("profit_exceptional_yoy", "Exceptional YoY (≥)", 30.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_exceptional_qoq", "Exceptional QoQ (≥)", 20.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_excellent_yoy", "Excellent YoY (≥)", 25.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_excellent_qoq", "Excellent QoQ (≥)", 15.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_strong_yoy", "Strong YoY (≥)", 20.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_strong_qoq", "Strong QoQ (≥)", 10.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_good_yoy", "Good YoY (≥)", 12.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_good_qoq", "Good QoQ (≥)", 6.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_moderate_yoy", "Moderate YoY, no QoQ data (≥)", 5.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_solo_strong", "YoY-only: Strong (≥)", 25.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_solo_good", "YoY-only: Good (≥)", 15.0, -50.0, 300.0, 1.0, 'f'),
        ("profit_solo_moderate", "YoY-only: Moderate (≥)", 8.0, -50.0, 300.0, 1.0, 'f'),
    ], False),
    ("💵 Profit Margin (%)", [
        ("margin_excellent", "Excellent (≥)", 20.0, 0.0, 100.0, 1.0, 'f'),
        ("margin_verygood", "Very Good (≥)", 15.0, 0.0, 100.0, 1.0, 'f'),
        ("margin_good", "Good (≥)", 10.0, 0.0, 100.0, 1.0, 'f'),
        ("margin_average", "Average (≥)", 5.0, 0.0, 100.0, 1.0, 'f'),
    ], False),
    ("🏦 Cash & Revenue Quality", [
        ("cash_to_mcap_strong", "Cash/MCap % — Strong (≥)", 15.0, 0.0, 200.0, 1.0, 'f'),
        ("cash_to_mcap_good", "Cash/MCap % — Good (≥)", 8.0, 0.0, 200.0, 1.0, 'f'),
        ("rev_to_mcap_strong", "Revenue/MCap ratio — Strong (≥)", 1.0, 0.0, 10.0, 0.1, 'f'),
        ("rev_to_mcap_good", "Revenue/MCap ratio — Good (≥)", 0.5, 0.0, 10.0, 0.1, 'f'),
    ], False),
    ("🏛️ FII/DII Activity Score", [
        ("fii_strong", "Strong Buying (≥)", 15, -50, 50, 1, 'i'),
        ("fii_good", "Good Buying (≥)", 10, -50, 50, 1, 'i'),
        ("fii_accum", "Accumulation (≥)", 5, -50, 50, 1, 'i'),
        ("fii_neutral", "Neutral (≥)", 0, -50, 50, 1, 'i'),
    ], False),
    ("📐 Consolidation — Weekly Change (%)", [
        ("consol_perfect_low", "Perfect base — low", -2.0, -50.0, 50.0, 0.1, 'f'),
        ("consol_perfect_high", "Perfect base — high", 0.3, -50.0, 50.0, 0.1, 'f'),
        ("consol_pullback_low", "Healthy pullback — low", -3.5, -50.0, 50.0, 0.1, 'f'),
        ("consol_pullback_high", "Healthy pullback — high", -2.0, -50.0, 50.0, 0.1, 'f'),
        ("consol_breakout_low", "Early breakout — low", 0.3, -50.0, 50.0, 0.1, 'f'),
        ("consol_breakout_high", "Early breakout — high", 1.5, -50.0, 50.0, 0.1, 'f'),
        ("consol_rallied_above", "Already rallied, above", 4.0, -50.0, 50.0, 0.1, 'f'),
    ], False),
    ("📉 RSI", [
        ("rsi_period", "RSI period", 14, 2, 60, 1, 'i'),
        ("rsi_low", "Lower bound (perfect entry)", 32, 0, 100, 1, 'i'),
        ("rsi_high", "Upper bound (perfect entry)", 38, 0, 100, 1, 'i'),
        ("rsi_band2_offset", "Momentum band offset", 7, 0, 50, 1, 'i'),
        ("rsi_band3_offset", "Early-momentum band offset", 12, 0, 50, 1, 'i'),
        ("rsi_band4_offset", "Neutral band offset", 17, 0, 50, 1, 'i'),
        ("rsi_overbought_offset", "Overbought cutoff offset", 24, 0, 50, 1, 'i'),
    ], False),
    ("📊 MACD", [
        ("macd_fast", "Fast EMA period", 12, 2, 50, 1, 'i'),
        ("macd_slow", "Slow EMA period", 26, 5, 100, 1, 'i'),
        ("macd_perfect_low", "Perfect crossover — low", -1.0, -20.0, 20.0, 0.5, 'f'),
        ("macd_perfect_high", "Perfect crossover — high", 1.0, -20.0, 20.0, 0.5, 'f'),
        ("macd_early_bullish_high", "Early bullish — high", 3.0, -20.0, 20.0, 0.5, 'f'),
        ("macd_about_turn_low", "About to turn — low", -3.0, -20.0, 20.0, 0.5, 'f'),
        ("macd_extended_above", "Extended, above", 6.0, -20.0, 20.0, 0.5, 'f'),
    ], False),
    ("📶 Bollinger Bands (%B)", [
        ("bb_period", "Period", 20, 5, 60, 1, 'i'),
        ("bb_std_mult", "Std-dev multiplier", 2.0, 0.5, 4.0, 0.1, 'f'),
        ("bb_lower_low", "Lower-band bounce — low", 8.0, 0.0, 100.0, 1.0, 'f'),
        ("bb_lower_high", "Lower-band bounce — high", 20.0, 0.0, 100.0, 1.0, 'f'),
        ("bb_below_mid_high", "Below middle — high", 30.0, 0.0, 100.0, 1.0, 'f'),
        ("bb_middle_high", "Middle zone — high", 45.0, 0.0, 100.0, 1.0, 'f'),
        ("bb_upper_above", "Upper band, above", 65.0, 0.0, 100.0, 1.0, 'f'),
    ], False),
    ("📦 Volume Multiple", [
        ("vol_window", "Averaging window (days)", 20, 5, 90, 1, 'i'),
        ("vol_perfect_low", "Perfect accumulation — low", 1.3, 0.0, 10.0, 0.1, 'f'),
        ("vol_perfect_high", "Perfect accumulation — high", 1.8, 0.0, 10.0, 0.1, 'f'),
        ("vol_building_high", "Building interest — high", 2.2, 0.0, 10.0, 0.1, 'f'),
        ("vol_toohigh_above", "Too high, above", 2.8, 0.0, 10.0, 0.1, 'f'),
        ("vol_avg_low", "Average — low", 1.0, 0.0, 10.0, 0.1, 'f'),
    ], False),
    ("📅 Today's Price Change (%)", [
        ("today_perfect_low", "Perfect entry — low", -1.5, -50.0, 50.0, 0.1, 'f'),
        ("today_perfect_high", "Perfect entry — high", 0.3, -50.0, 50.0, 0.1, 'f'),
        ("today_early_high", "Early move — high", 1.2, -50.0, 50.0, 0.1, 'f'),
        ("today_dip_low", "Dip — low", -2.5, -50.0, 50.0, 0.1, 'f'),
        ("today_rallied_above", "Already rallied, above", 2.5, -50.0, 50.0, 0.1, 'f'),
    ], False),
    ("🗓️ Monthly Trend (%)", [
        ("monthly_recover_low", "Recovering from dip — low", -8.0, -80.0, 80.0, 0.5, 'f'),
        ("monthly_recover_high", "Recovering from dip — high", -2.0, -80.0, 80.0, 0.5, 'f'),
        ("monthly_base_high", "Base building — high", 2.0, -80.0, 80.0, 0.5, 'f'),
        ("monthly_moderate_high", "Moderate gain — high", 6.0, -80.0, 80.0, 0.5, 'f'),
        ("monthly_extended_above", "Extended, above", 10.0, -80.0, 80.0, 0.5, 'f'),
    ], False),
    ("📆 3-Month Performance (%)", [
        ("three_m_correction_low", "Perfect correction — low", -15.0, -100.0, 100.0, 1.0, 'f'),
        ("three_m_correction_high", "Perfect correction — high", -5.0, -100.0, 100.0, 1.0, 'f'),
        ("three_m_sideways_high", "Sideways base — high", 5.0, -100.0, 100.0, 1.0, 'f'),
        ("three_m_moderate_high", "Moderate rise — high", 15.0, -100.0, 100.0, 1.0, 'f'),
        ("three_m_overextended_above", "Overextended, above", 25.0, -100.0, 100.0, 1.0, 'f'),
    ], False),
    ("🚀 Upside Potential (Price Range)", [
        ("upside_move_pct", "Target move — % of price", 10.0, 0.0, 100.0, 0.5, 'f'),
        ("upside_floor_rs", "Minimum target move (₹)", 20.0, 0.0, 1000.0, 5.0, 'f'),
        ("upside_excellent", "Excellent — % upside (≥)", 12.0, 0.0, 100.0, 0.5, 'f'),
        ("upside_verygood", "Very Good — % upside (≥)", 10.0, 0.0, 100.0, 0.5, 'f'),
        ("upside_good", "Good — % upside (≥)", 8.0, 0.0, 100.0, 0.5, 'f'),
    ], False),
    ("🚨 Operator / Pump Detection", [
        ("vol_spike_extreme_mult", "Extreme volume spike, ×avg", 5.0, 1.0, 20.0, 0.5, 'f'),
        ("vol_spike_high_mult", "High volume spike, ×avg", 3.0, 1.0, 20.0, 0.5, 'f'),
        ("swing_extreme_pct", "Extreme daily swing (%)", 8.0, 0.0, 50.0, 0.5, 'f'),
        ("swing_avg_extreme_pct", "Extreme swing — avg floor (%)", 3.0, 0.0, 50.0, 0.5, 'f'),
        ("swing_high_pct", "High daily swing (%)", 5.0, 0.0, 50.0, 0.5, 'f'),
        ("swing_avg_high_pct", "High swing — avg floor (%)", 2.0, 0.0, 50.0, 0.5, 'f'),
        ("circuit_change_pct", "Circuit-hit day change (%)", 9.0, 0.0, 30.0, 0.5, 'f'),
        ("circuit_hits_extreme", "Circuit hits — extreme (≥)", 3, 1, 20, 1, 'i'),
        ("circuit_hits_high", "Circuit hits — high (≥)", 2, 1, 20, 1, 'i'),
        ("operated_risk_cutoff", "Risk score — flag as OPERATED (≥)", 40, 0, 100, 5, 'i'),
    ], False),
    ("🗄️ Data Window", [
        ("lookback_trading_days", "Daily-history lookback (trading days)", 65, 20, 250, 5, 'i'),
        ("cache_ttl_sec", "Fundamentals cache TTL (sec)", 300, 30, 3600, 30, 'i'),
    ], False),
]


def _params_ui(mode_key: str) -> dict:
    params: dict = {}
    for title, rows, expanded in _PARAM_GROUPS:
        with st.sidebar.expander(title, expanded=expanded):
            for key, label, default, mn, mx, step, kind in rows:
                if kind == 'i':
                    params[key] = st.number_input(label, min_value=int(mn), max_value=int(mx),
                                                    value=int(default), step=int(step),
                                                    key=sskey(mode_key, f"p_{key}"))
                else:
                    params[key] = st.number_input(label, min_value=float(mn), max_value=float(mx),
                                                    value=float(default), step=float(step),
                                                    key=sskey(mode_key, f"p_{key}"))

    with st.sidebar.expander("🌐 Fundamentals Source", expanded=False):
        st.caption(
            "Market cap, P/E, revenue/profit growth, and cash come from "
            "screener.in by default — one HTTP call per stock instead of "
            "four separate Yahoo calls, and no Yahoo rate limit attached. "
            "Falls back to yfinance automatically for any stock screener.in "
            "doesn't have a page for."
        )
        params["use_screener"] = st.checkbox(
            "Use screener.in for fundamentals", value=True,
            help="Uncheck to go back to yfinance for market cap/financials on every stock "
                 "(slower, more Yahoo calls — useful if screener.in itself is unreachable).",
            key=sskey(mode_key, "use_screener"),
        )
        screener_cache_ttl = st.number_input(
            "screener.in cache TTL (hours)", min_value=1, max_value=48, value=6, step=1,
            help="Fundamentals barely change intraday — a long TTL means a re-scan of the "
                 "same universe later the same day reuses disk-cached results with zero "
                 "new requests to screener.in.",
            key=sskey(mode_key, "screener_cache_ttl_hours"),
        )
        screener_min_delay = st.number_input(
            "Min delay between screener.in requests (sec)", min_value=0.5, max_value=10.0,
            value=2.0, step=0.5,
            help="Floor on the gap between outgoing screener.in requests, shared across "
                 "all workers — kept polite by default since this is someone else's site, "
                 "not a rate-limited API.",
            key=sskey(mode_key, "screener_min_delay"),
        )
        screener.configure(min_delay=screener_min_delay, cache_ttl=screener_cache_ttl * 3600)
        sc_status = screener.get_status()
        if sc_status["requests"] or sc_status["cache_hits"]:
            st.caption(
                f"This session: {sc_status['requests']} request(s) · "
                f"{sc_status['cache_hits']} cache hit(s) · "
                f"{sc_status['not_found']} not on screener.in · "
                f"{sc_status['failures']} failed"
            )

    return params


def _split_yf_symbol(yf_symbol: str) -> "tuple[str, str] | tuple[None, None]":
    """'RELIANCE.NS' -> ('RELIANCE', 'NSE'); '500325.BO' -> ('500325', 'BSE')."""
    if yf_symbol.endswith(".NS"):
        return yf_symbol[:-3], "NSE"
    if yf_symbol.endswith(".BO"):
        return yf_symbol[:-3], "BSE"
    return None, None


_BSE_CODE_MAP: "dict | None" = None
_BSE_CODE_LOCK = threading.Lock()


def _bse_code_for(yf_symbol: str) -> "str | None":
    """screener.in indexes most BSE-only small-caps by ticker symbol like NSE,
    but a few only resolve by their numeric BSE scrip code — this gives
    screener.get_fundamentals() that code as a second slug to try."""
    global _BSE_CODE_MAP
    if not yf_symbol.endswith(".BO"):
        return None
    with _BSE_CODE_LOCK:
        if _BSE_CODE_MAP is None:
            _BSE_CODE_MAP = {r["yf_symbol"]: r.get("bse_code") for r in _tickers.load_bse_universe()}
    return _BSE_CODE_MAP.get(yf_symbol)


# ── CORE LOGIC: fetch ────────────────────────────────────────────────────
def fetch_stock_data(yf_symbol: str, params: dict):
    """History for technicals comes from NSE/BSE's own bhavcopy first (see
    bhavcopy.py) — free EOD data with no Yahoo rate limit attached. Falls
    back to yfinance's ticker.history() if bhavcopy has no usable data.

    Fundamentals (market cap, P/E, annual + quarterly P&L, cash) come from
    screener.in first (see screener.py) — one HTTP GET instead of four
    separate Yahoo calls, and no Yahoo rate limit attached either. Falls
    back to yfinance's financial statements only when screener.py returns
    None (page not found, or the sidebar "🌐 Fundamentals Source" toggle in
    _params_ui() disables it). When both bhavcopy and screener succeed for
    a stock, this function makes zero Yahoo calls for it at all. ticker.info
    is intentionally never called — the most throttled Yahoo endpoint, and
    everything here is derivable from the financial statements + fast_info."""
    if sc.is_known_dead(yf_symbol):
        return None

    cache_key = f"{MODE_KEY}:{yf_symbol}"
    cached = sc.cache_get(cache_key, params["cache_ttl_sec"])
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(yf_symbol)

        bhav = bhavcopy.get_daily_series(yf_symbol, trading_days=params["lookback_trading_days"])
        if bhav is not None:
            closes = bhav['close'].values
            highs = bhav['high'].values
            lows = bhav['low'].values
            volumes = bhav['volume'].values
        else:
            hist = ticker.history(period="3mo", interval="1d")
            if hist.empty:
                sc.mark_dead_symbol(yf_symbol)
                return None
            closes = hist['Close'].values
            highs = hist['High'].values
            lows = hist['Low'].values
            volumes = hist['Volume'].values

        price = closes[-1]
        prev_close = closes[-2] if len(closes) > 1 else price
        change = ((price - prev_close) / prev_close) * 100

        fundamentals_source = "yfinance"
        screener_data = None
        if params.get("use_screener", True):
            plain_symbol, _exch = _split_yf_symbol(yf_symbol)
            screener_data = (
                screener.get_fundamentals(plain_symbol, bse_code=_bse_code_for(yf_symbol))
                if plain_symbol else None
            )

        if screener_data is not None:
            fundamentals_source = "screener"
            market_cap = screener_data["market_cap"]
            pe_ratio = screener_data["pe_ratio"]
            total_cash = screener_data["total_cash"]
            latest_fy_revenue = screener_data["latest_fy_revenue"]
            profit_margin = screener_data["profit_margin"]
            qoq_revenue_growth = screener_data["qoq_revenue_growth"]
            yoy_revenue_growth = screener_data["yoy_revenue_growth"]
            qoq_profit_growth = screener_data["qoq_profit_growth"]
            yoy_profit_growth = screener_data["yoy_profit_growth"]
            historical_data = screener_data["historical_data"]
        else:
            fi = ticker.fast_info
            market_cap = getattr(fi, 'market_cap', None) or 0

            annual_inc = None
            try:
                annual_inc = ticker.income_stmt if hasattr(ticker, 'income_stmt') else ticker.financials
            except Exception:
                pass

            annual_bs = None
            try:
                annual_bs = ticker.balance_sheet
            except Exception:
                pass

            q_inc = None
            try:
                q_inc = ticker.quarterly_income_stmt if hasattr(ticker, 'quarterly_income_stmt') else ticker.quarterly_financials
            except Exception:
                pass

            latest_fy_revenue = 0
            if annual_inc is not None and not annual_inc.empty and 'Total Revenue' in annual_inc.index:
                v = annual_inc.loc['Total Revenue'].iloc[0]
                latest_fy_revenue = 0 if pd.isna(v) else v

            total_cash = 0
            if annual_bs is not None and not annual_bs.empty:
                for cash_key in ('Cash And Cash Equivalents',
                                  'Cash Cash Equivalents And Short Term Investments',
                                  'Cash And Short Term Investments'):
                    if cash_key in annual_bs.index:
                        v = annual_bs.loc[cash_key].iloc[0]
                        total_cash = 0 if pd.isna(v) else v
                        break

            profit_margin = None
            if annual_inc is not None and not annual_inc.empty:
                try:
                    rev = annual_inc.loc['Total Revenue'].iloc[0] if 'Total Revenue' in annual_inc.index else None
                    net = annual_inc.loc['Net Income'].iloc[0] if 'Net Income' in annual_inc.index else None
                    if rev and net and not pd.isna(rev) and not pd.isna(net) and rev != 0:
                        profit_margin = net / rev
                except Exception:
                    pass

            pe_ratio = getattr(fi, 'p_e_ratio', None)

            qoq_revenue_growth = yoy_revenue_growth = None
            qoq_profit_growth = yoy_profit_growth = None

            if q_inc is not None and not q_inc.empty:
                if 'Total Revenue' in q_inc.index:
                    revenues = [r for r in q_inc.loc['Total Revenue'].values if not pd.isna(r)]
                    if len(revenues) >= 2:
                        qoq_revenue_growth = ((revenues[0] - revenues[1]) / abs(revenues[1])) * 100 if revenues[1] != 0 else None
                    if len(revenues) >= 4:
                        yoy_revenue_growth = ((revenues[0] - revenues[3]) / abs(revenues[3])) * 100 if revenues[3] != 0 else None
                if 'Net Income' in q_inc.index:
                    profits = [p for p in q_inc.loc['Net Income'].values if not pd.isna(p)]
                    if len(profits) >= 2:
                        qoq_profit_growth = ((profits[0] - profits[1]) / abs(profits[1])) * 100 if profits[1] != 0 else None
                    if len(profits) >= 4:
                        yoy_profit_growth = ((profits[0] - profits[3]) / abs(profits[3])) * 100 if profits[3] != 0 else None

            historical_data = get_historical_financials_from_data(annual_inc, annual_bs, market_cap)

        cash_on_hand_to_mcap = (total_cash / market_cap * 100) if market_cap > 0 and total_cash > 0 else 0
        latest_fy_revenue_to_mcap = (latest_fy_revenue / market_cap) if market_cap > 0 and latest_fy_revenue > 0 else 0

        fii_dii_activity = indicators.detect_institutional_activity(volumes, closes)
        rsi = indicators.rsi(closes, period=params["rsi_period"])
        macd = indicators.macd(closes, fast=params["macd_fast"], slow=params["macd_slow"])
        bb_position = indicators.bollinger_position(closes, period=params["bb_period"], std_mult=params["bb_std_mult"])
        vol_multiple = indicators.volume_multiple(volumes, window=params["vol_window"])
        trend = indicators.detect_trend(closes)

        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 5 and closes[0] != 0 else 0

        result = {
            'symbol': yf_symbol, 'price': price, 'change': change,
            'weekly_change': weekly_change, 'monthly_change': monthly_change,
            'three_month_change': three_month_change, 'rsi': rsi, 'macd': macd,
            'bb_position': bb_position, 'vol_multiple': vol_multiple, 'trend': trend,
            'closes': closes, 'highs': highs, 'lows': lows, 'volumes': volumes,
            'fii_dii_score': fii_dii_activity, 'market_cap': market_cap,
            'profit_margin': profit_margin, 'pe_ratio': pe_ratio,
            'total_cash': total_cash, 'latest_fy_revenue': latest_fy_revenue,
            'cash_on_hand_to_mcap': cash_on_hand_to_mcap,
            'latest_fy_revenue_to_mcap': latest_fy_revenue_to_mcap,
            'historical_data': historical_data,
            'qoq_revenue_growth': qoq_revenue_growth, 'yoy_revenue_growth': yoy_revenue_growth,
            'qoq_profit_growth': qoq_profit_growth, 'yoy_profit_growth': yoy_profit_growth,
            'fundamentals_source': fundamentals_source,
        }
        sc.cache_set(cache_key, result)
        return result

    except Exception as e:
        if any(kw in str(e).lower() for kw in ("delisted", "not found", "no data found")):
            sc.mark_dead_symbol(yf_symbol)
        return None


def get_historical_financials_from_data(annual_inc, annual_bs, current_mcap):
    historical = {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []}
    try:
        if annual_inc is None or annual_inc.empty:
            return historical
        years = list(annual_inc.columns[:3]) if len(annual_inc.columns) >= 3 else list(annual_inc.columns)
        for year in years:
            year_str = year.strftime('%Y') if hasattr(year, 'strftime') else str(year)
            historical['years'].append(year_str)
            if 'Total Revenue' in annual_inc.index:
                v = annual_inc.loc['Total Revenue', year]
                historical['revenues'].append(0 if pd.isna(v) else v)
            else:
                historical['revenues'].append(0)
            cash = 0
            if annual_bs is not None and not annual_bs.empty and year in annual_bs.columns:
                for cash_key in ('Cash And Cash Equivalents',
                                  'Cash Cash Equivalents And Short Term Investments',
                                  'Cash And Short Term Investments'):
                    if cash_key in annual_bs.index:
                        v = annual_bs.loc[cash_key, year]
                        cash = 0 if pd.isna(v) else v
                        break
            historical['cash_amounts'].append(cash)
        for revenue in historical['revenues']:
            historical['sales_to_mcap'].append(revenue / current_mcap if current_mcap > 0 and revenue > 0 else 0)
    except Exception:
        pass
    return historical


def fetch_live_price(yf_symbol: str):
    """NSE/BSE's own live-quote endpoint first (free, no Yahoo rate limit) —
    see bhavcopy.get_live_quote(). Falls back to yfinance's 1-minute-bar
    call only if that returns nothing."""
    quote = bhavcopy.get_live_quote(yf_symbol, bse_code=_bse_code_for(yf_symbol))
    if quote is not None:
        return quote
    try:
        data = yf.Ticker(yf_symbol).history(period="1d", interval="1m")
        if data is not None and not data.empty:
            return data['Close'].iloc[-1]
        return None
    except Exception:
        return None


# ── CORE LOGIC: score ───────────────────────────────────────────────────
def analyze_stock(data, min_market_cap, min_price, max_price, p):
    """`p` is the full adjustable-parameters dict from _params_ui()."""
    try:
        if not data:
            return None

        price = data['price']; change = data['change']; rsi = data['rsi']; macd = data['macd']
        bb = data['bb_position']; vol = data['vol_multiple']; trend = data['trend']; closes = data['closes']

        if price < min_price or price > max_price:
            return None

        market_cap = data['market_cap'] / 10000000 if data['market_cap'] else 0
        if market_cap < min_market_cap:
            return None

        is_operated, operator_flags, operator_risk = indicators.detect_operator_activity(
            closes, data['volumes'],
            vol_spike_extreme_mult=p["vol_spike_extreme_mult"], vol_spike_high_mult=p["vol_spike_high_mult"],
            swing_extreme_pct=p["swing_extreme_pct"], swing_avg_extreme_pct=p["swing_avg_extreme_pct"],
            swing_high_pct=p["swing_high_pct"], swing_avg_high_pct=p["swing_avg_high_pct"],
            circuit_change_pct=p["circuit_change_pct"], circuit_hits_extreme=p["circuit_hits_extreme"],
            circuit_hits_high=p["circuit_hits_high"], operated_risk_cutoff=p["operated_risk_cutoff"],
        )

        weekly_change = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if len(closes) >= 5 and closes[-5] != 0 else 0
        monthly_change = ((closes[-1] - closes[-20]) / closes[-20]) * 100 if len(closes) >= 20 and closes[-20] != 0 else 0
        three_month_change = ((closes[-1] - closes[0]) / closes[0]) * 100 if len(closes) >= 5 and closes[0] != 0 else 0

        potential_rs = max(p["upside_floor_rs"], price * (p["upside_move_pct"] / 100.0))
        potential_pct = (potential_rs / price) * 100 if price != 0 else 0

        score = 0
        criteria = []

        if is_operated:
            score -= 70
            criteria.append(f'🚨 OPERATOR DETECTED: Risk Score {operator_risk}/100 - AVOID [-70 pts]')
        elif operator_risk >= 30:
            score -= 40
            criteria.append(f'🚨 VERY HIGH RISK: Major manipulation signs (Risk: {operator_risk}/100) [-40 pts]')
        elif operator_risk >= 20:
            score -= 25
            criteria.append(f'⚠️ HIGH RISK: Manipulation signs detected (Risk: {operator_risk}/100) [-25 pts]')
        elif operator_risk >= 12:
            score -= 12
            criteria.append(f'⚠️ MODERATE RISK: Some volatility flags (Risk: {operator_risk}/100) [-12 pts]')

        # 1. MARKET CAP (15 pts)
        if market_cap >= p["mcap_large"]:
            score += 15
            criteria.append(f'✅ Market Cap: Large Cap (₹{market_cap:.0f} Cr) [15 pts]')
        elif market_cap >= p["mcap_mid"]:
            score += 7
            criteria.append(f'⚠ Market Cap: Small-Mid Cap (₹{market_cap:.0f} Cr) [7 pts]')
        else:
            criteria.append(f'❌ Market Cap: Small Cap (₹{market_cap:.0f} Cr) [0 pts]')

        # 2. REVENUE GROWTH (25 pts)
        yoy_rev = data['yoy_revenue_growth']
        qoq_rev = data['qoq_revenue_growth']
        if yoy_rev is not None and qoq_rev is not None:
            if yoy_rev >= p["rev_exceptional_yoy"] and qoq_rev >= p["rev_exceptional_qoq"]:
                score += 25; criteria.append(f'✅ Revenue: EXCEPTIONAL Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [25 pts]')
            elif yoy_rev >= p["rev_excellent_yoy"] and qoq_rev >= p["rev_excellent_qoq"]:
                score += 22; criteria.append(f'✅ Revenue: Excellent Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [22 pts]')
            elif yoy_rev >= p["rev_strong_yoy"] and qoq_rev >= p["rev_strong_qoq"]:
                score += 18; criteria.append(f'✅ Revenue: Strong Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [18 pts]')
            elif yoy_rev >= p["rev_good_yoy"] and qoq_rev >= p["rev_good_qoq"]:
                score += 12; criteria.append(f'⚠ Revenue: Good Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [12 pts]')
            elif yoy_rev >= p["rev_moderate_yoy"]:
                score += 5; criteria.append(f'⚠ Revenue: Moderate Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Revenue: Weak/Negative Growth (YoY: {yoy_rev:.1f}%, QoQ: {qoq_rev:.1f}%) [0 pts]')
        elif yoy_rev is not None:
            if yoy_rev >= p["rev_solo_strong"]:
                score += 20; criteria.append(f'✅ Revenue: Strong YoY Growth ({yoy_rev:.1f}%) [20 pts]')
            elif yoy_rev >= p["rev_solo_good"]:
                score += 15; criteria.append(f'✅ Revenue: Good YoY Growth ({yoy_rev:.1f}%) [15 pts]')
            elif yoy_rev >= p["rev_solo_moderate"]:
                score += 8; criteria.append(f'⚠ Revenue: Moderate Growth ({yoy_rev:.1f}%) [8 pts]')
            else:
                criteria.append(f'❌ Revenue: Weak Growth ({yoy_rev:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Revenue: Data not available [0 pts]')

        # 3. PROFIT GROWTH (25 pts)
        yoy_profit = data['yoy_profit_growth']
        qoq_profit = data['qoq_profit_growth']
        profit_margin = data['profit_margin']
        if yoy_profit is not None and qoq_profit is not None:
            if yoy_profit >= p["profit_exceptional_yoy"] and qoq_profit >= p["profit_exceptional_qoq"]:
                score += 25; criteria.append(f'✅ Profit: EXCEPTIONAL Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [25 pts]')
            elif yoy_profit >= p["profit_excellent_yoy"] and qoq_profit >= p["profit_excellent_qoq"]:
                score += 22; criteria.append(f'✅ Profit: Excellent Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [22 pts]')
            elif yoy_profit >= p["profit_strong_yoy"] and qoq_profit >= p["profit_strong_qoq"]:
                score += 18; criteria.append(f'✅ Profit: Strong Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [18 pts]')
            elif yoy_profit >= p["profit_good_yoy"] and qoq_profit >= p["profit_good_qoq"]:
                score += 12; criteria.append(f'⚠ Profit: Good Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [12 pts]')
            elif yoy_profit >= p["profit_moderate_yoy"]:
                score += 5; criteria.append(f'⚠ Profit: Moderate Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Profit: Weak/Negative Growth (YoY: {yoy_profit:.1f}%, QoQ: {qoq_profit:.1f}%) [0 pts]')
        elif yoy_profit is not None:
            if yoy_profit >= p["profit_solo_strong"]:
                score += 20; criteria.append(f'✅ Profit: Strong YoY Growth ({yoy_profit:.1f}%) [20 pts]')
            elif yoy_profit >= p["profit_solo_good"]:
                score += 15; criteria.append(f'✅ Profit: Good YoY Growth ({yoy_profit:.1f}%) [15 pts]')
            elif yoy_profit >= p["profit_solo_moderate"]:
                score += 8; criteria.append(f'⚠ Profit: Moderate Growth ({yoy_profit:.1f}%) [8 pts]')
            else:
                criteria.append(f'❌ Profit: Weak Growth ({yoy_profit:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Profit: Data not available [0 pts]')

        # 4. PROFIT MARGIN (15 pts)
        if profit_margin is not None:
            pm = profit_margin * 100
            if pm >= p["margin_excellent"]:
                score += 15; criteria.append(f'✅ Profit Margin: Excellent ({pm:.1f}%) [15 pts]')
            elif pm >= p["margin_verygood"]:
                score += 12; criteria.append(f'✅ Profit Margin: Very Good ({pm:.1f}%) [12 pts]')
            elif pm >= p["margin_good"]:
                score += 10; criteria.append(f'✅ Profit Margin: Good ({pm:.1f}%) [10 pts]')
            elif pm >= p["margin_average"]:
                score += 5; criteria.append(f'⚠ Profit Margin: Average ({pm:.1f}%) [5 pts]')
            else:
                criteria.append(f'❌ Profit Margin: Low ({pm:.1f}%) [0 pts]')
        else:
            criteria.append('❌ Profit Margin: Data not available [0 pts]')

        # 5. CASH & REVENUE QUALITY (15 pts)
        cash_pct = data.get('cash_on_hand_to_mcap', 0)
        rev_ratio = data.get('latest_fy_revenue_to_mcap', 0)
        if cash_pct >= p["cash_to_mcap_strong"] and rev_ratio >= p["rev_to_mcap_strong"]:
            score += 15; criteria.append(f'✅ Cash/Revenue: Strong (Cash/MCap: {cash_pct:.1f}%, Rev/MCap: {rev_ratio:.2f}x) [15 pts]')
        elif cash_pct >= p["cash_to_mcap_good"] or rev_ratio >= p["rev_to_mcap_good"]:
            score += 8; criteria.append(f'⚠ Cash/Revenue: Good (Cash/MCap: {cash_pct:.1f}%, Rev/MCap: {rev_ratio:.2f}x) [8 pts]')
        else:
            criteria.append(f'❌ Cash/Revenue: Weak (Cash/MCap: {cash_pct:.1f}%, Rev/MCap: {rev_ratio:.2f}x) [0 pts]')

        # 6. FII/DII ACTIVITY (20 pts)
        fii_score = data['fii_dii_score']
        if fii_score >= p["fii_strong"]:
            score += 20; criteria.append(f'✅ FII/DII: Strong Buying ({fii_score}) [20 pts]')
        elif fii_score >= p["fii_good"]:
            score += 15; criteria.append(f'✅ FII/DII: Good Buying ({fii_score}) [15 pts]')
        elif fii_score >= p["fii_accum"]:
            score += 10; criteria.append(f'✅ FII/DII: Accumulation ({fii_score}) [10 pts]')
        elif fii_score >= p["fii_neutral"]:
            score += 5; criteria.append(f'⚠ FII/DII: Neutral ({fii_score}) [5 pts]')
        else:
            criteria.append(f'❌ FII/DII: Selling ({fii_score}) [0 pts]')

        # 7. CONSOLIDATION (20 pts)
        if p["consol_perfect_low"] <= weekly_change <= p["consol_perfect_high"]:
            score += 20; criteria.append(f'✅ Consolidation: Perfect base ({weekly_change:+.1f}% weekly) [20 pts]')
        elif p["consol_pullback_low"] <= weekly_change < p["consol_pullback_high"]:
            score += 18; criteria.append(f'✅ Consolidation: Healthy pullback ({weekly_change:+.1f}% weekly) [18 pts]')
        elif p["consol_breakout_low"] < weekly_change <= p["consol_breakout_high"]:
            score += 15; criteria.append(f'✅ Consolidation: Early breakout ({weekly_change:+.1f}% weekly) [15 pts]')
        elif weekly_change > p["consol_rallied_above"]:
            criteria.append(f'❌ Already rallied ({weekly_change:+.1f}% weekly) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ Consolidation: Weak ({weekly_change:+.1f}% weekly) [5 pts]')

        # 8. RSI (20 pts)
        rsi_low = p['rsi_low']; rsi_high = p['rsi_high']
        if rsi_low <= rsi <= rsi_high:
            score += 20; criteria.append(f'✅ RSI: Perfect oversold entry ({rsi:.0f}) [20 pts]')
        elif rsi_high < rsi <= rsi_high + p["rsi_band2_offset"]:
            score += 17; criteria.append(f'✅ RSI: Building momentum ({rsi:.0f}) [17 pts]')
        elif rsi_high + p["rsi_band2_offset"] < rsi <= rsi_high + p["rsi_band3_offset"]:
            score += 12; criteria.append(f'✅ RSI: Early momentum ({rsi:.0f}) [12 pts]')
        elif rsi_high + p["rsi_band3_offset"] < rsi <= rsi_high + p["rsi_band4_offset"]:
            score += 8; criteria.append(f'⚠ RSI: Neutral ({rsi:.0f}) [8 pts]')
        elif rsi > rsi_high + p["rsi_overbought_offset"]:
            criteria.append(f'❌ RSI: Overbought ({rsi:.0f}) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ RSI: Moderate ({rsi:.0f}) [5 pts]')

        # 9. MACD (15 pts)
        if p["macd_perfect_low"] <= macd <= p["macd_perfect_high"]:
            score += 15; criteria.append(f'✅ MACD: Perfect crossover ({macd:.1f}) [15 pts]')
        elif p["macd_perfect_high"] < macd <= p["macd_early_bullish_high"]:
            score += 12; criteria.append(f'✅ MACD: Early bullish ({macd:.1f}) [12 pts]')
        elif p["macd_about_turn_low"] <= macd < p["macd_perfect_low"]:
            score += 10; criteria.append(f'✅ MACD: About to turn ({macd:.1f}) [10 pts]')
        elif macd > p["macd_extended_above"]:
            criteria.append(f'❌ MACD: Extended ({macd:.1f}) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ MACD: Weak ({macd:.1f}) [5 pts]')

        # 10. BOLLINGER BANDS (15 pts)
        if p["bb_lower_low"] <= bb <= p["bb_lower_high"]:
            score += 15; criteria.append(f'✅ BB: Lower band bounce ({bb:.0f}%) [15 pts]')
        elif p["bb_lower_high"] < bb <= p["bb_below_mid_high"]:
            score += 12; criteria.append(f'✅ BB: Below middle ({bb:.0f}%) [12 pts]')
        elif p["bb_below_mid_high"] < bb <= p["bb_middle_high"]:
            score += 8; criteria.append(f'⚠ BB: Middle zone ({bb:.0f}%) [8 pts]')
        elif bb > p["bb_upper_above"]:
            criteria.append(f'❌ BB: Upper band ({bb:.0f}%) [0 pts]')
        else:
            score += 5; criteria.append(f'⚠ BB: Neutral ({bb:.0f}%) [5 pts]')

        # 11. VOLUME (15 pts)
        if p["vol_perfect_low"] <= vol <= p["vol_perfect_high"]:
            score += 15; criteria.append(f'✅ Volume: Perfect accumulation ({vol:.1f}x) [15 pts]')
        elif p["vol_perfect_high"] < vol <= p["vol_building_high"]:
            score += 12; criteria.append(f'✅ Volume: Building interest ({vol:.1f}x) [12 pts]')
        elif vol > p["vol_toohigh_above"]:
            score += 5; criteria.append(f'⚠ Volume: Too high ({vol:.1f}x) [5 pts]')
        elif p["vol_avg_low"] <= vol < p["vol_perfect_low"]:
            score += 7; criteria.append(f'⚠ Volume: Average ({vol:.1f}x) [7 pts]')
        else:
            criteria.append(f'❌ Volume: Too low ({vol:.1f}x) [0 pts]')

        # 12. TODAY'S PRICE (10 pts)
        if p["today_perfect_low"] <= change <= p["today_perfect_high"]:
            score += 10; criteria.append(f"✅ Today: Perfect entry ({change:+.1f}%) [10 pts]")
        elif p["today_perfect_high"] < change <= p["today_early_high"]:
            score += 8; criteria.append(f"✅ Today: Early move ({change:+.1f}%) [8 pts]")
        elif p["today_dip_low"] <= change < p["today_perfect_low"]:
            score += 7; criteria.append(f"⚠ Today: Dip ({change:+.1f}%) [7 pts]")
        elif change > p["today_rallied_above"]:
            criteria.append(f"❌ Today: Already rallied ({change:+.1f}%) [0 pts]")
        else:
            score += 4; criteria.append(f"⚠ Today: Moderate ({change:+.1f}%) [4 pts]")

        # 13. MONTHLY TREND (10 pts)
        if p["monthly_recover_low"] <= monthly_change <= p["monthly_recover_high"]:
            score += 10; criteria.append(f'✅ Monthly: Recovering from dip ({monthly_change:+.1f}%) [10 pts]')
        elif p["monthly_recover_high"] < monthly_change <= p["monthly_base_high"]:
            score += 8; criteria.append(f'✅ Monthly: Base building ({monthly_change:+.1f}%) [8 pts]')
        elif p["monthly_base_high"] < monthly_change <= p["monthly_moderate_high"]:
            score += 5; criteria.append(f'⚠ Monthly: Moderate gain ({monthly_change:+.1f}%) [5 pts]')
        elif monthly_change > p["monthly_extended_above"]:
            criteria.append(f'❌ Monthly: Extended ({monthly_change:+.1f}%) [0 pts]')
        else:
            score += 3; criteria.append(f'⚠ Monthly: Weak ({monthly_change:+.1f}%) [3 pts]')

        # 14. 3-MONTH PERFORMANCE (10 pts)
        if p["three_m_correction_low"] <= three_month_change <= p["three_m_correction_high"]:
            score += 10; criteria.append(f'✅ 3-Month: Perfect correction ({three_month_change:+.1f}%) [10 pts]')
        elif p["three_m_correction_high"] < three_month_change <= p["three_m_sideways_high"]:
            score += 8; criteria.append(f'✅ 3-Month: Sideways base ({three_month_change:+.1f}%) [8 pts]')
        elif p["three_m_sideways_high"] < three_month_change <= p["three_m_moderate_high"]:
            score += 5; criteria.append(f'⚠ 3-Month: Moderate rise ({three_month_change:+.1f}%) [5 pts]')
        elif three_month_change > p["three_m_overextended_above"]:
            criteria.append(f'❌ 3-Month: Overextended ({three_month_change:+.1f}%) [0 pts]')
        else:
            score += 3; criteria.append(f'⚠ 3-Month: Weak ({three_month_change:+.1f}%) [3 pts]')

        # 15. UPSIDE POTENTIAL (10 pts)
        if potential_pct >= p["upside_excellent"]:
            score += 10; criteria.append(f'✅ Upside: Excellent ({potential_pct:.1f}%) [10 pts]')
        elif potential_pct >= p["upside_verygood"]:
            score += 8; criteria.append(f'✅ Upside: Very Good ({potential_pct:.1f}%) [8 pts]')
        elif potential_pct >= p["upside_good"]:
            score += 5; criteria.append(f'⚠ Upside: Good ({potential_pct:.1f}%) [5 pts]')
        else:
            criteria.append(f'❌ Upside: Low ({potential_pct:.1f}%) [0 pts]')

        if is_operated:
            status = '🚨 OPERATED - AVOID'; rating = 'Operated - Avoid'
        elif score >= p["th_exceptional"]:
            status = '🌟 EXCEPTIONAL BUY'; rating = 'Exceptional Buy'
        elif score >= p["th_prime"]:
            status = '🚀 PRIME BUY'; rating = 'Prime Buy'
        elif score >= p["th_excellent"]:
            status = '💎 EXCELLENT BUY'; rating = 'Excellent Buy'
        elif score >= p["th_strong"]:
            status = '✅ STRONG BUY'; rating = 'Strong Buy'
        elif score >= p["th_good"]:
            status = '👍 GOOD BUY'; rating = 'Good Buy'
        elif score >= p["th_watchlist"]:
            status = '📋 WATCHLIST'; rating = 'Watchlist'
        else:
            status = '❌ SKIP'; rating = 'Skip'

        qualified = score >= p["th_excellent"] and not is_operated
        bare_symbol = data['symbol'].replace('.NS', '').replace('.BO', '')

        return {
            'symbol': data['symbol'], 'price': price, 'change': change,
            'weekly_change': weekly_change, 'monthly_change': monthly_change,
            'three_month_change': three_month_change, 'potential_rs': potential_rs,
            'potential_pct': potential_pct, 'rsi': rsi, 'macd': macd, 'bb': bb, 'vol': vol,
            'trend': trend, 'score': score, 'qualified': qualified, 'status': status,
            'rating': rating, 'criteria': criteria,
            'met_count': len([c for c in criteria if '✅' in c]),
            'sector': SECTOR_MAP.get(bare_symbol, 'Other'),
            'is_operated': is_operated, 'operator_risk': operator_risk, 'operator_flags': operator_flags,
            'market_cap': market_cap, 'yoy_revenue_growth': yoy_rev, 'qoq_revenue_growth': qoq_rev,
            'yoy_profit_growth': yoy_profit, 'qoq_profit_growth': qoq_profit,
            'profit_margin': profit_margin * 100 if profit_margin else None,
            'total_cash': data.get('total_cash', 0), 'latest_fy_revenue': data.get('latest_fy_revenue', 0),
            'cash_on_hand_to_mcap': data.get('cash_on_hand_to_mcap', 0),
            'latest_fy_revenue_to_mcap': data.get('latest_fy_revenue_to_mcap', 0),
            'historical_data': data.get('historical_data', {'years': [], 'revenues': [], 'cash_amounts': [], 'sales_to_mcap': []}),
        }
    except Exception:
        return None


# ── UI ───────────────────────────────────────────────────────────────────
def render() -> None:
    st.markdown('<p class="main-header">🎯 Positional Scanner — NSE & BSE Ultra-Strict</p>', unsafe_allow_html=True)
    st.markdown("*Choose NSE, BSE, or BOTH | Only stocks with EXCEPTIONAL fundamentals + technicals qualify*")

    scan_nse, scan_bse, universe = sc.render_exchange_selector(MODE_KEY)
    stocks_to_scan = sc.render_scan_mode_selector(MODE_KEY, universe)
    rate_cfg = sc.render_rate_limit_controls(MODE_KEY)

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Filters")
    min_market_cap = st.sidebar.slider("Minimum Market Cap (₹ Crores)", 0, 100000, 5000, 1000,
                                        help="Filter stocks by minimum market capitalization",
                                        key=sskey(MODE_KEY, "min_mcap"))
    pc1, pc2 = st.sidebar.columns(2)
    with pc1:
        min_price = st.number_input("Min Price (₹)", min_value=0.0, max_value=1000000.0, value=0.0, step=10.0,
                                     key=sskey(MODE_KEY, "min_price"))
    with pc2:
        max_price = st.number_input("Max Price (₹)", min_value=0.0, max_value=10000000.0, value=1000000.0, step=100.0,
                                     key=sskey(MODE_KEY, "max_price"))

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Scoring Parameters")
    st.sidebar.caption("Every cutoff below drives the scoring bands — adjust freely, the point weights per band stay fixed.")
    params = _params_ui(MODE_KEY)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 ULTRA-STRICT Criteria")
    st.sidebar.info(f"""*Only top 1-3% qualify!* **TOTAL: 265 Points**

**Fundamentals (95):** Market Cap 15 · Revenue Growth 25 · Profit Growth 25 · Profit Margin 15 · Cash/Revenue 15
**Technicals (170):** FII/DII 20 · Consolidation 20 · RSI 20 · MACD 15 · BB 15 · Volume 15 · Today 10 · Monthly 10 · 3-Month 10 · Upside 10

**Qualification:** Exceptional ≥{params['th_exceptional']} · Prime {params['th_prime']}-{params['th_exceptional']-1} · Excellent {params['th_excellent']}-{params['th_prime']-1} ✅ · Strong {params['th_strong']}-{params['th_excellent']-1}
**Penalties:** Operated -70 · High Risk -25 to -40
""")

    def fetch_and_analyze(rec):
        data = sc.bulletproof_fetch(fetch_stock_data, rec["yf_symbol"], params)
        if data is None:
            return 'failed', None
        analysis = analyze_stock(data, min_market_cap, min_price, max_price, params)
        if analysis is None:
            return 'filtered', None
        analysis['name'] = rec['name']
        return 'ok', analysis

    do_scan, resume_scan, checkpoint, _sig = sc.render_scan_trigger(
        MODE_KEY, stocks_to_scan, "🚀 FIND EXCEPTIONAL STOCKS")

    if do_scan:
        sc.run_scan(MODE_KEY, stocks_to_scan, fetch_and_analyze, rate_cfg, resume_scan, checkpoint)

    _render_results(params)
    sc.footer("<strong>NSE & BSE Positional Scanner with Fundamentals</strong> | Top 1-3% Only")


def _render_results(params) -> None:
    results = get_state(MODE_KEY, "results")
    if not results:
        st.info("👈 Configure and click 'FIND EXCEPTIONAL STOCKS' to start")
        return

    scan_time = get_state(MODE_KEY, "timestamp")
    st.markdown("---")

    col_r1, col_r2, col_r3 = st.columns([2, 2, 6])
    with col_r1:
        auto_refresh = st.checkbox("🔄 Auto-refresh prices", value=False,
                                    help="Continuously update prices every 30 seconds without resetting",
                                    key=sskey(MODE_KEY, "auto_refresh"))
    with col_r2:
        last_refresh = get_state(MODE_KEY, "last_refresh")
        if last_refresh:
            st.caption(f"📡 Updated {int((pd.Timestamp.now() - pd.Timestamp(last_refresh)).total_seconds())}s ago")
        else:
            st.caption("📡 Not refreshed yet")
    with col_r3:
        if auto_refresh:
            if st.button("⏸️ Pause Refresh", key=sskey(MODE_KEY, "pause_refresh")):
                set_state(MODE_KEY, "auto_refresh_paused", True)
                st.rerun()

    st.subheader("📈 Exceptional Stock Opportunities")
    if scan_time:
        st.caption(f"Initial scan: {scan_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if auto_refresh and not get_state(MODE_KEY, "auto_refresh_paused", False):
        last_refresh = get_state(MODE_KEY, "last_refresh")
        if last_refresh is None:
            set_state(MODE_KEY, "last_refresh", pd.Timestamp.now())
            last_refresh = get_state(MODE_KEY, "last_refresh")
        elapsed = (pd.Timestamp.now() - pd.Timestamp(last_refresh)).total_seconds()
        if elapsed >= 30:
            with st.spinner("🔄 Refreshing live prices..."):
                updated = 0
                for r in results:
                    try:
                        new_price = fetch_live_price(r['symbol'])
                        if new_price and new_price != r['price']:
                            prev = r['price']
                            r['price'] = new_price
                            r['change'] = ((new_price - prev) / prev) * 100 if prev != 0 else 0
                            updated += 1
                    except Exception:
                        pass
                set_state(MODE_KEY, "last_refresh", pd.Timestamp.now())
                if updated > 0:
                    st.toast(f"✅ Updated {updated} prices", icon="🔄")
            st.rerun()
        else:
            st.caption(f"⏱️ Next price refresh in {int(30 - elapsed)}s")

    df = pd.DataFrame([{
        'Symbol': r['symbol'], 'Name': r.get('name', ''),
        'Exchange': 'NSE' if '.NS' in r['symbol'] else 'BSE' if '.BO' in r['symbol'] else 'N/A',
        'Price (₹)': r['price'], 'Today (%)': r['change'], 'Weekly (%)': r['weekly_change'],
        'Monthly (%)': r['monthly_change'], '3M (%)': r['three_month_change'],
        'Market Cap (₹Cr)': r['market_cap'],
        'Cash/Hand (₹Cr)': r.get('total_cash', 0) / 10000000 if r.get('total_cash') else 0,
        'CashHand/MCap (%)': r.get('cash_on_hand_to_mcap', 0),
        'LatestFY Rev/MCap': r.get('latest_fy_revenue_to_mcap', 0),
        'Rev YoY (%)': r['yoy_revenue_growth'], 'Rev QoQ (%)': r['qoq_revenue_growth'],
        'Profit YoY (%)': r['yoy_profit_growth'], 'Profit QoQ (%)': r['qoq_profit_growth'],
        'Margin (%)': r['profit_margin'], 'RSI': r['rsi'], 'MACD': r['macd'], 'BB (%)': r['bb'],
        'Vol': f"{r['vol']:.1f}x", 'Score': r['score'], 'Rating': r['rating'], 'Status': r['status'],
        'Sector': r['sector'], 'Operated': '🚨 YES' if r['is_operated'] else '✅ Safe', 'Risk': r['operator_risk'],
    } for r in results])

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    threshold_exceptional = params['th_exceptional']
    threshold_prime = params['th_prime']
    threshold_excellent = params['th_excellent']
    threshold_strong = params['th_strong']

    exceptional = df[(df['Score'] >= threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    prime = df[(df['Score'] >= threshold_prime) & (df['Score'] < threshold_exceptional) & (df['Operated'] == '✅ Safe')]
    excellent = df[(df['Score'] >= threshold_excellent) & (df['Score'] < threshold_prime) & (df['Operated'] == '✅ Safe')]
    strong = df[(df['Score'] >= threshold_strong) & (df['Score'] < threshold_excellent) & (df['Operated'] == '✅ Safe')]
    operated_stocks = df[df['Operated'] == '🚨 YES']

    c1.metric("Total Scanned", len(df))
    c2.metric("🚨 Operated", len(operated_stocks))
    c3.metric(f"🌟 Exceptional (≥{threshold_exceptional})", len(exceptional))
    c4.metric(f"🚀 Prime ({threshold_prime}-{threshold_exceptional - 1})", len(prime))
    c5.metric(f"💎 Excellent ({threshold_excellent}-{threshold_prime - 1})", len(excellent))
    c6.metric(f"✅ Strong ({threshold_strong}-{threshold_excellent - 1})", len(strong))

    st.markdown("---")
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("📊 NSE Stocks", len(df[df['Exchange'] == 'NSE']))
    ec2.metric("📊 BSE Stocks", len(df[df['Exchange'] == 'BSE']))
    qualified_total = len(exceptional) + len(prime) + len(excellent)
    ec3.metric(f"🎯 Qualified (≥{threshold_excellent})", qualified_total)

    st.success(f"""
    **🎯 ULTRA-STRICT RESULTS:** Only **{qualified_total}** stocks qualified (Score ≥{threshold_excellent} + Safe) out of {len(df)}.
    That's the top **{(qualified_total/len(df)*100) if len(df) > 0 else 0:.1f}%** - truly exceptional opportunities with strong fundamentals!
    """)

    st.markdown("---")
    st.subheader("🔍 Filter Results")
    f1, f2, f3, f4, f5 = st.columns(5)

    with f1:
        st.markdown("**📊 Rating**")
        rating_opts = ["Exceptional Buy", "Prime Buy", "Excellent Buy", "Strong Buy", "Good Buy", "Watchlist", "Skip"]
        rating_filter = [r for r in rating_opts if st.checkbox(r, value=True, key=sskey(MODE_KEY, f"flt_rating_{r}"))]
    with f2:
        st.markdown("**📈 Exchange**")
        exchange_filter = []
        if st.checkbox("NSE", value=True, key=sskey(MODE_KEY, "flt_exc_nse")):
            exchange_filter.append("NSE")
        if st.checkbox("BSE", value=True, key=sskey(MODE_KEY, "flt_exc_bse")):
            exchange_filter.append("BSE")
    with f3:
        st.markdown("**🛡️ Safety**")
        safety_vals = []
        if st.checkbox("✅ Safe", value=True, key=sskey(MODE_KEY, "flt_safe")):
            safety_vals.append("✅ Safe")
        if st.checkbox("🚨 Operated", value=False, key=sskey(MODE_KEY, "flt_oper")):
            safety_vals.append("🚨 YES")
    with f4:
        st.markdown("**🏭 Sector**")
        all_sectors = sorted(df['Sector'].unique().tolist())
        sector_filter = [s for s in all_sectors if st.checkbox(s, value=True, key=sskey(MODE_KEY, f"flt_sector_{s}"))]
    with f5:
        min_score_filter = st.number_input("Min Score", 0, 300, threshold_excellent, 10,
                                            key=sskey(MODE_KEY, "flt_min_score"))

    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df['Rating'].isin(rating_filter)] if rating_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Exchange'].isin(exchange_filter)] if exchange_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Operated'].isin(safety_vals)] if safety_vals else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Sector'].isin(sector_filter)] if sector_filter else filtered_df.iloc[0:0]
    filtered_df = filtered_df[filtered_df['Score'] >= min_score_filter]

    st.info(f"📊 Showing *{len(filtered_df)}* stocks (filtered from {len(df)} total)")

    st.subheader("📋 Stock Analysis Table")

    def highlight_rating(row):
        if row['Operated'] == '🚨 YES':
            return ['background-color: #ff6b6b; color: white; font-weight: bold'] * len(row)
        elif row['Score'] >= 180:
            return ['background-color: #00e676; color: black; font-weight: bold'] * len(row)
        elif row['Score'] >= 160:
            return ['background-color: #69f0ae; font-weight: bold'] * len(row)
        elif row['Score'] >= 140:
            return ['background-color: #b9f6ca; font-weight: bold'] * len(row)
        elif row['Score'] >= 120:
            return ['background-color: #e1f5fe'] * len(row)
        elif row['Score'] >= 100:
            return ['background-color: #fff9c4'] * len(row)
        return ['background-color: #ffebee'] * len(row)

    styled = filtered_df.style.apply(highlight_rating, axis=1).format({
        'Price (₹)': '₹{:.2f}', 'Today (%)': '{:+.2f}%', 'Weekly (%)': '{:+.2f}%',
        'Monthly (%)': '{:+.2f}%', '3M (%)': '{:+.2f}%', 'Market Cap (₹Cr)': '₹{:.0f}',
        'Cash/Hand (₹Cr)': '₹{:.0f}', 'CashHand/MCap (%)': '{:.2f}%', 'LatestFY Rev/MCap': '{:.2f}x',
        'Rev YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Rev QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Profit YoY (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Profit QoQ (%)': lambda x: f'{x:+.1f}%' if pd.notna(x) else 'N/A',
        'Margin (%)': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A',
        'RSI': '{:.1f}', 'MACD': '{:.2f}', 'BB (%)': '{:.0f}%',
    })
    st.dataframe(styled, use_container_width=True, height=600)

    st.markdown("---")
    st.subheader("🔍 Detailed Stock Analysis")

    if len(filtered_df) > 0:
        options = filtered_df.apply(lambda r: f"{r['Symbol']} — {r['Name']}" if r['Name'] else r['Symbol'], axis=1).tolist()
        symbol_by_option = dict(zip(options, filtered_df['Symbol'].tolist()))
        selected_option = st.selectbox("Select stock for details", options, key=sskey(MODE_KEY, "detail_select"))
        selected_symbol = symbol_by_option[selected_option]
        selected_result = next((r for r in results if r['symbol'] == selected_symbol), None)

        if selected_result:
            st.markdown(f"### {selected_symbol} — {selected_result.get('name', '')} · {selected_result['status']}")

            if selected_result['is_operated']:
                st.error(f"🚨 **OPERATOR DETECTED** - Risk: {selected_result['operator_risk']}/100")
                for flag in selected_result['operator_flags']:
                    st.warning(flag)

            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Score", selected_result['score'])
            d2.metric("Price", f"₹{selected_result['price']:.2f}")
            d3.metric("Market Cap", f"₹{selected_result['market_cap']:.0f}Cr")
            d4.metric("Rev YoY", f"{selected_result['yoy_revenue_growth']:+.1f}%" if selected_result['yoy_revenue_growth'] else "N/A")
            d5.metric("Profit YoY", f"{selected_result['yoy_profit_growth']:+.1f}%" if selected_result['yoy_profit_growth'] else "N/A")

            st.markdown("---")
            st.markdown("**💵 Financial Ratios**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Cash on Hand", f"₹{selected_result.get('total_cash', 0)/10000000:.0f}Cr")
            cc2.metric("Cash/MCap Ratio", f"{selected_result.get('cash_on_hand_to_mcap', 0):.2f}%")
            cc3.metric("LatestFY Rev/MCap", f"{selected_result.get('latest_fy_revenue_to_mcap', 0):.2f}x")

            if selected_result.get('historical_data') and selected_result['historical_data']['years']:
                st.markdown("---")
                st.markdown("**📈 3-Year Historical Trends**")
                historical = selected_result['historical_data']
                fig = make_subplots(rows=3, cols=1,
                                     subplot_titles=('YoY Revenue (₹ Cr)', 'Cash Amounts (₹ Cr)', 'Sales to Market Cap Ratio'),
                                     vertical_spacing=0.12)
                if historical['revenues']:
                    fig.add_trace(go.Bar(x=historical['years'], y=[r/10000000 for r in historical['revenues']],
                                          name='Revenue', marker_color='lightblue',
                                          text=[f"₹{r/10000000:.0f}Cr" for r in historical['revenues']],
                                          textposition='auto'), row=1, col=1)
                if historical['cash_amounts']:
                    fig.add_trace(go.Bar(x=historical['years'], y=[c/10000000 for c in historical['cash_amounts']],
                                          name='Cash', marker_color='lightgreen',
                                          text=[f"₹{c/10000000:.0f}Cr" for c in historical['cash_amounts']],
                                          textposition='auto'), row=2, col=1)
                if historical['sales_to_mcap']:
                    fig.add_trace(go.Scatter(x=historical['years'], y=historical['sales_to_mcap'],
                                              name='Sales/MCap', mode='lines+markers',
                                              line=dict(color='orange', width=3), marker=dict(size=10),
                                              text=[f"{s:.2f}x" for s in historical['sales_to_mcap']],
                                              textposition='top center'), row=3, col=1)
                fig.update_layout(height=900, showlegend=False, title_text=f"{selected_symbol} - 3-Year Financial Trends")
                fig.update_yaxes(title_text="Revenue (₹ Cr)", row=1, col=1)
                fig.update_yaxes(title_text="Cash (₹ Cr)", row=2, col=1)
                fig.update_yaxes(title_text="Ratio", row=3, col=1)
                fig.update_xaxes(title_text="Year", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 Historical data not available for this stock")

            st.markdown("---")
            st.markdown("#### Detailed Scoring Breakdown")
            for criterion in selected_result['criteria']:
                if '🚨' in criterion:
                    st.error(criterion)
                elif '✅' in criterion:
                    st.success(criterion)
                elif '⚠' in criterion:
                    st.warning(criterion)
                else:
                    st.error(criterion)

    sc.download_buttons(MODE_KEY, filtered_df, df, "positional_scan")
