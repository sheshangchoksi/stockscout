"""
indicators.py — Shared technical-indicator math for the positional scanner.

Every numeric period/threshold here takes a keyword argument with a
default matching the scanner's original behaviour — mode_positional.py's
sidebar passes its user-adjustable values through explicitly so nothing
in here is hardcoded from the caller's point of view.

All functions are defensive (never raise) and take/return plain numpy
arrays or floats.
"""

from __future__ import annotations

import numpy as np


def rsi(prices: np.ndarray, period: int = 14) -> float:
    try:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return 50.0


def ema(prices: np.ndarray, period: int) -> float:
    try:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < period:
            return float(np.mean(prices)) if len(prices) else 0.0
        multiplier = 2 / (period + 1)
        e = np.mean(prices[:period])
        for price in prices[period:]:
            e = (price - e) * multiplier + e
        return float(e)
    except Exception:
        return 0.0


def macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
    try:
        if len(prices) < slow:
            return 0.0
        return ema(prices, fast) - ema(prices, slow)
    except Exception:
        return 0.0


def bollinger_position(prices: np.ndarray, period: int = 20, std_mult: float = 2.0) -> float:
    """0 = at lower band, 100 = at upper band, 50 = mid."""
    try:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < period:
            return 50.0
        recent = prices[-period:]
        sma = np.mean(recent)
        std = np.std(recent)
        upper = sma + (std_mult * std)
        lower = sma - (std_mult * std)
        if upper == lower:
            return 50.0
        position = ((prices[-1] - lower) / (upper - lower)) * 100
        return float(max(0, min(100, position)))
    except Exception:
        return 50.0


def volume_multiple(volumes: np.ndarray, window: int = 20) -> float:
    try:
        volumes = np.asarray(volumes, dtype=float)
        if len(volumes) < window:
            return 1.0
        avg = np.mean(volumes[-window:])
        if avg == 0:
            return 1.0
        return float(volumes[-1] / avg)
    except Exception:
        return 1.0


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """Average True Range."""
    try:
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(closes) < 2:
            return 0.0
        prev_close = np.roll(closes, 1)
        prev_close[0] = closes[0]
        tr1 = highs - lows
        tr2 = np.abs(highs - prev_close)
        tr3 = np.abs(lows - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        window = tr[-period:] if len(tr) >= period else tr
        return float(np.mean(window)) if len(window) else 0.0
    except Exception:
        return 0.0


def detect_trend(prices: np.ndarray) -> str:
    try:
        prices = np.asarray(prices, dtype=float)
        if len(prices) < 5:
            return "Sideways"
        recent = prices[-5:]
        ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        if ups >= 4:
            return "Strong Uptrend"
        elif ups >= 3:
            return "Uptrend"
        elif ups <= 1:
            return "Downtrend"
        return "Sideways"
    except Exception:
        return "Sideways"


def detect_institutional_activity(volumes: np.ndarray, closes: np.ndarray) -> int:
    """Detect FII/DII-style accumulation/distribution from volume + price action."""
    try:
        volumes = np.asarray(volumes, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(volumes) < 20 or len(closes) < 20:
            return 0
        score = 0
        recent_days = 10
        avg60 = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
        for i in range(-recent_days, 0):
            if i >= -len(volumes) and i >= -len(closes):
                vol_ratio = volumes[i] / avg60 if avg60 else 0
                if vol_ratio == 0 or np.isnan(vol_ratio):
                    continue
                if i > -len(closes) and closes[i - 1] != 0:
                    price_change = ((closes[i] - closes[i - 1]) / closes[i - 1]) * 100
                else:
                    price_change = 0
                if vol_ratio > 1.5 and price_change > 1:
                    score += 2
                elif vol_ratio > 1.2 and price_change > 0.5:
                    score += 1
                elif vol_ratio > 1.5 and price_change < -1:
                    score -= 2
                elif vol_ratio > 1.2 and price_change < -0.5:
                    score -= 1
        return score
    except Exception:
        return 0


def detect_operator_activity(
    closes: np.ndarray,
    volumes: np.ndarray,
    *,
    vol_spike_extreme_mult: float = 5.0,
    vol_spike_high_mult: float = 3.0,
    swing_extreme_pct: float = 8.0,
    swing_avg_extreme_pct: float = 3.0,
    swing_high_pct: float = 5.0,
    swing_avg_high_pct: float = 2.0,
    circuit_change_pct: float = 9.0,
    circuit_hits_extreme: int = 3,
    circuit_hits_high: int = 2,
    operated_risk_cutoff: int = 40,
):
    """Detect signs of pump/manipulation activity. Returns (is_operated, flags, risk_score)."""
    try:
        closes = np.asarray(closes, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        if len(closes) < 20:
            return False, [], 0

        flags = []
        risk = 0

        avg_vol = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
        if avg_vol == 0:
            return False, [], 0
        max_recent_vol = np.max(volumes[-10:])
        if max_recent_vol > avg_vol * vol_spike_extreme_mult:
            flags.append(f"🚨 EXTREME volume spike (>{vol_spike_extreme_mult:g}x avg) - Possible pump")
            risk += 30
        elif max_recent_vol > avg_vol * vol_spike_high_mult:
            flags.append(f"⚠️ High volume spike (>{vol_spike_high_mult:g}x avg) - Monitor closely")
            risk += 15

        recent_prices = closes[-10:]
        swings = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i - 1] != 0:
                swings.append(abs((recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]) * 100)
        avg_swing = np.mean(swings) if swings else 0
        max_swing = np.max(swings) if swings else 0
        if max_swing > swing_extreme_pct and avg_swing > swing_avg_extreme_pct:
            flags.append(f"🚨 Extreme volatility (>{swing_extreme_pct:g}% swings) - Operator activity likely")
            risk += 25
        elif max_swing > swing_high_pct and avg_swing > swing_avg_high_pct:
            flags.append(f"⚠️ High volatility (>{swing_high_pct:g}% swings) - Possible manipulation")
            risk += 12

        circuit_hits = 0
        for i in range(-20, 0):
            if i >= -len(closes) and i > -len(closes) and closes[i - 1] != 0:
                daily_change = abs((closes[i] - closes[i - 1]) / closes[i - 1]) * 100
                if daily_change > circuit_change_pct:
                    circuit_hits += 1
        if circuit_hits >= circuit_hits_extreme:
            flags.append("🚨 Multiple circuit hits - Highly manipulated")
            risk += 30
        elif circuit_hits >= circuit_hits_high:
            flags.append("⚠️ Circuit hits detected - High risk")
            risk += 15

        return risk >= operated_risk_cutoff, flags, risk
    except Exception:
        return False, [], 0
