#!/usr/bin/env python3
"""
features.py — Shared indicator & feature calculations.

All functions are pure, vectorized, and causality-safe:
  - Indicators use .shift(1) where the value must not include the current bar
  - Rolling/EWM use min_periods to avoid premature values
  - NaN propagates naturally; strategies decide how to handle it

Strategies compose these freely. Core/engine never imports this file.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict

# ══════════════════════════════════════════════════════════
# 1. CORE PRICE INDICATORS
# ══════════════════════════════════════════════════════════

def atr_series(df: pd.DataFrame, period: int = 14, alpha: Optional[float] = None) -> pd.Series:
    """
    Average True Range using Wilder smoothing (EWM alpha=1/period).
    NOT shifted — represents the ATR value known at bar close.
    Callers that need "ATR available before bar i" must .shift(1).
    """
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = 0.0
    if alpha is None:
        alpha = 1.0 / period
    return tr.ewm(alpha=alpha, adjust=False).mean()


def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder-style RSI, shifted 1 bar.
    The returned value at index i represents RSI computed through bar i-1,
    available for decision at bar i.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_np = rsi.to_numpy(dtype=np.float64)
    ag = avg_gain.to_numpy(dtype=np.float64)
    al = avg_loss.to_numpy(dtype=np.float64)
    rsi_np = np.where(al == 0.0, 100.0, rsi_np)
    rsi_np = np.where((ag == 0.0) & (al == 0.0), 50.0, rsi_np)
    return pd.Series(rsi_np, index=close.index).shift(1)


def sma_series(close: pd.Series, period: int) -> pd.Series:
    """Simple moving average, shifted 1 bar."""
    return close.rolling(period, min_periods=period).mean().shift(1)


def ema_series(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average, shifted 1 bar."""
    return close.ewm(span=period, adjust=False, min_periods=period).mean().shift(1)


def bb_ratio(df: pd.DataFrame, atr: pd.Series, period: int = 20) -> pd.Series:
    """
    Bollinger Band width ratio: (4 * std) / ATR.
    Low values = squeeze, high values = expansion.
    """
    std = df["close"].astype(float).rolling(period, min_periods=period).std()
    return (4.0 * std) / atr.replace(0, np.nan)


def bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Raw Bollinger Band width in price units."""
    std = close.rolling(period, min_periods=period).std()
    return 2.0 * num_std * std


def bb_position(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Position of close within Bollinger Bands: -1 = lower band, +1 = upper band.
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    band_range = (upper - lower).replace(0, np.nan)
    return ((close - lower) / band_range) * 2.0 - 1.0


def bb_change(close: pd.Series, period: int = 20, lookback: int = 5) -> pd.Series:
    """Change in BB width over `lookback` bars."""
    w = bb_width(close, period)
    return w.diff(lookback)


def rolling_high(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Highest high over past `period` bars, shifted 1 (excludes current bar)."""
    return df["high"].astype(float).rolling(period, min_periods=period).max().shift(1)


def rolling_low(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Lowest low over past `period` bars, shifted 1 (excludes current bar)."""
    return df["low"].astype(float).rolling(period, min_periods=period).min().shift(1)


# ══════════════════════════════════════════════════════════
# 2. VELOCITY & MOMENTUM
# ══════════════════════════════════════════════════════════

def velocity(close: pd.Series, window: int) -> pd.Series:
    """
    Normalized velocity: (close - close[window]) / close[window].
    Positive = upward momentum, negative = downward.
    """
    prev = close.shift(window)
    return (close - prev) / prev.replace(0, np.nan)


def velocity_acceleration(close: pd.Series, window: int = 5) -> pd.Series:
    """
    Acceleration of velocity: vel[window] - vel[window] shifted by window.
    Positive acceleration = momentum building.
    """
    v = velocity(close, window)
    return v - v.shift(window)


def compute_velocities(close: pd.Series, windows: Tuple[int, ...] = (1, 3, 5, 10, 20)) -> pd.DataFrame:
    """Compute velocity for multiple windows. Returns DataFrame with vel{w} columns."""
    result = pd.DataFrame(index=close.index)
    for w in windows:
        result[f"vel{w}"] = velocity(close, w)
    return result


def momentum_shift(close: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    """Difference between fast and slow velocity."""
    return velocity(close, fast) - velocity(close, slow)


# ══════════════════════════════════════════════════════════
# 3. ADX & DIRECTIONAL INDICATORS
# ══════════════════════════════════════════════════════════

def adx_indicators(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute ADX, +DI, -DI, DI spread.
    Returns DataFrame with columns: adx, plus_di, minus_di, di_spread.
    All values are shifted 1 bar for causality.
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    result = pd.DataFrame({
        "adx": adx.shift(1),
        "plus_di": plus_di.shift(1),
        "minus_di": minus_di.shift(1),
        "di_spread": (plus_di - minus_di).shift(1),
    }, index=df.index)

    return result


# ══════════════════════════════════════════════════════════
# 4. VOLATILITY REGIME
# ══════════════════════════════════════════════════════════

def vol_regime(atr: pd.Series, lookback: int = 50) -> pd.Series:
    """
    Volatility regime ratio: current ATR / ATR[lookback].
    > 1 = volatility expanding, < 1 = compressing.
    """
    return atr / atr.shift(lookback).replace(0, np.nan)


def atr_ratio(atr: pd.Series, long_window: int = 50) -> pd.Series:
    """ATR ratio: current ATR / rolling mean ATR over long_window."""
    return atr / atr.rolling(long_window, min_periods=long_window).mean().replace(0, np.nan)


def atr_percentile(atr: pd.Series, lookback: int = 500) -> pd.Series:
    """
    Percentile rank of current ATR within its recent history.
    Returns value 0-100.
    """
    return atr.rolling(lookback, min_periods=50).rank(pct=True) * 100.0


# ══════════════════════════════════════════════════════════
# 5. TREND QUALITY
# ══════════════════════════════════════════════════════════

def normalized_trend_slope(y: np.ndarray) -> float:
    """
    Price-normalized trend quality: (slope / mean_price) * r².
    Returns a dimensionless value so forex and indices share one scale.
    """
    arr = np.asarray(y, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:
        return 0.0
    try:
        from scipy.stats import linregress
        x = np.arange(len(arr))
        slope, _, r_value, _, _ = linregress(x, arr)
        price_level = float(np.mean(arr))
        norm_slope = slope / price_level if price_level > 0 else 0.0
        return float(norm_slope * (r_value ** 2))
    except Exception:
        return 0.0


def trend_quality_series(close: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling trend quality over `window` bars.
    Positive = uptrend, negative = downtrend, magnitude = strength.
    """
    def _tq(x):
        return normalized_trend_slope(x)
    return close.rolling(window, min_periods=window).apply(_tq, raw=True)


# ══════════════════════════════════════════════════════════
# 6. RANGE & STRUCTURE
# ══════════════════════════════════════════════════════════

def range_position(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Position of current close within the recent high-low range.
    0 = at range low, 1 = at range high.
    """
    h = df["high"].astype(float).rolling(lookback, min_periods=lookback).max()
    l = df["low"].astype(float).rolling(lookback, min_periods=lookback).min()
    c = df["close"].astype(float)
    rng = (h - l).replace(0, np.nan)
    return (c - l) / rng


def pullback_depth(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Depth of the most recent pullback from the swing high.
    Negative values indicate pullback from high.
    """
    h = df["high"].astype(float).rolling(lookback, min_periods=lookback).max()
    c = df["close"].astype(float)
    return (c - h) / h.replace(0, np.nan)


def retracement_depth(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Fibonacci-style retracement depth from recent swing.
    0 = at swing extreme, 1 = full retracement.
    """
    h = df["high"].astype(float).rolling(lookback, min_periods=lookback).max()
    l = df["low"].astype(float).rolling(lookback, min_periods=lookback).min()
    c = df["close"].astype(float)
    swing_range = (h - l).replace(0, np.nan)
    return (h - c) / swing_range


def swing_distance(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Distance from current close to the rolling low, in ATR units."""
    l = df["low"].astype(float).rolling(period, min_periods=period).min().shift(1)
    c = df["close"].astype(float)
    atr = atr_series(df)
    return (c - l) / atr.replace(0, np.nan)


def break_strength_up(df: pd.DataFrame, atr: pd.Series, period: int = 20) -> pd.Series:
    """Upward breakout strength: (close - rolling_high) / ATR."""
    rh = rolling_high(df, period)
    return (df["close"].astype(float) - rh) / atr.replace(0, np.nan)


def break_strength_down(df: pd.DataFrame, atr: pd.Series, period: int = 20) -> pd.Series:
    """Downward breakout strength: (rolling_low - close) / ATR."""
    rl = rolling_low(df, period)
    return (rl - df["close"].astype(float)) / atr.replace(0, np.nan)


# ══════════════════════════════════════════════════════════
# 7. FVG & SUPPORT/RESISTANCE
# ══════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, min_gap_atr: float = 0.3) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (3-bar imbalance zones).
    Returns DataFrame with columns: fvg_top, fvg_bottom, fvg_mid, fvg_type.
    fvg_type: 1 = bullish FVG, -1 = bearish FVG, 0 = none.
    """
    result = pd.DataFrame(index=df.index)
    result["fvg_top"] = np.nan
    result["fvg_bottom"] = np.nan
    result["fvg_mid"] = np.nan
    result["fvg_type"] = 0

    if len(df) < 4:
        return result

    atr = atr_series(df)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    for i in range(2, len(df)):
        a = atr.iloc[i] if np.isfinite(atr.iloc[i]) else 0.0
        if a <= 0:
            continue
        # Bullish FVG: bar[i-2] high < bar[i] low (gap up)
        if l.iloc[i] > h.iloc[i - 2]:
            gap = l.iloc[i] - h.iloc[i - 2]
            if gap > min_gap_atr * a:
                result.iloc[i, result.columns.get_loc("fvg_top")] = l.iloc[i]
                result.iloc[i, result.columns.get_loc("fvg_bottom")] = h.iloc[i - 2]
                result.iloc[i, result.columns.get_loc("fvg_mid")] = (l.iloc[i] + h.iloc[i - 2]) / 2.0
                result.iloc[i, result.columns.get_loc("fvg_type")] = 1
        # Bearish FVG: bar[i-2] low > bar[i] high (gap down)
        elif h.iloc[i] < l.iloc[i - 2]:
            gap = l.iloc[i - 2] - h.iloc[i]
            if gap > min_gap_atr * a:
                result.iloc[i, result.columns.get_loc("fvg_top")] = l.iloc[i - 2]
                result.iloc[i, result.columns.get_loc("fvg_bottom")] = h.iloc[i]
                result.iloc[i, result.columns.get_loc("fvg_mid")] = (l.iloc[i - 2] + h.iloc[i]) / 2.0
                result.iloc[i, result.columns.get_loc("fvg_type")] = -1

    return result


def fvg_distance(df: pd.DataFrame, lookback: int = 80) -> pd.Series:
    """
    Distance from current close to the nearest active FVG midpoint, in ATR units.
    Positive = FVG below price (bullish), negative = FVG above (bearish).
    """
    fvg = detect_fvg(df)
    atr = atr_series(df)
    c = df["close"].astype(float)
    result = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df)):
        a = atr.iloc[i] if np.isfinite(atr.iloc[i]) else 0.0
        if a <= 0:
            continue
        # Look back for the most recent FVG
        start = max(0, i - lookback)
        recent = fvg.iloc[start:i + 1]
        active = recent[recent["fvg_type"] != 0]
        if len(active) == 0:
            continue
        last_fvg = active.iloc[-1]
        mid = last_fvg["fvg_mid"]
        if np.isfinite(mid):
            result.iloc[i] = (c.iloc[i] - mid) / a

    return result


def dynamic_sr_levels(df: pd.DataFrame, lookback: int = 150, min_touches: int = 3,
                      tolerance_pct: float = 0.3) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect dynamic support/resistance levels from swing points.
    Returns (resistance_levels, support_levels), each a list of dicts
    with 'price', 'touches', 'strength'.
    """
    if len(df) < lookback:
        return [], []

    atr = atr_series(df)
    current_atr = atr.iloc[-1] if np.isfinite(atr.iloc[-1]) else 1.0
    tolerance = current_atr * tolerance_pct

    swing_highs = []
    swing_lows = []
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values

    for i in range(5, len(df) - 5):
        if h[i] == max(h[i - 5:i + 6]):
            swing_highs.append(h[i])
        if l[i] == min(l[i - 5:i + 6]):
            swing_lows.append(l[i])

    def cluster_prices(prices, tol):
        if not prices:
            return []
        prices = sorted(prices)
        clusters = []
        current = []
        for p in prices:
            if not current or abs(p - np.mean(current)) <= tol:
                current.append(p)
            else:
                if len(current) >= 2:
                    clusters.append({"price": np.mean(current), "touches": len(current), "strength": len(current)})
                current = [p]
        if len(current) >= 2:
            clusters.append({"price": np.mean(current), "touches": len(current), "strength": len(current)})
        return clusters

    resistance = cluster_prices(swing_highs, tolerance)
    support = cluster_prices(swing_lows, tolerance)

    resistance = [lv for lv in resistance if lv["touches"] >= min_touches]
    support = [lv for lv in support if lv["touches"] >= min_touches]

    current_price = df["close"].astype(float).iloc[-1]
    resistance.sort(key=lambda x: abs(x["price"] - current_price))
    support.sort(key=lambda x: abs(x["price"] - current_price))

    return resistance[:3], support[:3]


# ══════════════════════════════════════════════════════════
# 8. SESSION & TIME
# ══════════════════════════════════════════════════════════

def session_name(hour_utc: int) -> str:
    """Map UTC hour to session name."""
    if hour_utc < 7:
        return "Asian"
    if hour_utc < 12:
        return "London"
    if hour_utc < 17:
        return "NY"
    if hour_utc < 21:
        return "NY-PM"
    return "Off"


def session_id(hour_utc: int) -> int:
    """Map UTC hour to numeric session ID (0=Asian, 1=London, 2=NY, 3=Off)."""
    if hour_utc < 7:
        return 0
    if hour_utc < 13:
        return 1
    if hour_utc < 21:
        return 2
    return 3


def hour_sin(hour_utc: int) -> float:
    """Cyclical encoding of hour (sin component)."""
    return float(np.sin(2.0 * np.pi * hour_utc / 24.0))


def hour_cos(hour_utc: int) -> float:
    """Cyclical encoding of hour (cos component)."""
    return float(np.cos(2.0 * np.pi * hour_utc / 24.0))


# ══════════════════════════════════════════════════════════
# 9. RESAMPLING
# ══════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Resample M5 OHLC to a higher timeframe.
    Assumes input timestamps are bar OPEN times (MT5 convention).
    """
    rule_map = {
        "M15": "15min",
        "M30": "30min",
        "H1": "1h",
        "H4": "4h",
    }
    rule = rule_map.get(target_tf.upper())
    if rule is None:
        return df.copy()

    return (
        df.resample(rule, label="left", closed="left")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna(how="any")
    )


# ══════════════════════════════════════════════════════════
# 10. FEATURE ASSEMBLY HELPERS
# ══════════════════════════════════════════════════════════

def compute_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full M5 feature set used by strategies.
    Returns a DataFrame aligned with df.index.
    """
    if df is None or len(df) < 50:
        return pd.DataFrame()

    close = df["close"].astype(float)
    result = pd.DataFrame(index=df.index)

    # ATR
    atr = atr_series(df)
    result["atr"] = atr
    result["atr_ratio"] = atr_ratio(atr)

    # Velocity
    for w in (1, 3, 5, 10, 20):
        result[f"vel{w}"] = velocity(close, w)
    result["vel_acc"] = velocity_acceleration(close, 5)

    # RSI
    result["rsi"] = rsi_series(close, 14)

    # BB
    result["bb_ratio"] = bb_ratio(df, atr, 20)
    result["bb_pos"] = bb_position(close, 20)
    result["bb_width"] = bb_width(close, 20)
    result["bb_change"] = bb_change(close, 20, 5)

    # ADX
    adx_df = adx_indicators(df)
    result["adx"] = adx_df["adx"]
    result["plus_di"] = adx_df["plus_di"]
    result["minus_di"] = adx_df["minus_di"]
    result["di_spread"] = adx_df["di_spread"]

    # Vol regime
    result["vol_regime"] = vol_regime(atr, 50)

    # Trend
    result["trend_quality"] = trend_quality_series(close, 30)

    # Range
    result["range_pos"] = range_position(df, 20)
    result["pullback"] = pullback_depth(df, 20)
    result["retracement"] = retracement_depth(df, 50)
    result["swing_dist"] = swing_distance(df, 20)

    # Breakout
    result["break_up"] = break_strength_up(df, atr, 20)
    result["break_dn"] = break_strength_down(df, atr, 20)

    # FVG distance
    result["fvg_dist"] = fvg_distance(df, 80)

    # HTF premium/discount placeholder (needs H4 data, filled by strategy)
    result["htf_premium_discount"] = np.nan

    # Session/time
    if isinstance(df.index, pd.DatetimeIndex):
        hours = df.index.hour
        result["hour_utc"] = hours
        result["session_id"] = hours.map(session_id)
        result["hour_sin"] = hours.map(hour_sin)
        result["hour_cos"] = hours.map(hour_cos)
    else:
        result["hour_utc"] = 0
        result["session_id"] = 0
        result["hour_sin"] = 0.0
        result["hour_cos"] = 1.0

    return result


def compute_htf_features(df_htf: pd.DataFrame) -> pd.DataFrame:
    """
    Compute higher-timeframe context features (M15/M30/H1).
    Returns a DataFrame aligned with df_htf.index.
    """
    if df_htf is None or len(df_htf) < 30:
        return pd.DataFrame()

    close = df_htf["close"].astype(float)
    result = pd.DataFrame(index=df_htf.index)

    atr = atr_series(df_htf)
    result["atr"] = atr
    result["atr_ratio"] = atr_ratio(atr)

    for w in (5, 10, 20):
        result[f"vel{w}"] = velocity(close, w)
    result["vel_acc"] = velocity_acceleration(close, 5)

    adx_df = adx_indicators(df_htf)
    result["adx"] = adx_df["adx"]
    result["di_spread"] = adx_df["di_spread"]

    result["bb_width"] = bb_width(close, 20)
    result["range_pos"] = range_position(df_htf, 20)
    result["vol_regime"] = vol_regime(atr, 50)
    result["trend_quality"] = trend_quality_series(close, 30)

    return result
