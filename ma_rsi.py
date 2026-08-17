#!/usr/bin/env python3
"""
MA+RSI Extreme Crossover Strategy
===================================
Secondary strategy: 6 frozen patterns from Stage 2 research.
Exit strategies from Stage 3: hold_invalidation, fixed_2to1, tp2_inv_sl, tp1_inv_sl.

Ported from retf_stage3_secondary_fixed.py into the StrategyBase interface.
Preserves all original logic:
- MA+RSI extreme crossover detection
- Invalidation-based exits
- Fixed TP/SL exits
- Embargo between signals
- Warmup guards on all indicators
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_base import StrategyBase

from core import Signal, ExitAction, Direction

from features import (
    atr_series,
    rsi_series,
    sma_series,
    ema_series,
    session_name,
    resample_ohlcv,
)


class MARSIStrategy(StrategyBase):
    strategy_name = "ma_rsi"
    name = "MA+RSI Extreme Crossover"
    version = "3.0.0"

    # Backtest (80k M5 bars, ~280d, cost 10pts RT): PF 0.70 overall, net -898R.
    # Only XAUUSD (PF 1.13) and NAS100 (PF 1.07) are net-positive. All other
    # pairs lose (PF 0.34-0.93). Restrict to the beneficial pairs only.
    ALLOWED_SYMBOLS = {"XAUUSD", "NAS100"}

    EMBARGO_BARS = 20
    MAX_HOLD_BARS = 500

    # Exit strategy: one of hold_invalidation, fixed_2to1, tp2_inv_sl, tp1_inv_sl
    EXIT_STRATEGY = "tp2_inv_sl"

    # Hard stop (0 = disabled)
    HARD_STOP_R = 0.0

    # 6 frozen Stage 2 patterns
    PATTERNS = [
        {
            "name": "MA+RSI_EMA_50_RSI35/65_M15",
            "ma_type": "EMA", "ma_period": 50,
            "rsi_low": 35.0, "rsi_high": 65.0,
            "rsi_period": 14, "tf": "M15",
        },
        {
            "name": "MA+RSI_SMA_50_RSI35/65_M15",
            "ma_type": "SMA", "ma_period": 50,
            "rsi_low": 35.0, "rsi_high": 65.0,
            "rsi_period": 14, "tf": "M15",
        },
        {
            "name": "MA+RSI_EMA_20_RSI30/70_M5",
            "ma_type": "EMA", "ma_period": 20,
            "rsi_low": 30.0, "rsi_high": 70.0,
            "rsi_period": 14, "tf": "M5",
        },
        {
            "name": "MA+RSI_SMA_20_RSI35/65_M15",
            "ma_type": "SMA", "ma_period": 20,
            "rsi_low": 35.0, "rsi_high": 65.0,
            "rsi_period": 14, "tf": "M15",
        },
        {
            "name": "MA+RSI_SMA_100_RSI35/65_M15",
            "ma_type": "SMA", "ma_period": 100,
            "rsi_low": 35.0, "rsi_high": 65.0,
            "rsi_period": 14, "tf": "M15",
        },
        {
            "name": "MA+RSI_EMA_100_RSI35/65_M5",
            "ma_type": "EMA", "ma_period": 100,
            "rsi_low": 35.0, "rsi_high": 65.0,
            "rsi_period": 14, "tf": "M5",
        },
    ]

    # Exit specs
    EXIT_SPECS = {
        "hold_invalidation": {"tp_r": None, "sl_r": None, "invalidation": True},
        "fixed_2to1": {"tp_r": 2.0, "sl_r": 1.0, "invalidation": False},
        "tp2_inv_sl": {"tp_r": 2.0, "sl_r": None, "invalidation": True},
        "tp1_inv_sl": {"tp_r": 1.0, "sl_r": None, "invalidation": True},
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._last_signal_ts: Dict[str, str] = {}
        self._last_entry_ts: Dict[str, str] = {}

    # ── MA/RSI computation ──

    def _get_ma(self, close: pd.Series, ma_type: str, period: int) -> pd.Series:
        if ma_type.upper() == "EMA":
            return ema_series(close, period)
        return sma_series(close, period)

    def _compute_indicators(
        self, df: pd.DataFrame, ma_type: str, ma_period: int, rsi_period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (ma, rsi, atr) all shifted 1 bar."""
        close = df["close"].astype(float)
        ma = self._get_ma(close, ma_type, ma_period)
        rsi = rsi_series(close, rsi_period)
        atr = atr_series(df, alpha=1.0 / 14.0).shift(1)
        return ma, rsi, atr

    # ── Signal detection ──

    def _detect_signals(
        self,
        df: pd.DataFrame,
        pattern: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        MA+RSI extreme crossover detector.

        LONG:  RSI < rsi_low AND close[i-1] < MA AND close[i] >= MA
        SHORT: RSI > rsi_high AND close[i-1] > MA AND close[i] <= MA

        Returns list of dicts with bar index, direction, and metadata.
        """
        if df is None or len(df) < 220:
            return []

        ma, rsi, atr = self._compute_indicators(
            df, pattern["ma_type"], pattern["ma_period"], pattern["rsi_period"]
        )

        C = df["close"].astype(float).values
        O = df["open"].astype(float).values
        n = len(C)

        ma_vals = ma.values
        rsi_vals = rsi.values
        atr_vals = atr.values

        C_prev = np.roll(C, 1)
        if n > 0:
            C_prev[0] = np.nan

        valid = (
            np.isfinite(C)
            & np.isfinite(C_prev)
            & np.isfinite(ma_vals)
            & np.isfinite(rsi_vals)
        )

        # Warmup guard
        warmup = max(pattern["ma_period"] + 5, pattern["rsi_period"] + 5, 20)
        if n > warmup:
            valid[:warmup] = False

        rsi_low = pattern["rsi_low"]
        rsi_high = pattern["rsi_high"]

        long_sig = valid & (rsi_vals < rsi_low) & (C_prev < ma_vals) & (C >= ma_vals)
        short_sig = valid & (rsi_vals > rsi_high) & (C_prev > ma_vals) & (C <= ma_vals)

        results = []
        last_kept = -10**9

        long_bars = np.where(long_sig)[0]
        short_bars = np.where(short_sig)[0]

        all_bars = np.concatenate([long_bars, short_bars])
        all_dirs = np.concatenate([
            np.ones(len(long_bars), dtype=int),
            -np.ones(len(short_bars), dtype=int),
        ])
        order = np.argsort(all_bars, kind="stable")
        all_bars = all_bars[order]
        all_dirs = all_dirs[order]

        # Apply embargo
        for idx in range(len(all_bars)):
            b = int(all_bars[idx])
            d = int(all_dirs[idx])

            if b - last_kept < self.EMBARGO_BARS:
                continue

            ent = b + 1
            if ent >= n:
                continue

            ep = O[ent]
            eatr = atr_vals[ent] if np.isfinite(atr_vals[ent]) else 0.0

            if ep <= 0 or eatr <= 1e-12:
                continue

            last_kept = b

            results.append({
                "sig_bar": b,
                "entry_bar": ent,
                "dir": d,
                "entry_price": float(ep),
                "atr": float(eatr),
            })

        return results

    # ── Exit logic ──

    def _check_invalidation(
        self,
        direction: Direction,
        rsi_val: float,
        close_val: float,
        ma_val: float,
        rsi_low: float,
        rsi_high: float,
    ) -> bool:
        if not (np.isfinite(rsi_val) and np.isfinite(ma_val)):
            return False
        if direction == Direction.LONG:
            return rsi_val >= rsi_low and close_val < ma_val
        else:
            return rsi_val <= rsi_high and close_val > ma_val

    def _compute_tp_sl(
        self, direction: Direction, entry: float, atr: float
    ) -> Tuple[float, float]:
        spec = self.EXIT_SPECS.get(self.EXIT_STRATEGY, {})
        tp_r = spec.get("tp_r")
        sl_r = spec.get("sl_r")
        dir_sign = 1 if direction == Direction.LONG else -1

        tp = entry + dir_sign * tp_r * atr if tp_r is not None else 0.0
        sl = entry - dir_sign * sl_r * atr if sl_r is not None else 0.0

        # Hard stop override
        if self.HARD_STOP_R > 0.0 and sl == 0.0:
            sl = entry - dir_sign * self.HARD_STOP_R * atr

        return float(tp), float(sl)

    # ── StrategyBase interface ──

    def scan(self, data_feed, symbols: List[str]) -> List[Signal]:
        all_signals = []

        for sym in symbols:
            if not self.symbol_allowed(sym):
                continue
            for pat in self.PATTERNS:
                tf = pat["tf"]
                df = data_feed.get(sym, tf)

                if df is None or len(df) < 220:
                    continue

                # Work on completed bars only (drop current forming bar)
                lookback = max(pat["ma_period"] + pat["rsi_period"] + 100, 300)
                df_closed = df.iloc[:-1].tail(lookback).copy()

                if df_closed is None or len(df_closed) < 220:
                    continue

                raw = self._detect_signals(df_closed, pat)

                if not raw:
                    continue

                # Only take the most recent signal
                latest = raw[-1]
                sig_bar_idx = latest["sig_bar"]

                # Must be on the most recent completed bar
                if sig_bar_idx != len(df_closed) - 1:
                    continue

                # Deduplicate
                dedup_key = f"{sym}_{pat['name']}"
                sig_ts = df_closed.index[sig_bar_idx]
                sig_ts_str = str(sig_ts)

                if self._last_signal_ts.get(dedup_key) == sig_ts_str:
                    continue

                # Embargo check against last entry
                last_entry = self._last_entry_ts.get(dedup_key)
                if last_entry:
                    try:
                        last_entry_ts = pd.Timestamp(last_entry)
                        embargo_td = timedelta(minutes=self.EMBARGO_BARS * 5)
                        if pd.Timestamp(sig_ts) <= last_entry_ts + embargo_td:
                            continue
                    except Exception:
                        pass

                direction = Direction.LONG if latest["dir"] > 0 else Direction.SHORT
                entry_price = latest["entry_price"]
                atr_val = latest["atr"]

                tp, sl = self._compute_tp_sl(direction, entry_price, atr_val)

                ts_dt = sig_ts.to_pydatetime()
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)

                hour = int(sig_ts.hour)
                sess = session_name(hour)

                sig = Signal(
                    strategy=self.strategy_name,
                    symbol=sym,
                    direction=direction,
                    entry_price=entry_price,
                    atr=atr_val,
                    stop_price=sl,
                    tp_price=tp,
                    signal_bar_time=ts_dt,
                    meta={
                        "pattern": pat["name"],
                        "tf": tf,
                        "ma_type": pat["ma_type"],
                        "ma_period": pat["ma_period"],
                        "rsi_low": pat["rsi_low"],
                        "rsi_high": pat["rsi_high"],
                        "rsi_period": pat["rsi_period"],
                        "exit_strategy": self.EXIT_STRATEGY,
                        "session": sess,
                        "hour_utc": hour,
                        "hold_bars": self.MAX_HOLD_BARS,
                    },
                )

                self._last_signal_ts[dedup_key] = sig_ts_str
                self._last_entry_ts[dedup_key] = sig_ts_str
                all_signals.append(sig)

        return all_signals

    def manage_exits(
        self, open_positions: Dict[str, Any], data_feed
    ) -> List[ExitAction]:
        actions = []

        for oid, pos in open_positions.items():
            if pos.strategy != self.strategy_name:
                continue

            pattern_name = pos.meta.get("pattern", "")
            tf = pos.meta.get("tf", "M5")
            exit_strat = pos.meta.get("exit_strategy", self.EXIT_STRATEGY)
            rsi_low = pos.meta.get("rsi_low", 35.0)
            rsi_high = pos.meta.get("rsi_high", 65.0)
            rsi_period = pos.meta.get("rsi_period", 14)
            ma_type = pos.meta.get("ma_type", "EMA")
            ma_period = pos.meta.get("ma_period", 50)
            hold_bars = pos.meta.get("hold_bars", self.MAX_HOLD_BARS)

            spec = self.EXIT_SPECS.get(exit_strat, {})
            use_inv = spec.get("invalidation", False)

            df = data_feed.get(pos.symbol, tf)
            if df is None or len(df) < 50:
                continue

            # Get current bid/ask for observed exit prices
            bid, ask = None, None
            try:
                import MetaTrader5 as mt5
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick:
                    bid, ask = float(tick.bid), float(tick.ask)
            except Exception:
                pass

            if bid is None or ask is None:
                last_close = float(df["close"].iloc[-1])
                bid = last_close
                ask = last_close

            exit_price = None
            reason = None

            # 1. TP hit
            if pos.tp_price > 0.0:
                if pos.direction == Direction.LONG and bid >= pos.tp_price:
                    exit_price = bid
                    reason = "tp"
                elif pos.direction == Direction.SHORT and ask <= pos.tp_price:
                    exit_price = ask
                    reason = "tp"

            # 2. SL hit
            if exit_price is None and pos.stop_price > 0.0:
                if pos.direction == Direction.LONG and bid <= pos.stop_price:
                    exit_price = bid
                    reason = "sl"
                elif pos.direction == Direction.SHORT and ask >= pos.stop_price:
                    exit_price = ask
                    reason = "sl"

            # 3. Invalidation on newly closed bars
            if exit_price is None and use_inv:
                full_closed = df.iloc[:-1]
                if full_closed is not None and len(full_closed) > 50:
                    last_inv_ts = pos.meta.get("last_inv_bar_ts")
                    if last_inv_ts is not None:
                        new_bars = full_closed[full_closed.index > last_inv_ts]
                    else:
                        new_bars = full_closed.tail(1)

                    if len(new_bars) > 0:
                        # Update hold bars
                        pos.meta["hold_bars_elapsed"] = (
                            pos.meta.get("hold_bars_elapsed", 0) + len(new_bars)
                        )
                        pos.meta["last_inv_bar_ts"] = new_bars.index[-1]

                        # Compute indicators on latest closed bar
                        lookback = max(ma_period + rsi_period + 80, 300)
                        df_closed = full_closed.tail(lookback).copy()

                        ma_s, rsi_s, _ = self._compute_indicators(
                            df_closed, ma_type, ma_period, rsi_period
                        )

                        if len(ma_s) > 0 and len(rsi_s) > 0:
                            ma_val = float(ma_s.iloc[-1])
                            rsi_val = float(rsi_s.iloc[-1])
                            close_val = float(df_closed["close"].iloc[-1])

                            if self._check_invalidation(
                                pos.direction, rsi_val, close_val,
                                ma_val, rsi_low, rsi_high,
                            ):
                                exit_price = bid if pos.direction == Direction.LONG else ask
                                reason = "invalidation"

            # 4. Max hold
            if exit_price is None:
                elapsed = pos.meta.get("hold_bars_elapsed", 0)
                if elapsed >= hold_bars:
                    exit_price = bid if pos.direction == Direction.LONG else ask
                    reason = "max_hold"

            if exit_price is not None and reason is not None:
                actions.append(ExitAction(
                    oid=oid,
                    reason=reason,
                    price=float(exit_price),
                ))

        return actions
