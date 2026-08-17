#!/usr/bin/env python3
"""
BB-Squeeze Breakout Strategy
=============================
Primary combo: M30_bb + M30_vr
Execution/sizing ATR: M5 ONLY

Ported from retf.py into the StrategyBase interface.
Preserves all original logic:
- Bug A fix: M30 lookahead prevention via time-slicing
- Bug B fix: Chronological session risk tracking
- Bug C fix: Live tail embargo clearing
- Pre-decision stop tagging
- Quantile calibration embedded (no external JSON)
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_base import StrategyBase

# core.py imports (Signal, ExitAction, Direction are defined there)
from core import Signal, ExitAction, Direction

# features.py imports
from features import (
    atr_series,
    bb_ratio,
    rolling_high,
    rolling_low,
    vol_regime,
    session_name,
    resample_ohlcv,
)


class BBSqueezeStrategy(StrategyBase):
    strategy_name = "bb_squeeze"
    name = "BB-Squeeze Breakout"
    version = "7.0.0"

    # Backtest (80k M5 bars, ~280d, cost 10pts RT): PF 2.26 overall, net +472R.
    # Beneficial on nearly every pair; only SP500 lost (n=6, PF 0.67). Restrict
    # to the proven-beneficial pairs (exclude SP500).
    ALLOWED_SYMBOLS = {
        "AUDUSD", "CAC40", "CHFJPY", "EURGBP", "EURJPY", "EURUSD",
        "FTSE100", "GBPJPY", "GBPUSD", "NAS100", "NZDUSD", "USDCAD",
        "USDCHF", "USDJPY", "XAUUSD",
    }

    # ══════════════════════════════════════════════════════
    # EMBEDDED CALIBRATION BINS
    # Single source of truth — no external JSON.
    # ══════════════════════════════════════════════════════
    CALIBRATION = {
        "bb_ratio": {
            "labels": ["Loose", "Medium", "Tight"],
            "bins": [5.59744e-05, 1.5302320333, 1.79991342, 1.9999664605],
        },
        "vol_regime": {
            "labels": ["Low Vol", "Med Vol", "High Vol"],
            "bins": [0.0254095629, 0.902792942, 1.088520753, 206.4265001307],
        },
        "breakout_str": {
            "labels": ["Weak BS", "Med BS", "Strong BS"],
            "bins": [0.0019719145037118, 0.11208060666465787,
                     0.28578183326691986, 7.885800714565764],
        },
    }

    # ── Frozen research parameters ──
    BB_PERIOD = 20
    ATR_ALPHA = 1.0 / 14.0
    SQUEEZE_MAX_RATIO_M5 = 2.0
    BREAKOUT_TRIGGER_ATR = 0.5
    BREAKOUT_LOOKAHEAD = 5
    TRIGGER_EMBARGO_BARS = 20
    DECISION_BARS = 10
    HOLD_BARS = 40
    STOP_R = 1.0

    # Session filtering
    SESSION_BLOCK = {"NY-PM", "Off"}
    HOUR_ALLOW_NY = {12, 13, 14, 15, 16}
    HOUR_ALLOW_LONDON = {7, 8, 9, 10, 11}
    HOUR_ALLOW_ASIAN = {0, 1, 2, 3, 4, 5, 6}
    LONDON_CAUTION = True

    # M30 filter thresholds
    M30_VR_ALLOW = {"Low Vol", "Med Vol"}
    M30_VR_NY_ALLOW_HIGH = True
    M30_BB_MAX = 2.5
    M30_VR_LOOKBACK = 50

    # Squeeze filter
    SQ_ALLOW = {"Medium", "Tight"}
    SQ_NY_ALLOW_LOOSE = True

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._last_exit: Dict[str, int] = {}
        self._last_signal_ts: Dict[str, str] = {}

    # ── Quantile classifier ──

    def _classify(self, col: str, value: float) -> str:
        if col not in self.CALIBRATION or not np.isfinite(value):
            return "Unknown"
        bins = self.CALIBRATION[col]["bins"]
        labels = self.CALIBRATION[col]["labels"]
        if len(bins) < 2:
            return "Unknown"
        if value <= bins[0]:
            return labels[0]
        if value >= bins[-1]:
            return labels[-1]
        idx = int(np.searchsorted(bins[1:-1], value))
        return labels[min(idx, len(labels) - 1)]

    # ── Session check ──

    def _session_allowed(self, hour_utc: int, sess: str) -> bool:
        if sess in self.SESSION_BLOCK:
            return False
        allowed = None
        if sess == "NY":
            allowed = self.HOUR_ALLOW_NY
        elif sess == "London":
            allowed = self.HOUR_ALLOW_LONDON
        elif sess == "Asian":
            allowed = self.HOUR_ALLOW_ASIAN
        if allowed is not None and hour_utc not in allowed:
            return False
        return True

    # ── M30 filter (Bug A fix: time-sliced) ──

    def _apply_m30_filter(
        self,
        m30: pd.DataFrame,
        signal_ts: datetime,
        hour_utc: int,
        sess: str,
        breakout_str: float,
    ) -> Dict[str, Any]:
        result = {
            "passed": False,
            "verdict": "block",
            "reason": "",
            "m30_bb": np.nan,
            "m30_vr": np.nan,
            "vr_label": "Unknown",
        }

        if not self._session_allowed(hour_utc, sess):
            result["reason"] = f"session={sess} blocked"
            return result

        if m30 is None or len(m30) < 60:
            result["reason"] = "M30 data insufficient"
            return result

        # Bug A fix: slice M30 up to signal time
        try:
            if isinstance(m30.index, pd.DatetimeIndex) and signal_ts is not None:
                sig_ts = signal_ts
                if m30.index.tzinfo is not None and sig_ts.tzinfo is None:
                    sig_ts = sig_ts.replace(tzinfo=timezone.utc)
                elif m30.index.tzinfo is None and sig_ts.tzinfo is not None:
                    sig_ts = sig_ts.replace(tzinfo=None)
                m30_slice = m30[m30.index <= sig_ts]
            else:
                m30_slice = m30
        except Exception:
            m30_slice = m30

        if len(m30_slice) < 60:
            result["reason"] = "M30 data insufficient at signal time"
            return result

        m30_atr = atr_series(m30_slice, alpha=self.ATR_ALPHA)
        m30_bb = bb_ratio(m30_slice, m30_atr, self.BB_PERIOD)
        m30_vr = vol_regime(m30_atr, lookback=self.M30_VR_LOOKBACK)

        last_bb = float(m30_bb.iloc[-1]) if len(m30_bb) > 0 else np.nan
        last_vr = float(m30_vr.iloc[-1]) if len(m30_vr) > 0 else np.nan

        result["m30_bb"] = last_bb
        result["m30_vr"] = last_vr

        # M30 BB max check
        if not np.isfinite(last_bb) or last_bb >= self.M30_BB_MAX:
            result["reason"] = f"M30 bb_ratio={last_bb:.3f} >= {self.M30_BB_MAX}"
            return result

        # Vol regime label
        vr_label = self._classify("vol_regime", last_vr)
        result["vr_label"] = vr_label

        if vr_label == "Unknown":
            result["reason"] = "vol_regime unknown/calibration missing"
            return result

        if vr_label not in self.M30_VR_ALLOW:
            if sess == "NY" and self.M30_VR_NY_ALLOW_HIGH and vr_label == "High Vol":
                pass  # NY exception
            else:
                result["reason"] = f"M30_vr={vr_label} not allowed"
                return result

        # Squeeze filter
        sq_label = self._classify("bb_ratio", last_bb)
        if sq_label == "Unknown":
            result["reason"] = "squeeze unknown/calibration missing"
            return result

        if sq_label not in self.SQ_ALLOW:
            if sess == "NY" and self.SQ_NY_ALLOW_LOOSE and sq_label == "Loose":
                pass  # NY exception
            else:
                result["reason"] = f"sq={sq_label} not allowed"
                return result

        # Verdict
        verdict = "pass"
        if sess == "London" and self.LONDON_CAUTION:
            verdict = "caution"

        result["passed"] = True
        result["verdict"] = verdict
        result["reason"] = (
            f"M30_bb={last_bb:.2f}; M30_vr={vr_label}; sq={sq_label}"
        )
        return result

    # ── Signal detection ──

    def _detect_signals(self, m5: pd.DataFrame, symbol: str) -> List[Signal]:
        if m5 is None or len(m5) < 100:
            return []

        C = m5["close"].astype(float).values
        H = m5["high"].astype(float).values
        L = m5["low"].astype(float).values
        O = m5["open"].astype(float).values
        n = len(C)

        atr = atr_series(m5, alpha=self.ATR_ALPHA).values
        bb = bb_ratio(m5, atr_series(m5, alpha=self.ATR_ALPHA), self.BB_PERIOD).values
        hh20 = rolling_high(m5, self.BB_PERIOD).values
        ll20 = rolling_low(m5, self.BB_PERIOD).values

        has_dt_index = isinstance(m5.index, pd.DatetimeIndex)
        signals = []
        last_exit = self._last_exit.get(symbol, 0)
        scan_start = max(self.BB_PERIOD + 50, 50)

        for i in range(scan_start, n):
            if not np.isfinite(bb[i]) or bb[i] <= 0 or bb[i] >= self.SQUEEZE_MAX_RATIO_M5:
                continue
            if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
                continue

            trig = -1
            direction = Direction.LONG
            thresh_up = C[i] + self.BREAKOUT_TRIGGER_ATR * atr[i]
            for j in range(1, self.BREAKOUT_LOOKAHEAD + 1):
                jj = i + j
                if jj < n and np.isfinite(H[jj]) and H[jj] >= thresh_up:
                    trig = jj
                    break

            if trig == -1:
                thresh_dn = C[i] - self.BREAKOUT_TRIGGER_ATR * atr[i]
                for j in range(1, self.BREAKOUT_LOOKAHEAD + 1):
                    jj = i + j
                    if jj < n and np.isfinite(L[jj]) and L[jj] <= thresh_dn:
                        trig = jj
                        break
                direction = Direction.SHORT

            if trig == -1:
                continue

            ent = trig + 1
            if ent >= n or ent < last_exit + self.TRIGGER_EMBARGO_BARS:
                continue

            ep = O[ent]
            eatr = atr[ent]
            if ep <= 0 or eatr <= 1e-12:
                continue

            if direction == Direction.LONG:
                stop = ep - self.STOP_R * eatr
                bs = (H[i] - hh20[i]) / eatr if np.isfinite(hh20[i]) else 0.0
            else:
                stop = ep + self.STOP_R * eatr
                bs = (ll20[i] - L[i]) / eatr if np.isfinite(ll20[i]) else 0.0

            t = ent + self.DECISION_BARS
            fb = t + self.HOLD_BARS
            if fb >= n:
                continue

            # Pre-decision stop tagging (Bug #1 fix)
            pre_stop_bar = None
            for k in range(ent + 1, min(t + 1, n)):
                if direction == Direction.LONG:
                    if np.isfinite(L[k]) and L[k] <= stop:
                        pre_stop_bar = k
                        break
                else:
                    if np.isfinite(H[k]) and H[k] >= stop:
                        pre_stop_bar = k
                        break

            if has_dt_index and i < len(m5.index):
                ts_val = m5.index[i]
                hour = int(ts_val.hour)
                ts = ts_val.to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
            else:
                hour = datetime.now(timezone.utc).hour
                ts = datetime.now(timezone.utc)

            sess = session_name(hour)

            tp = 0.0  # No fixed TP; time-based exit
            sig = Signal(
                strategy=self.strategy_name,
                symbol=symbol,
                direction=direction,
                entry_price=ep,
                atr=eatr,
                stop_price=stop,
                tp_price=tp,
                signal_bar_time=ts,
                meta={
                    "bb_ratio_m5": float(bb[i]),
                    "breakout_str": float(bs),
                    "hour_utc": hour,
                    "session": sess,
                    "bar_idx": ent,
                    "trig_idx": trig,
                    "pre_decision_stop_bar": pre_stop_bar,
                    "hold_bars": self.HOLD_BARS,
                    "decision_bars": self.DECISION_BARS,
                },
            )
            signals.append(sig)
            last_exit = fb
            self._last_exit[symbol] = last_exit

        return signals

    # ── StrategyBase interface ──

    def scan(self, data_feed, symbols: List[str]) -> List[Signal]:
        all_signals = []
        for sym in symbols:
            if not self.symbol_allowed(sym):
                continue
            m5 = data_feed.get(sym, "M5")
            m30 = data_feed.get(sym, "M30")

            if m5 is None or len(m5) < 100:
                continue

            # Bug C fix: clear stale embargo for live tail scanning
            self._last_exit.pop(sym, None)

            raw_signals = self._detect_signals(m5.tail(120).copy(), sym)

            if not raw_signals:
                continue

            sig = raw_signals[-1]

            # Deduplicate: skip if same signal bar already processed
            sig_key = str(sig.signal_bar_time)
            if self._last_signal_ts.get(sym) == sig_key:
                continue

            # Apply M30 filter
            hour = sig.meta.get("hour_utc", 0)
            sess = sig.meta.get("session", "Off")
            bs = sig.meta.get("breakout_str", 0.0)

            filt = self._apply_m30_filter(m30, sig.signal_bar_time, hour, sess, bs)

            if not filt["passed"]:
                continue

            # Skip pre-decision stops
            if sig.meta.get("pre_decision_stop_bar") is not None:
                continue

            sig.meta["m30_bb"] = filt["m30_bb"]
            sig.meta["m30_vr"] = filt["m30_vr"]
            sig.meta["m30_vr_label"] = filt["vr_label"]
            sig.meta["filter_reason"] = filt["reason"]
            sig.meta["filter_verdict"] = filt["verdict"]

            self._last_signal_ts[sym] = sig_key
            all_signals.append(sig)

        return all_signals

    def manage_exits(
        self, open_positions: Dict[str, Any], data_feed
    ) -> List[ExitAction]:
        """
        BB-Squeeze uses broker-side SL (set at entry).
        Time-based exit after HOLD_BARS is checked here.
        """
        actions = []
        now = datetime.now(timezone.utc)

        for oid, pos in open_positions.items():
            if pos.strategy != self.strategy_name:
                continue

            hold_bars = pos.meta.get("hold_bars", self.HOLD_BARS)
            entry_time = pos.entry_time

            if entry_time is None:
                continue

            # Approximate bar count from elapsed time (M5 = 5 min)
            elapsed_min = (now - entry_time).total_seconds() / 60.0
            elapsed_bars = elapsed_min / 5.0

            if elapsed_bars >= hold_bars:
                actions.append(ExitAction(
                    oid=oid,
                    reason="time_exit",
                    price=None,  # market close
                ))

        return actions
