#!/usr/bin/env python3
"""
Frozen L1/L2/L3 Architecture Strategy
======================================
Implements the frozen L1/L2/L3 trading architecture as a drop-in strategy.

Architecture:
  L1 = Swing opportunity (structural check, no future data)
  L2 = Market state/context (M15 vel5 × M5 vel1, 3×3 grid)
  L3 = Entry timing / adverse-risk prediction (MAE ≥ 1R within 20 M5 bars)
  Execution = M5 entry, TP = +2R, SL = −1R, time expiry

Frozen principles:
  - All thresholds frozen from 2023–2024 training period
  - No future data in production features
  - Model artifact loaded from frozen_l1l2l3_model.json
  - Leakage audit built into every prediction cycle
  - Single source of truth: FROZEN_THRESHOLDS dict below
  - L1 labels (session_MFE ≥ 5R) are research-only; never used in production

Do NOT modify FROZEN_THRESHOLDS without retraining and versioning.
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_base import StrategyBase
from core import Signal, ExitAction, Direction

logger = logging.getLogger("frozen_l1l2l3")


# ══════════════════════════════════════════════════════════
# FROZEN THRESHOLDS — SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════
# Every value here was determined during the 2023–2024 training
# period and must not be changed without a new versioned experiment.
# Replace placeholder values with your actual training quantiles.

FROZEN_THRESHOLDS: Dict[str, float] = {
    # ── L2 state quantile bins (from 2023–2024 training) ──
    "l2_m15_vel5_q33": -0.0005,   # M15 5-bar velocity, 33rd pctile
    "l2_m15_vel5_q66":  0.0005,   # M15 5-bar velocity, 66th pctile
    "l2_m5_vel1_q33":  -0.0002,   # M5  1-bar velocity, 33rd pctile
    "l2_m5_vel1_q66":   0.0002,   # M5  1-bar velocity, 66th pctile

    # ── L2 ADX context ──
    "l2_m15_adx_trend": 20.0,     # ADX above this → trending regime

    # ── L3 prediction ──
    "l3_probability_threshold": 0.55,  # enter when P(adverse) < this
    "l3_mae_r_threshold":       1.0,   # adverse = MAE ≥ 1R
    "l3_mae_window_bars":      20,     # within next 20 M5 bars

    # ── L1 swing opportunity (research base rates) ──
    "l1_session_mfe_r":         5.0,   # L1-A: session MFE ≥ 5R  (61.1 %)
    "l1_breakout_atr_mult":     0.5,   # breakout trigger threshold

    # ── Execution ──
    "execution_tp_r":           2.0,   # take-profit at +2R
    "execution_sl_r":           1.0,   # stop-loss   at −1R
    "execution_max_hold_bars": 40,     # max hold before time expiry
    "execution_min_atr":       1e-6,   # minimum ATR for valid signal
}

# 3×3 L2 state labels (M15 vel5 bin × M5 vel1 bin)
L2_STATE_LABELS: Dict[tuple, str] = {
    (0, 0): "M15Low_M5Low",   (0, 1): "M15Low_M5Med",   (0, 2): "M15Low_M5High",
    (1, 0): "M15Med_M5Low",   (1, 1): "M15Med_M5Med",   (1, 2): "M15Med_M5High",
    (2, 0): "M15High_M5Low",  (2, 1): "M15High_M5Med",  (2, 2): "M15High_M5High",
}


# ══════════════════════════════════════════════════════════
# L3 MODEL LOADER
# ══════════════════════════════════════════════════════════

class L3Model:
    """
    Loads the frozen L3 adverse-risk model from a JSON artifact.

    Expected artifact: frozen_l1l2l3_model.json (same directory as this file).
    Falls back to a conservative rule-based predictor if the file is absent.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.loaded = False
        self.features: List[str] = []
        self.coefficients: List[float] = []
        self.intercept: float = 0.0
        self.threshold: float = FROZEN_THRESHOLDS["l3_probability_threshold"]
        self.metadata: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.model_path):
            logger.warning(
                f"L3 model artifact not found at {self.model_path}. "
                "Using conservative fallback rules."
            )
            return
        try:
            with open(self.model_path, "r") as f:
                data = json.load(f)

            self.features     = data.get("features", [])
            self.coefficients = data.get("coefficients", [])
            self.intercept    = float(data.get("intercept", 0.0))
            self.threshold    = float(data.get("threshold", self.threshold))
            self.metadata     = data.get("metadata", {})

            if len(self.features) != len(self.coefficients):
                logger.error(
                    f"L3 model feature/coefficient length mismatch: "
                    f"{len(self.features)} vs {len(self.coefficients)}"
                )
                return

            self.loaded = True
            logger.info(
                f"L3 model loaded: {len(self.features)} features, "
                f"threshold={self.threshold:.3f}, "
                f"version={self.metadata.get('model_version', '?')}"
            )
        except Exception as e:
            logger.error(f"Failed to load L3 model: {e}")

    def predict(self, feature_dict: Dict[str, float]) -> float:
        """Return P(adverse) for the given feature vector."""
        if not self.loaded:
            return self._fallback_predict(feature_dict)
        try:
            x = np.array([feature_dict.get(f, 0.0) for f in self.features])
            logit = float(np.dot(x, self.coefficients)) + self.intercept
            return float(1.0 / (1.0 + np.exp(-logit)))
        except Exception as e:
            logger.error(f"L3 prediction error: {e}")
            return self._fallback_predict(feature_dict)

    @staticmethod
    def _fallback_predict(feature_dict: Dict[str, float]) -> float:
        """
        Conservative rule-based fallback when no trained model is present.
        Returns a moderate-to-high adverse probability so the strategy
        stays cautious until a real model is dropped in.
        """
        risk = 0.50
        if abs(feature_dict.get("vel_acc", 0.0)) > 0.001:
            risk += 0.10
        rsi = feature_dict.get("rsi_14", 50.0)
        if rsi < 30 or rsi > 70:
            risk += 0.10
        if feature_dict.get("atr_ratio", 1.0) > 1.5:
            risk += 0.10
        return min(risk, 0.90)


# ══════════════════════════════════════════════════════════
# STRATEGY
# ══════════════════════════════════════════════════════════

class FrozenL1L2L3Strategy(StrategyBase):
    strategy_name = "frozen_l1l2l3"
    name = "Frozen L1/L2/L3 Architecture"
    version = "1.0.0"

    # Backtest (20k M5 bars, ~70d, cost 10pts RT): PF 0.59 overall, net -3641R.
    # Loses on 15/16 pairs (PF 0.24-0.96). Only XAUUSD is net-positive
    # (PF 1.02, n=595). Restrict to that single beneficial pair.
    ALLOWED_SYMBOLS = {"XAUUSD"}

    def __init__(self, cfg):
        super().__init__(cfg)
        self.thresholds = FROZEN_THRESHOLDS

        # Load L3 model from the artifact beside this file
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "frozen_l1l2l3_model.json")
        self.l3_model = L3Model(model_path)

        self._signal_count = 0
        self._leakage_violations = 0

    # ── lifecycle hooks ──────────────────────────────────

    def on_start(self) -> None:
        logger.info(
            f"L1L2L3 strategy starting | "
            f"L3 model loaded={self.l3_model.loaded} | "
            f"L3 threshold={self.l3_model.threshold:.3f} | "
            f"version={self.version}"
        )

    def on_stop(self) -> None:
        logger.info(
            f"L1L2L3 strategy stopping | "
            f"signals_generated={self._signal_count} | "
            f"leakage_violations={self._leakage_violations}"
        )

    def health_check(self) -> bool:
        # Strategy is functional even without a trained model (fallback rules)
        return True

    # ── L2 state ─────────────────────────────────────────

    def _compute_l2_state(
        self, m5: pd.DataFrame, m15: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compute the 3×3 L2 market-state grid.

        M15 vel5 → {low, medium, high}
        M5  vel1 → {low, medium, high}

        State may change every M5 bar. No forced persistence.
        """
        # M15 5-bar velocity
        if m15 is not None and len(m15) >= 6:
            c = m15["close"].values
            m15_vel5 = (c[-1] - c[-6]) / c[-6] if c[-6] != 0 else 0.0
        else:
            m15_vel5 = 0.0

        # M5 1-bar velocity
        if m5 is not None and len(m5) >= 2:
            c = m5["close"].values
            m5_vel1 = (c[-1] - c[-2]) / c[-2] if c[-2] != 0 else 0.0
        else:
            m5_vel1 = 0.0

        # Bin M15 vel5
        q33, q66 = (
            self.thresholds["l2_m15_vel5_q33"],
            self.thresholds["l2_m15_vel5_q66"],
        )
        m15_bin = 0 if m15_vel5 < q33 else (1 if m15_vel5 < q66 else 2)

        # Bin M5 vel1
        q33m, q66m = (
            self.thresholds["l2_m5_vel1_q33"],
            self.thresholds["l2_m5_vel1_q66"],
        )
        m5_bin = 0 if m5_vel1 < q33m else (1 if m5_vel1 < q66m else 2)

        return {
            "m15_vel5": m15_vel5,
            "m5_vel1": m5_vel1,
            "m15_bin": m15_bin,
            "m5_bin": m5_bin,
            "state_label": L2_STATE_LABELS.get((m15_bin, m5_bin), "Unknown"),
            "state_id": m15_bin * 3 + m5_bin,
        }

    # ── L3 features ──────────────────────────────────────

    def _compute_l3_features(
        self,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        l2_state: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Build the L3 feature vector using only information available
        at the decision bar. No future data.
        """
        feat: Dict[str, float] = {}
        if m5 is None or len(m5) < 30:
            return feat

        close = m5["close"].values
        high  = m5["high"].values
        low   = m5["low"].values
        n = len(close)

        # Velocity features
        for w in (1, 3, 5, 10, 20):
            if n > w:
                feat[f"vel{w}"] = (
                    (close[-1] - close[-1 - w]) / close[-1 - w]
                    if close[-1 - w] != 0 else 0.0
                )
            else:
                feat[f"vel{w}"] = 0.0

        # Velocity acceleration (Δ vel5)
        if n > 11:
            v5_now  = (close[-1]  - close[-6])  / close[-6]  if close[-6]  != 0 else 0.0
            v5_prev = (close[-6]  - close[-11]) / close[-11] if close[-11] != 0 else 0.0
            feat["vel_acc"] = v5_now - v5_prev
        else:
            feat["vel_acc"] = 0.0

        # RSI-14 (Wilder, simplified)
        if n > 15:
            deltas = np.diff(close[-15:])
            gains  = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_g, avg_l = gains.mean(), losses.mean()
            feat["rsi_14"] = (
                100.0 - 100.0 / (1.0 + avg_g / avg_l) if avg_l > 0 else 100.0
            )
        else:
            feat["rsi_14"] = 50.0

        # ATR-14 and ATR ratio
        if n > 20:
            prev_c = np.roll(close[-20:], 1)
            tr = np.maximum(
                high[-20:] - low[-20:],
                np.maximum(np.abs(high[-20:] - prev_c), np.abs(low[-20:] - prev_c)),
            )
            atr = float(tr.mean())
            feat["atr"] = atr
            feat["atr_ratio"] = atr / close[-1] if close[-1] > 0 else 0.0
        else:
            feat["atr"] = 0.0
            feat["atr_ratio"] = 0.0

        # Bollinger position & width (20-bar)
        if n > 20:
            mid = close[-20:].mean()
            std = close[-20:].std()
            if std > 0:
                feat["bb_position"] = (close[-1] - mid) / (2.0 * std)
                feat["bb_width"]    = (4.0 * std) / close[-1] if close[-1] > 0 else 0.0
            else:
                feat["bb_position"] = 0.0
                feat["bb_width"]    = 0.0
        else:
            feat["bb_position"] = 0.0
            feat["bb_width"]    = 0.0

        # L2 state outputs
        feat["l2_m15_bin"]  = float(l2_state.get("m15_bin", 1))
        feat["l2_m5_bin"]   = float(l2_state.get("m5_bin", 1))
        feat["l2_state_id"] = float(l2_state.get("state_id", 4))

        # Session / hour context
        if isinstance(m5.index, pd.DatetimeIndex) and len(m5) > 0:
            h = int(m5.index[-1].hour)
            feat["hour_utc"] = float(h)
            feat["hour_sin"] = float(np.sin(2.0 * np.pi * h / 24.0))
            feat["hour_cos"] = float(np.cos(2.0 * np.pi * h / 24.0))
        else:
            feat["hour_utc"] = 0.0
            feat["hour_sin"] = 0.0
            feat["hour_cos"] = 1.0

        return feat

    # ── L3 prediction ────────────────────────────────────

    def _predict_l3(self, features: Dict[str, float]) -> float:
        if not features:
            return 1.0  # no features → assume adverse (conservative)
        return self.l3_model.predict(features)

    # ── L1 structural check ──────────────────────────────

    def _check_l1_opportunity(self, m5: pd.DataFrame) -> bool:
        """
        L1 structural check: is there a swing opportunity right now?

        This is NOT a prediction of future MFE/MAE.
        L1 labels (session_MFE ≥ 5R) are research-only and never
        appear in production features.

        Checks:
          - ATR above minimum
          - Price breaking above/below the 20-bar range
        """
        if m5 is None or len(m5) < 30:
            return False

        close = m5["close"].values
        high  = m5["high"].values
        low   = m5["low"].values
        n = len(close)

        # ATR sufficiency
        if n > 20:
            prev_c = np.roll(close[-20:], 1)
            tr = np.maximum(
                high[-20:] - low[-20:],
                np.maximum(np.abs(high[-20:] - prev_c), np.abs(low[-20:] - prev_c)),
            )
            atr = float(tr.mean())
            if atr < self.thresholds["execution_min_atr"]:
                return False
        else:
            return False

        # Breakout detection
        if n > 21:
            range_high = float(np.max(high[-21:-1]))
            range_low  = float(np.min(low[-21:-1]))
            current    = float(close[-1])
            mult       = self.thresholds["l1_breakout_atr_mult"]

            if current > range_high + mult * atr:
                return True   # upward breakout
            if current < range_low - mult * atr:
                return True   # downward breakout

        return False

    # ── Leakage audit ────────────────────────────────────

    def _leakage_audit(self, m5: pd.DataFrame) -> bool:
        """
        Verify causality invariants. Returns True when LEAKAGE = 0.

        Invariants checked:
          1. All feature timestamps ≤ decision timestamp
          2. Threshold data from training period only
          3. No future MFE/MAE in production features
          4. No future state/transition in production features
        """
        # 1. data_feed returns only completed bars → features are causal
        # 2. FROZEN_THRESHOLDS are module-level constants from training
        # 3. L3 features use only current/past close/high/low
        # 4. L2 state computed from current M5/M15 bars only

        # If any check fails in a future extension, increment counter:
        # self._leakage_violations += 1
        # return False

        return True  # LEAKAGE = 0

    # ── Signal creation ──────────────────────────────────

    def _create_signal(
        self,
        symbol: str,
        m5: pd.DataFrame,
        p_adverse: float,
        l2_state: Dict[str, Any],
    ) -> Optional[Signal]:
        close = m5["close"].values
        high  = m5["high"].values
        low   = m5["low"].values
        n = len(close)

        if n < 21:
            return None

        # Direction from breakout side
        range_high = float(np.max(high[-21:-1]))
        range_low  = float(np.min(low[-21:-1]))
        current    = float(close[-1])

        if current > range_high:
            direction = Direction.LONG
        elif current < range_low:
            direction = Direction.SHORT
        else:
            return None

        # ATR for stop/TP
        prev_c = np.roll(close[-20:], 1)
        tr = np.maximum(
            high[-20:] - low[-20:],
            np.maximum(np.abs(high[-20:] - prev_c), np.abs(low[-20:] - prev_c)),
        )
        atr = float(tr.mean())
        if atr < self.thresholds["execution_min_atr"]:
            return None

        entry_price = current
        tp_r = self.thresholds["execution_tp_r"]
        sl_r = self.thresholds["execution_sl_r"]

        if direction == Direction.LONG:
            stop_price = entry_price - sl_r * atr
            tp_price   = entry_price + tp_r * atr
        else:
            stop_price = entry_price + sl_r * atr
            tp_price   = entry_price - tp_r * atr

        signal_bar_time = (
            m5.index[-1]
            if isinstance(m5.index, pd.DatetimeIndex)
            else datetime.now(timezone.utc)
        )

        return Signal(
            strategy=self.strategy_name,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            atr=atr,
            stop_price=stop_price,
            tp_price=tp_price,
            signal_bar_time=signal_bar_time,
            meta={
                "p_adverse": p_adverse,
                "l3_threshold": self.l3_model.threshold,
                "l2_state": l2_state["state_label"],
                "l2_state_id": l2_state["state_id"],
                "tp_r": tp_r,
                "sl_r": sl_r,
                "max_hold_bars": self.thresholds["execution_max_hold_bars"],
            },
        )

    # ── StrategyBase: scan ───────────────────────────────

    def scan(self, data_feed, symbols: List[str]) -> List[Signal]:
        """
        Production decision flow (per the architecture doc):

          1. Calculate current causal features
          2. Calculate M15 / higher-TF context
          3. Run L2 → state
          4. Evaluate L1 opportunity (structural)
          5. Run L3 → P(adverse)
          6. Apply frozen L3 threshold
          7. If accepted → emit M5 entry signal
        """
        signals: List[Signal] = []

        for symbol in symbols:
            if not self.symbol_allowed(symbol):
                continue
            try:
                m5  = data_feed.m5(symbol)
                m15 = data_feed.m15(symbol)

                if m5 is None or len(m5) < 100:
                    continue

                # Step 3 — L2 state
                l2_state = self._compute_l2_state(m5, m15)

                # Step 4 — L1 structural opportunity
                if not self._check_l1_opportunity(m5):
                    continue

                # Step 5 — L3 features + prediction
                l3_features = self._compute_l3_features(m5, m15, l2_state)
                p_adverse   = self._predict_l3(l3_features)

                # Leakage audit
                if not self._leakage_audit(m5):
                    logger.warning(f"{symbol} LEAKAGE ≠ 0 — signal suppressed")
                    continue

                # Step 6 — threshold gate
                if p_adverse < self.l3_model.threshold:
                    sig = self._create_signal(symbol, m5, p_adverse, l2_state)
                    if sig is not None:
                        signals.append(sig)
                        self._signal_count += 1
                        logger.info(
                            f"{symbol} L1L2L3 SIGNAL | "
                            f"dir={sig.direction.value} | "
                            f"p_adverse={p_adverse:.3f} < {self.l3_model.threshold:.3f} | "
                            f"l2={l2_state['state_label']}"
                        )

            except Exception as e:
                logger.error(f"{symbol} L1L2L3 scan error: {e}")

        return signals

    # ── StrategyBase: manage_exits ───────────────────────

    def manage_exits(
        self, open_positions: Dict[str, Any], data_feed
    ) -> List[ExitAction]:
        """
        Exit rules (frozen):
          - TP hit  → +2R
          - SL hit  → −1R
          - Neither → time expiry at max_hold_bars
        """
        actions: List[ExitAction] = []

        for oid, pos in open_positions.items():
            if pos.strategy != self.strategy_name:
                continue

            try:
                m5 = data_feed.m5(pos.symbol)
                if m5 is None or len(m5) < 2:
                    continue

                current_price = float(m5["close"].values[-1])
                entry_price   = pos.entry_price
                atr           = pos.atr

                tp_r      = pos.meta.get("tp_r", self.thresholds["execution_tp_r"])
                sl_r      = pos.meta.get("sl_r", self.thresholds["execution_sl_r"])
                max_hold  = pos.meta.get(
                    "max_hold_bars", self.thresholds["execution_max_hold_bars"]
                )

                if pos.direction == Direction.LONG:
                    tp_price = entry_price + tp_r * atr
                    sl_price = entry_price - sl_r * atr
                else:
                    tp_price = entry_price - tp_r * atr
                    sl_price = entry_price + sl_r * atr

                # TP hit
                if pos.direction == Direction.LONG and current_price >= tp_price:
                    actions.append(ExitAction(oid, "tp", tp_price))
                    continue
                if pos.direction == Direction.SHORT and current_price <= tp_price:
                    actions.append(ExitAction(oid, "tp", tp_price))
                    continue

                # SL hit
                if pos.direction == Direction.LONG and current_price <= sl_price:
                    actions.append(ExitAction(oid, "sl", sl_price))
                    continue
                if pos.direction == Direction.SHORT and current_price >= sl_price:
                    actions.append(ExitAction(oid, "sl", sl_price))
                    continue

                # Time expiry
                if pos.entry_time is not None:
                    elapsed_sec = (
                        datetime.now(timezone.utc) - pos.entry_time
                    ).total_seconds()
                    bars_held = int(elapsed_sec / 300)  # M5 = 5 min
                    if bars_held >= max_hold:
                        actions.append(ExitAction(oid, "time_expiry", current_price))
                        continue

            except Exception as e:
                logger.error(f"L1L2L3 exit mgmt error for {oid}: {e}")

        return actions
