#!/usr/bin/env python3
"""
Sloppy ATR Momentum — ML-Gated Breakout Strategy
==================================================
Trades EURUSD only.

Entry trigger:
  M15 ATR ratio (ATR14 / 100-bar ATR mean) > 1.04
  + momentum direction (close / close[5] - 1)

Confirmation gate:
  Pre-trained RandomForest filter must approve.
  Gate = P(Lb) >= thr[Lb] AND P(plus2) >= thr[plus2]

Execution:
  TP = 1.5R, SL = 1.0R, time exit = 6 M15 bars
  R = 24 pts (EURUSD)
  0.01 lots fixed
  FOK fills with stops-level bump on retcode 10016

Requires:
  features_lib.py in same directory (single source of truth for columns)
  edge_models.pkl per symbol (trained via tf_sweep.py)
"""

import os
import sys
import time
import logging
import pickle
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_base import StrategyBase
from core import Direction, ExitAction

logger = logging.getLogger("sloppy_atr_momentum")

# Add strategies dir to path for features_lib import
_STRAT_DIR = os.path.dirname(os.path.abspath(__file__))
if _STRAT_DIR not in sys.path:
    sys.path.insert(0, _STRAT_DIR)


class SloppyAtrMomentumStrategy(StrategyBase):
    strategy_name = "sloppy_atr_momentum"
    name = "Sloppy ATR Momentum (ML-Gated)"
    version = "1.0.0"
    enabled = True

    # ══════════════════════════════════════════════════════
    # CONFIG
    # ══════════════════════════════════════════════════════
    SYMBOLS = ["EURUSD"]
    MAGIC = 20260811
    VOLUME = 0.01
    DEVIATION = 20

    # Entry parameters
    ATR_PERIOD = 14
    ATR_MA_PERIOD = 100
    MOM_PERIOD = 5
    ATR_RATIO_THRESHOLD = 1.04

    # R-points per symbol
    R_POINTS = {"EURUSD": 24.0, "GBPUSD": 21.0, "NZDUSD": 24.0}

    # Exit geometry
    TP_RATIO = 1.5
    SL_RATIO = 1.0
    TIME_EXIT_BARS = 6  # M15 bars

    # ── Trailing stop (A0.5 / D0.5) ──
    TRAIL_ACTIVATION = 0.5   # R of favorable excursion before trail arms
    TRAIL_DISTANCE   = 0.5   # R retrace from peak that triggers exit

    # Model paths
    MODEL_PATHS = {}  # Set in __init__

    def __init__(self, cfg):
        super().__init__(cfg)
        self.models = {}        # {symbol: loaded model meta}
        self.last_bar_time = {}  # {symbol: last M15 bar time}
        self.positions = {}     # {symbol: position dict}
        self._trail_peak: Dict[str, float] = {}   # oid → peak favorable R
        self._model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models"
        )
        self._setup_model_paths()
        self._load_ml_models()

    def _setup_model_paths(self):
        """Candidate model files per symbol (most-specific first).

        EURUSD may fall back to the generic ``edge_models.pkl`` for legacy
        single-file deployments. GBPUSD / XAUUSD MUST use their own file and
        never inherit the EURUSD bundle — reusing a model trained on a
        different symbol's distribution (and R) destroys the gate edge.
        """
        os.makedirs(self._model_dir, exist_ok=True)
        self.MODEL_CANDIDATES = {
            "EURUSD": ["edge_models_eurusd.pkl", "edge_models.pkl"],
            "GBPUSD": ["edge_models_gbpusd.pkl", "edge_models_gbp.pkl"],
            "XAUUSD": ["edge_models_xauusd.pkl", "edge_models_xau.pkl"],
            "AUDUSD": ["edge_models_audusd.pkl", "edge_models_aud.pkl"],
            "NZDUSD": ["edge_models_nzdusd.pkl", "edge_models_nzd.pkl"],
            "USDCAD": ["edge_models_usdcad.pkl", "edge_models_cad.pkl"],
        }

    def _load_ml_models(self):
        """
        Load pre-trained RF model bundles per symbol from models/.

        Bundle structure (flat, as saved by tf_sweep.py):
            {
                'models': {'La': clf, 'Lb': clf, 'Lc': clf, 'plus2': clf},
                'thr':    {'La': float, 'Lb': float, 'Lc': float, 'plus2': float},
                'col_order': [...],
                'symbol': 'EURUSD',   # optional, used for a safety check
                ...
            }
        """
        import pickle

        self.models = {}

        for sym in self.SYMBOLS:
            candidates = self.MODEL_CANDIDATES.get(sym, [])
            loaded = False

            for cand in candidates:
                model_path = os.path.join(self._model_dir, cand)
                if not os.path.exists(model_path):
                    continue

                try:
                    with open(model_path, "rb") as f:
                        meta = pickle.load(f)
                except Exception as e:
                    logger.error(f"[{sym}] Failed to read {cand}: {e}")
                    continue

                # Validate bundle structure
                required = {"models", "thr", "col_order"}
                if not required.issubset(meta.keys()):
                    logger.warning(
                        f"[{sym}] Model bundle {cand} missing keys: "
                        f"{required - set(meta.keys())} — skipping"
                    )
                    continue

                # Safety: never apply a bundle trained for another symbol
                if "symbol" in meta and meta["symbol"] != sym:
                    logger.warning(
                        f"[{sym}] Model bundle {cand} was trained for "
                        f"{meta['symbol']} (symbol/R mismatch) — skipping"
                    )
                    continue

                self.models[sym] = meta
                logger.info(
                    f"[{sym}] ML model loaded from {cand}: "
                    f"{len(meta.get('models', {}))} labels, "
                    f"thr_Lb={meta['thr'].get('Lb', '?'):.3f}, "
                    f"thr_plus2={meta['thr'].get('plus2', '?'):.3f}, "
                    f"features={len(meta.get('col_order', []))}"
                )
                loaded = True
                break

            if not loaded:
                logger.warning(
                    f"[{sym}] No model loaded — will trade ungated"
                )

    # ══════════════════════════════════════════════════════
    # FEATURE PIPELINE (delegates to features_lib)
    # ══════════════════════════════════════════════════════

    def _build_feature_row(self, symbol):
        """
        Build current-market feature row from recent M5 bars using features_lib.

        This is the SINGLE SOURCE OF TRUTH for feature column order.
        Must match exactly what tf_sweep.py used during training.

        Returns:
            (Xrow, col_order) or (None, None) on failure
        """
        try:
            # Import features_lib from the strategies directory
            # (features_lib.py must be in the same directory or on sys.path)
            from features_lib import build_blocks, feature_row, TFS

            # Fetch recent M5 bars from MT5
            import MetaTrader5 as mt5

            resolved = self._resolve_symbol(symbol)
            if resolved is None:
                return None, None

            rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M5, 0, 3000)
            if rates is None or len(rates) == 0:
                logger.warning(f"[{symbol}] No M5 data for ML features")
                return None, None

            df = pd.DataFrame(rates)
            ts = pd.to_datetime(df['time'], unit='s', utc=True)
            close = df['close'].to_numpy(float)
            high = df['high'].to_numpy(float)
            low = df['low'].to_numpy(float)

            # Build multi-TF feature blocks (M5 native, others resampled)
            blocks, block_inds = build_blocks(ts, close, high, low)

            # Extract last-bar feature vector
            Xrow, col_order = feature_row(blocks, block_inds)

            return Xrow, col_order

        except ImportError as e:
            logger.error(f"features_lib import failed: {e}")
            return None, None
        except Exception as e:
            logger.error(f"[{symbol}] Feature build failed: {e}")
            return None, None

    def _ml_confirm_gate(self, symbol: str, m5_df=None):
        """
        Run the ML confirmation gate for a symbol.

        Args:
            symbol: e.g. "EURUSD"
            m5_df: Optional DataFrame with M5 OHLC (paper/backtest path).
                   If None, fetches 3000 M5 bars from MT5 directly (live path).

        Returns:
            dict with probs, gate, gate_lb, gate_plus2, thr
            or None if model unavailable (trade ungated)
        """
        if symbol not in self.models:
            return None

        meta = self.models[symbol]

        # Build feature row from provided M5 data or fetch from MT5
        if m5_df is not None and len(m5_df) >= 500:
            try:
                from features_lib import build_blocks, feature_row
                ts = pd.to_datetime(m5_df.index, utc=True)
                close = m5_df["close"].to_numpy(float)
                high = m5_df["high"].to_numpy(float)
                low = m5_df["low"].to_numpy(float)
                blocks, block_inds = build_blocks(ts, close, high, low)
                Xrow, col_order = feature_row(blocks, block_inds)
            except Exception as e:
                logger.error(f"[{symbol}] Feature build from data_feed failed: {e}")
                return None
        else:
            # Fallback: fetch directly from MT5 (live only)
            Xrow, col_order = self._build_feature_row(symbol)
            if Xrow is None:
                return None

        # Validate column count matches model
        expected_cols = len(meta.get("col_order", []))
        if len(col_order) != expected_cols:
            logger.error(
                f"[{symbol}] ML gate: feature count mismatch: "
                f"live={len(col_order)} vs model={expected_cols}"
            )
            return None

        X = Xrow.reshape(1, -1)

        try:
            probs = {}
            for tgt in meta["models"]:
                probs[tgt] = float(meta["models"][tgt].predict_proba(X)[0, 1])

            thr = meta["thr"]
            gate_lb = probs.get("Lb", 0) >= thr.get("Lb", 0.5)
            gate_plus2 = probs.get("plus2", 0) >= thr.get("plus2", 0.5)
            gate = gate_lb and gate_plus2

            return {
                "probs": probs,
                "gate": gate,
                "gate_lb": gate_lb,
                "gate_plus2": gate_plus2,
                "thr": thr,
            }
        except Exception as e:
            logger.error(f"[{symbol}] ML gate prediction failed: {e}")
            return None

    def _resolve_symbol(self, symbol):
        """Resolve broker-specific symbol name."""
        try:
            import MetaTrader5 as mt5

            # Direct match first
            info = mt5.symbol_info(symbol)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(symbol, True)
                return symbol

            # Try common aliases
            aliases = {
                'EURUSD': ['EURUSD', 'EURUSD.m', 'EURUSD.raw'],
                'GBPUSD': ['GBPUSD', 'GBPUSD.m', 'GBPUSD.raw'],
                'XAUUSD': ['XAUUSD', 'GOLD', 'XAUUSD.m'],
            }

            candidates = aliases.get(symbol.upper(), [symbol])
            for c in candidates:
                info = mt5.symbol_info(c)
                if info is not None:
                    if not info.visible:
                        mt5.symbol_select(c, True)
                    return c

            logger.warning(f"Symbol not found on broker: {symbol}")
            return None
        except Exception:
            return symbol

    # ══════════════════════════════════════════════════════
    # ENTRY SIGNAL (ATR ratio trigger + momentum direction)
    # ══════════════════════════════════════════════════════

    def _compute_atr_ratio(self, m15_df):
        """Compute ATR14 / 100-bar ATR mean ratio from M15 data."""
        if m15_df is None or len(m15_df) < self.ATR_MA_PERIOD + 1:
            return np.nan, np.nan

        closes = m15_df["close"].to_numpy(float)
        highs = m15_df["high"].to_numpy(float)
        lows = m15_df["low"].to_numpy(float)

        # ATR14 (Wilder smoothing)
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        if len(tr) < self.ATR_PERIOD:
            return np.nan, np.nan

        atr_arr = np.empty(len(tr))
        atr_arr[self.ATR_PERIOD - 1] = tr[:self.ATR_PERIOD].mean()
        for i in range(self.ATR_PERIOD, len(tr)):
            atr_arr[i] = (atr_arr[i - 1] * (self.ATR_PERIOD - 1) + tr[i]) / self.ATR_PERIOD
        atr_arr[:self.ATR_PERIOD - 1] = np.nan

        if np.isnan(atr_arr[-1]):
            return np.nan, np.nan

        atr_ratio = atr_arr[-1] / np.nanmean(atr_arr[-self.ATR_MA_PERIOD:])
        mom = closes[-1] / closes[-1 - self.MOM_PERIOD] - 1.0

        return atr_ratio, mom

    # ══════════════════════════════════════════════════════
    # STRATEGY BASE INTERFACE
    # ══════════════════════════════════════════════════════

    def scan(self, data_feed, symbols):
        """
        Scan for entry signals on each symbol.
        Only evaluates on new M15 bars (not every poll).

        Gate behaviour:
          - paper/backtest: M5 pulled from data_feed, passed to gate
          - live: M5 fetched inside _build_feature_row via MT5
          - if neither available: trade proceeds ungated (logged)
        """
        signals = []

        for sym in self.SYMBOLS:
            if sym not in symbols:
                continue

            try:
                # ── Get M15 data for trigger detection ──
                m15_df = data_feed.get(sym, "M15")
                if m15_df is None or len(m15_df) < 150:
                    continue

                # ── Check if new M15 bar ──
                bar_time = m15_df.index[-1]
                if sym in self.last_bar_time and bar_time == self.last_bar_time[sym]:
                    continue  # Same bar, skip
                self.last_bar_time[sym] = bar_time

                # ── Compute ATR ratio trigger ──
                atr_ratio, mom = self._compute_atr_ratio(m15_df)

                if np.isnan(atr_ratio) or np.isnan(mom):
                    continue

                # ── Trigger check ──
                if atr_ratio <= self.ATR_RATIO_THRESHOLD:
                    continue

                # ── Direction from momentum ──
                if mom > 0:
                    direction = Direction.LONG
                elif mom < 0:
                    direction = Direction.SHORT
                else:
                    continue

                # ── ML confirmation gate ──
                # Try to get M5 from data_feed (works in paper/backtest)
                m5_df = data_feed.get(sym, "M5")

                fr = self._ml_confirm_gate(sym, m5_df=m5_df)

                if fr is not None and not fr["gate"]:
                    logger.info(
                        f"[{sym}] FILTER BLOCK {direction.value} | "
                        f"P(Lb)={fr['probs'].get('Lb', 0):.3f} "
                        f"P(plus2)={fr['probs'].get('plus2', 0):.3f}"
                    )
                    continue

                if fr is not None:
                    logger.info(
                        f"[{sym}] FILTER OK {direction.value} | "
                        f"P(Lb)={fr['probs'].get('Lb', 0):.3f} "
                        f"P(plus2)={fr['probs'].get('plus2', 0):.3f}"
                    )
                else:
                    logger.info(
                        f"[{sym}] No model or features unavailable — trading ungated"
                    )

                # ── Build signal ──
                current_price = float(m15_df["close"].iloc[-1])
                r_pts = self.R_POINTS.get(sym, 24.0)

                sl_dist = r_pts * self.SL_RATIO * 1e-5

                if direction == Direction.LONG:
                    sl_price = current_price - sl_dist
                else:
                    sl_price = current_price + sl_dist
                # Trailing: no broker TP — the trade rides; SL stays as a
                # crash-safety floor at -1R.
                tp_price = 0.0

                from core import Signal
                sig = Signal(
                    strategy=self.strategy_name,
                    symbol=sym,
                    direction=direction,
                    entry_price=current_price,
                    atr=r_pts * 1e-5,  # R as ATR unit
                    stop_price=sl_price,
                    tp_price=tp_price,
                    signal_bar_time=bar_time,
                    meta={
                        "bb_ratio_m5": atr_ratio,
                        "breakout_str": mom,
                        "bar_idx": len(m15_df) - 1,
                        "trig_idx": len(m15_df) - 1,
                    },
                )
                signals.append(sig)

            except Exception as e:
                logger.error(f"[{sym}] Scan error: {e}")

        return signals

    def manage_exits(self, open_positions, data_feed):
        """
        Trailing-stop exit manager (A0.5 / D0.5).

        Pre-activation  (peak < 0.5R): broker SL at -1R is the only exit.
        Post-activation (peak >= 0.5R): trail = peak - 0.5R, floored at 0R.
        Time exit: 6 M15 bars, unchanged.
        """
        actions: List[ExitAction] = []

        for oid, pos in open_positions.items():
            if pos.strategy != self.strategy_name:
                continue

            sym = pos.symbol
            if sym not in self.SYMBOLS:
                continue

            try:
                # ── current price (M15 close; tick fallback not needed here) ──
                m15_df = data_feed.get(sym, "M15")
                if m15_df is None or len(m15_df) < 2:
                    continue
                current_price = float(m15_df["close"].iloc[-1])

                entry_price = pos.entry_price
                if entry_price <= 0:
                    continue

                RP = self.R_POINTS.get(sym, 24.0) * 1e-5
                sign = 1.0 if pos.direction == Direction.LONG else -1.0

                # ── favourable excursion in R ──
                fav = sign * (current_price - entry_price) / RP

                # ── update peak ──
                prev_peak = self._trail_peak.get(oid, 0.0)
                peak = max(prev_peak, fav)
                self._trail_peak[oid] = peak

                # ── time exit (6 M15 bars) ──
                entry_time = pos.entry_time
                if entry_time is not None:
                    current_bar_time = m15_df.index[-1]
                    bars_held = (current_bar_time - entry_time).total_seconds() / 900.0
                    if bars_held >= self.TIME_EXIT_BARS:
                        self._trail_peak.pop(oid, None)
                        actions.append(ExitAction(oid, "TIME", current_price))
                        continue

                # ── pre-activation: peak < TRAIL_ACTIVATION ──
                if peak < self.TRAIL_ACTIVATION:
                    # Broker SL at -1R is the only protection; detect if hit.
                    if (sign > 0 and current_price <= pos.stop_price) or \
                       (sign < 0 and current_price >= pos.stop_price):
                        self._trail_peak.pop(oid, None)
                        actions.append(ExitAction(oid, "SL", current_price))
                    continue

                # ── post-activation: trailing stop ──
                trail_r = max(peak - self.TRAIL_DISTANCE, 0.0)
                trail_price = entry_price + sign * trail_r * RP

                crossed = (
                    (sign > 0 and current_price <= trail_price)
                    or (sign < 0 and current_price >= trail_price)
                )

                if crossed:
                    self._trail_peak.pop(oid, None)
                    actions.append(ExitAction(oid, "TRAIL", current_price))
                    continue

                # ── optional: push trail to broker (enhancement, disabled) ──
                # self._sync_broker_sl(pos, trail_price, sym)

            except Exception as e:
                logger.error(f"manage_exits error {sym} {oid}: {e}")

        return actions

    def _sync_broker_sl(self, pos, trail_price: float, sym: str):
        """
        Move the broker SL to the trailing level so the position is
        protected even while the bot sleeps or lags. Only modifies when
        the new SL differs by > epsilon. Leave uncalled until the
        in-process trailing is verified in live.
        """
        try:
            import MetaTrader5 as mt5

            live = mt5.positions_get(ticket=pos.mt5_ticket)
            if live is None or len(live) == 0:
                return

            current_sl = live[0].sl
            epsilon = self.R_POINTS.get(sym, 24.0) * 1e-5 * 0.05  # 5% of R

            if abs(current_sl - trail_price) < epsilon:
                return  # already close enough

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": pos.mt5_ticket,
                "symbol": sym,
                "sl": trail_price,
                "tp": 0.0,   # no TP — trailing manages exit
            }
            result = mt5.order_send(request)
            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"TRAIL-SYNC {sym} ticket={pos.mt5_ticket}: "
                    f"SL moved {current_sl:.5f} -> {trail_price:.5f}"
                )
            else:
                logger.warning(
                    f"TRAIL-SYNC {sym} ticket={pos.mt5_ticket} failed: "
                    f"{result.retcode if result else 'None'}"
                )
        except Exception as e:
            logger.warning(f"TRAIL-SYNC {sym} error: {e}")

    def health_check(self):
        """Check if models are loaded and strategy is operational."""
        models_loaded = len(self.models) > 0
        if not models_loaded:
            logger.warning("No ML models loaded — trading will be ungated")
        return True  # Strategy can still trade ungated
