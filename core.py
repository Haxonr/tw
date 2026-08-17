#!/usr/bin/env python3
"""
core.py — Trading framework infrastructure.

Knows NOTHING about specific strategies (L1/L2/L3, BB-Squeeze, MA-RSI).
Only knows:

    Strategy → Signal → Risk → Execution → Position Management → Logging

Strategies are loaded from strategies/ via the StrategyBase interface.
"""

import os
import sys
import json
import time
import logging
import threading
import importlib
import inspect
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Protocol
from datetime import datetime, timezone, timedelta
from enum import Enum
from logging.handlers import RotatingFileHandler
from collections import defaultdict
from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════
# .ENV LOADING
# ══════════════════════════════════════════════════════════

def _load_dotenv_fallback(path: str = ".env", override: bool = True) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2:
                if (value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'"):
                    value = value[1:-1]
            else:
                if " #" in value:
                    value = value.split(" #", 1)[0].strip()
                else:
                    if value.startswith("#"):
                        value = ""
            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
        env_path = find_dotenv(filename=".env", usecwd=True)
        if env_path:
            load_dotenv(env_path, override=True)
            return
        script_env = Path(__file__).resolve().parent / ".env"
        if script_env.exists():
            load_dotenv(script_env, override=True)
            return
    except ImportError:
        pass
    for p in (Path(".env"), Path(__file__).resolve().parent / ".env"):
        if p.exists():
            _load_dotenv_fallback(str(p), override=True)
            break


_bootstrap_env()


def _raw(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is None or str(val).strip() == "":
        return default
    return str(val).strip()


def _str(key: str, default: str = "") -> str:
    return str(_raw(key, default))


def _int(key: str, default: int) -> int:
    try:
        return int(float(_raw(key, str(default))))
    except Exception:
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(_raw(key, str(default)))
    except Exception:
        return default


def _bool(key: str, default: bool = False) -> bool:
    v = str(_raw(key, str(default))).strip().lower()
    return v in ("1", "true", "yes", "on", "y", "t")


def _list(key: str, default: List[str]) -> List[str]:
    v = _raw(key, "")
    if not v:
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]


# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════

LOG_FILE = _str("LOG_FILE", "trading_framework.log")
_LOG_TO_FILE = _bool("LOG_TO_FILE", True)
_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

logger = logging.getLogger("trading_framework")
logger.setLevel(_str("LOG_LEVEL", "INFO").upper())
logger.handlers.clear()

if _LOG_TO_FILE:
    _fh = RotatingFileHandler(LOG_FILE, maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8")
    _fh.setFormatter(_fmt)
    logger.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
logger.addHandler(_sh)


# ══════════════════════════════════════════════════════════
# ENUMS & DATA TYPES
# ══════════════════════════════════════════════════════════

class Direction(Enum):
    LONG = "long"
    SHORT = "short"


class Signal:
    """
    Strategy-agnostic signal emitted by any strategy.
    Core only reads the fields it needs for risk + execution.
    Strategy-specific metadata lives in `meta`.
    """
    def __init__(
        self,
        strategy: str,
        symbol: str,
        direction: Direction,
        entry_price: float,
        atr: float,
        stop_price: float,
        tp_price: float,
        signal_bar_time: datetime,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.atr = atr
        self.stop_price = stop_price
        self.tp_price = tp_price
        self.signal_bar_time = signal_bar_time
        self.meta = meta or {}
        self.ts = datetime.now(timezone.utc)


@dataclass
class Position:
    """An open position managed by the engine."""
    oid: str
    strategy: str
    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    tp_price: float
    qty: float
    atr: float
    cost_r: float
    entry_time: datetime
    signal_bar_time: datetime
    mt5_ticket: int = 0
    status: str = "open"
    meta: Dict[str, Any] = field(default_factory=dict)
    peak_price: float = 0.0
    trough_price: float = 0.0


class ExitAction:
    """Emitted by a strategy to request closing a position."""
    def __init__(self, oid: str, reason: str, price: Optional[float] = None):
        self.oid = oid
        self.reason = reason
        self.price = price  # None = use current market price


# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

@dataclass
class Config:
    # MT5
    mt5_login: int = field(default_factory=lambda: _int("MT5_LOGIN", 0))
    mt5_password: str = field(default_factory=lambda: _str("MT5_PASSWORD", ""))
    mt5_server: str = field(default_factory=lambda: _str("MT5_SERVER", ""))

    # Symbols
    symbols: List[str] = field(default_factory=lambda: _list(
        "SYMBOLS", ["XAUUSD", "EURUSD", "GBPUSD", "NAS100"]
    ))

    # Execution
    mode: str = field(default_factory=lambda: _str("MODE", "paper"))
    cost_bps: float = field(default_factory=lambda: _float("COST_BPS", 0.7))
    magic_number: int = field(default_factory=lambda: _int("MAGIC_NUMBER", 445566))

    # Risk
    risk_per_trade_pct: float = field(default_factory=lambda: _float("RISK_PER_TRADE_PCT", 0.01))
    max_open_positions: int = field(default_factory=lambda: _int("MAX_OPEN_POSITIONS", 5))
    max_per_symbol: int = field(default_factory=lambda: _int("MAX_SAME_SYMBOL", 2))
    max_daily_loss_pct: float = field(default_factory=lambda: _float("MAX_DAILY_LOSS_PCT", 0.03))
    max_weekly_loss_pct: float = field(default_factory=lambda: _float("MAX_WEEKLY_LOSS_PCT", 0.05))
    max_peak_dd_pct: float = field(default_factory=lambda: _float("MAX_PEAK_DD_PCT", 0.05))
    max_session_risk_r: float = field(default_factory=lambda: _float("MAX_SESSION_RISK_R", 3.0))

    # Lot limits
    min_lot_size: float = field(default_factory=lambda: _float("MIN_LOT_SIZE", 0.01))
    max_lot_size: float = field(default_factory=lambda: _float("MAX_LOT_SIZE", 0.50))
    max_lot_xauusd: float = field(default_factory=lambda: _float("MAX_LOT_XAUUSD", 0.02))
    max_lot_indices: float = field(default_factory=lambda: _float("MAX_LOT_INDICES", 0.10))

    # Data
    cache_ttl: int = field(default_factory=lambda: _int("CACHE_TTL", 15))
    output_dir: str = field(default_factory=lambda: _str("OUTPUT_DIR", "./results"))

    # State / audit
    state_file: str = field(default_factory=lambda: _str("STATE_FILE", "bot_state.json"))
    trade_history_path: str = field(default_factory=lambda: _str("TRADE_LEDGER_FILE", "trade_ledger.jsonl"))

    # Telegram
    telegram_enabled: bool = field(default_factory=lambda: _bool("TELEGRAM_ENABLED", False))
    tg_token: str = field(default_factory=lambda: _str("TELEGRAM_BOT_TOKEN", ""))
    tg_chat: str = field(default_factory=lambda: _str("TELEGRAM_CHAT_ID", ""))

    # Engine
    loop_sleep: int = field(default_factory=lambda: _int("LOOP_SLEEP", 15))
    max_consecutive_errors: int = field(default_factory=lambda: _int("MAX_CONSECUTIVE_ERRORS", 5))

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.isabs(self.state_file):
            self.state_file = os.path.join(self.output_dir, self.state_file)
        if not os.path.isabs(self.trade_history_path):
            self.trade_history_path = os.path.join(self.output_dir, self.trade_history_path)


# ══════════════════════════════════════════════════════════
# SYMBOL RESOLVER
# ══════════════════════════════════════════════════════════

def resolve_symbol(name: str) -> Optional[str]:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return name

    aliases = {
        "NAS100": ["NAS100", "NQ100", "US100", "USTEC", "NASDAQ", "US_TECH100"],
        "SP500": ["SP500", "US500", "SPX500", "US500.cash"],
        "US30": ["US30", "YM100", "DJI30", "WS30", "DOW30"],
        "DAX40": ["DAX40", "GER40", "DE40", "GERMANY40"],
        "FTSE100": ["FTSE100", "UK100", "GB100"],
        "XAGUSD": ["XAGUSD", "SILVER"],
        "XAUUSD": ["XAUUSD", "GOLD"],
    }

    candidates = aliases.get(name.upper(), [name])
    for c in candidates:
        try:
            info = mt5.symbol_info(c)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(c, True)
                return c
        except Exception:
            continue

    logger.warning(f"Symbol not found on broker: {name}")
    return None


# ══════════════════════════════════════════════════════════
# STATE MANAGER
# ══════════════════════════════════════════════════════════

class StateManager:
    def __init__(self, path: str):
        self.path = path
        self.state: Dict = {}
        tmp = f"{self.path}.tmp"
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

    def load(self) -> Dict:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r") as f:
                    self.state = json.load(f)
                logger.info(f"State loaded: peak={self.state.get('peak_balance')}")
            else:
                logger.info("No state file — fresh start")
        except Exception as e:
            logger.error(f"State load failed: {e}")
        return self.state

    def save(self) -> bool:
        try:
            tmp = f"{self.path}.tmp"
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
            os.replace(tmp, self.path)
            return True
        except Exception as e:
            logger.error(f"State save failed: {e}")
            return False

    def get(self, k, default=None):
        return self.state.get(k, default)

    def set(self, k, v):
        self.state[k] = v


# ══════════════════════════════════════════════════════════
# DATA FEED (MT5 ONLY)
# ══════════════════════════════════════════════════════════

class DataFeed:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
        self._lock = threading.RLock()
        self._resolved: Dict[str, str] = {}

    def get(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        key = f"{symbol}_{tf}"
        now = time.time()
        with self._lock:
            if key in self._cache:
                ts, df = self._cache[key]
                if now - ts < self.cfg.cache_ttl:
                    return df
        df = self._fetch(symbol, tf)
        if df is not None:
            with self._lock:
                self._cache[key] = (now, df)
        return df

    def _fetch(self, symbol: str, tf: str) -> Optional[pd.DataFrame]:
        try:
            import MetaTrader5 as mt5

            if symbol not in self._resolved:
                resolved = resolve_symbol(symbol)
                if resolved is None:
                    return None
                self._resolved[symbol] = resolved
            resolved = self._resolved[symbol]

            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1,
            }
            mt5_tf = tf_map.get(tf.upper())
            if mt5_tf is None:
                logger.error(f"Unknown timeframe: {tf}")
                return None

            rates = mt5.copy_rates_from_pos(resolved, mt5_tf, 0, 500)
            if rates is None or len(rates) == 0:
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            return df

        except ImportError:
            logger.critical("MetaTrader5 not installed.")
            return None
        except Exception as e:
            logger.error(f"Data fetch error {symbol} {tf}: {e}")
            return None

    def m5(self, sym: str) -> Optional[pd.DataFrame]:
        return self.get(sym, "M5")

    def m15(self, sym: str) -> Optional[pd.DataFrame]:
        return self.get(sym, "M15")

    def m30(self, sym: str) -> Optional[pd.DataFrame]:
        return self.get(sym, "M30")

    def h1(self, sym: str) -> Optional[pd.DataFrame]:
        return self.get(sym, "H1")


# ══════════════════════════════════════════════════════════
# RISK MANAGER
# ══════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, cfg: Config, state: StateManager):
        self.cfg = cfg
        self.state = state
        self.peak = state.get("peak_balance")
        self.daily_start = state.get("daily_start")
        self.weekly_start = state.get("weekly_start")
        self.last_daily = state.get("last_daily")
        self.last_weekly = state.get("last_weekly")
        self.session_used: Dict[str, float] = defaultdict(float)
        self._session_date = None

    def check_reset(self, balance: float):
        now = datetime.now(timezone.utc)
        today = str(now.date())

        if self.last_daily != today:
            self.daily_start = balance
            self.last_daily = today
            self._persist()
            logger.info("Daily P&L reset")

        wk = str((now - timedelta(days=now.weekday())).date())
        if self.last_weekly != wk:
            self.weekly_start = balance
            self.last_weekly = wk
            self._persist()
            logger.info("Weekly P&L reset")

        if self.peak is None or balance > self.peak:
            self.peak = balance
            self._persist()

    def allowed(self, balance: float) -> Tuple[bool, str]:
        if self.peak and self.peak > 0:
            dd = (self.peak - balance) / self.peak
            if dd > self.cfg.max_peak_dd_pct:
                return False, f"Peak DD {dd:.2%} > {self.cfg.max_peak_dd_pct:.2%}"
        if self.daily_start and self.daily_start > 0:
            dd = (self.daily_start - balance) / self.daily_start
            if dd > self.cfg.max_daily_loss_pct:
                return False, f"Daily loss {dd:.2%}"
        if self.weekly_start and self.weekly_start > 0:
            dd = (self.weekly_start - balance) / self.weekly_start
            if dd > self.cfg.max_weekly_loss_pct:
                return False, f"Weekly loss {dd:.2%}"
        return True, "ok"

    @staticmethod
    def _session_key(hour: int) -> str:
        if hour < 7:
            return "ASIA"
        if hour < 12:
            return "LONDON"
        if hour < 17:
            return "NY"
        if hour < 21:
            return "NY_PM"
        return "OFF"

    def session_ok(self, risk_r: float, at_time: datetime = None) -> bool:
        now = at_time or datetime.now(timezone.utc)
        today = now.date()
        if self._session_date != today:
            self.session_used = defaultdict(float)
            self._session_date = today
        key = self._session_key(now.hour)
        if self.session_used[key] + risk_r > self.cfg.max_session_risk_r:
            return False
        return True

    def session_add(self, risk_r: float, at_time: datetime = None):
        now = at_time or datetime.now(timezone.utc)
        key = self._session_key(now.hour)
        self.session_used[key] += risk_r

    def _persist(self):
        self.state.set("peak_balance", self.peak)
        self.state.set("daily_start", self.daily_start)
        self.state.set("weekly_start", self.weekly_start)
        self.state.set("last_daily", self.last_daily)
        self.state.set("last_weekly", self.last_weekly)
        self.state.save()


# ══════════════════════════════════════════════════════════
# EXECUTOR
# ══════════════════════════════════════════════════════════

class Executor:
    def __init__(self, cfg: Config, mode: str = "paper"):
        self.cfg = cfg
        self.mode = mode
        self.open_positions: Dict[str, Position] = {}

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5

            if mt5.initialize():
                info = mt5.account_info()
                if info:
                    logger.info(
                        f"MT5 attached (account {info.login}, balance {info.balance:.2f})"
                    )
                    return True

            logger.info("No running terminal. Attempting login...")
            if mt5.initialize(
                login=self.cfg.mt5_login,
                password=self.cfg.mt5_password,
                server=self.cfg.mt5_server,
            ):
                info = mt5.account_info()
                if info:
                    logger.info(f"MT5 logged in (account {info.login})")
                    return True

            error = mt5.last_error()
            logger.critical(f"MT5 CONNECTION FAILED: {error}")
            return False

        except ImportError:
            logger.critical("MetaTrader5 package not installed.")
            return False
        except Exception as e:
            logger.critical(f"MT5 connection error: {e}")
            return False

    def get_balance(self) -> float:
        if self.mode == "paper":
            return 10000.0
        try:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            return info.balance if info else 0.0
        except Exception:
            return 0.0

    def _get_filling_mode(self, symbol: str):
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info is None:
                return mt5.ORDER_FILLING_IOC
            mode = int(getattr(info, "filling_mode", 0))
            if mode & 2:
                return mt5.ORDER_FILLING_IOC
            if mode & 1:
                return mt5.ORDER_FILLING_FOK
            if mode & 4:
                return mt5.ORDER_FILLING_RETURN
            return mt5.ORDER_FILLING_RETURN
        except Exception:
            import MetaTrader5 as mt5
            return mt5.ORDER_FILLING_IOC

    def submit(self, signal: Signal, qty: float) -> Optional[Position]:
        """Submit an order. Returns Position on success, None on failure."""
        oid = f"{signal.strategy}_{signal.symbol}_{int(time.time() * 1000)}"

        pos = Position(
            oid=oid,
            strategy=signal.strategy,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            tp_price=signal.tp_price,
            qty=qty,
            atr=signal.atr,
            cost_r=self._cost_r(signal),
            entry_time=datetime.now(timezone.utc),
            signal_bar_time=signal.signal_bar_time,
            meta=signal.meta,
            peak_price=signal.entry_price,
            trough_price=signal.entry_price,
        )

        if self.mode == "paper":
            pos.status = "filled"
            self.open_positions[oid] = pos
            return pos

        # Live submission
        try:
            import MetaTrader5 as mt5

            resolved = resolve_symbol(signal.symbol)
            if resolved is None:
                return None

            info = mt5.symbol_info(resolved)
            if not info:
                return None
            if not info.visible:
                mt5.symbol_select(resolved, True)

            tick = mt5.symbol_info_tick(resolved)
            if tick is None:
                return None

            if signal.direction == Direction.LONG:
                ot = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                ot = mt5.ORDER_TYPE_SELL
                price = tick.bid

            filling = self._get_filling_mode(resolved)

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": resolved,
                "volume": round(qty, 2),
                "type": ot,
                "price": price,
                "sl": signal.stop_price,
                "tp": signal.tp_price if signal.tp_price > 0 else 0.0,
                "deviation": 20,
                "magic": self.cfg.magic_number,
                "comment": f"{signal.strategy[:12]}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            res = mt5.order_send(req)

            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed {signal.symbol}: {res}")
                return None

            pos.status = "filled"
            pos.mt5_ticket = int(res.order)
            pos.entry_price = float(getattr(res, "price", price) or price)
            self.open_positions[oid] = pos

            logger.info(
                f"MT5 FILL {signal.symbol} ticket={res.order} "
                f"filling={filling} qty={qty}"
            )
            return pos

        except Exception as e:
            logger.error(f"Submit error {signal.symbol}: {e}")
            return None

    def close_position(self, pos: Position, exit_price: float, reason: str) -> bool:
        """Close a position. Returns True on success."""
        if self.mode == "paper":
            pos.status = "closed"
            self.open_positions.pop(pos.oid, None)
            return True

        # ── Broker reconciliation: position may already be gone (SL/TP hit) ──
        try:
            import MetaTrader5 as mt5

            live_positions = mt5.positions_get(ticket=pos.mt5_ticket)
            if live_positions is None or len(live_positions) == 0:
                # Broker already closed it server-side — reconcile state.
                pos.status = "closed"
                self.open_positions.pop(pos.oid, None)
                logger.info(
                    f"RECONCILE {pos.symbol} ticket={pos.mt5_ticket}: "
                    f"position already closed broker-side ({reason})"
                )
                return True
        except Exception as recon_err:
            logger.warning(
                f"Reconciliation check failed for {pos.symbol} "
                f"ticket={pos.mt5_ticket}: {recon_err} — proceeding to close"
            )
        # ── end reconciliation ──

        try:
            import MetaTrader5 as mt5

            if pos.mt5_ticket == 0:
                self.open_positions.pop(pos.oid, None)
                return True

            resolved = resolve_symbol(pos.symbol)
            if resolved is None:
                return False

            tick = mt5.symbol_info_tick(resolved)
            if tick is None:
                return False

            if pos.direction == Direction.LONG:
                close_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                close_type = mt5.ORDER_TYPE_BUY
                price = tick.ask

            filling = self._get_filling_mode(resolved)

            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": resolved,
                "volume": pos.qty,
                "type": close_type,
                "position": pos.mt5_ticket,
                "price": price,
                "deviation": 50,
                "magic": self.cfg.magic_number,
                "comment": "close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            res = mt5.order_send(req)

            if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Close failed {pos.symbol}: {res}")
                return False

            pos.status = "closed"
            self.open_positions.pop(pos.oid, None)
            return True

        except Exception as e:
            logger.error(f"Close error {pos.symbol}: {e}")
            return False

    def _cost_r(self, signal: Signal) -> float:
        if signal.atr <= 0:
            return 0.0
        one_way = signal.entry_price * self.cfg.cost_bps / 10000.0
        return one_way / signal.atr

    def position_count(self) -> int:
        return len(self.open_positions)

    def symbol_position_count(self, symbol: str) -> int:
        return sum(1 for p in self.open_positions.values() if p.symbol == symbol)


# ══════════════════════════════════════════════════════════
# POSITION SIZER
# ══════════════════════════════════════════════════════════

class PositionSizer:
    def __init__(self, cfg: Config, risk: RiskManager):
        self.cfg = cfg
        self.risk = risk

    def _contract_size(self, symbol: str) -> float:
        try:
            import MetaTrader5 as mt5
            resolved = resolve_symbol(symbol)
            info = mt5.symbol_info(resolved) if resolved else None
            if info is not None:
                cs = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
                if cs > 0:
                    return cs
        except Exception:
            pass
        return 1.0

    def size(self, signal: Signal, balance: float) -> float:
        if signal.atr <= 1e-12:
            return 0.0
        if not self.risk.session_ok(1.0):
            return 0.0

        stop_dist = abs(signal.entry_price - signal.stop_price)
        if stop_dist <= 0:
            return 0.0

        risk_amount = balance * max(self.cfg.risk_per_trade_pct, 0.0)
        contract_size = self._contract_size(signal.symbol)

        if contract_size > 0:
            qty = risk_amount / (stop_dist * contract_size)
        else:
            qty = risk_amount / stop_dist

        # Apply lot caps
        s = signal.symbol.upper()
        if any(x in s for x in ("XAU", "GOLD")):
            max_lot = min(self.cfg.max_lot_size, self.cfg.max_lot_xauusd)
        elif any(x in s for x in ("SP500", "SPX", "NAS", "NQ", "US100", "US30", "DJI", "DAX", "GER40", "FTSE", "UK100")):
            max_lot = min(self.cfg.max_lot_size, self.cfg.max_lot_indices)
        else:
            max_lot = self.cfg.max_lot_size

        qty = min(qty, max_lot)
        qty = max(qty, self.cfg.min_lot_size)

        # Broker step
        try:
            import MetaTrader5 as mt5
            resolved = resolve_symbol(signal.symbol)
            info = mt5.symbol_info(resolved) if resolved else None
            if info is not None:
                step = float(getattr(info, "volume_step", 0.01))
                if step > 0:
                    qty = int(qty / step) * step
        except Exception:
            pass

        return round(qty, 2)


# ══════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════

def send_telegram(cfg: Config, msg: str) -> bool:
    if not cfg.telegram_enabled or not cfg.tg_token or not cfg.tg_chat:
        return False
    try:
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{cfg.tg_token}/sendMessage",
            json={"chat_id": cfg.tg_chat, "text": msg[:4000]},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════
# TRADE HISTORY LOGGER
# ══════════════════════════════════════════════════════════

class TradeHistoryLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def log_entry(self, pos: Position):
        rec = {
            "event": "entry",
            "ts": datetime.now(timezone.utc).isoformat(),
            "oid": pos.oid,
            "strategy": pos.strategy,
            "symbol": pos.symbol,
            "direction": pos.direction.value,
            "entry": pos.entry_price,
            "stop": pos.stop_price,
            "tp": pos.tp_price,
            "qty": pos.qty,
            "atr": pos.atr,
            "signal_bar_time": str(pos.signal_bar_time),
            "meta": pos.meta,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def log_exit(self, pos: Position, exit_price: float, reason: str,
                 pnl_r: float, pnl_net_r: float):
        rec = {
            "event": "exit",
            "ts": datetime.now(timezone.utc).isoformat(),
            "oid": pos.oid,
            "strategy": pos.strategy,
            "symbol": pos.symbol,
            "direction": pos.direction.value,
            "entry": pos.entry_price,
            "exit": exit_price,
            "reason": reason,
            "pnl_r": pnl_r,
            "pnl_net_r": pnl_net_r,
            "hold_sec": (datetime.now(timezone.utc) - pos.entry_time).total_seconds(),
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# PERFORMANCE ANALYZER
# ══════════════════════════════════════════════════════════

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl_r": 0.0,
            "by_strategy": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_r": 0.0}),
            "by_symbol": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_r": 0.0}),
        }
        self._lock = threading.Lock()

    def update(self, pos: Position, pnl_net_r: float):
        with self._lock:
            self.metrics["total_trades"] += 1
            self.metrics["total_pnl_r"] += pnl_net_r

            if pnl_net_r > 0.01:
                self.metrics["winning_trades"] += 1
            elif pnl_net_r < -0.01:
                self.metrics["losing_trades"] += 1

            self.metrics["by_strategy"][pos.strategy]["trades"] += 1
            self.metrics["by_strategy"][pos.strategy]["pnl_r"] += pnl_net_r
            if pnl_net_r > 0.01:
                self.metrics["by_strategy"][pos.strategy]["wins"] += 1

            self.metrics["by_symbol"][pos.symbol]["trades"] += 1
            self.metrics["by_symbol"][pos.symbol]["pnl_r"] += pnl_net_r
            if pnl_net_r > 0.01:
                self.metrics["by_symbol"][pos.symbol]["wins"] += 1

    def report(self) -> str:
        with self._lock:
            m = self.metrics
            total = m["total_trades"]
            if total == 0:
                return "No trades recorded."

            wr = m["winning_trades"] / total * 100
            lines = [
                "=" * 55,
                "PERFORMANCE REPORT",
                "=" * 55,
                f"Total Trades: {total}",
                f"Win Rate: {wr:.1f}%",
                f"Total Net R: {m['total_pnl_r']:+.3f}R",
                "=" * 55,
                "BY STRATEGY:",
            ]
            for strat, perf in sorted(m["by_strategy"].items()):
                swr = (perf["wins"] / perf["trades"] * 100) if perf["trades"] > 0 else 0
                lines.append(
                    f"  {strat}: {perf['trades']} trades | {swr:.1f}% WR | {perf['pnl_r']:+.3f}R"
                )
            lines.append("=" * 55)
            lines.append("BY SYMBOL:")
            for sym, perf in sorted(m["by_symbol"].items(), key=lambda x: -x[1]["pnl_r"]):
                swr = (perf["wins"] / perf["trades"] * 100) if perf["trades"] > 0 else 0
                lines.append(
                    f"  {sym}: {perf['trades']} trades | {swr:.1f}% WR | {perf['pnl_r']:+.3f}R"
                )
            lines.append("=" * 55)
            return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# HEARTBEAT
# ══════════════════════════════════════════════════════════

class Heartbeat:
    def __init__(self, interval: int = 300):
        self.interval = interval
        self.last = time.time()
        self.start = time.time()
        self.cycles = 0

    def beat(self):
        self.cycles += 1
        now = time.time()
        if now - self.last >= self.interval:
            uptime = now - self.start
            h = int(uptime // 3600)
            m = int((uptime % 3600) // 60)
            logger.info(f"HEARTBEAT | cycles={self.cycles} | uptime={h}h {m}m")
            self.last = now


# ══════════════════════════════════════════════════════════
# STRATEGY LOADER
# ══════════════════════════════════════════════════════════

def load_strategies(cfg: Config, enabled_names: Optional[List[str]] = None) -> List[Any]:
    """
    Discovers and instantiates all strategies in strategies/.

    Each strategy module must contain a class that subclasses StrategyBase.
    The class is found via inspection, not by name convention.
    """
    strategies_dir = Path(__file__).resolve().parent / "strategies"
    if not strategies_dir.exists():
        logger.error(f"strategies/ directory not found at {strategies_dir}")
        return []

    loaded = []

    for py_file in sorted(strategies_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        if py_file.name == "strategy_base.py":
            continue

        module_name = f"strategies.{py_file.stem}"

        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            logger.error(f"Failed to import {module_name}: {e}")
            continue

        # Find StrategyBase subclass in module
        from strategies.strategy_base import StrategyBase

        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, StrategyBase) and obj is not StrategyBase:
                if enabled_names and obj.strategy_name not in enabled_names:
                    logger.info(f"Skipping {obj.strategy_name} (not in --strategies)")
                    continue
                try:
                    instance = obj(cfg)
                    loaded.append(instance)
                    logger.info(f"Loaded strategy: {instance.name} (v{instance.version})")
                except Exception as e:
                    logger.error(f"Failed to instantiate {name}: {e}")
                break

    return loaded


# ══════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════

class Engine:
    """
    Strategy-agnostic engine.

    Pipeline:
        Strategy.scan() → Risk → Execution
        Strategy.manage_exits() → Execution.close()
    """

    def __init__(self, cfg: Config, strategies: List[Any]):
        self.cfg = cfg
        self.strategies = strategies
        self.running = False

        self.state = StateManager(cfg.state_file)
        self.state.load()

        self.data = DataFeed(cfg)
        self.risk = RiskManager(cfg, self.state)
        self.sizer = PositionSizer(cfg, self.risk)
        self.executor = Executor(cfg, cfg.mode)
        self.perf = PerformanceAnalyzer()
        self.history = TradeHistoryLogger(cfg.trade_history_path)
        self.heartbeat = Heartbeat(interval=300)

        self.consecutive_errors = 0

    def run_live(self):
        logger.info(f"Engine starting | mode={self.cfg.mode}")

        if not self.executor.connect():
            logger.critical("Cannot start without MT5 connection.")
            return

        send_telegram(
            self.cfg,
            f"Trading framework started\n"
            f"Mode: {self.cfg.mode}\n"
            f"Strategies: {[s.name for s in self.strategies]}\n"
            f"Symbols: {', '.join(self.cfg.symbols)}",
        )

        self.running = True

        while self.running:
            try:
                self.heartbeat.beat()

                balance = self.executor.get_balance()
                self.risk.check_reset(balance)

                ok, why = self.risk.allowed(balance)
                if not ok:
                    logger.warning(f"Trading halted: {why}")
                    time.sleep(60)
                    continue

                # ── 1. Let each strategy manage exits ──
                for strat in self.strategies:
                    try:
                        exit_actions = strat.manage_exits(
                            self.executor.open_positions, self.data
                        )
                        if exit_actions:
                            for action in exit_actions:
                                pos = self.executor.open_positions.get(action.oid)
                                if pos is None:
                                    continue
                                exit_price = action.price or pos.entry_price
                                ok_close = self.executor.close_position(
                                    pos, exit_price, action.reason
                                )
                                if ok_close:
                                    pnl_r = self._pnl_r(pos, exit_price)
                                    pnl_net = pnl_r - 2.0 * pos.cost_r
                                    self.perf.update(pos, pnl_net)
                                    self.history.log_exit(
                                        pos, exit_price, action.reason, pnl_r, pnl_net
                                    )
                                    logger.info(
                                        f"EXIT {pos.symbol} {pos.direction.value} "
                                        f"{action.reason} | {pnl_net:+.3f}R"
                                    )
                    except Exception as e:
                        logger.error(f"Exit mgmt error ({strat.name}): {e}")

                # ── 2. Let each strategy scan for entries ──
                for strat in self.strategies:
                    try:
                        signals = strat.scan(self.data, self.cfg.symbols)
                    except Exception as e:
                        logger.error(f"Scan error ({strat.name}): {e}")
                        continue

                    if not signals:
                        continue

                    for sig in signals:
                        # Position limits (shared across all strategies)
                        if self.executor.position_count() >= self.cfg.max_open_positions:
                            continue
                        if self.executor.symbol_position_count(sig.symbol) >= self.cfg.max_per_symbol:
                            continue

                        # Risk
                        balance = self.executor.get_balance()
                        ok, why = self.risk.allowed(balance)
                        if not ok:
                            continue
                        if not self.risk.session_ok(1.0):
                            continue

                        # Size
                        qty = self.sizer.size(sig, balance)
                        if qty <= 0:
                            continue

                        # Execute
                        pos = self.executor.submit(sig, qty)
                        if pos is not None:
                            self.risk.session_add(1.0)
                            self.history.log_entry(pos)
                            logger.info(
                                f"ENTRY {sig.symbol} {sig.direction.value} "
                                f"qty={qty} @{sig.entry_price:.5f} "
                                f"[{sig.strategy}]"
                            )
                            send_telegram(
                                self.cfg,
                                f"✅ {sig.strategy} {sig.symbol} {sig.direction.value}\n"
                                f"@ {sig.entry_price:.5f} qty={qty}",
                            )

                self.consecutive_errors = 0
                time.sleep(self.cfg.loop_sleep)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(
                    f"Engine error ({self.consecutive_errors}/{self.cfg.max_consecutive_errors}): {e}"
                )
                if self.consecutive_errors >= self.cfg.max_consecutive_errors:
                    logger.critical("Max consecutive errors reached. Shutting down.")
                    self.running = False
                else:
                    time.sleep(30)

    def run_backtest(self):
        logger.info("Backtest mode not yet implemented for multi-strategy engine.")
        logger.info("Use strategy-specific backtest scripts for research.")

    def stop(self):
        logger.info("Engine stopping...")
        self.running = False
        logger.info(self.perf.report())
        self.state.save()

    def _pnl_r(self, pos: Position, exit_price: float) -> float:
        if pos.direction == Direction.LONG:
            return (exit_price - pos.entry_price) / pos.atr
        else:
            return (pos.entry_price - exit_price) / pos.atr
