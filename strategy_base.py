#!/usr/bin/env python3
"""
strategies/strategy_base.py — Strategy interface.

Every strategy module in strategies/ must contain exactly one class
that subclasses StrategyBase. The loader in core.py discovers it
automatically via inspection.

Contract:
  - strategy_name : unique key for --strategies filtering
  - name          : human-readable display name
  - version       : strategy version string
  - scan()        : returns List[Signal] for new entries
  - manage_exits(): returns List[ExitAction] for open positions

The engine calls scan() then manage_exits() every cycle.
Strategies must be stateless between cycles except for internal
caches keyed by (symbol, bar_time) to avoid duplicate signals.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""

    # ── Class-level identity (required) ──
    strategy_name: str = ""       # unique key, e.g. "bb_squeeze"
    name: str = ""                # display name, e.g. "BB-Squeeze Breakout"
    version: str = "1.0.0"       # strategy version

    # ── Per-symbol allow-list ──
    # If None, the strategy may trade any symbol passed to scan().
    # If a set, the strategy only trades symbols in the set. This is the
    # backtest-derived "only trade pairs that benefit" gate.
    ALLOWED_SYMBOLS: Optional[set] = None

    def symbol_allowed(self, sym: str) -> bool:
        """True if `sym` is permitted to trade under ALLOWED_SYMBOLS."""
        if self.ALLOWED_SYMBOLS is None:
            return True
        return sym in self.ALLOWED_SYMBOLS

    def __init__(self, cfg):
        """
        Called once at startup by the engine.

        Args:
            cfg: Config object from core.py with all .env settings.
        """
        self.cfg = cfg

    @abstractmethod
    def scan(self, data_feed, symbols: List[str]) -> List[Any]:
        """
        Scan for new entry signals across all symbols.

        Called every engine cycle. Must return a list of core.Signal
        objects (or empty list). Each Signal must have:
            strategy, symbol, direction, entry_price, atr,
            stop_price, tp_price, signal_bar_time, meta

        Args:
            data_feed: core.DataFeed instance for fetching OHLC data.
            symbols: list of symbol strings to scan.

        Returns:
            List of Signal objects for new entries.
        """
        pass

    @abstractmethod
    def manage_exits(self, open_positions: Dict[str, Any], data_feed) -> List[Any]:
        """
        Manage open positions belonging to this strategy.

        Called every engine cycle. Must return a list of core.ExitAction
        objects (or empty list). Each ExitAction must have:
            oid, reason, price (optional; None = use market price)

        Only positions where position.strategy == self.strategy_name
        should be managed.

        Args:
            open_positions: dict of {oid: Position} from core.Executor.
            data_feed: core.DataFeed instance for fetching OHLC data.

        Returns:
            List of ExitAction objects requesting position closure.
        """
        pass

    # ── Optional overrides ──

    def on_start(self):
        """Called once after construction, before the first cycle."""
        pass

    def on_stop(self):
        """Called once during graceful shutdown."""
        pass

    def health_check(self) -> bool:
        """Return True if the strategy is healthy and ready to trade."""
        return True
