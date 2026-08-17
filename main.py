#!/usr/bin/env python3
"""
main.py — Single entry point for the trading framework.

Discovers and loads all strategies from strategies/, boots the engine,
and runs the main loop.

Usage:
    python main.py                    # paper trading (default)
    python main.py --mode live        # live trading
    python main.py --mode backtest    # backtest
    python main.py --strategies bb_squeeze,ma_rsi   # subset
"""

import argparse
import sys
import os

# Ensure strategies/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import Config, Engine, load_strategies, logger


def main():
    parser = argparse.ArgumentParser(description="Multi-strategy trading framework")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["paper", "live", "backtest"],
        help="Override MODE from .env",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names to enable (default: all discovered)",
    )
    parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols to trade (overrides .env)",
    )
    args = parser.parse_args()

    cfg = Config()

    # CLI overrides
    if args.mode:
        cfg.mode = args.mode
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    enabled_names = None
    if args.strategies:
        enabled_names = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # Discover and instantiate strategies
    strategies = load_strategies(cfg, enabled_names=enabled_names)

    if not strategies:
        logger.critical("No strategies loaded. Check strategies/ directory.")
        sys.exit(1)

    logger.info(f"Loaded {len(strategies)} strategies: {[s.name for s in strategies]}")
    logger.info(f"Mode: {cfg.mode} | Symbols: {cfg.symbols}")

    # Build and run engine
    engine = Engine(cfg, strategies)

    try:
        if cfg.mode == "backtest":
            engine.run_backtest()
        else:
            engine.run_live()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
