import MetaTrader5 as mt5
mt5.initialize()
syms = "XAUUSD EURUSD GBPUSD GBPJPY AUDUSD NZDUSD USDJPY NAS100 SP500 XAGUSD USDCHF EURJPY AUDJPY NZDJPY US30 DAX40 FTSE100".split()
for s in syms:
    r = mt5.copy_rates_from_pos(s, mt5.TIMEFRAME_M5, 0, 10)
    ok = r is not None and len(r) > 0
    info = mt5.symbol_info(s)
    pt = info.point if info else "-"
    sp = info.spread if info else "-"
    print(f"{s:8} {'OK' if ok else 'NO '} pt={pt} spread={sp}")
mt5.shutdown()
