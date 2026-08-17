import MetaTrader5 as mt5
import time

if not mt5.initialize():
    print("init failed:", mt5.last_error()); raise SystemExit

info = mt5.account_info()
print(f"ACCOUNT {info.login}  balance={info.balance:.2f}  equity={info.equity:.2f}  profit={info.profit:.2f}")

print("\n=== OPEN POSITIONS ===")
pos = mt5.positions_get()
if not pos:
    print("  (none)")
else:
    for p in pos:
        # magic number identifies the strategy/bot
        print(f"  {p.symbol} {'BUY' if p.type==0 else 'SELL'}  vol={p.volume}  "
              f"open={p.price_open:.5f}  sl={p.sl:.5f}  tp={p.tp:.5f}  "
              f"cur={p.price_current:.5f}  pnl={p.profit:.2f}  magic={p.magic}  ticket={p.ticket}")

print("\n=== DEALS (last 40) ===")
to = int(time.time())
frm = to - 7*24*3600
deals = mt5.history_deals_get(frm, to)
if not deals:
    print("  (none in last 7d)")
else:
    deals = sorted(deals, key=lambda d: d.time, reverse=True)[:40]
    for d in deals:
        t = time.strftime("%m-%d %H:%M", time.gmtime(d.time))
        print(f"  {t}  {d.symbol} {'BUY' if d.type==0 else 'SELL'}  "
              f"vol={d.volume}  price={d.price:.5f}  pnl={d.profit:.2f}  magic={d.magic}  ticket={d.ticket}")
mt5.shutdown()
