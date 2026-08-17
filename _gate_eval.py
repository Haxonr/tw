import numpy as np, pandas as pd, pickle, time
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
ATR_TH = 1.04
WARM, H = 300, 96
SYMS = [
    ("EURUSD", r"C:\Users\huzey\Downloads\research handoff\EURUSD_M5_2023_2026.csv",
     r"C:\Users\huzey\fsv\models\edge_models_eurusd.pkl"),
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_nzdusd.pkl"),
]


def atr_wilder(h, l, c, p=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    out = np.empty(len(tr)); out[p - 1] = tr[:p].mean()
    for i in range(p, len(tr)):
        out[i] = (out[i - 1] * (p - 1) + tr[i]) / p
    return np.concatenate([[np.nan], out])


for sym, fp, pkl in SYMS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close)
    blocks, block_inds = build_blocks(df["ts"].to_numpy(), close, high, low)
    col_order = [(tf, ind) for _, tf in TFS for ind in block_inds[tf]]
    Xall = np.hstack([blocks[tf] for _, tf in TFS])

    # M15 series for vol trigger + direction
    r = pd.DataFrame({"ts": df["ts"], "open": close, "high": high, "low": low, "close": close})
    r = r.set_index("ts").resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    m15_ts = r["ts"].to_numpy(); m15_h = r["high"].to_numpy(float); m15_l = r["low"].to_numpy(float); m15_c = r["close"].to_numpy(float)
    atr = atr_wilder(m15_h, m15_l, m15_c, 14)
    atr_mean = pd.Series(atr).rolling(100, min_periods=100).mean().to_numpy()
    atr_ratio = atr / atr_mean
    mom5 = m15_c / np.roll(m15_c, 5) - 1
    m15_year = r["ts"].dt.year.to_numpy()
    m5ts = pd.to_datetime(df["ts"]).to_numpy()
    e_idx = np.searchsorted(m5ts, m15_ts) - 1

    meta = pickle.load(open(pkl, "rb"))
    mdl_lb = meta["models"]["Lb"]; mdl_p2 = meta["models"]["plus2"]
    thr_lb = meta["thr"]["Lb"]; thr_p2 = meta["thr"]["plus2"]

    # collect candidate bars (vol-burst + valid direction + enough history)
    cands = []
    triggered = 0; bad_dir = 0
    for k in range(150, len(m15_ts)):
        ar = atr_ratio[k]; m = mom5[k]
        if np.isnan(ar) or np.isnan(m) or ar <= ATR_TH:
            continue
        triggered += 1
        d = 1 if m > 0 else (-1 if m < 0 else 0)
        if d == 0:
            bad_dir += 1
            continue
        e = e_idx[k]
        if e < WARM or e + H >= n:
            continue
        cands.append((e, d, int(m15_year[k])))
    triggered = triggered  # total vol-burst bars
    cands = np.array(cands) if cands else np.empty((0, 3), int)
    blocked = 0; pnl = []; yr = []
    if len(cands):
        E = cands[:, 0].astype(int); D = cands[:, 1].astype(int); Y = cands[:, 2].astype(int)
        Xc = Xall[E]
        PLB = mdl_lb.predict_proba(Xc)[:, 1]
        PP2 = mdl_p2.predict_proba(Xc)[:, 1]
        passmask = (PLB >= thr_lb) & (PP2 >= thr_p2)
        blocked = int((~passmask).sum())
        TE = 18  # 6 M15 bars = 90 min time exit (in M5 bars)
        for idx in np.where(passmask)[0]:
            e = int(E[idx]); d = int(D[idx]); yv = int(Y[idx]); ep = close[e]
            if d == 1:
                tp = high[e + 1:e + 1 + TE] >= ep + 1.5 * RP
                sl = low[e + 1:e + 1 + TE] <= ep - 1.0 * RP
            else:
                tp = low[e + 1:e + 1 + TE] <= ep - 1.5 * RP
                sl = high[e + 1:e + 1 + TE] >= ep + 1.0 * RP
            ft = np.argmax(tp) if tp.any() else 10**9
            fs = np.argmax(sl) if sl.any() else 10**9
            if ft < fs:
                p = 1.5
            elif fs < ft:
                p = -1.0
            else:
                p = d * (close[e + TE] - ep) / RP  # time exit at market
            pnl.append(p); yr.append(yv)

    pnl = np.array(pnl); yr = np.array(yr)
    print(f"\n===== {sym} (FULL filter: atr_ratio>1.04 AND gate[Lb&plus2]) =====", flush=True)
    print(f"triggered(vol-burst)={triggered}  blocked_by_gate={blocked}  passed={len(pnl)}", flush=True)
    for label, mask in [("FULL 23-26", np.ones(len(pnl), bool)),
                        ("EVAL 25-26", yr >= 2025)]:
        sub = pnl[mask] if len(pnl) else pnl
        if len(sub) == 0:
            print(f"  {label}: no trades", flush=True); continue
        gw = sub[sub > 0].sum(); gl = -sub[sub < 0].sum()
        pf = gw / gl if gl > 0 else 0
        win = (sub > 0).mean() * 100
        print(f"  {label}: n={len(sub)} win={win:.1f}%  netR/trade={sub.mean():.3f}  PF={pf:.3f}", flush=True)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)
print("\nFULL-FILTER EVAL DONE")
