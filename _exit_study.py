import numpy as np, pandas as pd, pickle, time
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
ATR_TH = 1.04
WARM, WIN = 300, 18   # 90-min holding window (matches strategy time-exit)
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


def sim_fixed(fh, fl, fc, tp, sl):
    for i in range(len(fh)):
        if fh[i] >= tp:
            return tp
        if fl[i] <= -sl:
            return -sl
    return fc[-1]


def sim_trail(fh, fl, fc, A, D, initSL):
    activated = False; peak = 0.0; stop = -initSL
    for i in range(len(fh)):
        if not activated:
            if fl[i] <= -initSL:
                return -initSL
            if fh[i] >= A:
                activated = True; peak = fh[i]; stop = max(peak - D, 0.0)
                if fl[i] <= stop:
                    return stop
        else:
            peak = max(peak, fh[i]); stop = max(peak - D, 0.0)
            if fl[i] <= stop:
                return stop
    return fc[-1]


SCHEMES = [
    ("FIX 1.5:1 (base)", "fixed", dict(tp=1.5, sl=1.0)),
    ("FIX 4:2",          "fixed", dict(tp=4.0, sl=2.0)),
    ("FIX 4:3",          "fixed", dict(tp=4.0, sl=3.0)),
    ("TRAIL A0.5 D0.5",  "trail", dict(A=0.5, D=0.5, initSL=1.0)),
    ("TRAIL A1.0 D0.5",  "trail", dict(A=1.0, D=0.5, initSL=1.0)),
    ("TRAIL A1.5 D0.5",  "trail", dict(A=1.5, D=0.5, initSL=1.0)),
    ("TRAIL A2.0 D0.5",  "trail", dict(A=2.0, D=0.5, initSL=1.0)),
    ("TRAIL A1.0 D1.0",  "trail", dict(A=1.0, D=1.0, initSL=1.0)),
]

all_pnl = {s[0]: [] for s in SCHEMES}
for sym, fp, pkl in SYMS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close)
    r = pd.DataFrame({"ts": df["ts"], "open": close, "high": high, "low": low, "close": close})
    r = r.set_index("ts").resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    m15_h = r["high"].to_numpy(float); m15_l = r["low"].to_numpy(float); m15_c = r["close"].to_numpy(float)
    atr = atr_wilder(m15_h, m15_l, m15_c, 14)
    atr_mean = pd.Series(atr).rolling(100, min_periods=100).mean().to_numpy()
    atr_ratio = atr / atr_mean
    mom5 = m15_c / np.roll(m15_c, 5) - 1
    m15_ts = r["ts"].to_numpy()
    m5ts = pd.to_datetime(df["ts"]).to_numpy()
    e_idx = np.searchsorted(m5ts, r["ts"].to_numpy()) - 1

    meta = pickle.load(open(pkl, "rb"))
    mdl_lb = meta["models"]["Lb"]; mdl_p2 = meta["models"]["plus2"]
    thr_lb = meta["thr"]["Lb"]; thr_p2 = meta["thr"]["plus2"]

    E = []; Dd = []
    for k in range(150, len(m15_ts)):
        ar = atr_ratio[k]; m = mom5[k]
        if np.isnan(ar) or np.isnan(m) or ar <= ATR_TH:
            continue
        d = 1 if m > 0 else (-1 if m < 0 else 0)
        if d == 0:
            continue
        e = e_idx[k]
        if e < WARM or e + WIN >= n:
            continue
        E.append(e); Dd.append(d)
    E = np.array(E); Dd = np.array(Dd)
    blocks, _ = build_blocks(df["ts"].to_numpy(), close, high, low)
    Xc = np.hstack([blocks[tf][E] for _, tf in TFS])
    PLB = mdl_lb.predict_proba(Xc)[:, 1]; PP2 = mdl_p2.predict_proba(Xc)[:, 1]
    keep = (PLB >= thr_lb) & (PP2 >= thr_p2)
    E = E[keep]; Dd = Dd[keep]
    ep = close[E]; dsign = Dd
    # forward windows in favorability-R coordinates (positive = in-trade direction)
    fwd = E[:, None] + np.arange(1, WIN + 1)
    Hw = high[fwd]; Lw = low[fwd]; Cw = close[fwd]
    fh = (Hw - ep[:, None]) / RP * dsign[:, None]
    fl = (Lw - ep[:, None]) / RP * dsign[:, None]
    fc = (Cw - ep[:, None]) / RP * dsign[:, None]

    print(f"\n===== {sym} (gated n={len(E)}) =====", flush=True)
    print(f"{'scheme':<18}{'win%':>7}{'netR':>8}{'PF':>8}{'P(loss)':>9}", flush=True)
    for name, kind, kw in SCHEMES:
        pnl = np.empty(len(E))
        for j in range(len(E)):
            if kind == "fixed":
                pnl[j] = sim_fixed(fh[j], fl[j], fc[j], kw["tp"], kw["sl"])
            else:
                pnl[j] = sim_trail(fh[j], fl[j], fc[j], kw["A"], kw["D"], kw["initSL"])
        gw = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
        pf = gw / gl if gl > 0 else 0
        pl = (pnl < 0).mean() * 100
        print(f"{name:<18}{(pnl>0).mean()*100:>6.1f}%{pnl.mean():>8.3f}{pf:>8.3f}{pl:>8.1f}%", flush=True)
        all_pnl[name].append(pnl)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)

print("\n===== COMBINED =====", flush=True)
print(f"{'scheme':<18}{'win%':>7}{'netR':>8}{'PF':>8}{'P(loss)':>9}", flush=True)
for name, kind, kw in SCHEMES:
    pnl = np.concatenate(all_pnl[name])
    gw = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else 0
    print(f"{name:<18}{(pnl>0).mean()*100:>6.1f}%{pnl.mean():>8.3f}{pf:>8.3f}{(pnl<0).mean()*100:>8.1f}%", flush=True)
print("DONE")
