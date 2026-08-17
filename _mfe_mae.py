import numpy as np, pandas as pd, pickle, time
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
ATR_TH = 1.04
WARM, WIN = 300, 18
THR = [0.2, 0.5, 0.8, 1.0, 1.5]
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


all_mfe = []; all_mae = []
for sym, fp, pkl in SYMS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close)
    blocks, _ = build_blocks(df["ts"].to_numpy(), close, high, low)
    r = pd.DataFrame({"ts": df["ts"], "open": close, "high": high, "low": low, "close": close})
    r = r.set_index("ts").resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
    m15_ts = r["ts"].to_numpy(); m15_h = r["high"].to_numpy(float); m15_l = r["low"].to_numpy(float); m15_c = r["close"].to_numpy(float)
    atr = atr_wilder(m15_h, m15_l, m15_c, 14)
    atr_mean = pd.Series(atr).rolling(100, min_periods=100).mean().to_numpy()
    atr_ratio = atr / atr_mean
    mom5 = m15_c / np.roll(m15_c, 5) - 1
    m5ts = pd.to_datetime(df["ts"]).to_numpy()
    e_idx = np.searchsorted(m5ts, m15_ts) - 1

    meta = pickle.load(open(pkl, "rb"))
    mdl_lb = meta["models"]["Lb"]; mdl_p2 = meta["models"]["plus2"]
    thr_lb = meta["thr"]["Lb"]; thr_p2 = meta["thr"]["plus2"]

    cands = []
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
        cands.append((e, d))
    cands = np.array(cands) if cands else np.empty((0, 2), int)
    mfe = []; mae = []
    if len(cands):
        E = cands[:, 0].astype(int); D = cands[:, 1].astype(int)
        Xc = np.hstack([blocks[tf][E] for _, tf in TFS])
        PLB = mdl_lb.predict_proba(Xc)[:, 1]; PP2 = mdl_p2.predict_proba(Xc)[:, 1]
        for idx in np.where((PLB >= thr_lb) & (PP2 >= thr_p2))[0]:
            e = int(E[idx]); d = int(D[idx]); ep = close[e]
            hi = high[e + 1:e + 1 + WIN]; lo = low[e + 1:e + 1 + WIN]
            if d == 1:
                fav = (hi - ep) / RP; adv = (ep - lo) / RP
            else:
                fav = (ep - lo) / RP; adv = (hi - ep) / RP
            mfe.append(float(fav.max())); mae.append(float(adv.max()))
    mfe = np.array(mfe); mae = np.array(mae)
    all_mfe.append(mfe); all_mae.append(mae)
    print(f"\n===== {sym} (gated entries, n={len(mfe)}) =====", flush=True)
    print(f"{'R thr':>6} | {'MFE>=':>8} {'MAE>=':>8}", flush=True)
    for t in THR:
        print(f"{t:>6.1f} | { (mfe>=t).mean()*100:>7.1f}% {(mae>=t).mean()*100:>7.1f}%", flush=True)
    win = (mfe >= 1.5) & (mae < 1.0)
    print(f"median MFE={np.median(mfe):.2f}R  median MAE={np.median(mae):.2f}R  "
          f"P(win: MFE>=1.5R AND MAE<1R)={win.mean()*100:.1f}%", flush=True)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)

combo_mfe = np.concatenate(all_mfe); combo_mae = np.concatenate(all_mae)
print(f"\n===== COMBINED (n={len(combo_mfe)}) =====", flush=True)
print(f"{'R thr':>6} | {'MFE>=':>8} {'MAE>=':>8}", flush=True)
for t in THR:
    print(f"{t:>6.1f} | { (combo_mfe>=t).mean()*100:>7.1f}% {(combo_mae>=t).mean()*100:>7.1f}%", flush=True)
print("DONE")
