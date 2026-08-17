import numpy as np, pandas as pd, pickle, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
ATR_TH = 1.04
WIN = 18  # 90 min in M5 bars

SYMS = [
    ("EURUSD", r"C:\Users\huzey\Downloads\research handoff\EURUSD_M5_2023_2026.csv",
     r"C:\Users\huzey\fsv\models\edge_models_eurusd.pkl"),
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_nzdusd.pkl"),
]

def sim_fixed(fh, fl, fv, tp=1.5, sl=1.0):
    hmax = fh.max(); lmin = fl.min()
    if hmax >= tp:
        return fv[fh.argmax()]
    if lmin <= -sl:
        return fv[fl.argmin()]
    return fv[-1]

def atr_wilder(h, l, c, p=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    out = np.empty(len(tr)); out[p - 1] = tr[:p].mean()
    for i in range(p, len(tr)):
        out[i] = (out[i - 1] * (p - 1) + tr[i]) / p
    return np.concatenate([[np.nan], out])

def stats(pnl):
    pnl = np.asarray(pnl)
    gw = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    pf = gw / gl if gl > 0 else 0
    return (pnl > 0).mean() * 100, pnl.mean(), pf

for sym, fp, pkl in SYMS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close)

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

    blocks, _ = build_blocks(df["ts"].to_numpy(), close, high, low)

    E_all, D_all = [], []
    for k in range(150, len(m15_ts)):
        ar = atr_ratio[k]; m = mom5[k]
        if np.isnan(ar) or np.isnan(m) or ar <= ATR_TH:
            continue
        d = 1 if m > 0 else (-1 if m < 0 else 0)
        if d == 0:
            continue
        e = e_idx[k]
        if e < 300 or e + 96 >= n:
            continue
        E_all.append(e); D_all.append(d)
    E_all = np.array(E_all); D_all = np.array(D_all)
    Xc = np.hstack([blocks[tf][E_all] for _, tf in TFS])
    PLB = mdl_lb.predict_proba(Xc)[:, 1]; PP2 = mdl_p2.predict_proba(Xc)[:, 1]
    gmask = (PLB >= thr_lb) & (PP2 >= thr_p2)

    def analyze(E, D, label):
        ep = close[E]
        fwd = E[:, None] + np.arange(1, WIN + 1)
        Hw = high[fwd]; Lw = low[fwd]; Cw = close[fwd]

        fav = D[:, None] * (Hw - ep[:, None]) / RP
        adv = D[:, None] * (Lw - ep[:, None]) / RP
        clo = D[:, None] * (Cw - ep[:, None]) / RP

        pnl_actual = np.array([sim_fixed(fav[i], adv[i], clo[i]) for i in range(len(E))])
        pnl_flip = np.array([sim_fixed(-fav[i], -adv[i], -clo[i]) for i in range(len(E))])

        rng = np.random.default_rng(7)
        prnd = []
        for _ in range(5):
            s = rng.choice([-1, 1], size=len(E))
            fa = s[:, None] * (Hw - ep[:, None]) / RP
            la = s[:, None] * (Lw - ep[:, None]) / RP
            ca = s[:, None] * (Cw - ep[:, None]) / RP
            prnd.append(np.array([sim_fixed(fa[i], la[i], ca[i]) for i in range(len(E))]))
        pnl_rand = np.mean(prnd, axis=0)

        mfe = fav.max(axis=1); mae = adv.min(axis=1)
        dir_right = mfe >= -mae

        print(f"\n  --- {label} (n={len(E)}) ---", flush=True)
        print(f"  directional dominance (MFE>=|MAE|): {dir_right.mean()*100:.1f}%  "
              f"median MFE={np.median(mfe):.2f}R  median|MAE|={np.median(-mae):.2f}R", flush=True)
        for nm, pnl in [("ACTUAL ", pnl_actual), ("FLIPPED", pnl_flip), ("RANDOM ", pnl_rand)]:
            w, nv, pf = stats(pnl)
            print(f"  {nm} dir: win={w:5.1f}%  netR={nv:+.3f}  PF={pf:.3f}", flush=True)
        if len(pnl_actual[dir_right]):
            print(f"  -> right-dir (n={len(pnl_actual[dir_right])}): win={(pnl_actual[dir_right]>0).mean()*100:5.1f}% netR={pnl_actual[dir_right].mean():+.3f}", flush=True)
        if len(pnl_actual[~dir_right]):
            print(f"  -> wrong-dir (n={len(pnl_actual[~dir_right])}): win={(pnl_actual[~dir_right]>0).mean()*100:5.1f}% netR={pnl_actual[~dir_right].mean():+.3f}", flush=True)

    analyze(E_all, D_all, "UNGATED (vol+momentum)")
    analyze(E_all[gmask], D_all[gmask], "GATED (vol+mom+ML)")
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)
print("\nDIR-EDGE DONE")
