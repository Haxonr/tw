import numpy as np, pandas as pd, pickle, time
from sklearn.ensemble import RandomForestClassifier
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
PAIRS = [
    ("AUDUSD", r"C:\Users\huzey\Downloads\trading_system\data\AUDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_audusd.pkl"),
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_nzdusd.pkl"),
    ("USDCAD", r"C:\Users\huzey\Downloads\trading_system\data\USDCAD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_usdcad.pkl"),
]
WARM, STEP, H = 300, 48, 96


def entries(df):
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close); rows = []
    for i in range(WARM, n - H, STEP):
        d = 1 if (close[i] - close[i - 20]) > 0 else -1; ep = close[i]
        mfe = 0.0; mae = 0.0; t2r = 10**9; t4r = 10**9; t1a = 10**9; t2a = 10**9
        for j in range(1, H + 1):
            hi = high[i + j]; lo = low[i + j]
            mf = (hi - ep) / RP if d == 1 else (ep - lo) / RP
            ma = (ep - lo) / RP if d == 1 else (hi - ep) / RP
            if mf > mfe: mfe = mf
            if ma > mae: mae = ma
            if mf >= 2 and j < t2r: t2r = j
            if mf >= 4 and j < t4r: t4r = j
            if ma >= 1 and j < t1a: t1a = j
            if ma >= 2 and j < t2a: t2a = j
        La = 1 if (mae > 0 and mfe >= 2 * mae) else 0
        Lc = 1 if (3 <= mfe <= 6) else 0
        Lb = 1 if (t2r < t1a or t4r < t2a) else 0
        if d == 1:
            tp = (high[i + 1:i + H + 1] >= ep + 2 * RP); sl = (low[i + 1:i + H + 1] <= ep - RP)
        else:
            tp = (low[i + 1:i + H + 1] <= ep - 2 * RP); sl = (high[i + 1:i + H + 1] >= ep + RP)
        plus2 = 0
        if tp.any() or sl.any():
            ft = np.argmax(tp) if tp.any() else 10**9
            fs = np.argmax(sl) if sl.any() else 10**9
            plus2 = 1 if ft < fs else 0
        rows.append([i, int(df["ts"].iloc[i].year), d, La, Lb, Lc, plus2])
    return pd.DataFrame(rows, columns=["ent_bar", "year", "dir", "La", "Lb", "Lc", "plus2"])


def eval_tf(X, y, year, tag):
    tr = year.isin([2023, 2024]).to_numpy(); te = year.isin([2025, 2026]).to_numpy()
    if len(np.unique(y[tr])) < 2 or te.sum() == 0:
        return None
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[tr])[:, 1]; cut = float(np.quantile(p, 0.66))
    pte = clf.predict_proba(X[te])[:, 1]; sel = pte >= cut
    if sel.sum() == 0:
        return None
    hit = y[te][sel]
    def pf(c):
        nw = (hit * (2 - c)).sum(); nl = ((1 - hit) * (1 + c)).sum()
        return nw / nl if nl > 0 else 0
    return dict(hit=hit.mean(), n=int(sel.sum()), pf6=pf(6/24), pf10=pf(10/24), pf20=pf(20/24))


for sym, fp, _ in PAIRS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    blocks, _ = build_blocks(df["ts"].to_numpy(), df["close"].to_numpy(float),
                             df["high"].to_numpy(float), df["low"].to_numpy(float))
    F = entries(df); eb = F["ent_bar"].to_numpy()
    print(f"\n=== {sym} (per-TF plus2, train 23-24 / eval 25-26) ===", flush=True)
    print(f"{'TF':<6}{'top%+':>8}{'n':>7}{'PF@6':>8}{'PF@10':>8}{'PF@20':>8}", flush=True)
    for rule, tf in TFS:
        res = eval_tf(blocks[tf][eb], F["plus2"].to_numpy(), F["year"], tf)
        if res is None:
            print(f"{tf:<6}{'n/a':>8}", flush=True)
        else:
            print(f"{tf:<6}{res['hit']*100:>7.1f}%{res['n']:>7}{res['pf6']:>8.3f}"
                  f"{res['pf10']:>8.3f}{res['pf20']:>8.3f}", flush=True)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)
print("\nALL PER-TF SWEEPS DONE")
