import numpy as np, pandas as pd, time
from sklearn.ensemble import RandomForestClassifier
from features_lib import POINT, TFS, build_blocks

RP = 24.0 * POINT
SYMS = [
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv"),
    ("EURUSD", r"C:\Users\huzey\Downloads\research handoff\EURUSD_M5_2023_2026.csv"),
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


def eval_stats(X, y, yr):
    tr = np.isin(yr, [2023, 2024]); te = np.isin(yr, [2025, 2026])
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=50,
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
    return dict(rate=hit.mean(), n=int(sel.sum()), pf6=pf(6/24), pf10=pf(10/24), pf20=pf(20/24))


for sym, fp in SYMS:
    t0 = time.time()
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    blocks, block_inds = build_blocks(df["ts"].to_numpy(), df["close"].to_numpy(float),
                                      df["high"].to_numpy(float), df["low"].to_numpy(float))
    F = entries(df); eb = F["ent_bar"].to_numpy()
    col_order = [(tf, ind) for _, tf in TFS for ind in block_inds[tf]]
    Xall = np.hstack([blocks[tf][eb] for _, tf in TFS])
    y = F["plus2"].to_numpy(); yr = F["year"].to_numpy()

    # importance model -> best TF per indicator
    clf0 = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=50,
                                  max_features="sqrt", class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    tr0 = np.isin(yr, [2023, 2024]); clf0.fit(Xall[tr0], y[tr0])
    imp = clf0.feature_importances_
    fdf = pd.DataFrame({"tf": [c[0] for c in col_order], "ind": [c[1] for c in col_order], "imp": imp})
    gi = fdf.groupby(["ind", "tf"])["imp"].sum().reset_index()
    best_tf = gi.loc[gi.groupby("ind")["imp"].idxmax()].set_index("ind")["tf"].to_dict()
    mask = np.array([(tf == best_tf[ind]) for tf, ind in col_order])
    Xsel = Xall[:, mask]
    print(f"\n===== {sym} =====", flush=True)
    print(f"features: all-TF={Xall.shape[1]}  pruned(best-TF/ind)={Xsel.shape[1]}", flush=True)

    base = y[np.isin(yr, [2025, 2026])].mean()
    a = eval_stats(Xall, y, yr); s = eval_stats(Xsel, y, yr)
    print(f"baseline plus2 (eval)= {base*100:.1f}%", flush=True)
    print(f"{'model':<14}{'rate':>7}{'n':>7}{'PF@6':>8}{'PF@10':>8}{'PF@20':>8}", flush=True)
    print(f"{'all-TF':<14}{a['rate']*100:>6.1f}%{a['n']:>7}{a['pf6']:>8.3f}{a['pf10']:>8.3f}{a['pf20']:>8.3f}", flush=True)
    print(f"{'optimal-TF':<14}{s['rate']*100:>6.1f}%{s['n']:>7}{s['pf6']:>8.3f}{s['pf10']:>8.3f}{s['pf20']:>8.3f}", flush=True)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)
print("\nOPTIMAL-TF APPLY DONE")
