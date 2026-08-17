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
    tr = np.isin(yr, [2023, 2024]); te = np.isin(yr, [2025, 2026])
    clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=50,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(Xall[tr], y[tr])
    imp = clf.feature_importances_
    fdf = pd.DataFrame({"tf": [c[0] for c in col_order],
                        "ind": [c[1] for c in col_order],
                        "imp": imp})
    print(f"\n===== {sym}: feature-importance TF decomposition =====", flush=True)
    share = fdf.groupby("tf")["imp"].sum().sort_values(ascending=False)
    share = (share / share.sum() * 100).round(1)
    print("TF importance share (% of total model importance):", flush=True)
    print(share.to_string(), flush=True)

    # best TF per indicator (sum importance across its lag/diff/z variants per TF)
    gi = fdf.groupby(["ind", "tf"])["imp"].sum().reset_index()
    best = gi.loc[gi.groupby("ind")["imp"].idxmax()]
    best = best.sort_values("imp", ascending=False)
    print(f"\nTop 20 indicators by total importance (with their OPTIMAL TF):", flush=True)
    print(f"{'indicator':<12}{'best_TF':>8}{'imp%':>8}   per-TF imp%",
          flush=True)
    tot = fdf.groupby('ind')['imp'].sum()
    for _, r in best.head(20).iterrows():
        ind = r["ind"]
        row = gi[gi.ind == ind].set_index("tf")["imp"].sort_index()
        per = " ".join(f"{t}:{row.get(t,0)*100/tot[ind]:.0f}" for t in ["M5","M15","M30","H1","H4"])
        print(f"{ind:<12}{r['tf']:>8}{r['imp']*100:>7.1f}%  {per}", flush=True)
    print(f"[{sym}] done in {time.time()-t0:.0f}s", flush=True)
print("\nFEATURE TF SWEEP DONE")
