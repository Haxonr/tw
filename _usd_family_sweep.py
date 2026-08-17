import numpy as np, pandas as pd, time, pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from features_lib import (POINT, Rp, TFS, build_blocks, feature_row)

# USD-family FX pairs quoted in 1e-5 (same scale as EURUSD/GBPUSD).
# Same labels (La/Lb/Lc/plus2) as tf_sweep.py. Rp=24e-5 reference R.
PAIRS = [
    ("AUDUSD", r"C:\Users\huzey\Downloads\trading_system\data\AUDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_audusd.pkl"),
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_nzdusd.pkl"),
    ("USDCAD", r"C:\Users\huzey\Downloads\trading_system\data\USDCAD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_usdcad.pkl"),
]


def train(symbol, fp, outpkl):
    rp = 24.0 * POINT  # same reference R as the other sweeps
    t0 = time.time()
    df = pd.read_csv(fp)
    df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)  # infers "YYYY-MM-DD HH:MM:SS"
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    n = len(close)
    print(f"[{symbol}] loaded n={n}", flush=True)

    blocks, block_inds = build_blocks(df["ts"].to_numpy(), close, high, low)
    print(f"[{symbol}] built all TF blocks ({time.time()-t0:.0f}s)", flush=True)

    COSTS = [6.0 / 24.0, 10.0 / 24.0, 20.0 / 24.0]
    COST_LBL = ["typ(6pt RT)", "prior(10pt RT)", "max(20pt RT)"]
    WARM, STEP, H = 300, 48, 96
    rows = []
    for i in range(WARM, n - H, STEP):
        d = 1 if (close[i] - close[i - 20]) > 0 else -1
        ep = close[i]
        mfe = 0.0; mae = 0.0; t2r = 10 ** 9; t4r = 10 ** 9; t1a = 10 ** 9; t2a = 10 ** 9
        for j in range(1, H + 1):
            hi = high[i + j]; lo = low[i + j]
            mf = (hi - ep) / rp if d == 1 else (ep - lo) / rp
            ma = (ep - lo) / rp if d == 1 else (hi - ep) / rp
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
            tp = (high[i + 1:i + H + 1] >= ep + 2 * rp)
            sl = (low[i + 1:i + H + 1] <= ep - 1 * rp)
        else:
            tp = (low[i + 1:i + H + 1] <= ep - 2 * rp)
            sl = (high[i + 1:i + H + 1] >= ep + 1 * rp)
        plus2 = 0
        if tp.any() or sl.any():
            ft = np.argmax(tp) if tp.any() else 10 ** 9
            fs = np.argmax(sl) if sl.any() else 10 ** 9
            plus2 = 1 if ft < fs else 0
        rows.append([i, int(df["ts"].iloc[i].year), d, La, Lb, Lc, plus2])
    F = pd.DataFrame(rows, columns=["ent_bar", "year", "dir", "La", "Lb", "Lc", "plus2"])
    print(f"[{symbol}] entries n={len(F)} base: La={F.La.mean()*100:.1f}% "
          f"Lb={F.Lb.mean()*100:.1f}% Lc={F.Lc.mean()*100:.1f}% plus2={F.plus2.mean()*100:.1f}%",
          flush=True)

    eb = F["ent_bar"].to_numpy()
    Xtf = {tf: blocks[tf][eb] for tf in blocks}
    Xall = np.hstack([blocks[tf][eb] for _, tf in TFS])
    meta_all = [(tf, ind) for _, tf in TFS for ind in block_inds[tf]]

    def pf_net(plus2, cost_R):
        nw = (plus2 * (2 - cost_R)).sum()
        nl = ((1 - plus2) * (1 + cost_R)).sum()
        return nw / nl if nl > 0 else 0

    def eval_model(X, y, yname, tag):
        tr_idx = F.year.isin([2023, 2024])
        te_idx = F.year.isin([2025, 2026])
        if len(np.unique(y[tr_idx])) < 2:
            return None
        clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr_idx], y[tr_idx])
        te = F[te_idx].copy()
        te["p"] = clf.predict_proba(X[te_idx])[:, 1]
        base = te[yname].mean()
        pred = (te.p >= 0.5).astype(int)
        acc = (pred == te[yname].values).mean()
        maj = max(base, 1 - base)
        cut = te.p.quantile(0.66)
        hi = te[te.p >= cut]
        return dict(base=base, acc=acc, maj=maj, top_plus2=hi[yname].to_numpy(), n=len(hi))

    models = {}; thr = {}
    for tgt in ["La", "Lb", "Lc", "plus2"]:
        y = F[tgt].to_numpy()
        tr_idx = F.year.isin([2023, 2024])
        clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(Xall[tr_idx], y[tr_idx])
        p = clf.predict_proba(Xall[tr_idx])[:, 1]
        models[tgt] = clf
        thr[tgt] = float(np.quantile(p, 0.66))
        print(f"  [{symbol} {tgt}]: train_pos={y[tr_idx].mean()*100:.1f}% "
              f"thr(0.66q)={thr[tgt]:.3f}", flush=True)

    meta = dict(models=models, thr=thr, col_order=meta_all, tfs=TFS, rp=rp,
                labels=["La", "Lb", "Lc", "plus2"], symbol=symbol)
    with open(outpkl, "wb") as f:
        pickle.dump(meta, f)
    print(f"[{symbol}] saved {outpkl} ({time.time()-t0:.0f}s)\n", flush=True)


if __name__ == "__main__":
    for sym, fp, out in PAIRS:
        train(sym, fp, out)
    print("ALL USD-FAMILY SWEEPS DONE")
