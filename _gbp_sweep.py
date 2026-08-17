"""
TF SWEEP -- EURUSD. For each label (La/Lb/Lc/plus2), find the optimal timeframe per feature.
  - compute ~29 base indicators on TFs M5/M15/M30/H1/H4, expand each to a feature block (lags/diffs/rolling-z)
  - single-TF sweep: best TF per label by OOS skill + top-tercile net PF
  - all-TF model: feature importances aggregated per (indicator, TF) -> optimal TF per feature
  - SAVES trained all-TF RF models (La/Lb/Lc/plus2) + top-tercile thresholds to edge_models.pkl
    for live_confirm.py / sloppy_live_bot confirmation filter.

R = 24 points (2.4 pips), cost = 10 points = 0.42R.  Lookahead-free (features causal, labels are targets).
"""
import numpy as np, pandas as pd, time, re, pickle
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from features_lib import (POINT, Rp, TFS, build_blocks, feature_row)

# GBPUSD override: R = 21 points (matches strategy R_POINTS["GBPUSD"])
Rp = 21.0 * POINT

t0 = time.time()
fp = r"C:\Users\huzey\Downloads\research handoff\GBPUSD_M5_2023_2026.csv"
df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
df = df.sort_values("ts").reset_index(drop=True)
close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float); n = len(close)
print(f"loaded n={n}", flush=True)

# ---------- build per-TF blocks, mapped to M5 bars (shared with live_filter) ----------
blocks, block_inds = build_blocks(df["ts"].to_numpy(), close, high, low)
print(f"built all TF blocks ({time.time()-t0:.0f}s)", flush=True)

# ---------- entries + labels (R=24pts, cost 0.42R) ----------
COSTS = [6.0 / 21.0, 10.0 / 21.0, 20.0 / 21.0]
COST_LBL = ["typ(6pt RT)", "prior(10pt RT)", "max(20pt RT)"]
WARM, STEP, H = 300, 48, 96
rows = []
for i in range(WARM, n - H, STEP):
    d = 1 if (close[i] - close[i - 20]) > 0 else -1
    ep = close[i]
    mfe = 0.0; mae = 0.0; t2r = 10 ** 9; t4r = 10 ** 9; t1a = 10 ** 9; t2a = 10 ** 9
    for j in range(1, H + 1):
        hi = high[i + j]; lo = low[i + j]
        mf = (hi - ep) / Rp if d == 1 else (ep - lo) / Rp
        ma = (ep - lo) / Rp if d == 1 else (hi - ep) / Rp
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
        tp = (high[i + 1:i + H + 1] >= ep + 2 * Rp); sl = (low[i + 1:i + H + 1] <= ep - 1 * Rp)
    else:
        tp = (low[i + 1:i + H + 1] <= ep - 2 * Rp); sl = (high[i + 1:i + H + 1] >= ep + 1 * Rp)
    plus2 = 0
    if tp.any() or sl.any():
        ft = np.argmax(tp) if tp.any() else 10 ** 9; fs = np.argmax(sl) if sl.any() else 10 ** 9
        plus2 = 1 if ft < fs else 0
    rows.append([i, int(df["ts"].iloc[i].year), d, La, Lb, Lc, plus2])
F = pd.DataFrame(rows, columns=["ent_bar", "year", "dir", "La", "Lb", "Lc", "plus2"])
print(f"entries n={len(F)} base: La={F.La.mean()*100:.1f}% Lb={F.Lb.mean()*100:.1f}% "
      f"Lc={F.Lc.mean()*100:.1f}% plus2={F.plus2.mean()*100:.1f}%", flush=True)

# align feature blocks to entries
eb = F["ent_bar"].to_numpy()
Xtf = {tf: blocks[tf][eb] for tf in blocks}
Xall = np.hstack([Xtf[tf] for _, tf in TFS])
meta_all = [(tf, ind) for _, tf in TFS for ind in block_inds[tf]]
print(f"Xall shape={Xall.shape} ({time.time()-t0:.0f}s)", flush=True)

def pf_net(plus2, cost_R):
    nw = (plus2 * (2 - cost_R)).sum(); nl = ((1 - plus2) * (1 + cost_R)).sum()
    return nw / nl if nl > 0 else 0

def eval_model(X, y, yname, tag):
    tr_idx = F.year.isin([2023, 2024]); te_idx = F.year.isin([2025, 2026])
    if len(np.unique(y[tr_idx])) < 2:
        return None
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                 max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X[tr_idx], y[tr_idx])
    te = F[te_idx].copy(); te["p"] = clf.predict_proba(X[te_idx])[:, 1]
    base = te[yname].mean(); pred = (te.p >= 0.5).astype(int)
    acc = (pred == te[yname].values).mean(); maj = max(base, 1 - base)
    cut = te.p.quantile(0.66); hi = te[te.p >= cut]
    return dict(base=base, acc=acc, maj=maj, top_plus2=hi[yname].to_numpy(), n=len(hi))

# single-TF sweep
print("\n=== SINGLE-TF SWEEP (OOS 25-26): best TF per label ===", flush=True)
sweep_res = {}
for tgt in ["La", "Lb", "Lc", "plus2"]:
    y = F[tgt].to_numpy()
    best = None
    print(f" [{tgt}] base={y[F.year.isin([2025, 2026])].mean()*100:.1f}%", flush=True)
    for tf in blocks:
        r = eval_model(Xtf[tf], y, tgt, tf)
        if r:
            pfs = " ".join(f"{lbl}={pf_net(r['top_plus2'], cr):.2f}" for lbl, cr in zip(COST_LBL, COSTS))
            print(f"    {tf:4s}: acc={r['acc']*100:.1f}% vs maj={r['maj']*100:.1f}% skill={(r['acc']-r['maj'])*100:+.1f}pp  TOP1/3 PF [{pfs}]", flush=True)
            if best is None or pf_net(r['top_plus2'], COSTS[0]) > best['pf0']:
                best = dict(tf=tf, pf0=pf_net(r['top_plus2'], COSTS[0]))
    sweep_res[tgt] = best

# all-TF: optimal TF per feature
print("\n=== ALL-TF MODEL: optimal TF per feature (top by importance) ===", flush=True)
for tgt in ["La", "Lb", "Lc", "plus2"]:
    y = F[tgt].to_numpy()
    tr_idx = F.year.isin([2023, 2024]); te_idx = F.year.isin([2025, 2026])
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                 max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xall[tr_idx], y[tr_idx])
    imp = clf.feature_importances_
    agg = defaultdict(float)
    for k, (tf, ind) in enumerate(meta_all):
        agg[(ind, tf)] += imp[k]
    best_tf = {}; best_val = {}; tot = defaultdict(float)
    for (ind, tf), v in agg.items():
        tot[ind] += v
        if ind not in best_val or v > best_val[ind]:
            best_val[ind] = v; best_tf[ind] = tf
    all_inds = sorted(set(ind for (_, ind) in meta_all))
    cols = []
    for ind in all_inds:
        tf = best_tf[ind]
        mask = [j for j, ii in enumerate(block_inds[tf]) if ii == ind]
        cols.append(blocks[tf][eb][:, mask])
    Xsel = np.hstack(cols)
    rsel = eval_model(Xsel, y, tgt, "selTF")
    if rsel:
        pfs = " ".join(f"{lbl}={pf_net(rsel['top_plus2'], cr):.2f}" for lbl, cr in zip(COST_LBL, COSTS))
        print(f"    >> OPTIMAL-TF-ASSEMBLED: acc={rsel['acc']*100:.1f}% vs maj={rsel['maj']*100:.1f}% "
              f"skill={(rsel['acc']-rsel['maj'])*100:+.1f}pp  TOP1/3 PF [{pfs}]", flush=True)
    top = sorted(tot, key=lambda x: -tot[x])[:22]
    print(f"\n [{tgt}] optimal-TF-per-feature (top {len(top)} by total importance):", flush=True)
    for ind in top:
        print(f"    {ind:12s} bestTF={best_tf[ind]:4s}  imp%={tot[ind]*100:.2f}", flush=True)

# ---------- save all-TF models for live confirmation filter ----------
print("\n=== SAVING MODELS ===", flush=True)
models = {}; thr = {}
for tgt in ["La", "Lb", "Lc", "plus2"]:
    y = F[tgt].to_numpy()
    tr_idx = F.year.isin([2023, 2024])
    clf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=50,
                                 max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xall[tr_idx], y[tr_idx])
    p = clf.predict_proba(Xall[tr_idx])[:, 1]
    models[tgt] = clf
    thr[tgt] = float(np.quantile(p, 0.66))
    print(f"  {tgt}: train_pos={y[tr_idx].mean()*100:.1f}% thr(0.66q)={thr[tgt]:.3f}", flush=True)
meta = dict(models=models, thr=thr, col_order=meta_all, tfs=TFS, rp=Rp,
            labels=["La", "Lb", "Lc", "plus2"], symbol="GBPUSD")
with open(r"C:\Users\huzey\fsv\models\edge_models_gbp.pkl", "wb") as f:
    pickle.dump(meta, f)
print(f"saved edge_models.pkl ({time.time()-t0:.0f}s)", flush=True)
print(f"\ndone ({time.time()-t0:.0f}s)", flush=True)
