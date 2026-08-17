import numpy as np, pandas as pd, pickle, time
from features_lib import POINT, Rp, TFS, build_blocks, feature_row

PAIRS = [
    ("AUDUSD", r"C:\Users\huzey\Downloads\trading_system\data\AUDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_audusd.pkl"),
    ("NZDUSD", r"C:\Users\huzey\Downloads\trading_system\data\NZDUSD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_nzdusd.pkl"),
    ("USDCAD", r"C:\Users\huzey\Downloads\trading_system\data\USDCAD_M5_aligned.csv",
     r"C:\Users\huzey\fsv\models\edge_models_usdcad.pkl"),
]
RP = 24.0 * POINT

def entries(df):
    df = df.sort_values("ts").reset_index(drop=True)
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    n = len(close); rows = []
    for i in range(300, n - 96, 48):
        d = 1 if (close[i] - close[i-20]) > 0 else -1; ep = close[i]
        mfe = 0.0; mae = 0.0; t2r = 10**9; t4r = 10**9; t1a = 10**9; t2a = 10**9
        for j in range(1, 97):
            hi = high[i+j]; lo = low[i+j]
            mf = (hi-ep)/RP if d == 1 else (ep-lo)/RP
            ma = (ep-lo)/RP if d == 1 else (hi-ep)/RP
            if mf > mfe: mfe = mf
            if ma > mae: mae = ma
            if mf >= 2 and j < t2r: t2r = j
            if mf >= 4 and j < t4r: t4r = j
            if ma >= 1 and j < t1a: t1a = j
            if ma >= 2 and j < t2a: t2a = j
        plus2 = 0
        if d == 1:
            tp = (high[i+1:i+97] >= ep+2*RP); sl = (low[i+1:i+97] <= ep-RP)
        else:
            tp = (low[i+1:i+97] <= ep-2*RP); sl = (high[i+1:i+97] >= ep+RP)
        if tp.any() or sl.any():
            ft = np.argmax(tp) if tp.any() else 10**9
            fs = np.argmax(sl) if sl.any() else 10**9
            plus2 = 1 if ft < fs else 0
        rows.append([i, int(df["ts"].iloc[i].year), d, plus2])
    return pd.DataFrame(rows, columns=["ent_bar","year","dir","plus2"])

for sym, fp, pkl in PAIRS:
    m = pickle.load(open(pkl,"rb"))
    assert m["symbol"] == sym, (m["symbol"], sym)
    assert len(m["col_order"]) == 1885, len(m["col_order"])
    assert abs(m["rp"] - RP) < 1e-12
    df = pd.read_csv(fp); df.columns = [c.lower() for c in df.columns]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    blocks, _ = build_blocks(df["ts"].to_numpy(), df["close"].to_numpy(float),
                             df["high"].to_numpy(float), df["low"].to_numpy(float))
    F = entries(df); eb = F["ent_bar"].to_numpy()
    Xall = np.hstack([blocks[tf][eb] for _, tf in TFS])
    clf = m["models"]["plus2"]; thr = m["thr"]["plus2"]
    te = F[F.year.isin([2025,2026])]; tei = F.year.isin([2025,2026]).to_numpy()
    p = clf.predict_proba(Xall[tei])[:,1]
    sel = p >= thr
    base = te["plus2"].mean()
    if sel.sum() == 0:
        print(f"{sym}: INTSCT empty; base plus2={base*100:.1f}%"); continue
    hit = te["plus2"].to_numpy()[sel]
    for cdesc, c in [("6pt",6/24),("10pt",10/24),("20pt",20/24)]:
        nw = (hit*(2-c)).sum(); nl = ((1-hit)*(1+c)).sum()
        pf = nw/nl if nl>0 else 0
        print(f"{sym}: top-tercile plus2 rate={hit.mean()*100:.1f}% (n={sel.sum()}) "
              f"PF@{cdesc} RT={pf:.3f}")
    print(f"{sym}: bundle OK (symbol={sym}, col_order=1885, rp=24e-5)\n", flush=True)
print("DONE")
