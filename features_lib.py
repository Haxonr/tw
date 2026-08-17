"""
Shared indicators + per-TF feature-block builder for the EURUSD lookahead-free edge models.
Used by BOTH tf_sweep.py (training) and live_filter.py (live prediction) so the feature
column order is identical between train and serve.

R = 24 points = 0.00024 = 2.4 pips (POINT = 1e-5). Reference only.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

POINT = 1e-5
Rp = 24.0 * POINT  # 24 points = 2.4 pips (reference R used by labels / cost)

# ----------------------------------------------------------------- indicator helpers
def atr(h, l, c, p=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    return pd.Series(tr).rolling(p, min_periods=p).mean().to_numpy()

def rmean(s, w):
    return pd.Series(s).rolling(w, min_periods=w).mean().to_numpy()

def rstd(s, w):
    return pd.Series(s).rolling(w, min_periods=w).std().to_numpy()

def ema(s, sp):
    return pd.Series(s).ewm(span=sp, min_periods=sp).mean().to_numpy()

def shift(s, k):
    return pd.Series(s).shift(k).to_numpy()

def rsi(c, p):
    d = pd.Series(c).diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    return (100 - 100 / (1 + g.rolling(p, min_periods=p).mean() / (l.rolling(p, min_periods=p).mean() + 1e-9))).to_numpy()

def stoch(h, l, c, p=14):
    ll = pd.Series(l).rolling(p, min_periods=p).min(); hh = pd.Series(h).rolling(p, min_periods=p).max()
    return (100 * (c - ll) / (hh - ll + 1e-9)).to_numpy()

def cci(h, l, c, p=20):
    tp = (h + l + c) / 3
    return (tp - rmean(tp, p)) / (0.015 * (rstd(tp, p) + 1e-9))

def adx_di(h, l, c, p=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    up = pd.Series(h).diff(); dn = -pd.Series(l).diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0); minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_ = pd.Series(tr).rolling(p, min_periods=p).mean()
    pdi = 100 * pd.Series(plus).rolling(p, min_periods=p).sum() / atr_
    mdi = 100 * pd.Series(minus).rolling(p, min_periods=p).sum() / atr_
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-9)
    return pd.Series(dx).rolling(p, min_periods=p).mean().to_numpy(), pdi.to_numpy(), mdi.to_numpy()

def detect_fvgs(h, l, ts):
    out = []
    for i in range(2, len(h)):
        if l[i] > h[i - 2]:
            out.append((i, ts[i], l[i], h[i - 2]))
        if h[i] < l[i - 2]:
            out.append((i, ts[i], l[i - 2], h[i]))
    return out

def fvg_self(h, l, ts, atr_m, K=200):
    fvgs = detect_fvgs(h, l, ts); m = len(h); dist = np.full(m, np.nan); ptr = 0; bb = None; ba = None
    for t in range(m):
        px = h[t]; a = atr_m[t]
        while ptr < len(fvgs) and fvgs[ptr][0] <= t:
            _, _, bot, top = fvgs[ptr]
            if top <= px and (bb is None or top > bb):
                bb = top
            if bot >= px and (ba is None or bot < ba):
                ba = bot
            ptr += 1
        if a > 0:
            if bb is not None:
                dist[t] = (px - bb) / a
            elif ba is not None:
                dist[t] = (ba - px) / a
    return dist

def swing(px, lb=3):
    hi = np.full(len(px), np.nan); lo = np.full(len(px), np.nan)
    for i in range(lb, len(px) - lb):
        if px[i] == np.max(px[i - lb:i + lb + 1]):
            hi[i] = px[i]
        if px[i] == np.min(px[i - lb:i + lb + 1]):
            lo[i] = px[i]
    return hi, lo

def base_indicators(c, h, l, ts):
    n = len(c); S = {}
    atr14 = atr(h, l, c, 14)
    S['ret1'] = (c - np.roll(c, 1)) / (atr14 + 1e-9); S['ret3'] = (c - np.roll(c, 3)) / (atr14 + 1e-9)
    S['ret5'] = (c - np.roll(c, 5)) / (atr14 + 1e-9); S['ret10'] = (c - np.roll(c, 10)) / (atr14 + 1e-9)
    S['ret20'] = (c - np.roll(c, 20)) / (atr14 + 1e-9); S['ret50'] = (c - np.roll(c, 50)) / (atr14 + 1e-9)
    S['rsi7'] = rsi(c, 7); S['rsi14'] = rsi(c, 14); S['rsi21'] = rsi(c, 21)
    bb_mid = rmean(c, 20); bb_std = rstd(c, 20)
    S['bb_pos'] = (c - bb_mid) / (2 * bb_std + 1e-9); S['bb_w'] = (4 * bb_std) / (bb_mid + 1e-9)
    S['atr_lvl'] = atr14 / (c + 1e-9) * 1e5; S['atr_ratio'] = atr14 / (rmean(atr14, 50) + 1e-9)
    e12 = ema(c, 12); e26 = ema(c, 26); macd = e12 - e26
    S['macd'] = macd / (atr14 + 1e-9); S['macd_hist'] = (macd - ema(macd, 9)) / (atr14 + 1e-9)
    S['mom5'] = c / np.roll(c, 5) - 1; S['mom10'] = c / np.roll(c, 10) - 1; S['mom20'] = c / np.roll(c, 20) - 1; S['mom50'] = c / np.roll(c, 50) - 1
    S['stochK'] = stoch(h, l, c, 14); S['stochD'] = rmean(S['stochK'], 3)
    S['cci20'] = cci(h, l, c, 20)
    a5, p5, m5 = adx_di(h, l, c, 14); S['adx'] = a5; S['diplus'] = p5; S['diminus'] = m5
    S['fvg'] = fvg_self(h, l, ts, atr(h, l, c, 14))
    shi, _ = swing(h, 3); _, slo = swing(l, 3); shd = np.full(n, np.nan); sld = np.full(n, np.nan)
    for i in range(60, n):
        wa = shi[i - 60:i + 1]; wb = slo[i - 60:i + 1]; wa = wa[~np.isnan(wa)]; wb = wb[~np.isnan(wb)]
        if len(wa):
            shd[i] = (np.min(wa) - c[i]) / (atr14[i] + 1e-9)
        if len(wb):
            sld[i] = (c[i] - np.max(wb)) / (atr14[i] + 1e-9)
    S['swing'] = np.minimum(np.abs(shd), np.abs(sld))
    hr = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    S['hour_sin'] = np.sin(2 * np.pi * hr / 24); S['hour_cos'] = np.cos(2 * np.pi * hr / 24)
    return S

def expand(S, tf):
    bases = list(S.keys()); mat = []; cols = []; inds = []
    LAGS = [1, 2, 3, 5, 8, 13, 21]
    for name in bases:
        arr = np.asarray(S[name], float); ser = pd.Series(arr)
        mat.append(ser.fillna(0).to_numpy()); cols.append(f"{tf}__{name}"); inds.append(name)
        for L in LAGS:
            mat.append(ser.shift(L).fillna(0).to_numpy()); cols.append(f"{tf}__{name}_lag{L}"); inds.append(name)
        mat.append((ser - ser.shift(1)).fillna(0).to_numpy()); cols.append(f"{tf}__{name}_d1"); inds.append(name)
        mat.append((ser - ser.shift(3)).fillna(0).to_numpy()); cols.append(f"{tf}__{name}_d3"); inds.append(name)
        mat.append((ser - ser.shift(5)).fillna(0).to_numpy()); cols.append(f"{tf}__{name}_d5"); inds.append(name)
        z20 = ((ser - ser.rolling(20, min_periods=20).mean()) / (ser.rolling(20, min_periods=20).std() + 1e-9)).fillna(0).to_numpy(); mat.append(z20); cols.append(f"{tf}__{name}_z20"); inds.append(name)
        z50 = ((ser - ser.rolling(50, min_periods=50).mean()) / (ser.rolling(50, min_periods=50).std() + 1e-9)).fillna(0).to_numpy(); mat.append(z50); cols.append(f"{tf}__{name}_z50"); inds.append(name)
    return np.stack(mat, 1), cols, inds

TFS = [("5min", "M5"), ("15min", "M15"), ("30min", "M30"), ("1h", "H1"), ("4h", "H4")]

def build_blocks(m5_ts, m5_close, m5_high, m5_low):
    """Build per-TF feature blocks mapped to every M5 bar.

    m5_ts: datetime64 array (or pd.Series) of M5 bar timestamps.
    m5_close/high/low: float arrays of M5 OHLC.
    Returns (blocks, block_inds): blocks[tf] is an (n_m5, n_feat) array aligned to M5 bars,
    block_inds[tf] is the per-column indicator name list (in column order).
    """
    ts = pd.Series(m5_ts)
    blocks = {}; block_inds = {}
    for rule, tf in TFS:
        if rule == "5min":
            c, h, l, t = m5_close, m5_high, m5_low, ts
        else:
            r = pd.DataFrame({"ts": ts, "open": m5_close, "high": m5_high, "low": m5_low, "close": m5_close})
            r = r.set_index("ts").resample(rule).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()
            c, h, l, t = r["close"].to_numpy(float), r["high"].to_numpy(float), r["low"].to_numpy(float), r["ts"]
        S = base_indicators(c, h, l, t)
        mat, cols, inds = expand(S, tf)
        mts = pd.to_datetime(t).to_numpy()
        pos = mts.searchsorted(pd.to_datetime(m5_ts).to_numpy(), side="right") - 1
        pos = np.clip(pos, 0, len(mat) - 1)
        blocks[tf] = mat[pos]; block_inds[tf] = inds
    return blocks, block_inds

def feature_row(blocks, block_inds):
    """Feature vector for the LAST M5 bar (current market state) + column-order metadata."""
    Xrow = np.hstack([blocks[tf][-1] for _, tf in TFS])
    col_order = [(tf, ind) for _, tf in TFS for ind in block_inds[tf]]
    return Xrow, col_order
