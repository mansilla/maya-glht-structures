#!/usr/bin/env python
"""
Scaling analysis of the tau = 0.16 catalogue for the "Scaling properties of
ancient Maya settlement" manuscript.

Everything the paper's Sections IV-V need, recomputed by likelihood rather
than by R^2 on rank plots, plus the sensitivity analyses the v1 draft left as
future work:

  1. de-duplication of tile-overlap doubles (< 3 m, different tiles)
  2. DBSCAN grid: eps 40/50/60 m x min_samples 3/4
  3. CSR null: same per-tile counts, uniform inside each tile's detection hull
  4. strip truncation: GLAS vs NFI, interior agglomerations (>= 300 m from edge)
  5. n_structures (the population analogue) alongside built area
  6. tau sensitivity of the individual-footprint distribution (gated, 0.10,
     0.16, 0.24)
  7. MLE tails (Clauset et al. 2009: KS-selected xmin, bootstrap CI, GOF p,
     Vuong LR tests vs lognormal / exponential / truncated PL / Weibull)
  8. MLE full-window density fits with BIC and KS (Tables 3, 5)
  9. rank-size NLS fits with bootstrap CIs (Tables 2, 4), kept as descriptive

Outputs -> analysis/*.csv, results.json and analysis/figures/*.png.

Run:  python tools/scaling_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats, special, integrate
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
FIG = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260927)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def local_metres(lat, lon):
    return np.column_stack([lon * 111_320.0 * np.cos(np.radians(lat)), lat * 110_540.0])


def load():
    d = pd.read_csv(ROOT / "catalogue/detections_tau016.csv")
    g = pd.read_csv(ROOT / "intermediate/gated_candidates.csv")
    g = g[g.shape_type == "rectangular"].copy()
    tiles = pd.read_csv(ROOT / "intermediate/tile_inventory.csv")
    for df in (d, g):
        df["geom"] = df.source_folder.str.extract(r"(GLAS|NFI|Kuiic)")[0]
        m = df.merge(tiles[["filename", "min_x", "max_x", "min_y", "max_y"]], on="filename", how="left")
        df["edge_dist_m"] = np.minimum.reduce([
            m.easting.values - m.min_x.values, m.max_x.values - m.easting.values,
            m.northing.values - m.min_y.values, m.max_y.values - m.northing.values])
    return d, g, tiles


def dedup(df: pd.DataFrame, r: float = 3.0):
    """Drop the lower-scored member of every cross-tile pair closer than r m."""
    xy = local_metres(df.latitude.values, df.longitude.values)
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r, output_type="ndarray")
    fn = df.filename.values
    sc = df.building_confidence.values
    cross = pairs[fn[pairs[:, 0]] != fn[pairs[:, 1]]]
    drop = set()
    for i, j in cross:
        drop.add(i if sc[i] < sc[j] else j)
    keep = np.ones(len(df), bool)
    keep[list(drop)] = False
    return df[keep].reset_index(drop=True), {"pairs_lt_r": int(len(pairs)),
                                             "cross_tile_pairs": int(len(cross)),
                                             "dropped": int(len(drop)), "r_m": r}


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
def hull_ha(pts):
    if len(pts) < 3:
        return 0.0
    try:
        return float(ConvexHull(pts).volume) / 1e4
    except Exception:
        return 0.0


def cluster(df: pd.DataFrame, eps=50.0, min_samples=3, xy=None) -> pd.DataFrame:
    if xy is None:
        xy = local_metres(df.latitude.values, df.longitude.values)
    lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
    area = df.area_m2.values
    geom = df.geom.values
    edge = df.edge_dist_m.values
    fn = df.filename.values
    ds = df.source_folder.values
    rows = []
    order = np.argsort(lab)
    lab_s = lab[order]
    bounds = np.flatnonzero(np.diff(lab_s)) + 1
    groups = np.split(order, bounds)
    for idx in groups:
        cid = lab[idx[0]]
        if cid == -1:
            continue
        pts = xy[idx]
        span = pts.max(0) - pts.min(0)
        vals, cnt = np.unique(geom[idx], return_counts=True)
        rows.append({
            "n": len(idx), "area_sum": float(area[idx].sum()),
            "area_max": float(area[idx].max()),
            "hull_ha": hull_ha(pts), "extent_m": float(np.hypot(*span)),
            "geom": vals[np.argmax(cnt)], "edge_min_m": float(np.nanmin(edge[idx])),
            "n_tiles": int(len(np.unique(fn[idx]))),
            "dataset": pd.Series(ds[idx]).mode().iloc[0],
        })
    a = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    a["grouped_frac"] = a.n.sum() / len(df)
    return a


# --------------------------------------------------------------------------
# power-law tails (Clauset, Shalizi & Newman 2009)
# --------------------------------------------------------------------------
def pl_alpha(x, xmin):
    x = x[x >= xmin]
    return 1.0 + len(x) / np.log(x / xmin).sum()


def pl_ks(x, xmin, alpha):
    x = np.sort(x[x >= xmin])
    n = len(x)
    cdf = 1.0 - (x / xmin) ** (1.0 - alpha)
    emp_hi = np.arange(1, n + 1) / n
    emp_lo = np.arange(0, n) / n
    return float(max(np.abs(cdf - emp_hi).max(), np.abs(cdf - emp_lo).max()))


def pl_fit(x, xmin_grid=None, min_tail=25):
    x = np.asarray(x, float)
    if xmin_grid is None:
        u = np.unique(x)
        xmin_grid = u[(u >= np.quantile(x, 0.30)) & (u <= np.sort(x)[-min_tail])]
    best = None
    for xm in xmin_grid:
        a = pl_alpha(x, xm)
        D = pl_ks(x, xm, a)
        if best is None or D < best[2]:
            best = (xm, a, D)
    xm, a, D = best
    n = int((x >= xm).sum())
    return {"xmin": float(xm), "alpha": float(a), "ks": D, "n_tail": n,
            "sigma": float((a - 1) / np.sqrt(n))}


def pl_bootstrap(x, reps=1000, fixed_xmin=None):
    x = np.asarray(x, float)
    out = []
    for _ in range(reps):
        s = RNG.choice(x, len(x))
        if fixed_xmin is None:
            out.append(pl_fit(s)["alpha"])
        else:
            if (s >= fixed_xmin).sum() < 5:
                continue
            out.append(pl_alpha(s, fixed_xmin))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def pl_gof(x, fit, reps=1000):
    """Semi-parametric bootstrap p-value (Clauset et al. 2009, sec. 4.1)."""
    x = np.asarray(x, float)
    xm, a, D0 = fit["xmin"], fit["alpha"], fit["ks"]
    tail = x[x >= xm]
    body = x[x < xm]
    n = len(x)
    p_tail = len(tail) / n
    worse = 0
    for _ in range(reps):
        k = RNG.binomial(n, p_tail)
        synth_tail = xm * (1 - RNG.random(k)) ** (-1.0 / (a - 1))
        synth_body = RNG.choice(body, n - k) if len(body) else np.array([])
        s = np.concatenate([synth_tail, synth_body])
        f = pl_fit(s)
        if f["ks"] >= D0:
            worse += 1
    return worse / reps


def _loglik_alternatives(x, xmin, alpha):
    """Tail-conditional log-likelihoods per point for PL and alternatives."""
    t = x[x >= xmin]
    n = len(t)
    lpl = np.log((alpha - 1) / xmin) - alpha * np.log(t / xmin)
    out = {"power_law": lpl}
    # exponential, truncated at xmin
    lam = 1.0 / (t - xmin).mean()
    out["exponential"] = np.log(lam) - lam * (t - xmin)
    # lognormal, truncated at xmin
    lt = np.log(t)

    def nll_ln(p):
        mu, ls = p
        s = np.exp(ls)
        z = (np.log(xmin) - mu) / s
        norm = stats.norm.logsf(z)
        return -(stats.norm.logpdf(lt, mu, s) - lt - norm).sum()
    r = optimize.minimize(nll_ln, [lt.mean(), np.log(lt.std() + 1e-3)], method="Nelder-Mead")
    mu, s = r.x[0], np.exp(r.x[1])
    out["lognormal"] = stats.norm.logpdf(lt, mu, s) - lt - stats.norm.logsf((np.log(xmin) - mu) / s)
    # truncated power law  p ~ x^-a e^{-l x}
    def nll_tpl(p):
        a, ll = p[0], np.exp(p[1])
        if a <= 0:
            return 1e30
        Z, _ = integrate.quad(lambda u: u ** (-a) * np.exp(-ll * u), xmin, np.inf, limit=200)
        if not np.isfinite(Z) or Z <= 0:
            return 1e30
        return -(-a * np.log(t) - ll * t - np.log(Z)).sum()
    r = optimize.minimize(nll_tpl, [alpha, np.log(0.1 / t.mean())], method="Nelder-Mead")
    a, ll = r.x[0], np.exp(r.x[1])
    Z, _ = integrate.quad(lambda u: u ** (-a) * np.exp(-ll * u), xmin, np.inf, limit=200)
    out["truncated_power_law"] = -a * np.log(t) - ll * t - np.log(Z)
    # Weibull (stretched exponential), truncated at xmin
    def nll_wb(p):
        k, lam = np.exp(p)
        return -(stats.weibull_min.logpdf(t, k, scale=lam) - stats.weibull_min.logsf(xmin, k, scale=lam)).sum()
    r = optimize.minimize(nll_wb, [np.log(0.5), np.log(t.mean())], method="Nelder-Mead")
    k, lam = np.exp(r.x)
    out["stretched_exponential"] = stats.weibull_min.logpdf(t, k, scale=lam) - stats.weibull_min.logsf(xmin, k, scale=lam)
    return out, n


def vuong(x, fit):
    ll, n = _loglik_alternatives(np.asarray(x, float), fit["xmin"], fit["alpha"])
    res = {}
    for name, l2 in ll.items():
        if name == "power_law":
            continue
        d = ll["power_law"] - l2
        R = d.sum()
        sd = d.std(ddof=1)
        if name == "truncated_power_law":      # nested: LR test, 1 df
            p = float(stats.chi2.sf(max(0.0, -2 * R), 1))
            res[name] = {"R": float(R), "p": p, "nested": True}
        else:
            Rn = R / (sd * np.sqrt(n)) if sd > 0 else 0.0
            res[name] = {"R": float(R), "Rn": float(Rn), "p": float(2 * stats.norm.sf(abs(Rn))), "nested": False}
    return res


# discrete power law for n_structures
def dpl_alpha(k, kmin):
    k = k[k >= kmin]
    n = len(k)
    slk = np.log(k).sum()
    f = lambda a: n * np.log(special.zeta(a, kmin)) + a * slk
    r = optimize.minimize_scalar(f, bounds=(1.01, 6.0), method="bounded")
    return float(r.x)


def dpl_fit(k, kmin_grid=range(3, 30), min_tail=25):
    k = np.asarray(k, float)
    best = None
    for km in kmin_grid:
        t = k[k >= km]
        if len(t) < min_tail:
            break
        a = dpl_alpha(k, km)
        ks_x = np.arange(km, t.max() + 1)
        cdf = np.cumsum(ks_x ** (-a)) / special.zeta(a, km)
        emp = np.searchsorted(np.sort(t), ks_x, side="right") / len(t)
        D = float(np.abs(cdf - emp).max())
        if best is None or D < best[2]:
            best = (km, a, D, len(t))
    return {"xmin": int(best[0]), "alpha": best[1], "ks": best[2], "n_tail": int(best[3])}


def dpl_sample(kmin, alpha, size, kmax=100000):
    ks = np.arange(kmin, kmax + 1, dtype=float)
    pm = ks ** (-alpha); pm /= pm.sum()
    return ks[np.searchsorted(np.cumsum(pm), RNG.random(size))]


def dpl_bootstrap(k, reps=1000):
    out = [dpl_fit(RNG.choice(k, len(k)))["alpha"] for _ in range(reps)]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def dpl_gof(k, fit, reps=1000):
    """Semi-parametric bootstrap p-value for the discrete power law (Clauset et al. 2009)."""
    k = np.asarray(k, float)
    km, a, D0 = fit["xmin"], fit["alpha"], fit["ks"]
    body = k[k < km]; n = len(k); p_tail = (k >= km).mean()
    worse = 0
    for _ in range(reps):
        m = RNG.binomial(n, p_tail)
        s = np.concatenate([dpl_sample(km, a, m), RNG.choice(body, n - m) if len(body) else np.array([])])
        if dpl_fit(s)["ks"] >= D0:
            worse += 1
    return worse / reps


# --------------------------------------------------------------------------
# full-window density models (MLE with truncation to [lo, hi])
# --------------------------------------------------------------------------
class Model:
    def __init__(self, name, k, logpdf, cdf, init, transform=None):
        self.name, self.k, self.logpdf, self.cdf, self.init = name, k, logpdf, cdf, init

    def logZ(self, p, lo, hi):
        """log of the window mass, integrated numerically from logpdf itself so
        that the normalisation is always consistent with the density even where
        the closed-form CDF loses precision (GB2 with p -> inf, IGG with
        alpha -> 1)."""
        f = lambda u: np.exp(self.logpdf(np.array([np.exp(u)]), p)[0] + u)
        pts = np.linspace(np.log(lo), np.log(hi), 9)[1:-1]
        val, _ = integrate.quad(f, np.log(lo), np.log(hi), points=list(pts), limit=400)
        return np.log(val) if val > 0 else -np.inf

    def num_cdf(self, xq, p, lo, hi):
        """Window-normalised CDF by cumulative numerical integration of the
        density on a fine log grid (robust where the closed-form CDF is not)."""
        g = np.exp(np.linspace(np.log(lo), np.log(min(hi, max(np.max(xq), lo) * 1.0001)), 6000))
        pdf = np.exp(self.logpdf(g, p) - self.logZ(p, lo, hi))
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(g))])
        return np.interp(xq, g, cum)

    def fit(self, x, lo, hi):
        lx = np.asarray(x, float)

        def nll(p):
            if np.any(np.abs(p) > 14):
                return 1e30
            try:
                lZ = self.logZ(p, lo, hi)
                if not np.isfinite(lZ):
                    return 1e30
                v = self.logpdf(lx, p)
                if not np.all(np.isfinite(v)):
                    return 1e30
                return -(v.sum() - len(lx) * lZ)
            except Exception:
                return 1e30
        best = None
        for p0 in self.init(lx):
            r = optimize.minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
            r = optimize.minimize(nll, r.x, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        lZ = self.logZ(p, lo, hi)
        n = len(lx)
        lnL = -best.fun
        # KS on the discrete support: areas are multiples of 4 m^2 (2 m cells), so
        # the empirical CDF is compared with the model mass up to the bin's
        # right edge (value + 2), which removes the discreteness artefact.
        u, cnt = np.unique(lx, return_counts=True)
        Femp = np.cumsum(cnt) / n
        Fmod = self.num_cdf(np.minimum(u + 2.0, hi), p, lo, hi)
        D = float(np.abs(Fmod - Femp).max())
        return {"model": self.name, "k": self.k, "lnL": float(lnL), "AIC": float(2 * self.k - 2 * lnL),
                "BIC": float(self.k * np.log(n) - 2 * lnL), "KS": D, "params": [float(v) for v in p],
                "per_point": self.logpdf(lx, p) - lZ}


def gb2_logpdf(x, p):
    a, b, pp, q = np.exp(p)
    return (np.log(a) + (a * pp - 1) * np.log(x) - a * pp * np.log(b) - special.betaln(pp, q)
            - (pp + q) * np.log1p((x / b) ** a))


def gb2_cdf(x, p):
    a, b, pp, q = np.exp(p)
    z = (x / b) ** a
    return special.betainc(pp, q, z / (1 + z))


def igg_logpdf(x, p):          # inverse generalised gamma: x^-alpha exp(-(theta/x)^c)
    lal, lth, lc = p
    al, th, c = np.exp(lal) + 1.0, np.exp(lth), np.exp(lc)
    # normalisation on (0, inf): c * th^(1-al) / Gamma((al-1)/c)
    logC = np.log(c) + (al - 1) * np.log(th) - special.gammaln((al - 1) / c)
    return logC - al * np.log(x) - (th / x) ** c


def igg_cdf(x, p):
    lal, lth, lc = p
    al, th, c = np.exp(lal) + 1.0, np.exp(lth), np.exp(lc)
    s = (al - 1) / c
    # P(X<=x) = P(Y>=1/x) with Y ~ gengamma: Q(s, (th/x)^c)
    return special.gammaincc(s, (th / x) ** c)


_LO = [1.0]
MODELS = [
    Model("Lognormal", 2, lambda x, p: stats.norm.logpdf(np.log(x), p[0], np.exp(p[1])) - np.log(x),
          lambda x, p: stats.norm.cdf(np.log(x), p[0], np.exp(p[1])),
          lambda x: [[np.log(x).mean(), np.log(np.log(x).std())]]),
    Model("Gamma", 2, lambda x, p: stats.gamma.logpdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x, p: stats.gamma.cdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x: [[0.0, np.log(x.mean())], [np.log(0.5), np.log(x.mean())]]),
    Model("Weibull (stretched exponential)", 2, lambda x, p: stats.weibull_min.logpdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x, p: stats.weibull_min.cdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x: [[np.log(0.7), np.log(x.mean())], [np.log(1.5), np.log(x.mean())]]),
    Model("Exponential", 1, lambda x, p: stats.expon.logpdf(x, scale=np.exp(p[0])),
          lambda x, p: stats.expon.cdf(x, scale=np.exp(p[0])), lambda x: [[np.log(x.mean())]]),
    Model("Power law (Pareto)", 1, lambda x, p: stats.pareto.logpdf(x, np.exp(p[0]), scale=_LO[0] * 0.999),
          lambda x, p: stats.pareto.cdf(x, np.exp(p[0]), scale=_LO[0] * 0.999), lambda x: [[np.log(1.0)], [np.log(2.0)]]),
    Model("Lomax (Zipf-Mandelbrot analogue)", 2, lambda x, p: stats.lomax.logpdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x, p: stats.lomax.cdf(x, np.exp(p[0]), scale=np.exp(p[1])),
          lambda x: [[np.log(1.5), np.log(x.mean())], [np.log(2.5), np.log(x.mean() / 3)]]),
    Model("Inverse generalised gamma", 3, igg_logpdf, igg_cdf,
          lambda x: [[np.log(2.0), np.log(x.mean()), np.log(0.5)], [np.log(1.0), np.log(3 * x.mean()), np.log(0.4)],
                     [np.log(3.0), np.log(x.mean() / 2), np.log(1.0)]]),
    Model("GB2", 4, gb2_logpdf, gb2_cdf,
          lambda x: [[np.log(1.5), np.log(np.median(x)), np.log(3.0), np.log(1.5)],
                     [np.log(1.0), np.log(np.median(x)), np.log(5.0), np.log(1.0)],
                     [np.log(2.0), np.log(np.median(x) / 2), np.log(1.5), np.log(2.0)]]),
]


def fit_density_table(x, lo, hi, label):
    _LO[0] = lo
    fits = [m.fit(x, lo, hi) for m in MODELS]
    fits.sort(key=lambda f: f["BIC"])
    best = fits[0]
    rows = []
    n = len(x)
    for f in fits:
        d = best["per_point"] - f["per_point"]
        sd = d.std(ddof=1)
        Rn = d.sum() / (sd * np.sqrt(n)) if sd > 0 and f is not best else 0.0
        limit_pair = {best["model"], f["model"]} == {"GB2", "Inverse generalised gamma"}
        if f is best:
            pval, test = None, None
        elif limit_pair:      # IGG is the p -> inf limit of the GB2: not a Vuong comparison; LR with 1 df
            pval, test = round(float(stats.chi2.sf(max(0.0, 2 * abs(best["lnL"] - f["lnL"])), 1)), 4), "LR"
        else:
            pval, test = round(float(2 * stats.norm.sf(abs(Rn))), 4), "Vuong"
        rows.append({"model": f["model"], "k": f["k"], "lnL": round(f["lnL"], 1), "BIC": round(f["BIC"], 1),
                     "dBIC": round(f["BIC"] - best["BIC"], 1), "KS": round(f["KS"], 4),
                     "Vuong_vs_best_p": pval, "test": test, "params": f["params"]})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / f"density_fits_{label}.csv", index=False)
    return t, fits


# --------------------------------------------------------------------------
# rank-size NLS (descriptive), with bootstrap CIs
# --------------------------------------------------------------------------
def rank_models(N):
    r = np.arange(1, N + 1, dtype=float)
    return {
        "Zipf": (lambda rr, lA, a: lA - a * np.log(rr), [10, 0.7]),
        "Zipf-Mandelbrot": (lambda rr, lA, a, q: lA - a * np.log(rr + q), [12, 1.0, 10.0]),
        "DGBD": (lambda rr, lA, a, b: lA - a * np.log(rr) + b * np.log(N + 1 - rr), [8, 0.5, 0.3]),
        "Stretched exponential": (lambda rr, lA, lc, beta: lA - np.exp(lc) * rr ** beta, [9, -2, 0.4]),
        "Lavalette": (lambda rr, lA, a: lA - a * np.log(N * rr / (N + 1 - rr)), [10, 0.5]),
    }, r


def fit_rank(vals, reps=200):
    v = np.sort(np.asarray(vals, float))[::-1]
    N = len(v)
    models, r = rank_models(N)
    ly = np.log(v)
    out = []
    for name, (f, p0) in models.items():
        try:
            p, _ = optimize.curve_fit(f, r, ly, p0=p0, maxfev=20000)
        except Exception:
            out.append({"model": name, "R2": np.nan}); continue
        res = ly - f(r, *p)
        R2 = 1 - res.var() / ly.var()
        boots = []
        for _ in range(reps):
            s = np.sort(RNG.choice(v, N))[::-1]
            try:
                pb, _ = optimize.curve_fit(f, r, np.log(s), p0=p, maxfev=20000)
                boots.append(pb)
            except Exception:
                pass
        boots = np.array(boots)
        ci = np.percentile(boots, [2.5, 97.5], axis=0) if len(boots) else np.full((2, len(p)), np.nan)
        out.append({"model": name, "params": [float(a) for a in p], "ci_lo": [float(a) for a in ci[0]],
                    "ci_hi": [float(a) for a in ci[1]], "R2": float(R2), "RMSE_log10": float(np.sqrt((res ** 2).mean()) / np.log(10))})
    return out


def scaling_exponent(a, ycol, xcol="n", nmin=10):
    s = a[(a.n >= nmin) & (a[ycol] > 0)]
    if len(s) < 5:
        return np.nan, np.nan, len(s)
    X = np.log(s[xcol].values); Y = np.log(s[ycol].values)
    b, i = np.polyfit(X, Y, 1)
    boots = [np.polyfit(*(lambda idx: (X[idx], Y[idx]))(RNG.integers(0, len(X), len(X))), 1)[0] for _ in range(300)]
    return float(b), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), len(s)


def tail_summary(a, label, boot=True):
    x = a.area_sum.values.astype(float)
    f = pl_fit(x)
    res = {"label": label, "N": len(a), "grouped_frac": float(a.grouped_frac.iloc[0]),
           "n_ge_1000": int((x >= 1000).sum()), "max3": [float(v) for v in np.sort(x)[-3:][::-1]],
           "xmin_ks": f["xmin"], "alpha_ks": f["alpha"], "n_tail_ks": f["n_tail"], "ks": f["ks"],
           "alpha_1000": float(pl_alpha(x, 1000.0)) if (x >= 1000).sum() >= 10 else np.nan,
           "n_1000": int((x >= 1000).sum())}
    if boot:
        res["alpha_1000_ci"] = pl_bootstrap(x, 1000, fixed_xmin=1000.0)
    hb, hci, hn = scaling_exponent(a, "hull_ha")
    eb, eci, en = scaling_exponent(a, "extent_m")
    res.update({"hull_exp": hb, "hull_ci": hci, "extent_exp": eb, "extent_ci": eci, "D": 1 / eb if eb else np.nan, "n_ge10": hn})
    return res


# --------------------------------------------------------------------------
# CSR null
# --------------------------------------------------------------------------
def csr_sample(df, xy):
    """Same number of points per tile, uniform inside that tile's detection hull."""
    out = xy.copy()
    for fn, idx in df.groupby("filename").indices.items():
        pts = xy[idx]
        if len(idx) < 3:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        lo, hi = pts.min(0), pts.max(0)
        eq = hull.equations
        need = len(idx)
        got = []
        while need > 0:
            cand = lo + RNG.random((max(4 * need, 64), 2)) * (hi - lo)
            inside = np.all(cand @ eq[:, :2].T + eq[:, 2] <= 1e-9, axis=1)
            sel = cand[inside][:need]
            got.append(sel); need -= len(sel)
        out[idx] = np.vstack(got)
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False})


def log_bins_pdf(x, nb=24):
    e = np.logspace(np.log10(x.min()), np.log10(x.max() * 1.0001), nb + 1)
    h, _ = np.histogram(x, e)
    w = np.diff(e); c = np.sqrt(e[1:] * e[:-1])
    p = h / (len(x) * w); err = np.sqrt(h) / (len(x) * w)
    m = h > 0
    return c[m], p[m], err[m]


def pdf_of(model_name, p, lo, hi, xs):
    m = next(mm for mm in MODELS if mm.name == model_name)
    return np.exp(m.logpdf(xs, p) - m.logZ(p, lo, hi))


def poisson_chain_null(eps=50.0, min_samples=3, reps=6):
    """Homogeneous Poisson points in a 1.35 x 8 km strip over a range of
    densities; DBSCAN chaining at the same radius. Answers: when chaining of
    random points *does* produce a cluster of n points, how do its hull area
    and extent scale with n?"""
    W, L = 1350.0, 8000.0
    rows = []
    for lam in (1.5e-4, 2.5e-4, 3.5e-4, 4.5e-4, 5.5e-4, 6.5e-4):
        for _ in range(reps):
            n = RNG.poisson(lam * W * L)
            xy = np.column_stack([RNG.random(n) * W, RNG.random(n) * L])
            lab = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xy)
            for cid in np.unique(lab):
                if cid < 0:
                    continue
                pts = xy[lab == cid]
                if len(pts) < 10:
                    continue
                span = pts.max(0) - pts.min(0)
                rows.append({"lam": lam, "n": len(pts), "hull_ha": hull_ha(pts), "extent_m": float(np.hypot(*span))})
    t = pd.DataFrame(rows)
    t = t[(t.hull_ha > 0) & (t.n < 1000)]
    out = {"clusters_ge10": int(len(t))}
    for ycol in ("hull_ha", "extent_m"):
        X, Y = np.log(t.n.values), np.log(t[ycol].values)
        b = np.polyfit(X, Y, 1)[0]
        boots = [np.polyfit(X[idx], Y[idx], 1)[0] for idx in (RNG.integers(0, len(X), len(X)) for _ in range(300))]
        out[ycol] = {"exp": float(b), "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))}
    out["D"] = 1.0 / out["extent_m"]["exp"]
    return out, t


# --------------------------------------------------------------------------
def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style()
    R = {}
    d, g, tiles = load()
    log(f"loaded {len(d)} published, {len(g)} gated rectangular")

    # 1. de-duplication ----------------------------------------------------
    dd, dinfo = dedup(d, 3.0)
    R["dedup"] = dinfo | {"published": len(d), "after": len(dd)}
    log(f"dedup: {dinfo}")
    xy_d = local_metres(d.latitude.values, d.longitude.values)
    xy_dd = local_metres(dd.latitude.values, dd.longitude.values)

    # 2. reference clustering + grid ---------------------------------------
    A = cluster(d, 50, 3, xy_d)
    A.to_csv(OUT / "agglomerations_ref.csv", index=False)
    Add = cluster(dd, 50, 3, xy_dd)
    Add.to_csv(OUT / "agglomerations_dedup.csv", index=False)
    grid = []
    for eps in (40, 50, 60):
        for ms in (3, 4):
            a = cluster(d, eps, ms, xy_d)
            s = tail_summary(a, f"eps{eps}_ms{ms}")
            s.update({"eps": eps, "min_samples": ms, "catalogue": "published"})
            grid.append(s); log(f"grid eps={eps} ms={ms}: N={s['N']} alpha1000={s['alpha_1000']:.2f} hull={s['hull_exp']:.2f} D={s['D']:.2f}")
    s = tail_summary(Add, "dedup_eps50_ms3"); s.update({"eps": 50, "min_samples": 3, "catalogue": "deduplicated"}); grid.append(s)
    log(f"dedup ref: N={s['N']} alpha1000={s['alpha_1000']:.2f}")
    pd.DataFrame(grid).to_csv(OUT / "sensitivity_grid.csv", index=False)
    R["grid"] = grid

    # 3. CSR null -----------------------------------------------------------
    csr = []
    for rep in range(20):
        xy_n = csr_sample(d, xy_d)
        a = cluster(d, 50, 3, xy_n)
        s = tail_summary(a, f"csr{rep}", boot=False)
        csr.append(s)
    csr_df = pd.DataFrame(csr)
    csr_df.to_csv(OUT / "csr_null.csv", index=False)
    R["csr"] = {k: {"mean": float(np.nanmean(csr_df[k])), "sd": float(np.nanstd(csr_df[k]))}
                for k in ("N", "grouped_frac", "n_ge_1000", "n_ge10")}
    R["csr"]["max_area_mean"] = float(np.mean([c["max3"][0] for c in csr]))
    R["csr"]["max_n_mean"] = float(np.mean([cluster(d, 50, 3, csr_sample(d, xy_d)).n.max() for _ in range(5)]))
    log(f"CSR: {json.dumps(R['csr'])}")
    R["poisson_chain"], pc_t = poisson_chain_null()
    log(f"Poisson chaining null: {json.dumps(R['poisson_chain'])}")

    # 4. geometry / interior ------------------------------------------------
    ref = tail_summary(A, "reference")
    R["reference"] = ref
    subsets = {}
    for name, sub in {"GLAS": A[A.geom == "GLAS"], "NFI": A[A.geom == "NFI"],
                      "interior_300m": A[A.edge_min_m >= 300], "interior_NFI_300m": A[(A.geom == "NFI") & (A.edge_min_m >= 300)],
                      "single_tile": A[A.n_tiles == 1]}.items():
        sub = sub.copy(); sub["grouped_frac"] = np.nan
        s = tail_summary(sub, name); subsets[name] = s
        log(f"{name}: N={s['N']} alpha1000={s['alpha_1000']:.2f} (n={s['n_1000']}) hull={s['hull_exp']:.2f}")
    R["subsets"] = subsets

    # 5. tails: area_sum and n_structures ------------------------------------
    x = A.area_sum.values.astype(float)
    f = pl_fit(x); f["alpha_ci"] = pl_bootstrap(x, 1000); f["gof_p"] = pl_gof(x, f, 1000); f["vuong"] = vuong(x, f)
    f["alpha_fixed"] = {str(xm): {"alpha": float(pl_alpha(x, xm)), "n": int((x >= xm).sum()), "ci": pl_bootstrap(x, 1000, xm)} for xm in (500, 1000, 2000)}
    # block bootstrap by dataset (resample the 16 datasets with replacement): are the
    # agglomerations exchangeable, or does one region drive the tail?
    ds_names = A.dataset.unique()
    blk = []
    for _ in range(1000):
        pick = RNG.choice(ds_names, len(ds_names))
        xb = np.concatenate([A.area_sum.values[A.dataset.values == nm] for nm in pick]).astype(float)
        if (xb >= f["xmin"]).sum() >= 20:
            blk.append(pl_alpha(xb, f["xmin"]))
    f["alpha_block_ci"] = (float(np.percentile(blk, 2.5)), float(np.percentile(blk, 97.5)))
    f["decades_above_1000"] = float(np.log10(x.max() / 1000))
    R["tail_area"] = f
    log(f"tail area: {json.dumps({k: v for k, v in f.items() if k != 'vuong'})}")
    log(f"  vuong: {json.dumps(f['vuong'])}")
    k = A.n.values
    fk = dpl_fit(k)
    fk["alpha_ci"] = dpl_bootstrap(k.astype(float), 1000)
    fk["gof_p"] = dpl_gof(k, fk, 1000)
    fk["alpha_from3"] = dpl_alpha(k.astype(float), 3)
    fk["alpha_from10"] = dpl_alpha(k.astype(float), 10)
    b10 = [dpl_alpha(RNG.choice(k, len(k)).astype(float), 10) for _ in range(1000)]
    fk["alpha_from10_ci"] = (float(np.percentile(b10, 2.5)), float(np.percentile(b10, 97.5)))
    fk["n_from10"] = int((k >= 10).sum())
    R["tail_n"] = fk
    log(f"tail n: {fk}")

    # 6. density tables ------------------------------------------------------
    t3, fits3 = fit_density_table(x, 72.0, 1e7, "agglomeration_area")
    R["table3"] = t3.drop(columns="params").to_dict("records")
    log("Table 3 (agglomeration area) by BIC:\n" + t3.drop(columns="params").to_string())
    xs = d.area_m2.values.astype(float)
    t5, fits5 = fit_density_table(xs, 24.0, 900.0, "structure_area")
    R["table5"] = t5.drop(columns="params").to_dict("records")
    log("Table 5 (structure area) by BIC:\n" + t5.drop(columns="params").to_string())

    def local_slopes(fit):
        m = next(mm for mm in MODELS if mm.name == fit["model"])
        out = {}
        for x0 in (50, 200, 800):
            h = 1e-3
            lp = lambda v: m.logpdf(np.array([v]), fit["params"])[0]
            out[str(x0)] = float((lp(x0 * (1 + h)) - lp(x0 * (1 - h))) / (np.log(1 + h) - np.log(1 - h)))
        return out
    best5 = fits5[0]
    R["table5_local_slopes"] = local_slopes(best5)
    ign = next(f for f in fits5 if f["model"] == "Inverse generalised gamma")
    R["table5_igg_slopes"] = local_slopes(ign)

    # 7. tau sensitivity of the footprint distribution -----------------------
    taus = []
    for label, sub in {"gated (no tau)": g, "tau 0.10": g[g.building_confidence >= 0.10],
                       "tau 0.16": g[g.building_confidence >= 0.16], "tau 0.24": g[g.building_confidence >= 0.24]}.items():
        xa = sub.area_m2.values.astype(float)
        _LO[0] = 24.0
        igg = next(mm for mm in MODELS if mm.name == "GB2").fit(xa, 24.0, 900.0)
        ln = next(mm for mm in MODELS if mm.name == "Lognormal").fit(xa, 24.0, 900.0)
        sl = local_slopes(igg)
        taus.append({"catalogue": label, "N": len(xa), "median_m2": float(np.median(xa)),
                     "frac_gt200": float((xa > 200).mean()), "frac_gt400": float((xa > 400).mean()),
                     "slope_50": sl["50"], "slope_200": sl["200"], "slope_800": sl["800"],
                     "lognormal_mu": ln["params"][0], "lognormal_sigma": float(np.exp(ln["params"][1])),
                     "dBIC_gb2_minus_lognormal": igg["BIC"] - ln["BIC"]})
        log(f"tau {label}: {taus[-1]}")
    pd.DataFrame(taus).to_csv(OUT / "tau_sensitivity_structures.csv", index=False)
    R["tau_sensitivity"] = taus

    # 7b. tau sensitivity of the agglomeration tail ---------------------------
    tau_ag = []
    for label, tau in (("gated (no tau)", 0.0), ("tau 0.10", 0.10), ("tau 0.16", 0.16), ("tau 0.24", 0.24)):
        sub = g[g.building_confidence >= tau].reset_index(drop=True)
        a = cluster(sub, 50, 3)
        s = tail_summary(a, label)
        s.update({"tau": tau, "N_structures": int(len(sub)), "alpha_ks_ci": pl_bootstrap(a.area_sum.values.astype(float), 1000)})
        tau_ag.append(s)
        log(f"tau agglomerations {label}: N={s['N']} grouped={s['grouped_frac']:.3f} alpha1000={s['alpha_1000']:.2f} (n={s['n_1000']}) alpha_ks={s['alpha_ks']:.2f} xmin={s['xmin_ks']:.0f}")
    pd.DataFrame(tau_ag).to_csv(OUT / "tau_sensitivity_agglomerations.csv", index=False)
    R["tau_agglomerations"] = tau_ag

    # 7c. archive densities, to state what regime the Poisson chaining null runs in
    dens = []
    for fn, idx in d.groupby("filename").indices.items():
        pts = xy_d[idx]
        if len(idx) >= 3:
            h = hull_ha(pts) * 1e4
            if h > 0:
                dens.append(len(idx) / h)
    big = A[A.n >= 10]
    R["poisson_chain"]["archive_density"] = {
        "per_tile_in_hull_median": float(np.median(dens)), "per_tile_in_hull_p90": float(np.percentile(dens, 90)),
        "intra_agglomeration_median": float((big.n / (big.hull_ha * 1e4)).median()),
        "intra_agglomeration_p10_p90": [float(v) for v in (big.n / (big.hull_ha * 1e4)).quantile([0.1, 0.9])]}
    log(f"archive densities: {json.dumps(R['poisson_chain']['archive_density'])}")

    # 8. rank-size tables ------------------------------------------------------
    R["table2"] = fit_rank(x, reps=1000)
    R["table2_n"] = fit_rank(k.astype(float), reps=1000)
    R["table4"] = fit_rank(xs, reps=200)
    for key in ("table2", "table2_n", "table4"):
        pd.DataFrame(R[key]).to_csv(OUT / f"rank_fits_{key}.csv", index=False)
        log(f"{key}: " + "; ".join(f"{r['model']} R2={r.get('R2', np.nan):.4f} p={np.round(r.get('params', []), 3).tolist()}" for r in R[key]))

    # 9. Mexican cities benchmark (rank-size of n_structures vs population) - skipped, no data file

    # 10. summary numbers -----------------------------------------------------
    R["summary"] = {
        "N_agglomerations": int(len(A)), "grouped": int(A.n.sum()), "grouped_frac": float(A.n.sum() / len(d)),
        "median_n": float(A.n.median()), "median_area_sum": float(A.area_sum.median()),
        "min_area_sum": float(A.area_sum.min()), "max_area_sum": float(A.area_sum.max()),
        "largest_n": [int(v) for v in A.n.values[:10]],
        "structures": {"N": int(len(d)), "min": float(xs.min()), "max": float(xs.max()), "median": float(np.median(xs)),
                       "unique_values": int(len(np.unique(xs))), "frac_gt200": float((xs > 200).mean())},
        "canopy": d.canopy_class.value_counts(normalize=True).round(4).to_dict(),
        "area_max_vs_n_exp": scaling_exponent(A, "area_max")[0],
        "area_sum_vs_n_exp": scaling_exponent(A, "area_sum")[0],
        "area_sum_vs_n_exp_all": float(np.polyfit(np.log(A.n), np.log(A.area_sum), 1)[0]),
    }
    log(f"summary: {json.dumps(R['summary'])}")

    # ---------------------------------------------------------------- figures
    # F1 rank-size of agglomeration built area with three models + residuals
    v = np.sort(x)[::-1]; N = len(v); r = np.arange(1, N + 1)
    models, _ = rank_models(N)
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(6.5, 6.2), height_ratios=[3, 1.2], sharex=True)
    ax.scatter(r, v, s=6, color=INK, alpha=0.6, label=f"Agglomerations (N = {N:,})", zorder=3)
    for (name, col, ls) in (("Zipf", C2, "-"), ("Zipf-Mandelbrot", C3, "--"), ("DGBD", C1, "-.")):
        p = next(t for t in R["table2"] if t["model"] == name)["params"]
        yf = np.exp(models[name][0](r.astype(float), *p))
        ax.plot(r, yf, color=col, ls=ls, lw=1.6, label=name)
        axr.plot(r, np.log10(v / yf), color=col, ls=ls, lw=1.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("Built area of agglomeration (m²)")
    ax.legend(loc="lower left"); axr.axhline(0, color=INK, lw=0.8); axr.set_ylabel("log₁₀(data / fit)"); axr.set_xlabel("Rank")
    axr.set_ylim(-0.8, 0.8)
    fig.tight_layout(); fig.savefig(FIG / "fig_rank_agglomerations.png", dpi=300); plt.close(fig)

    # F2 CCDF tails with MLE power law (area and n)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, data, ff, lab, disc in ((axs[0], x, R["tail_area"], "Built area of agglomeration (m²)", False),
                                    (axs[1], k.astype(float), R["tail_n"], "Structures per agglomeration", True)):
        s = np.sort(data); cc = 1 - np.arange(len(s)) / len(s)
        ax.step(s, cc, where="post", color=INK, lw=1.2, label="Empirical CCDF")
        xm, al = ff["xmin"], ff["alpha"]
        xx = np.logspace(np.log10(xm), np.log10(s.max()), 100)
        if disc:
            base = (s >= xm).mean(); yy = base * special.zeta(al, xx) / special.zeta(al, xm)
        else:
            base = (s >= xm).mean(); yy = base * (xx / xm) ** (1 - al)
        ax.plot(xx, yy, color=C2, lw=2, label=f"Power law, α = {al:.2f}, x_min = {xm:g}")
        ax.axvline(xm, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(lab); ax.set_ylabel("P(X ≥ x)"); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_tails_ccdf.png", dpi=300); plt.close(fig)

    # F3 density of agglomeration area: best two MLE models
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    c, p, e = log_bins_pdf(x, 24)
    ax.errorbar(c, p, e, fmt="o", ms=4, color=INK, lw=0.8, label="Empirical density (24 log bins)")
    xx = np.logspace(np.log10(72), np.log10(x.max()), 200)
    for f_, col, ls in ((fits3[0], C1, "-"), (fits3[1], C2, "--")):
        ax.plot(xx, pdf_of(f_["model"], f_["params"], 72.0, 1e7, xx), color=col, ls=ls, lw=1.8,
                label=f"{f_['model']} (MLE, BIC {f_['BIC']:.0f})")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Built area of agglomeration (m²)"); ax.set_ylabel("Probability density (m⁻²)")
    ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(FIG / "fig_density_agglomerations.png", dpi=300); plt.close(fig)

    # F4 structures: density at tau 0.16 + tau sensitivity of local slopes
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4))
    ax = axs[0]
    c, p, e = log_bins_pdf(xs, 29)
    ax.errorbar(c, p, e, fmt="o", ms=3.5, color=INK, lw=0.8, label="Empirical density (29 log bins)")
    xx = np.logspace(np.log10(24), np.log10(900), 200)
    for f_, col, ls in ((fits5[0], C1, "-"), (next(f for f in fits5 if f["model"] == "Lognormal"), C2, "--")):
        ax.plot(xx, pdf_of(f_["model"], f_["params"], 24.0, 900.0, xx), color=col, ls=ls, lw=1.8, label=f"{f_['model']} (MLE)")
    for gx in (24, 900):
        ax.axvline(gx, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Footprint area (m²)"); ax.set_ylabel("Probability density (m⁻²)"); ax.legend(loc="lower left")
    ax = axs[1]
    tt = pd.DataFrame(taus)
    for col, key, lab in ((C1, "slope_50", "at 50 m²"), (C2, "slope_200", "at 200 m²"), (C3, "slope_800", "at 800 m²")):
        ax.plot(range(len(tt)), tt[key], "o-", color=col, lw=1.6, ms=6, label=f"Local slope {lab}")
    ax.set_xticks(range(len(tt))); ax.set_xticklabels(tt.catalogue, rotation=15); ax.set_ylabel("d ln p / d ln x"); ax.legend(loc="lower left")
    ax.set_title("Effect of the score threshold τ", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "fig_density_structures.png", dpi=300); plt.close(fig)

    # F5 hull area and extent vs n, data vs Poisson-chaining null
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, ycol, ylab in ((axs[0], "hull_ha", "Convex-hull area (ha)"), (axs[1], "extent_m", "Extent (m)")):
        s = A[(A.n >= 10) & (A[ycol] > 0)]; sn = pc_t
        ax.scatter(sn.n, sn[ycol], s=8, color=C2, alpha=0.35, label="Poisson chaining null", zorder=2)
        ax.scatter(s.n, s[ycol], s=12, color=C1, alpha=0.85, label="Agglomerations (n ≥ 10)", zorder=3)
        for sub, col in ((s, C1), (sn, C2)):
            b, i = np.polyfit(np.log(sub.n), np.log(sub[ycol]), 1)
            nn = np.array([10, s.n.max()]); ax.plot(nn, np.exp(i) * nn ** b, color=col, lw=1.6, zorder=4)
            ax.annotate(f"∝ n^{b:.2f}", (nn[-1], np.exp(i) * nn[-1] ** b), color=col, fontsize=8, ha="right", va="bottom")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Structures per agglomeration, n"); ax.set_ylabel(ylab); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "fig_hull_extent_scaling.png", dpi=300); plt.close(fig)

    # F6 sensitivity: alpha and hull exponent across the DBSCAN grid
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    gd = pd.DataFrame(grid); gd = gd[gd.catalogue == "published"]
    for ax, key, cik, lab in ((axs[0], "alpha_1000", "alpha_1000_ci", "Tail exponent α (x ≥ 1,000 m²)"), (axs[1], "hull_exp", "hull_ci", "Hull-area exponent (n ≥ 10)")):
        for ms, col in ((3, C1), (4, C2)):
            sub = gd[gd.min_samples == ms]
            lo = np.array([c[0] for c in sub[cik]]); hi = np.array([c[1] for c in sub[cik]])
            ax.errorbar(sub.eps + (ms - 3) * 2, sub[key], yerr=[sub[key].values - lo, hi - sub[key].values], fmt="o", color=col, ms=6, capsize=3, label=f"min_samples = {ms}")
        if key == "hull_exp":
            lo_, hi_ = R["poisson_chain"]["hull_ha"]["ci"]
            ax.axhspan(lo_, hi_, color=C3, alpha=0.18, lw=0); ax.axhline(R["poisson_chain"]["hull_ha"]["exp"], color=C3, lw=1.2, label="Poisson chaining null (95% CI)")
        else:
            ax.axhline(2.0, color=MUTED, lw=0.8, ls=":"); ax.annotate("Zipf (α = 2)", (60, 2.02), color=MUTED, fontsize=8, ha="right")
        ax.set_xticks([40, 50, 60]); ax.set_xlabel("DBSCAN radius ε (m)"); ax.set_ylabel(lab); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(FIG / "fig_sensitivity.png", dpi=300); plt.close(fig)

    # F7 rank-size by geometry (GLAS / NFI), area and n
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, col_, lab in ((axs[0], "area_sum", "Built area (m²)"), (axs[1], "n", "Structures per agglomeration")):
        for gname, colr in (("GLAS", C1), ("NFI", C2)):
            s = np.sort(A[A.geom == gname][col_].values)[::-1]
            ax.plot(np.arange(1, len(s) + 1), s, color=colr, lw=1.8, label=f"{gname} (N = {len(s):,})")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Rank"); ax.set_ylabel(lab); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig_rank_by_geometry.png", dpi=300); plt.close(fig)

    def clean(o):
        if isinstance(o, dict):
            return {str(k_): clean(v_) for k_, v_ in o.items() if k_ != "per_point"}
        if isinstance(o, (list, tuple)):
            return [clean(v_) for v_ in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        return o
    (OUT / "results.json").write_text(json.dumps(clean(R), indent=1))
    log("done")


if __name__ == "__main__":
    main()
