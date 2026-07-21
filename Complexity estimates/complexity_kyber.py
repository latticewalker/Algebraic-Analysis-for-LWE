"""
Estimates the overall attack complexity for Kyber-512 / 768 / 1024 in a single
run, and prints the best (balanced) cost for each parameter set.

The cost model is the two-stage algorithm:
    total = log2(2^com1 + 2^com2)
"""

import os
import math
from math import log2
import numpy as np
from decimal import Decimal, getcontext
from concurrent.futures import ProcessPoolExecutor, as_completed

from util import matzov_short_vectors


def _matzov(beta, d):
    res = tuple(matzov_short_vectors(beta, d))
    if len(res) == 4:
        _, total_cost, n_vectors, beta_sieve = res
    elif len(res) == 3:
        total_cost, n_vectors, beta_sieve = res
    else:
        raise ValueError(
            f"matzov_short_vectors returned {len(res)} values; expected 3 or 4.")
    return total_cost, n_vectors, beta_sieve


getcontext().prec = 50
getcontext().rounding = 'ROUND_HALF_UP'

LN2 = math.log(2.0)

# =========================================================================== #
#  Global search settings
# =========================================================================== #
Y_MIN, Y_MAX      = 0, 6
K_I_MIN, K_I_MAX  = 1500, 2000
SURVIVE_PROB      = 0.8
SUPPORT_THRESHOLD = 1e-14
TRUNC_Q           = 6
P0_TABLE_POINTS   = 8000
WORKERS           = 24   # parallel worker processes

KYBER_PARAMS = {
    "Kyber512":  dict(n=512,  eta=3, beta=(580, 720, 2),   k_lat=(420, 505, 1)),
    "Kyber768":  dict(n=768,  eta=2, beta=(900, 1050, 2),  k_lat=(640, 761, 1)),
    "Kyber1024": dict(n=1024, eta=2, beta=(1300, 1500, 2), k_lat=(851, 1017, 1)),
}
Q = 3329


def cbd_coordinate_pmf(eta: int) -> dict:
    if eta < 0:
        raise ValueError("eta must be nonnegative.")
    if eta == 0:
        return {0: 1.0}
    denom = 2 ** (2 * eta)
    return {x: math.comb(2 * eta, x + eta) / denom for x in range(-eta, eta + 1)}


def cbd_square_pmf_array(eta: int) -> np.ndarray:
    pmf_x = cbd_coordinate_pmf(eta)
    arr = np.zeros(eta * eta + 1, dtype=np.float64)
    for x, p in pmf_x.items():
        arr[x * x] += p
    return arr


def precompute_norm_pmfs(eta: int, m: int, k_lats, threshold: float = SUPPORT_THRESHOLD):
    base = cbd_square_pmf_array(eta)
    targets = {m + k: k for k in k_lats}
    ordered = sorted(targets)
    max_d = ordered[-1]

    pmfs = {}
    result = np.array([1.0], dtype=np.float64)
    ti = 0
    for d in range(1, max_d + 1):
        result = np.convolve(result, base)
        if ti < len(ordered) and d == ordered[ti]:
            k = targets[d]
            r = result
            s = r.sum()
            if s > 0.0:
                r = r / s
            mask = r > threshold
            h_values = np.where(mask)[0].astype(np.int64)
            probs = r[mask].astype(np.float64).copy()
            pmfs[k] = (h_values, probs)
            ti += 1
    return pmfs


def rho_mod_q_zero_vec(q: int, sigmas: np.ndarray, trunc_q: int = TRUNC_Q,
                       cutoff: float = 45.0) -> np.ndarray:
    sig = np.asarray(sigmas, dtype=np.float64)
    inv = 0.5 / (sig * sig)
    B = trunc_q

    j = np.arange(1, B + 1, dtype=np.float64)
    num = 1.0 + 2.0 * np.exp(-((j * q) ** 2)[None, :] * inv[:, None]).sum(axis=1)

    z_full = B * q
    inv_min = float(inv.min())
    z_bound = min(z_full, int(math.ceil(math.sqrt(cutoff / inv_min))))
    z_bound = max(z_bound, 1)

    zsq = (np.arange(1, z_bound + 1, dtype=np.float64)) ** 2
    den = np.ones_like(sig)
    chunk = max(256, 8_000_000 // max(len(sig), 1))
    for s in range(0, len(zsq), chunk):
        blk = zsq[s:s + chunk]
        den += 2.0 * np.exp(-blk[None, :] * inv[:, None]).sum(axis=1)
    return num / den


def cal_l(n: int, m: int, beta1: int, beta2: int, k_lat: int, eta: int) -> float:
    q = Decimal(3329)
    pi = Decimal('3.141592653589793238462643383279502884197')
    e = Decimal('2.718281828459045235360287471352662497757')
    beta1_dec = Decimal(beta1)
    beta2_dec = Decimal(beta2)
    k_lat_dec = Decimal(k_lat)
    m_dec = Decimal(m)
    d = m_dec + k_lat_dec
    N = (Decimal(4) / Decimal(3)) ** (beta2_dec / Decimal(2))
    term1 = beta1_dec / (Decimal(2) * pi * e)
    term2 = (pi * beta1_dec) ** (Decimal(1) / beta1_dec)
    delta = (term1 * term2) ** (Decimal(1) / (beta1_dec - Decimal(1)))
    q_pow = q ** (k_lat_dec / d)
    N_pow = N ** (Decimal(1) / beta2_dec)
    sqrt_term = ((beta2_dec / (Decimal(2) * pi * e))
                 * ((pi * beta2_dec) ** (Decimal(1) / beta2_dec))).sqrt()
    delta_pow = delta ** ((d - beta2_dec) / Decimal(2))
    l = q_pow * N_pow * sqrt_term * delta_pow
    return float(l)


def precompute_lattice(n, m, eta, betas, k_lats):
    table = {}
    for k_lat in k_lats:
        d = m + k_lat
        for beta in betas:
            total_cost, n_vectors, beta_sieve = _matzov(beta, d)
            if not math.isfinite(total_cost) or total_cost <= 0:
                continue
            com2 = math.log2(total_cost)
            log2_N = math.log2(n_vectors) if n_vectors > 0 else float('-inf')
            ell = cal_l(n, m, beta, int(beta_sieve), k_lat, eta)
            table[(beta, k_lat)] = (com2, log2_N, int(beta_sieve), ell)
    return table


def build_p0_table(q, eta, m, k_lats, pmfs, lattice_tbl, n_points=P0_TABLE_POINTS):
    sig_min, sig_max = math.inf, 0.0
    for k_lat, (h_values, _) in pmfs.items():
        d_star = m + k_lat
        hpos = h_values[h_values > 0]
        if hpos.size == 0:
            continue
        h_lo = float(hpos.min())
        h_hi = float(h_values.max())
        for beta_key in ((b, k_lat) for b in {kk[0] for kk in lattice_tbl if kk[1] == k_lat}):
            ell = lattice_tbl[beta_key][3]
            sig_min = min(sig_min, ell * math.sqrt(h_lo / d_star))
            sig_max = max(sig_max, ell * math.sqrt(h_hi / d_star))

    if not math.isfinite(sig_min) or sig_max <= 0.0:
        sig_min, sig_max = 1e-6, 1.0
    sig_min *= 0.9
    sig_max *= 1.1

    sigma_grid = np.logspace(math.log10(sig_min), math.log10(sig_max), n_points)
    p0_grid = rho_mod_q_zero_vec(q, sigma_grid, TRUNC_Q)
    np.maximum(p0_grid, 1.0 / q, out=p0_grid)
    return np.log(sigma_grid), p0_grid


def com1_value(n, k_lat, y, eta, w_val):
    D = 2 * eta + 1
    n_g = n - k_lat - y
    lv = n_g * log2(2 * D - 1)
    return (lv + log2(lv) + log2(log2(lv))
            + log2(w_val) + log2(D - 1) + (2 * D - 1))


def total_complexity(com1, com2):
    hi = com1 if com1 >= com2 else com2
    lo = com2 if com1 >= com2 else com1
    return hi + math.log2(1.0 + 2.0 ** (lo - hi))


_WS = {}


def _init_worker(cfg, lattice_tbl, log_sigma_grid, p0_grid):
    q = cfg['q']
    o = 2 * cfg['eta'] + 1

    i_arr = np.arange(K_I_MIN, K_I_MAX, dtype=np.float64)
    logK = i_arr / 100.0
    K_arr = np.power(2.0, logK)
    base_log_q = math.log1p(-1.0 / q)
    u2 = K_arr * base_log_q
    log_ob2 = np.where(u2 < -37.0, -np.exp(u2),
                       np.log(-np.expm1(np.minimum(u2, -1e-300))))
    log2_term2 = log_ob2 / LN2
    w_unit = (-math.log2(o)) / log2_term2

    _WS.update(dict(
        cfg=cfg, lattice_tbl=lattice_tbl,
        log_sigma_grid=log_sigma_grid, p0_grid=p0_grid,
        logK=logK, K_arr=K_arr,
        log2_o=math.log2(o), log2_q=math.log2(q),
        w_unit=w_unit, log2_wunit=np.log2(w_unit),
    ))


def eval_k_lat(task):
    k_lat, h_values, probs = task
    cfg = _WS['cfg']
    n, m, eta, q = cfg['n'], cfg['m'], cfg['eta'], cfg['q']
    lattice_tbl = _WS['lattice_tbl']
    log_sigma_grid, p0_grid = _WS['log_sigma_grid'], _WS['p0_grid']
    logK, K_arr = _WS['logK'], _WS['K_arr']
    w_unit, log2_wunit = _WS['w_unit'], _WS['log2_wunit']
    log2_q = _WS['log2_q']

    d_star = m + k_lat
    zero_mask = (h_values == 0)
    hfloat = h_values.astype(np.float64)
    ys = list(range(Y_MIN, Y_MAX + 1))

    best = None
    for beta in cfg['betas']:
        key = (beta, k_lat)
        if key not in lattice_tbl:
            continue
        com2, log2_N, beta2, ell = lattice_tbl[key]

        if best is not None and com2 >= best['obj']:
            break

        sig = ell * np.sqrt(hfloat / d_star)
        with np.errstate(divide='ignore'):
            log_sig = np.log(sig)
        p0 = np.interp(log_sig, log_sigma_grid, p0_grid)
        np.maximum(p0, 1.0 / q, out=p0)
        p0[zero_mask] = 1.0

        max_idx = -1
        budget = []
        for y in ys:
            n_g = n - k_lat - y
            if n_g < 1:
                budget.append(None)
                continue
            log2_budget = log2_N - y * log2_q
            log2_KW = logK + math.log2(n_g) + log2_wunit
            bmask = log2_KW <= log2_budget
            budget.append((n_g, log2_budget, bmask))
            if bmask.any():
                last = int(np.nonzero(bmask)[0][-1])
                if last > max_idx:
                    max_idx = last
        if max_idx < 0:
            continue
        M = max_idx + 1

        log1m_p = np.log1p(-np.clip(p0, 1e-300, 1.0 - 1e-15))
        u = K_arr[:M, None] * log1m_p[None, :]
        log_ob = np.where(u < -37.0, -np.exp(u),
                          np.log(-np.expm1(np.minimum(u, -1e-300))))
        G = w_unit[:M, None] * log_ob

        for y, b in zip(ys, budget):
            if b is None:
                continue
            n_g, log2_budget, bmask = b
            bmask = bmask[:M]
            if not bmask.any():
                continue

            log_phi = n_g * G
            phi = np.where(log_phi < -745.0, 0.0, np.exp(log_phi))
            phi_true = phi @ probs

            feasible = (phi_true >= SURVIVE_PROB) & bmask
            if not feasible.any():
                continue

            idx = int(np.argmax(feasible))
            w_val = n_g * float(w_unit[idx])
            com1 = com1_value(n, k_lat, y, eta, w_val)
            obj = total_complexity(com1, com2)
            if (best is None) or (obj < best['obj']):
                best = dict(
                    obj=obj, com1=com1, com2=com2,
                    beta=beta, beta2=beta2, k_lat=k_lat, y=y,
                    log2K=float(logK[idx]), log2W=math.log2(w_val),
                    phi_true=float(phi_true[idx]), ell=ell, log2N=log2_N,
                )
    return best


def run_variant(name, params, q=Q, workers=WORKERS):
    n = params['n']
    m = n
    eta = params['eta']
    betas = list(range(params['beta'][0], params['beta'][1] + 1, params['beta'][2]))
    k_lats = list(range(params['k_lat'][0], params['k_lat'][1] + 1, params['k_lat'][2]))

    print(f"\n################  {name}  "
          f"(n={n}, eta={eta}, q={q})  ################")
    print(f"[grid] |beta|={len(betas)}  |k_lat|={len(k_lats)}  "
          f"|y|={Y_MAX - Y_MIN + 1}  workers={workers}")

    pmfs = precompute_norm_pmfs(eta, m, k_lats, SUPPORT_THRESHOLD)
    lattice_tbl = precompute_lattice(n, m, eta, betas, k_lats)
    log_sigma_grid, p0_grid = build_p0_table(q, eta, m, k_lats, pmfs, lattice_tbl)

    cfg = dict(n=n, m=m, eta=eta, q=q, betas=betas)
    tasks = [(k_lat, pmfs[k_lat][0], pmfs[k_lat][1]) for k_lat in k_lats]

    results = []
    if workers <= 1:
        _init_worker(cfg, lattice_tbl, log_sigma_grid, p0_grid)
        for task in tasks:
            r = eval_k_lat(task)
            if r is not None:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(cfg, lattice_tbl, log_sigma_grid, p0_grid)) as ex:
            futs = {ex.submit(eval_k_lat, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                if r is not None:
                    results.append(r)

    if not results:
        print("  No feasible point in the given ranges; widen beta/k_lat/y.")
        return None

    results.sort(key=lambda x: x['obj'])
    best = results[0]
    print(f"\n----- {name}: optimum (min log2(2^com1 + 2^com2)) -----")
    print(f"  beta_bkz    = {best['beta']}")
    print(f"  beta_sieve  = {best['beta2']}")
    print(f"  k_lat       = {best['k_lat']}")
    print(f"  y           = {best['y']}")
    print(f"  log2(K)     = {best['log2K']:.4f}")
    print(f"  log2(W)     = {best['log2W']:.4f}")
    print(f"  phi_true    = {best['phi_true']:.6f}  (>= {SURVIVE_PROB})")
    print(f"  ell         = {best['ell']:.3f}")
    print(f"  com1        = {best['com1']:.4f}")
    print(f"  com2        = {best['com2']:.4f}")
    print(f"  |com1-com2| = {abs(best['com1'] - best['com2']):.4f}  (smaller = more balanced)")
    print(f"  TOTAL log2(2^com1 + 2^com2) = {best['obj']:.4f}")
    return best


def main(selected=None, workers=WORKERS):
    names = selected or list(KYBER_PARAMS.keys())
    summary = {}
    for name in names:
        best = run_variant(name, KYBER_PARAMS[name], q=Q, workers=workers)
        if best is not None:
            summary[name] = best

    print("\n=================  SUMMARY  =================")
    for name in names:
        if name in summary:
            b = summary[name]
            print(f"  {name:<10s}: total = {b['obj']:.3f} bits  "
                  f"(com1={b['com1']:.2f}, com2={b['com2']:.2f}, "
                  f"beta_bkz={b['beta']}, beta_sieve={b['beta2']}, "
                  f"k_lat={b['k_lat']}, y={b['y']})")
        else:
            print(f"  {name:<10s}: no feasible point")
    return summary


if __name__ == "__main__":
    main()
