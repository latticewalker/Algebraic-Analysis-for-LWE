"""
Two-stage algebraic attack complexity estimator for 7 parameter
sets plus binary-secret TFHE:
  * HES:       3 sets (uniform ternary secret + discrete Gaussian error)
  * NTRU-HPS:  3 sets (Ternary secret + Fixed_Type error)
  * NTRU-HRSS: 1 set
"""

import math
from decimal import Decimal, getcontext
from math import log2

import numpy as np
from scipy.special import roots_legendre
from scipy.stats import binom, chi2

from util import matzov_short_vectors

getcontext().prec = 50
getcontext().rounding = "ROUND_HALF_UP"

LN2 = math.log(2.0)
TINY_NEG = -np.finfo(np.float64).tiny

BETA_MIN, BETA_MAX, BETA_STEP = 300, 1000, 5
Y_MIN, Y_MAX = 0, 6
K_I_MIN, K_I_MAX, K_I_STEP = 600, 12000, 5

SURVIVE_PROB = 0.8
SUPPORT_THRESHOLD = 1e-14
QUAD_ORDER = 64
TRUNC_Q = 6
SIGMA_GRID_PTS = 256
PHI_BINS = 1024
SIGMA_CHUNK = 512
PHI_STATE_CHUNK = 4096

VALID_MODELS = {"hes", "ntru_hps", "ntru_hrss", "binary_gaussian"}


def normalize_secret_distribution(values, probs):
    values = np.asarray(values, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if values.ndim != 1 or probs.ndim != 1 or len(values) != len(probs):
        raise ValueError("secret_values and secret_probs must have equal 1-D lengths")
    if len(values) < 2 or np.any(probs < 0) or probs.sum() <= 0:
        raise ValueError("invalid secret distribution")

    merged = {}
    for value, prob in zip(values.tolist(), (probs / probs.sum()).tolist()):
        merged[value] = merged.get(value, 0.0) + prob
    values = np.array(sorted(merged), dtype=np.int64)
    probs = np.array([merged[int(value)] for value in values], dtype=np.float64)
    probs /= probs.sum()
    return values, probs


def secret_support_size(values) -> int:
    return int(len(np.unique(np.asarray(values, dtype=np.int64))))


def ternary_nonzero_probability(values, probs) -> float:
    values, probs = normalize_secret_distribution(values, probs)
    if set(values.tolist()) != {-1, 0, 1}:
        raise ValueError("this estimator only supports ternary secrets {-1,0,1}")
    return float(probs[values != 0].sum())


def _compress(states, weights, threshold=SUPPORT_THRESHOLD):
    states = np.asarray(states, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = weights > threshold
    states = states[mask]
    weights = weights[mask]
    weights /= weights.sum()
    return states, weights


def chi_square_probability_quadrature(df: int, order: int = QUAD_ORDER):
    if df <= 0 or order < 8:
        raise ValueError("invalid chi-square quadrature parameters")
    z, w = roots_legendre(order)
    u = 0.5 * (z + 1.0)
    nodes = chi2.ppf(u, df=df).astype(np.float64)
    weights = (0.5 * w).astype(np.float64)
    weights /= weights.sum()
    if not np.all(np.isfinite(nodes)):
        raise FloatingPointError("non-finite chi-square quadrature nodes")
    return nodes, weights


def target_norm_states(cfg, k_lat, quad_order=QUAD_ORDER):
    model = cfg["norm_model"]
    if model not in VALID_MODELS:
        raise ValueError(f"unsupported norm_model: {model}")

    m = int(cfg["m"])

    if model == "binary_gaussian":
        values, probs = normalize_secret_distribution(
            cfg["secret_values"], cfg["secret_probs"]
        )
        if set(values.tolist()) != {0, 1}:
            raise ValueError("binary_gaussian requires secret support {0,1}")
        p_one = float(probs[values == 1].sum())
        sigma_e = float(cfg["error_sigma"])
        error_nodes, error_weights = chi_square_probability_quadrature(m, quad_order)
        secret_norm = np.arange(k_lat + 1, dtype=np.int64)
        secret_weights = binom.pmf(secret_norm, k_lat, p_one)
        h = sigma_e**2 * error_nodes[None, :] + secret_norm[:, None]
        w = secret_weights[:, None] * error_weights[None, :]
        return _compress(h.ravel(), w.ravel(), 0.0)

    p_nz = ternary_nonzero_probability(cfg["secret_values"], cfg["secret_probs"])

    if model == "hes":
        sigma_e = float(cfg["error_sigma"])
        error_nodes, error_weights = chi_square_probability_quadrature(m, quad_order)
        secret_norm = np.arange(k_lat + 1, dtype=np.int64)
        secret_weights = binom.pmf(secret_norm, k_lat, p_nz)
        h = sigma_e**2 * error_nodes[None, :] + secret_norm[:, None]
        w = secret_weights[:, None] * error_weights[None, :]
        return _compress(h.ravel(), w.ravel(), 0.0)

    if model == "ntru_hps":
        w_g = int(cfg["w_g"])
        secret_norm = np.arange(k_lat + 1, dtype=np.int64)
        secret_weights = binom.pmf(secret_norm, k_lat, p_nz)
        h, w = _compress(secret_norm, secret_weights)
        return w_g + h, w

    total_active_coordinates = m + k_lat
    norm = np.arange(total_active_coordinates + 1, dtype=np.int64)
    weights = binom.pmf(norm, total_active_coordinates, p_nz)
    return _compress(norm, weights)


def rho_mod_q_zero_vec(q: int, sigmas: np.ndarray, trunc_q: int = TRUNC_Q,
                       sigma_chunk: int = SIGMA_CHUNK) -> np.ndarray:
    sig = np.asarray(sigmas, dtype=np.float64)
    if np.any(sig <= 0) or np.any(~np.isfinite(sig)):
        raise ValueError("all sigmas must be finite and positive")

    qf = float(q)
    correction = (
        1.0
        + 2.0 * np.exp(-2.0 * math.pi**2 * sig**2)
        + 2.0 * np.exp(-8.0 * math.pi**2 * sig**2)
    )
    out = np.empty_like(sig)

    primal_mask = sig / qf < 0.75
    if primal_mask.any():
        small = sig[primal_mask]
        denominator = math.sqrt(2.0 * math.pi) * small * correction[primal_mask]
        j_max = max(int(math.ceil(trunc_q * float(small.max()) / qf)) + 2, 1)
        jq_sq = (np.arange(1, j_max + 1, dtype=np.float64) * qf) ** 2
        inv = 0.5 / (small * small)
        small_out = np.empty_like(small)
        for start in range(0, len(small), sigma_chunk):
            end = min(start + sigma_chunk, len(small))
            numerator = 1.0 + 2.0 * np.exp(
                -jq_sq[None, :] * inv[start:end, None]
            ).sum(axis=1)
            small_out[start:end] = numerator / denominator[start:end]
        out[primal_mask] = small_out

    if (~primal_mask).any():
        large = sig[~primal_mask]
        ratio = large / qf
        theta = np.ones_like(large)
        for k in range(1, 64):
            term = 2.0 * np.exp(-2.0 * math.pi**2 * ratio**2 * k**2)
            theta += term
            if float(term.max()) < 1e-16:
                break
        out[~primal_mask] = theta / (qf * correction[~primal_mask])
    return out


def phi_matrix(p_row: np.ndarray, k_col: np.ndarray, w_col: np.ndarray) -> np.ndarray:
    p = np.clip(p_row, 1e-300, 1.0 - 1e-15)
    u = k_col[:, None] * np.log1p(-p)[None, :]
    log_one_minus = np.where(
        u < -37.0,
        -np.exp(u),
        np.log(-np.expm1(np.minimum(u, -1e-300))),
    )
    log_phi = w_col[:, None] * log_one_minus
    return np.where(log_phi < -745.0, 0.0, np.exp(log_phi))


def phi_expectation(p0_states, state_weights, k_arr, w_arr,
                    state_chunk=PHI_STATE_CHUNK):
    ans = np.zeros_like(k_arr, dtype=np.float64)
    for start in range(0, len(p0_states), state_chunk):
        end = min(start + state_chunk, len(p0_states))
        ans += (
            phi_matrix(p0_states[start:end], k_arr, w_arr)
            @ state_weights[start:end]
        )
    return ans


def cal_l(m: int, beta_bkz: int, beta_sieve: int, k_lat: int, q: int) -> float:
    q_dec = Decimal(int(q))
    pi = Decimal("3.141592653589793238462643383279502884197")
    e = Decimal("2.718281828459045235360287471352662497757")
    beta1_dec = Decimal(beta_bkz)
    beta2_dec = Decimal(beta_sieve)
    k_lat_dec = Decimal(k_lat)
    m_dec = Decimal(m)
    r = m_dec + k_lat_dec

    n_vectors = (Decimal(4) / Decimal(3)) ** (beta2_dec / Decimal(2))
    delta = (
        beta1_dec / (Decimal(2) * pi * e)
        * (pi * beta1_dec) ** (Decimal(1) / beta1_dec)
    ) ** (Decimal(1) / (beta1_dec - Decimal(1)))

    ell = (
        q_dec ** (k_lat_dec / r)
        * n_vectors ** (Decimal(1) / beta2_dec)
        * (
            beta2_dec / (Decimal(2) * pi * e)
            * (pi * beta2_dec) ** (Decimal(1) / beta2_dec)
        ).sqrt()
        * delta ** ((r - beta2_dec) / Decimal(2))
    )
    return float(ell)


def precompute_p0(m, q, beta_bkz, beta_sieve, k_lat, h_states, state_weights,
                  grid_pts=SIGMA_GRID_PTS, bins=PHI_BINS):
    ell = cal_l(m, beta_bkz, int(beta_sieve), k_lat, q)
    sigma = ell * np.sqrt(h_states / float(m + k_lat))
    inv_q = 1.0 / q

    lo, hi = float(sigma.min()), float(sigma.max())
    if hi > lo:
        grid = np.linspace(lo, hi, grid_pts)
        p0 = np.interp(sigma, grid, rho_mod_q_zero_vec(q, grid))
    else:
        p0 = np.full_like(sigma, rho_mod_q_zero_vec(q, np.array([lo]))[0])
    np.maximum(p0, inv_q, out=p0)

    p_lo, p_hi = float(p0.min()), float(p0.max())
    if p_hi > p_lo and bins > 1:
        edges = np.linspace(p_lo, p_hi, bins + 1)
        idx = np.clip(np.searchsorted(edges, p0, side="right") - 1, 0, bins - 1)
        w_c = np.bincount(idx, weights=state_weights, minlength=bins)
        pw_c = np.bincount(idx, weights=state_weights * p0, minlength=bins)
        keep = w_c > 0.0
        p0_c = pw_c[keep] / w_c[keep]
        w_c = w_c[keep]
    else:
        p0_c = np.array([p_lo])
        w_c = np.array([state_weights.sum()])
    return p0_c, w_c


def com1_value(n, k_lat, y, secret_degree, w_value):
    d = int(secret_degree)
    n_solve = n - k_lat - y
    if d < 2 or n_solve < 1:
        return float("inf")
    log2_v = n_solve * log2(2 * d - 1)
    return (
        log2_v
        + log2(log2_v)
        + log2(log2(log2_v))
        + log2(w_value)
        + log2(d - 1)
        + (2 * d - 1)
    )


def total_complexity(com1, com2):
    high = max(com1, com2)
    low = min(com1, com2)
    return high + math.log2(1.0 + 2.0 ** (low - high))


def eval_k_lat(k_lat, cfg):
    n = int(cfg["n"])
    m = int(cfg["m"])
    q = int(cfg["q"])

    secret_values, secret_probs = normalize_secret_distribution(
        cfg["secret_values"], cfg["secret_probs"]
    )
    secret_degree = secret_support_size(secret_values)
    log2_secret_space = math.log2(secret_degree)
    log2_q = math.log2(q)
    base_log_q = math.log1p(-1.0 / q)

    h_states, state_weights = target_norm_states(cfg, k_lat)

    i_arr = np.arange(K_I_MIN, K_I_MAX, K_I_STEP, dtype=np.float64)
    log_k = i_arr / 100.0
    k_arr = np.power(2.0, log_k)

    u2 = k_arr * base_log_q
    log_ob2 = np.where(
        u2 < -37.0,
        -np.exp(u2),
        np.log(-np.expm1(np.minimum(u2, -1e-300))),
    )
    log2_wrong_hit = np.where(log_ob2 / LN2 < 0.0, log_ob2 / LN2, TINY_NEG)

    best = None
    lattice_dim = m + k_lat

    for beta_bkz in range(BETA_MIN, BETA_MAX + 1, BETA_STEP):
        if beta_bkz > lattice_dim:
            continue

        total_cost, n_vectors, beta_sieve = matzov_short_vectors(
            beta_bkz, lattice_dim
        )
        com2 = math.log2(total_cost)
        if best is not None and com2 >= best["obj"]:
            continue

        log2_n_vectors = math.log2(n_vectors)
        p0_states, p0_weights = precompute_p0(
            m, q, beta_bkz, beta_sieve, k_lat, h_states, state_weights
        )

        for y in range(Y_MIN, Y_MAX + 1):
            n_solve = n - k_lat - y
            if n_solve < 1:
                continue

            log2_budget = log2_n_vectors - y * log2_q
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                w_arr = (-n_solve * log2_secret_space) / log2_wrong_hit
                log2_kw = log_k + np.log2(w_arr)

            cut = int(np.searchsorted(log2_kw, log2_budget, side="right"))
            if cut == 0:
                continue

            p_true = phi_expectation(
                p0_states, p0_weights, k_arr[:cut], w_arr[:cut]
            )
            feasible = p_true >= SURVIVE_PROB
            if not feasible.any():
                continue

            idx = int(np.argmax(feasible))
            w_value = float(w_arr[idx])
            com1 = com1_value(n, k_lat, y, secret_degree, w_value)
            obj = total_complexity(com1, com2)

            if best is None or obj < best["obj"]:
                best = {
                    "obj": obj,
                    "com1": com1,
                    "com2": com2,
                    "beta_bkz": beta_bkz,
                    "beta_sieve": int(beta_sieve),
                    "k_lat": k_lat,
                    "y": y,
                    "n_solve": n_solve,
                    "log2K": float(log_k[idx]),
                    "log2W": math.log2(w_value),
                    "p_true": float(p_true[idx]),
                }
    return best
