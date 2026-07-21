"""
Checks the 1/q approximation of Lemma 6 

For a wrong candidate v != s_solve  each linear factor of F_j
is
    L_j(v) = b'_j - <a'_j, v>
           = <x_j, A_solve (s_solve - v)> + <x_j, e> + <y_j, s_lat>   (mod q),
and Lemma 6 claims  Pr[ L_j(v) = 0 mod q ] ~= 1/q  over the sieve outputs
(x_j, y_j). This script measures that probability directly, pooling over many
candidates x many sieve vectors so the confidence interval is a tiny fraction
of 1/q.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import sys
from functools import partial
from math import ceil, sqrt
from multiprocessing import cpu_count, Pool

import numpy as np

from fpylll import IntegerMatrix, BKZ
from fpylll.algorithms.bkz2 import BKZReduction
from g6k import SieverParams
from g6k.siever import Siever


# ---------------------------------------------------------------------------
# First stage
# ---------------------------------------------------------------------------
class CentredBinomial:
    def __init__(self, eta=3):
        self.eta = eta

    def __call__(self):
        from random import randint
        return sum(randint(0, 1) for _ in range(2 * self.eta)) - self.eta


def generate_LWE_lattice(m, n, q):
    B = IntegerMatrix.random(m, "qary", k=m - n, q=q)
    A = B.submatrix(0, n, n, m)
    return A, B


def progressive_BKZ(B, beta, params, verbose=False):
    g6k = Siever(B, params)
    bkz = BKZReduction(g6k.M)
    for _beta in range(2, beta + 1):
        if verbose:
            print("\rBKZ_%d" % _beta, end=""); sys.stdout.flush()
        bkz(BKZ.Param(_beta, max_loops=2))
    if verbose:
        print("\nfinish bkz")
    return g6k


def progressive_sieve(g6k, l, r, verbose=False):
    g6k.initialize_local(l, max(l, r - 20), r)
    g6k(alg="gauss")
    while g6k.l > l:
        if verbose:
            print("\rSieving [%3d, %3d]..." % (g6k.l, g6k.r), end=""); sys.stdout.flush()
        g6k.extend_left()
        g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
    with g6k.temp_params(saturation_ratio=0.9, db_size_factor=6):
        g6k(alg="hk3")
    g6k.resize_db(ceil(1.0 * (4 / 3) ** ((r - l) / 2)))
    if verbose:
        print()
    return g6k


def change_basis(basis, vector):
    return basis.multiply_left(vector)


def run_first_stage(n, k_lat, q, eta, threads, betabkz, betasieve, verbose):
    """Build a fresh LWE instance, dual-reduce, and return a'_j, b'_j, s_solve."""
    k_solve = n - k_lat
    A, _ = generate_LWE_lattice(2 * n, n, q)
    A_solve, A_lat = A[:k_solve], A[k_solve:]
    A_solve.transpose()          # n x k_solve
    A_lat.transpose()            # n x k_lat

    dist = CentredBinomial(eta)
    secret = [dist() for _ in range(n)]
    error = [dist() for _ in range(n)]
    s_solve = np.array(secret[:k_solve], dtype=np.int64)
    s_lat = np.array(secret[k_solve:], dtype=np.int64)
    err = np.array(error, dtype=np.int64)

    target = np.array(A.multiply_left(secret), dtype=np.int64) + err
    A_solve_np = np.array([[A_solve[i, j] for j in range(k_solve)]
                           for i in range(n)], dtype=np.int64)

    B_dual = IntegerMatrix.identity(n + k_lat)
    for i in range(n, n + k_lat):
        B_dual[i, i] *= q
    for i in range(0, n):
        for j in range(0, k_lat):
            B_dual[i, n + j] = A_lat[i, j] % q

    print(f"[stage-1] BKZ-{betabkz}, sieve dim {betasieve}, r={n + k_lat}, {threads} threads")
    sp = SieverParams(threads=threads, dual_mode=False)
    g6k = progressive_BKZ(B_dual, betabkz, sp, verbose=verbose)
    progressive_sieve(g6k, 0, betasieve, verbose=verbose)

    Bmat = g6k.M.B
    try:
        with Pool(threads) as pool:
            db = pool.map(partial(change_basis, Bmat), g6k.itervalues())
    except Exception as ex:
        print(f"[stage-1] parallel collect failed ({ex}); serial fallback")
        db = [change_basis(Bmat, c) for c in g6k.itervalues()]

    W = np.array([list(w[:n + k_lat]) for w in db], dtype=np.int64)
    X, Y = W[:, :n], W[:, n:n + k_lat]
    N = X.shape[0]

    A_prime = (X @ A_solve_np) % q       # a'_j
    b_prime = (X @ target) % q           # b'_j
    e_prime = (X @ err + Y @ s_lat) % q  # e'_j

    if not np.array_equal((b_prime - A_prime @ s_solve) % q, e_prime):
        raise RuntimeError("identity b' - <a', s_solve> = e' FAILED")
    print(f"[stage-1] N = {N} sieve vectors; identity check OK")
    return A_prime, b_prime, s_solve, k_solve, N


def build_candidates(k_solve, T, s_solve, max_enum, num_sample, seed):
    total = (2 * T + 1) ** k_solve - 1
    if total <= max_enum:
        cands = [np.array(v, dtype=np.int64)
                 for v in itertools.product(range(-T, T + 1), repeat=k_solve)
                 if not np.array_equal(np.array(v, dtype=np.int64), s_solve)]
        return np.array(cands, dtype=np.int64), total, True
    rng = np.random.default_rng(seed)
    st = tuple(int(x) for x in s_solve)
    seen, out = set(), []
    while len(out) < num_sample:
        for v in rng.integers(-T, T + 1, size=(num_sample, k_solve)):
            t = tuple(int(x) for x in v)
            if t == st or t in seen:
                continue
            seen.add(t); out.append(v.astype(np.int64))
            if len(out) >= num_sample:
                break
    return np.array(out, dtype=np.int64), total, False


_Af = _bcol = _qf = None


def _count_zeros(V):
    M = _Af @ V.T.astype(np.float64)                   
    return int(np.count_nonzero(np.mod(_bcol - M, _qf) == 0.0))


def count_all(A_prime, b_prime, q, candidates, threads):
    global _Af, _bcol, _qf
    _Af = A_prime.astype(np.float64)                    
    _bcol = b_prime.astype(np.float64)[:, None]         
    _qf = float(q)
    N = A_prime.shape[0]
    step = max(1, 8_000_000 // N)
    chunks = [candidates[i:i + step] for i in range(0, len(candidates), step)]
    if threads <= 1 or len(chunks) == 1:
        parts = [_count_zeros(c) for c in chunks]
    else:
        with Pool(threads) as pool:
            parts = pool.map(_count_zeros, chunks)
    zeros = int(sum(parts))
    pairs = int(len(candidates)) * int(N)
    return zeros, pairs


def run_combo(solve, lat, args, threads):
    n, T, p0 = solve + lat, args.eta, 1.0 / args.q
    tot_z = tot_p = 0
    per_inst = []
    for rep in range(args.repeats):
        if args.seed is not None:                       
            from random import seed as rseed
            rseed(args.seed + rep); np.random.seed(args.seed + rep)
        A_prime, b_prime, s_solve, k_solve, N = run_first_stage(
            n, lat, args.q, args.eta, threads, args.bkz, args.sieve, args.v > 0)
        cands, total_wrong, enum = build_candidates(
            k_solve, T, s_solve, args.max_enum, args.num_candidates,
            seed=(args.seed or 0) + rep)
        print(f"  [cands] total_wrong={total_wrong} tested={len(cands)} enum_all={enum}")
        z, pr = count_all(A_prime, b_prime, args.q, cands, threads)
        tot_z += z; tot_p += pr
        per_inst.append((rep + 1, N, len(cands), pr, z / pr))
        print(f"  [inst {rep + 1}/{args.repeats}] p_hat={z / pr:.6e} "
              f"ratio={(z / pr) / p0:.4f}")
    ph = tot_z / tot_p
    se = sqrt(ph * (1 - ph) / tot_p)
    return {"solve": solve, "lat": lat, "trials": tot_p, "zeros": tot_z,
            "p_hat": ph, "ratio": ph / p0, "ci": (ph - 1.96 * se, ph + 1.96 * se),
            "z": (ph - p0) / se, "per_inst": per_inst}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', action='count', default=0)
    ap.add_argument('-j', type=int, default=24)
    ap.add_argument('--solve-list', type=str, default="8,10,12,14,16",
                    help='comma-separated n_solve values')
    ap.add_argument('--lat-list', type=str, default="40,45,50,55",
                    help='comma-separated n_lat values (need n_solve + 2*n_lat >= sieve)')
    ap.add_argument('-q', type=int, default=3329)
    ap.add_argument('--eta', type=int, default=1, help='T = eta')
    ap.add_argument('--bkz', type=int, default=40)
    ap.add_argument('--sieve', type=int, default=80)
    ap.add_argument('--repeats', type=int, default=1,
                    help='independent LWE instances pooled per (solve, lat)')
    ap.add_argument('--max-enum', type=int, default=60000,
                    help='enumerate ALL wrong candidates if their count <= this')
    ap.add_argument('--num-candidates', type=int, default=50000,
                    help='random wrong candidates per instance when enumeration is infeasible')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', type=str, default="Results/lemma6_grid.txt")
    args = ap.parse_args()

    threads = min(cpu_count(), args.j)
    p0 = 1.0 / args.q
    solve_vals = [int(x) for x in args.solve_list.split(",") if x.strip()]
    lat_vals = [int(x) for x in args.lat_list.split(",") if x.strip()]
    combos = list(itertools.product(solve_vals, lat_vals))

    print(f"grid: {len(combos)} (solve, lat) combos, {threads} threads, "
          f"BKZ={args.bkz}, sieve={args.sieve}, q={args.q}, repeats={args.repeats}")

    results = []
    for idx, (solve, lat) in enumerate(combos, 1):
        r = solve + 2 * lat                             # dual lattice dimension
        print("\n" + "#" * 64)
        print(f"# combo {idx}/{len(combos)}: n_solve={solve}, n_lat={lat}, r={r}")
        print("#" * 64)
        if r < args.sieve:                              # sieve needs r >= sieve dim
            print(f"  [skip] r={r} < sieve={args.sieve}; raise n_lat for this combo")
            continue
        results.append(run_combo(solve, lat, args, threads))

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"SUMMARY   Pr[b' - <a', v> = 0 mod q]  vs  1/q = {p0:.6e}")
    print("=" * 78)
    print(f"{'solve':>5} {'lat':>4} {'r':>4} {'trials':>16} "
          f"{'p_hat':>13} {'ratio':>8} {'z':>8}")
    for res in results:
        print(f"{res['solve']:>5} {res['lat']:>4} {res['solve'] + 2 * res['lat']:>4} "
              f"{res['trials']:>16,} {res['p_hat']:>13.6e} "
              f"{res['ratio']:>8.4f} {res['z']:>+8.2f}")
    print("=" * 78)
    print("ratio ~= 1 and |z| < 2 => Lemma 6's 1/q approximation is supported")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"target_1_over_q = {p0:.10e}\n")
        f.write(f"bkz = {args.bkz}, sieve = {args.sieve}, q = {args.q}, "
                f"eta = {args.eta}, repeats = {args.repeats}\n\n")
        f.write("# solve, lat, r, trials, zeros, p_hat, ratio, ci_lo, ci_hi, z\n")
        for res in results:
            lo, hi = res['ci']
            f.write(f"{res['solve']}, {res['lat']}, {res['solve'] + 2 * res['lat']}, "
                    f"{res['trials']}, {res['zeros']}, {res['p_hat']:.10e}, "
                    f"{res['ratio']:.6f}, {lo:.10e}, {hi:.10e}, {res['z']:.4f}\n")
    print(f"[saved] {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
