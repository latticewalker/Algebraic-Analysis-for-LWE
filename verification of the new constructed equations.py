import os

# Must be set before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import itertools
import random
from dataclasses import dataclass
from math import ceil, comb
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Sequence, Tuple, Optional

import numpy as np

# FPLLL / G6K
from fpylll import IntegerMatrix, BKZ
from fpylll.algorithms.bkz2 import BKZReduction
from g6k import SieverParams
from g6k.siever import Siever


# ============================================================
# Basic helpers
# ============================================================

def handle_num(a: int, q: int) -> int:
    a = int(a) % q
    if a > q / 2:
        a -= q
    return int(a)


def handle_array(arr: Sequence[int], q: int) -> np.ndarray:
    arr = np.array(arr, dtype=np.int64) % q
    mask = arr > q / 2
    arr[mask] -= q
    return arr.astype(np.int64)


def intmat_to_numpy(mat: IntegerMatrix) -> np.ndarray:
    rows, cols = mat.nrows, mat.ncols
    out = np.zeros((rows, cols), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = int(mat[i, j])
    return out


class CentredBinomial:
    """
    Generate samples from centered binomial distribution B_eta.
    """
    def __init__(self, eta: int = 1):
        self.eta = eta

    def support(self):
        return range(-self.eta, self.eta + 1)

    def PDF(self, outcome: int) -> float:
        return 0.25 ** self.eta * comb(2 * self.eta, outcome + self.eta)

    def __call__(self) -> int:
        return sum(random.randint(0, 1) for _ in range(2 * self.eta)) - self.eta


@dataclass
class InstanceData:
    a: np.ndarray               # shape (N, solve_dim)
    b: np.ndarray               # shape (N,)
    e: np.ndarray               # shape (N,)
    secret_solve: np.ndarray    # shape (solve_dim,)
    q: int
    solve_dim: int
    eta: int


# ============================================================
# Stage 1: Generate new LWE samples
# ============================================================

def generate_LWE_lattice(m: int, n: int, q: int) -> Tuple[IntegerMatrix, IntegerMatrix]:
    """
    Generate a random q-ary lattice basis.
    """
    B = IntegerMatrix.random(m, "qary", k=m - n, q=q)
    A = B.submatrix(0, n, n, m)
    return A, B


def progressive_BKZ(B: IntegerMatrix, beta: int, params: SieverParams, verbose: bool = False):
    g6k = Siever(B, params)
    bkz = BKZReduction(g6k.M)

    for cur_beta in range(2, beta + 1):
        if verbose:
            print(f"\rBKZ_{cur_beta}", end="", flush=True)
        bkz(BKZ.Param(cur_beta, max_loops=2))

    if verbose:
        print()
    print("finish bkz")
    return g6k


def progressive_sieve(g6k, l: int, r: int, verbose: bool = False):
    if verbose:
        print("\rSieving", end="", flush=True)

    g6k.initialize_local(l, max(l, r - 20), r)
    g6k(alg="gauss")

    while g6k.l > l:
        if verbose:
            print(f"\rSieving [{g6k.l:3d}, {g6k.r:3d}]...", end="", flush=True)
        g6k.extend_left()
        g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")

    with g6k.temp_params(saturation_ratio=0.9, db_size_factor=6):
        g6k(alg="hk3")

    g6k.resize_db(ceil((4 / 3) ** ((r - l) / 2)))

    if verbose:
        print()
    return g6k


def change_basis(basis: IntegerMatrix, vector: Sequence[int]):
    """
    Compute vector * basis.
    """
    return basis.multiply_left(vector)


def build_dual_basis(A_lat: np.ndarray, q: int) -> IntegerMatrix:
    """
    Build the dual lattice basis:
        [ I_n   A_lat ]
        [ 0     qI    ]
    where A_lat has shape (n, k_lat).
    """
    n, k_lat = A_lat.shape
    B_dual = IntegerMatrix.identity(n + k_lat)

    for i in range(n, n + k_lat):
        B_dual[i, i] = q

    for i in range(n):
        for j in range(k_lat):
            B_dual[i, n + j] = int(A_lat[i, j] % q)

    return B_dual


def short_vectors_sampling(
    basis: IntegerMatrix,
    threads: int,
    beta_bkz: int,
    beta_sieve: int,
    n: int,
    k_lat: int,
    verbose: bool = False
) -> List[np.ndarray]:
    """
    Sample short vectors from the dual lattice.
    """
    print(f"Using {threads} threads")
    sieve_params = SieverParams(threads=threads, dual_mode=False)

    g6k = progressive_BKZ(basis, beta_bkz, sieve_params, verbose=verbose)
    progressive_sieve(g6k, 0, beta_sieve, verbose=verbose)

    with Pool(threads) as pool:
        database = pool.starmap(
            change_basis,
            [(g6k.M.B, v) for v in g6k.itervalues()]
        )

    return [np.array(w[:n + k_lat], dtype=np.int64) for w in database]


def calculate_average_2norm(vector_list: List[np.ndarray]) -> float:
    if not vector_list:
        return 0.0
    total = 0.0
    for vec in vector_list:
        total += np.linalg.norm(vec)
    return total / len(vector_list)


def inner_vector_worker(args):
    vec, error, s_lat, n, q = args
    x = vec[:n]
    y = vec[n:]
    val = np.inner(x, error) + np.inner(y, s_lat)
    return handle_num(int(val), q)


def cal_new_lwe_worker(args):
    vec, target, error, s_lat, A_solve, n, q = args
    x = vec[:n]
    y = vec[n:]

    new_b = handle_num(int(np.inner(x, target)), q)
    new_a = handle_array(np.dot(x, A_solve), q)
    new_e = handle_num(int(np.inner(x, error) + np.inner(y, s_lat)), q)

    return np.concatenate([
        new_a,
        np.array([new_b], dtype=np.int64),
        np.array([new_e], dtype=np.int64)
    ])


def generate_one_instance(
    solve_dim: int,
    lat_dim: int,
    q: int,
    eta: int,
    beta_bkz: int,
    beta_sieve: int,
    threads: int,
    verbose: bool = False
) -> InstanceData:
    n = solve_dim + lat_dim

    A_block, _ = generate_LWE_lattice(2 * n, n, q)
    A_np = intmat_to_numpy(A_block)   # shape: (n, n)

    # ===== Correct split =====
    # original code:
    # A_solve, A_lat = A[:n-k_lat], A[n-k_lat:]
    # A_solve.transpose()
    # A_lat.transpose()
    #
    # equivalent numpy version:
    A_solve = A_np[:solve_dim, :].T.copy()   # shape: (n, solve_dim)
    A_lat = A_np[solve_dim:, :].T.copy()     # shape: (n, lat_dim)

    print("Generate s, e from the centered binomial distribution")
    secret_dist = CentredBinomial(eta)

    secret = np.array([secret_dist() for _ in range(n)], dtype=np.int64)
    error = np.array([secret_dist() for _ in range(n)], dtype=np.int64)

    s_solve = secret[:solve_dim]
    s_lat = secret[solve_dim:]

    # target = secret * A + error
    target = np.dot(secret, A_np) + error

    print("------------------------------------------------------")
    print("sample short vectors")
    B_dual = build_dual_basis(A_lat, q)

    L = short_vectors_sampling(
        basis=B_dual,
        threads=threads,
        beta_bkz=beta_bkz,
        beta_sieve=beta_sieve,
        n=n,
        k_lat=lat_dim,
        verbose=verbose
    )

    avg_norm = calculate_average_2norm(L)
    print(f"\nAverage Euclidean norm of vectors in L: {avg_norm:.6f}")
    print(f"Database contains {len(L)} dual vectors")

    print("------------------------------------------------------")
    print("calculate new lwe samples")

    with Pool(threads) as pool:
        error_new = pool.map(
            inner_vector_worker,
            [(vec, error, s_lat, n, q) for vec in L]
        )
        new_lwe_sample = pool.map(
            cal_new_lwe_worker,
            [(vec, target, error, s_lat, A_solve, n, q) for vec in L]
        )

    error_new = np.array(error_new, dtype=np.int64)
    new_lwe_sample = np.array(new_lwe_sample, dtype=np.int64)

    print("new_error statistics:")
    print(f"Mean: {np.mean(error_new):.6f}")
    print(f"Standard deviation: {np.std(error_new):.6f}")
    print(f"Maximum and minimum values: {np.max(error_new)} {np.min(error_new)}")

    flag = True
    zero_count = 0
    print("correct solution s_solve:", s_solve.tolist())

    for row in new_lwe_sample:
        new_a = row[:solve_dim]
        new_b = int(row[solve_dim])
        new_e = int(row[-1])

        if new_e == 0:
            zero_count += 1

        lhs = (new_b - int(np.inner(new_a, s_solve))) % q
        rhs = new_e % q
        if lhs != rhs:
            flag = False
            print("Mismatch found!")
            print("new_a =", new_a)
            print("new_b =", new_b)
            print("new_e =", new_e)
            print("lhs   =", lhs)
            print("rhs   =", rhs)
            break

    if not flag:
        raise RuntimeError("Verification of b - a*s = e failed!")

    print("Verification passed")
    print("Number of zeros in new_e:", zero_count)

    return InstanceData(
        a=(new_lwe_sample[:, :solve_dim] % q).astype(np.int64, order="C"),
        b=(new_lwe_sample[:, solve_dim] % q).astype(np.int64, order="C"),
        e=new_lwe_sample[:, -1].astype(np.int64, order="C"),
        secret_solve=(s_solve % q).astype(np.int64, order="C"),
        q=q,
        solve_dim=solve_dim,
        eta=eta
    )



# ============================================================
# Eq. (7): Verify true-secret probability
# ============================================================

def eq7_theory(p0: float, K: int, W: int) -> float:
    """
    Equation (7): probability that the true secret belongs to the solution set.
    """
    return (1.0 - (1.0 - p0) ** K) ** W


def eq7_single_trial(args):
    e_zero_mask, K, W, seed = args
    rng = random.Random(seed)
    N = len(e_zero_mask)

    idx = rng.sample(range(N), K * W)

    for i in range(W):
        group = idx[i * K:(i + 1) * K]
        if not np.any(e_zero_mask[group]):
            return 0
    return 1


def verify_eq7_for_instance(
    instance: InstanceData,
    K: int,
    W: int,
    trials: int = 4000,
    p0_theory: Optional[float] = None,
    seed: int = 42
):
    e_zero_mask = (instance.e == 0)
    N = len(e_zero_mask)
    zero_count = int(np.sum(e_zero_mask))

    if K * W > N:
        raise ValueError(f"K*W={K * W} exceeds total samples N={N}")

    p0_emp = zero_count / N
    p0_used = p0_theory if p0_theory is not None else p0_emp
    theory_val = eq7_theory(p0_used, K, W)

    seeds = [seed + i for i in range(trials)]

    total_score = 0
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(eq7_single_trial, (e_zero_mask, K, W, s))
            for s in seeds
        ]
        for fut in as_completed(futures):
            total_score += fut.result()

    empirical_val = total_score / trials

    return {
        "N": N,
        "zero_count": zero_count,
        "p0_emp": p0_emp,
        "p0_used": p0_used,
        "theory_eq7": theory_val,
        "empirical_eq7": empirical_val
    }


# ============================================================
# Lemma 6: Verify average number of solutions
# ============================================================

def enumerate_candidates(eta: int, solve_dim: int, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate all candidates in [-eta, ..., eta]^solve_dim.
    """
    values = list(range(-eta, eta + 1))
    candidates = list(itertools.product(values, repeat=solve_dim))

    X = np.array(candidates, dtype=np.int64)
    X[X < 0] += q
    X = X.astype(np.int64, order="C")
    return X, X.T.astype(np.int64, order="C")


def lemma6_theory(eta: int, solve_dim: int, q: int, p0: float, K: int, W: int) -> float:
    """
    Expected number of solutions from Lemma 6:
      E[#solutions] = true-secret contribution + wrong-secret contribution
    """
    total_candidates = (2 * eta + 1) ** solve_dim
    wrong_candidates = total_candidates - 1

    true_part = (1.0 - (1.0 - p0) ** K) ** W
    wrong_part = wrong_candidates * (1.0 - (1.0 - 1.0 / q) ** K) ** W
    return true_part + wrong_part


def lemma6_single_experiment(args):
    a_all, b_all, X_T, true_sol_idx, K, W, q, seed = args
    rng = random.Random(seed)

    idx = rng.sample(range(len(a_all)), K * W)
    a_selected = a_all[idx]
    b_selected = b_all[idx]

    valid = np.ones(X_T.shape[1], dtype=bool)

    for i in range(W):
        start, end = i * K, (i + 1) * K
        A_i = a_selected[start:end]
        b_i = b_selected[start:end]

        dot_products = A_i @ X_T
        matches = (dot_products % q) == b_i[:, np.newaxis]
        group_valid = np.any(matches, axis=0)
        valid &= group_valid

        if not np.any(valid):
            break

    solution_count = int(np.sum(valid))
    has_true_sol = bool(valid[true_sol_idx]) if solution_count > 0 else False
    return solution_count, has_true_sol


def verify_lemma6_for_instance(
    instance: InstanceData,
    K: int,
    W: int,
    experiments: int = 100,
    p0_theory: Optional[float] = None,
    seed: int = 42
):
    a_all = instance.a
    b_all = instance.b
    q = instance.q
    solve_dim = instance.solve_dim
    eta = instance.eta

    if K * W > len(a_all):
        raise ValueError(f"K*W={K * W} exceeds total sample count {len(a_all)}")

    X, X_T = enumerate_candidates(eta, solve_dim, q)
    idx_arr = np.where((X == instance.secret_solve).all(axis=1))[0]
    if len(idx_arr) != 1:
        raise RuntimeError("True secret not found in candidate set.")
    true_sol_idx = int(idx_arr[0])

    p0_emp = float(np.mean(instance.e == 0))
    p0_used = p0_theory if p0_theory is not None else p0_emp
    theory_val = lemma6_theory(eta, solve_dim, q, p0_used, K, W)

    task_args = [
        (a_all, b_all, X_T, true_sol_idx, K, W, q, seed + i)
        for i in range(experiments)
    ]

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(lemma6_single_experiment, arg) for arg in task_args]
        for fut in as_completed(futures):
            results.append(fut.result())

    solution_counts = np.array([cnt for cnt, _ in results], dtype=np.int64)
    true_sol_success = sum(1 for _, has_true in results if has_true)

    return {
        "candidate_count": int(X.shape[0]),
        "p0_emp": p0_emp,
        "p0_used": p0_used,
        "theory_lemma6": theory_val,
        "empirical_mean": float(np.mean(solution_counts)),
        "empirical_std": float(np.std(solution_counts)),
        "empirical_min": int(np.min(solution_counts)),
        "empirical_max": int(np.max(solution_counts)),
        "true_solution_hit_rate": true_sol_success / experiments
    }


# ============================================================
# Batch experiment runner
# ============================================================

def run_group_experiment(args):
    threads = min(cpu_count(), args.threads)

    eq7_all = []
    lemma6_all = []

    for inst_id in range(args.instances):
        print("\n" + "=" * 80)
        print(f"Instance {inst_id + 1}/{args.instances}")
        print("=" * 80)

        instance = generate_one_instance(
            solve_dim=args.solve,
            lat_dim=args.lat,
            q=args.q,
            eta=args.eta,
            beta_bkz=args.beta_bkz,
            beta_sieve=args.beta_sieve,
            threads=threads,
            verbose=args.verbose
        )

        eq7_res = verify_eq7_for_instance(
            instance=instance,
            K=args.K,
            W=args.W,
            trials=args.eq7_trials,
            p0_theory=args.p0_theory,
            seed=args.seed + inst_id
        )
        eq7_all.append(eq7_res)

        print("\n[Eq. (7)]")
        print(f"N                 = {eq7_res['N']}")
        print(f"zero_count        = {eq7_res['zero_count']}")
        print(f"p0_emp            = {eq7_res['p0_emp']:.6f}")
        print(f"p0_used           = {eq7_res['p0_used']:.6f}")
        print(f"theory            = {eq7_res['theory_eq7']:.6f}")
        print(f"empirical         = {eq7_res['empirical_eq7']:.6f}")

        lemma6_res = verify_lemma6_for_instance(
            instance=instance,
            K=args.K,
            W=args.W,
            experiments=args.lemma6_trials,
            p0_theory=args.p0_theory,
            seed=args.seed + 10000 + inst_id
        )
        lemma6_all.append(lemma6_res)

        print("\n[Lemma 6]")
        print(f"candidate_count       = {lemma6_res['candidate_count']}")
        print(f"p0_emp                = {lemma6_res['p0_emp']:.6f}")
        print(f"p0_used               = {lemma6_res['p0_used']:.6f}")
        print(f"theory                = {lemma6_res['theory_lemma6']:.6f}")
        print(f"empirical mean        = {lemma6_res['empirical_mean']:.6f}")
        print(f"empirical std         = {lemma6_res['empirical_std']:.6f}")
        print(f"empirical min/max     = {lemma6_res['empirical_min']}/{lemma6_res['empirical_max']}")
        print(f"true secret hit rate  = {100 * lemma6_res['true_solution_hit_rate']:.2f}%")

    print("\n" + "#" * 80)
    print("GROUP SUMMARY")
    print("#" * 80)

    eq7_emp = np.array([x["empirical_eq7"] for x in eq7_all], dtype=float)
    eq7_theory_vals = np.array([x["theory_eq7"] for x in eq7_all], dtype=float)

    lem_emp = np.array([x["empirical_mean"] for x in lemma6_all], dtype=float)
    lem_theory_vals = np.array([x["theory_lemma6"] for x in lemma6_all], dtype=float)
    lem_hit = np.array([x["true_solution_hit_rate"] for x in lemma6_all], dtype=float)

    print("\nEq. (7) summary:")
    print(f"Theory mean/std    : {np.mean(eq7_theory_vals):.6f} / {np.std(eq7_theory_vals):.6f}")
    print(f"Empirical mean/std : {np.mean(eq7_emp):.6f} / {np.std(eq7_emp):.6f}")

    print("\nLemma 6 summary:")
    print(f"Theory mean/std    : {np.mean(lem_theory_vals):.6f} / {np.std(lem_theory_vals):.6f}")
    print(f"Empirical mean/std : {np.mean(lem_emp):.6f} / {np.std(lem_emp):.6f}")
    print(f"Hit-rate mean/std  : {np.mean(lem_hit):.6f} / {np.std(lem_hit):.6f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--instances", type=int, default=20, help="number of random LWE instances")
    parser.add_argument("--solve", type=int, default=10, help="dimension of solving stage")
    parser.add_argument("--lat", type=int, default=60, help="dimension of dual lattice reduction stage")
    parser.add_argument("-q", type=int, default=3329, help="modulus q")
    parser.add_argument("--eta", type=int, default=1, help="eta for centered binomial distribution")
    parser.add_argument("--threads", type=int, default=32, help="max number of threads/processes")
    parser.add_argument("--beta-bkz", type=int, default=40, help="BKZ blocksize")
    parser.add_argument("--beta-sieve", type=int, default=80, help="sieve dimension")

    parser.add_argument("--K", type=int, default=1600, help="number of samples per polynomial")
    parser.add_argument("--W", type=int, default=15, help="number of polynomials")

    parser.add_argument("--eq7-trials", type=int, default=4000, help="number of Monte Carlo trials for Eq. (7)")
    parser.add_argument("--lemma6-trials", type=int, default=100, help="number of systems for Lemma 6")
    parser.add_argument("--p0-theory", type=float, default=0.002769542,
                        help="theoretical p0 from the paper; if omitted, empirical p0 is used")
    parser.add_argument("--seed", type=int, default=45, help="random seed")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.solve <= 0 or args.lat <= 0:
        raise ValueError("solve and lat must be positive")
    if args.K <= 0 or args.W <= 0:
        raise ValueError("K and W must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_group_experiment(args)


if __name__ == "__main__":
    main()
