import os

# -------------------------- 必须放在导入Numpy之前！ --------------------------
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
# ---------------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed
import itertools
import random

# -------------------------- Global Parameter Configuration --------------------------
MOD = 3329
FILE_PATH = "new_lwe_sample_n50_nlat40.txt"  # Data file path
K = 700  # Number of samples per equation
W = 8  # Number of equations in the system (must satisfy K*W ≤ total samples)
NUM_EXPERIMENTS = 100  # Total number of experiments
TRUE_SOLUTION = np.array(    [-1, 0, 0, -1, -1, 0, 1, 0, 1, 1], dtype=np.int64) % MOD  # true solution

# -----------------------------------------------------------------

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', dtype=np.int64)
    a = data[:, :10] % MOD
    b = data[:, 10] % MOD
    return a.astype(np.int64, order='C'), b.astype(np.int64, order='C')

def generate_candidates(true_sol):
    candidates = list(itertools.product([-1, 0, 1], repeat=10))

    X = np.array(candidates, dtype=np.int64)
    X[X == -1] = MOD - 1
    X = X.astype(np.int64, order='C')

    true_sol_idx = np.where((X == true_sol).all(axis=1))[0][0]
    print(f"Index of correct solution in candidate list：{true_sol_idx}")
    return X, X.T, true_sol_idx


def single_experiment(args):
    a_all, b_all, X_T, true_sol_idx, K, W, MOD = args
    idx = random.sample(range(len(a_all)), K * W)
    a_selected = a_all[idx]
    b_selected = b_all[idx]

    valid = np.ones(X_T.shape[1], dtype=bool)
    for i in range(W):
        start, end = i * K, (i + 1) * K
        A_i = a_selected[start:end]
        b_i = b_selected[start:end]
        dot_products = A_i @ X_T
        matches = (dot_products % MOD) == b_i[:, np.newaxis]
        group_valid = np.any(matches, axis=0)
        valid &= group_valid

        # Early termination: return immediately if no valid solution
        if not np.any(valid):
            break

    solution_count = np.sum(valid)
    has_true_sol = valid[true_sol_idx] if solution_count > 0 else False
    return solution_count, has_true_sol





def main():
    a_all, b_all = load_data(FILE_PATH)
    if K * W > len(a_all):
        print(f"Error: K*W={K * W} exceeds total sample count {len(a_all)}, please adjust parameters")
        return
    X, X_T, true_sol_idx = generate_candidates(TRUE_SOLUTION)
    print(f"Successfully loaded {len(a_all)} samples")

    # Construct parameters for parallel tasks
    task_args = [
        (a_all, b_all, X_T, true_sol_idx, K, W, MOD)
        for _ in range(NUM_EXPERIMENTS)
    ]

    print(f"\nStarting {NUM_EXPERIMENTS} parallel experiments...")
    results = Parallel(
        n_jobs=-1,
        verbose=10,
        backend='loky'
    )(delayed(single_experiment)(arg) for arg in task_args)

    solution_counts = []
    true_sol_success = 0
    for cnt, has_true in results:
        solution_counts.append(cnt)
        true_sol_success += 1 if has_true else 0

    # Output statistical report
    solution_counts = np.array(solution_counts)
    print("\n==================== Experiment Statistics Report (Fully Aligned with Serial Logic) ====================")
    print(f"Total number of experiments: {NUM_EXPERIMENTS}")
    print(
        f"Number of times containing the correct solution: {true_sol_success} (proportion: {true_sol_success / NUM_EXPERIMENTS * 100:.2f}%)")
    print(f"Solution count statistics:")
    print(f"  Average: {np.mean(solution_counts):.2f}")
    print(f"  Standard deviation: {np.std(solution_counts):.2f}")
    print(f"  Min/Max: {np.min(solution_counts)}/{np.max(solution_counts)}")
    # 保存结果
    result_array = np.column_stack([solution_counts, [1 if h else 0 for _, h in results]])
    np.savetxt(
        "the_number_of_solutions.txt",
        result_array,
        fmt="%d %d",
        header="the_number_of_solutions",
        comments=""
    )
    print("\nResults have been saved to the number of solutions.txt")


if __name__ == "__main__":
    main()