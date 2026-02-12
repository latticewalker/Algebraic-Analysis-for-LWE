import random
from typing import Optional
import numpy as np
from math import sqrt
def _single_trial(
        k: int,
        w: int,
        total_num: int,
        zero_count: int,
        seed: Optional[int] = None
) -> int:

    zero_remaining = zero_count
    non_zero_remaining = total_num - zero_count

    for _ in range(w):
        remaining_total = zero_remaining + non_zero_remaining
        sample_indices = random.sample(range(remaining_total), k)
        has_zero = any(idx < zero_remaining for idx in sample_indices)

        if not has_zero:
            return 0

        count_zero = sum(1 for idx in sample_indices if idx < zero_remaining)
        zero_remaining -= count_zero
        non_zero_remaining -= (k - count_zero)

    return 1  # All w trials contain at least one zero → success

from concurrent.futures import ProcessPoolExecutor, as_completed
import random

def calculate_score_parallel(
        k: int,
        w: int,
        total_num: int = 99438,
        zero_count: int = 373,
        trials: int = 4000,
        random_seed: Optional[int] = None
) -> int:

    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(w, int) or w < 1:
        raise ValueError("w must be a positive integer")
    if w * k > total_num:
        raise ValueError(f"w*k ({w * k}) exceeds total pool size {total_num}")
    trial_seeds = None
    if random_seed is not None:
        rng = random.Random(random_seed)
        trial_seeds = [rng.randint(0, 10**18) for _ in range(trials)]
    else:
        trial_seeds = [None] * trials

    total_score = 0
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                _single_trial,
                k, w, total_num, zero_count, trial_seeds[i]
            ) for i in range(trials)
        ]
        for future in as_completed(futures):
            total_score += future.result()

    return total_score


if __name__ == "__main__":
    # ====== Simulation Setup ======
    k = 700
    w = 8
    C = []
    a1 = [525, 528, 591, 503, 608, 490, 546, 471, 509, 499, 532, 503, 528, 572, 493, 565, 549, 574, 501, 489] #Data from the 20 randomly generated LWE instances.

    print("\n[Experiment 1] Configuration: k=700, w=8")
    print("Total samples N = 99438, trials per zero_count = 4000")
    print(f"Zero counts to test (length = {len(a1)}): {a1}\n")
    for i in range(len(a1)):
        c = calculate_score_parallel(k, w, total_num=99438, zero_count=a1[i], trials=4000, random_seed=42)
        c = c / 4000
        C.append(c)
    print(f"\n[Result] Group 2: mean success rate = {np.mean(C):.4f}, std = {np.std(C):.4f}\n")

    k, w = 1000, 9
    S = []
    a2 = [392, 343, 375, 340, 352, 365, 405, 332, 385, 350, 373, 351, 364, 352, 358, 365, 372, 383, 379,375] #Data from the 20 randomly generated LWE instances.
    print("\n[Experiment 2] Configuration: k=1000, w=9")
    print("Total samples N = 99438, trials per zero_count = 4000")
    print(f"Zero counts to test (length = {len(a2)}): {a2}\n")

    for i in range(len(a2)):
        s = calculate_score_parallel(k, w, total_num=99438, zero_count=a2[i], trials=4000, random_seed=42)
        s=s/4000
        S.append(s)
    print(f"\n[Result] Group 2: mean success rate = {np.mean(S):.4f}, std = {np.std(S):.4f}\n")

