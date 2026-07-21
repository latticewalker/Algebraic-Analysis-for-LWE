import time

import complexity_estimator as C


def sweep(name, cfg, *, klat_lo, klat_hi, klat_step,
          beta_lo, beta_hi, beta_step,
          y_max, ki_lo, ki_hi, ki_step=5, verbose=False):
    """Sweep one parameter set and print the best complexity found."""
    C.BETA_MIN, C.BETA_MAX, C.BETA_STEP = beta_lo, beta_hi, beta_step
    C.Y_MIN, C.Y_MAX = 0, y_max
    C.K_I_MIN, C.K_I_MAX, C.K_I_STEP = ki_lo, ki_hi, ki_step

    n = int(cfg["n"])
    best = None
    start = time.time()

    lower = max(0, klat_lo)
    upper = min(klat_hi, n - 1)
    for k_lat in range(lower, upper + 1, klat_step):
        result = C.eval_k_lat(k_lat, cfg)
        if result is not None and (best is None or result["obj"] < best["obj"]):
            best = result
            if verbose:
                print(
                    f"    k_lat={k_lat}: total={result['obj']:.2f} "
                    f"com1={result['com1']:.1f} "
                    f"com2={result['com2']:.1f} "
                    f"beta_bkz={result['beta_bkz']} y={result['y']}"
                )

    elapsed = time.time() - start
    if best is None:
        print(f"[{name}] no feasible point ({elapsed:.0f}s)")
        return None

    print(
        f"[{name}] total={best['obj']:.1f} "
        f"(com1={best['com1']:.1f}, com2={best['com2']:.1f}) "
        f"k_lat={best['k_lat']} n_solve={best['n_solve']} "
        f"y={best['y']} beta_bkz={best['beta_bkz']} "
        f"beta_sieve={best['beta_sieve']} "
        f"log2K={best['log2K']:.1f} "
        f"log2W={best['log2W']:.1f} "
        f"p_true={best['p_true']:.3f} ({elapsed:.0f}s)"
    )
    return best

