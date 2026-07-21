"""Run the NTRU hps/hrss HES and two binary-secret TFHE parameter sets."""
from driver import sweep

TERNARY = (-1, 0, 1)

# HES: the uniform ternary distribution of the mathematical model.
HES_TERNARY_PROBS = (1 / 3, 1 / 3, 1 / 3)
HES_ERROR_SIGMA = 3.19

NTRU_TERNARY_PROBS = (85 / 256, 86 / 256, 85 / 256)


def hes_config(name, n, q):
    return {
        "scheme": name,
        "norm_model": "hes",
        "n": n,
        "m": n,
        "q": q,
        "error_sigma": HES_ERROR_SIGMA,
        "secret_values": TERNARY,
        "secret_probs": HES_TERNARY_PROBS,
    }


def ntru_hps_config(name, ring_n, q):
    active_n = ring_n - 1
    return {
        "scheme": name,
        "norm_model": "ntru_hps",
        "ring_n": ring_n,
        "n": active_n,
        "m": active_n,
        "q": q,
        "w_g": q // 8 - 2,
        "secret_values": TERNARY,
        "secret_probs": NTRU_TERNARY_PROBS,
    }


def ntru_hrss_config(name, ring_n, q):
    active_n = ring_n - 1
    return {
        "scheme": name,
        "norm_model": "ntru_hrss",
        "ring_n": ring_n,
        "n": active_n,
        "m": active_n,
        "q": q,
        "secret_values": TERNARY,
        "secret_probs": NTRU_TERNARY_PROBS,
    }


def binary_gaussian_config(name, n, q, error_sigma):
    return {
        "scheme": name,
        "norm_model": "binary_gaussian",
        "n": n,
        "m": n,
        "q": q,
        "error_sigma": error_sigma,
        "secret_values": (0, 1),
        "secret_probs": (0.5, 0.5),
    }


def run_binary_tfhe():
    print("\n" + "=" * 78)
    print("Binary-secret TFHE parameter sets")
    print("=" * 78)

    sweep(
        "Concrete-TFHE512",
        binary_gaussian_config(
            "Concrete-TFHE512", 512, 2**32, 2 ** (-24.8) * 2**32
        ),
        klat_lo=450, klat_hi=490, klat_step=1,
        beta_lo=20, beta_hi=2 * 512 - 1, beta_step=1,
        y_max=3, ki_lo=3300, ki_hi=3800, ki_step=2,
    )

    sweep(
        "TFHE16-1024",
        binary_gaussian_config(
            "TFHE16-1024", 1024, 2**32, 3.73 * 10 ** (-9) * 2**32
        ),
        klat_lo=940, klat_hi=980, klat_step=1,
        beta_lo=20, beta_hi=2 * 1024 - 1, beta_step=1,
        y_max=3, ki_lo=3400, ki_hi=3900, ki_step=2,
    )


def main():
    print("=" * 78)
    print("HES: uniform ternary secret + discrete Gaussian error")
    print("=" * 78)

    sweep(
        "HES n=1024 q=2^27",
        hes_config("HES n=1024 q=2^27", 1024, 2**27),
        klat_lo=960, klat_hi=1023, klat_step=2,
        beta_lo=330, beta_hi=520, beta_step=5,
        y_max=3, ki_lo=800, ki_hi=3400,
    )
    sweep(
        "HES n=2048 q=2^54",
        hes_config("HES n=2048 q=2^54", 2048, 2**54),
        klat_lo=1985, klat_hi=2047, klat_step=2,
        beta_lo=330, beta_hi=480, beta_step=5,
        y_max=2, ki_lo=3000, ki_hi=6200,
    )
    sweep(
        "HES n=4096 q=2^109",
        hes_config("HES n=4096 q=2^109", 4096, 2**109),
        klat_lo=4010, klat_hi=4095, klat_step=2,
        beta_lo=330, beta_hi=520, beta_step=10,
        y_max=2, ki_lo=6000, ki_hi=11200,
    )

    print("\n" + "=" * 78)
    print("NTRU-HPS: Ternary f + Fixed_Type g")
    print("=" * 78)

    for name, ring_n, q, klo, khi, blo, bhi in (
        ("ntruhps2048509", 509, 2048, 440, 507, 480, 660),
        ("ntruhps2048677", 677, 2048, 570, 675, 700, 880),
        ("ntruhps4096821", 821, 4096, 700, 760, 820, 980),
    ):
        cfg = ntru_hps_config(name, ring_n, q)
        sweep(
            name,
            cfg,
            klat_lo=klo, klat_hi=khi, klat_step=2,
            beta_lo=blo, beta_hi=bhi, beta_step=10,
            y_max=6, ki_lo=600, ki_hi=3200,
        )

    print("\n" + "=" * 78)
    print("NTRU-HRSS: Ternary_Plus f, g0 in M_{h, 3*Phi_1}")
    print("=" * 78)

    sweep(
        "ntruhrss701",
        ntru_hrss_config("ntruhrss701", 701, 8192),
        klat_lo=600, klat_hi=699, klat_step=2,
        beta_lo=560, beta_hi=760, beta_step=10,
        y_max=6, ki_lo=700, ki_hi=3200,
    )

    run_binary_tfhe()


if __name__ == "__main__":
    main()
