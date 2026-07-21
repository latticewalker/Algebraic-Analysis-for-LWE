"""MATZOV cost model for BKZ reduction and sieving."""

import math


MATZOV_NN_AGPS = {
    "list_decoding-classical": {"a": 0.29613500308205365, "b": 20.387885985467914}
}

_A = MATZOV_NN_AGPS["list_decoding-classical"]["a"]
_B = MATZOV_NN_AGPS["list_decoding-classical"]["b"]
_C = 1.0 / (1.0 - 2.0 ** (-_A))  # asymptotic overhead factor


def d4f(beta):
    if beta <= 0:
        return 0.0
    log_term = math.log(4 / 3.0) / math.log(beta / (2 * math.pi * math.e))
    return max(float(beta * log_term), 0.0)


def lll_cost(d):
    return d ** 3


def matzov_bkz_cost(beta, d):
    beta_prime = beta - d4f(beta)
    svp_calls = _C * max(d - beta, 1)
    gate_count = _C * (2 ** (_A * beta_prime + _B))
    return lll_cost(d) + svp_calls * gate_count


def matzov_short_vectors(beta, d):
    beta_prime = beta - d4f(beta)

    beta_sieve = beta_prime
    if beta < d:
        log_term = math.log((d - beta) * _C, 2) / _A
        beta_sieve = min(d, math.floor(beta_prime + log_term))

    n_vectors = math.floor(2 ** (0.2075 * beta_sieve))
    sieve_cost = _C * (2 ** (_A * beta_sieve + _B))
    total_cost = matzov_bkz_cost(beta, d) + sieve_cost
    return total_cost, n_vectors, beta_sieve

