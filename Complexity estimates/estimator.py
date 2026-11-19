import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, ndtri
from scipy.fft import rfft, irfft, next_fast_len


MODULUS = 3329
SCHEMES = {512: 3, 768: 2, 1024: 2}
LIMITS = {512: 403, 768: 625, 1024: 877}
MULTIPLICATIONS = {512: 2832, 768: 152, 1024: 152}
SAMPLE_RATE = 0.5 * math.log2(4 / 3)
LN2 = math.log(2)
LOG_Q = math.log2(MODULUS)
SIEVE_A = 0.29613500308205365
SIEVE_B = 20.387885985467914
SIEVE_C = 1 / (1 - 2 ** (-SIEVE_A))
RADII = np.array([
    0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1,
    1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96,
    128, 160, 192, 256, 384, 512, 768, 1024
])


def logsum(*values):
    return float(np.logaddexp2.reduce(np.asarray(values, dtype=float)))


@lru_cache(maxsize=4096)
def log_delta(beta):
    return (math.log(beta / (2 * math.pi * math.e)) +
            math.log(math.pi * beta) / beta) / (2 * (beta - 1) * LN2)


def free_dimensions(beta):
    return max(beta * math.log(4 / 3) /
               math.log(beta / (2 * math.pi * math.e)), 0.0)


def automatic_sieve(beta, dimension):
    effective = beta - free_dimensions(beta)
    if beta < dimension:
        effective += math.log2((dimension - beta) * SIEVE_C) / SIEVE_A
    return min(dimension, math.floor(effective))


@lru_cache(maxsize=16384)
def lattice_cost(n, nlat, bkz, sieve):
    dimension = n + nlat
    if not (50 <= bkz <= dimension and 50 <= sieve <= dimension):
        return None
    effective = bkz - free_dimensions(bkz)
    bkz_cost = logsum(3 * math.log2(dimension),
                       2 * math.log2(SIEVE_C) +
                       math.log2(max(dimension - bkz, 1)) +
                       SIEVE_A * effective + SIEVE_B)
    sieve_cost = math.log2(SIEVE_C) + SIEVE_A * sieve + SIEVE_B
    samples = sieve * SAMPLE_RATE
    length = ((nlat / dimension) * LOG_Q +
              0.5 * math.log2(4 / 3) +
              (sieve - 1) * log_delta(sieve) +
              (dimension - sieve) * log_delta(bkz))
    return logsum(bkz_cost, sieve_cost, samples), samples, 2 ** length


def modular_zero(sigma):
    scalar = np.ndim(sigma) == 0
    values = np.atleast_1d(np.asarray(sigma, dtype=float))
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError('Gaussian standard deviations must be finite and positive')
    answer = np.empty_like(values)
    small = values < 3
    if np.any(small):
        current = values[small]
        indices = np.arange(1, 27, dtype=float)
        denominator = 1 + 2 * np.exp(-indices[None, :]**2 / (2 * current[:, None]**2)).sum(axis=1)
        answer[small] = 1 / denominator
    medium = (values >= 3) & (values < MODULUS / math.sqrt(2 * math.pi))
    if np.any(medium):
        current = values[medium]
        series = np.ones_like(current)
        for index in range(1, 7):
            series += 2 * np.exp(-0.5 * (index * MODULUS / current)**2)
        answer[medium] = series / (math.sqrt(2 * math.pi) * current)
    large = ~(small | medium)
    if np.any(large):
        current = values[large]
        series = np.ones_like(current)
        for index in range(1, 7):
            series += 2 * np.exp(-2 * math.pi**2 * (index * current / MODULUS)**2)
        answer[large] = series / MODULUS
    answer = np.clip(answer, 1 / MODULUS, 1)
    return float(answer[0]) if scalar else answer


def cbd_square(eta):
    values = np.zeros(eta**2 + 1)
    for coordinate in range(-eta, eta + 1):
        values[coordinate**2] += math.comb(2 * eta, eta + coordinate) / 4**eta
    return values


@lru_cache(maxsize=64)
def norm_distribution(dimension, eta):
    base = cbd_square(eta)
    size = next_fast_len(dimension * (len(base) - 1) + 1)
    frequencies = rfft(base, n=size)
    probabilities = irfft(frequencies**dimension, n=size)[:dimension * (len(base) - 1) + 1]
    probabilities = np.maximum(probabilities, 0)
    probabilities /= probabilities.sum()
    retained = probabilities > 1e-18
    weights = probabilities[retained]
    return np.flatnonzero(retained).astype(float), weights / weights.sum()


@lru_cache(maxsize=64)
def approximate_norm(dimension, eta, points=128):
    base = cbd_square(eta)
    squares = np.arange(len(base), dtype=float)
    mean = float(base @ squares)
    variance = float(base @ squares**2 - mean**2)
    z = ndtri((np.arange(points) + 0.5) / points)
    norms = np.maximum(0.05, dimension * mean + math.sqrt(dimension * variance) * z)
    return norms, np.full(points, 1 / points)


def logarithmic_failure(standard_deviation, dimension, norms):
    sigma = standard_deviation * np.sqrt(np.maximum(norms, 1e-12) / dimension)
    zeros = np.asarray(modular_zero(sigma))
    zeros = np.where(norms == 0, 1, zeros)
    with np.errstate(divide='ignore'):
        return np.log1p(-zeros)


def log_negative_log_success(log_failure):
    values = np.asarray(log_failure, dtype=float)
    output = np.array(values, copy=True)
    mask = values >= -20
    with np.errstate(divide='ignore', invalid='ignore'):
        output[mask] = np.log(-np.log(-np.expm1(values[mask])))
    return output


class TrueSurvival:
    def __init__(self, n, nlat, eta, length, exact=False):
        dimension = n + nlat
        norms, self.weights = (norm_distribution(dimension, eta) if exact else
                               approximate_norm(dimension, eta))
        self.log_failure = logarithmic_failure(length, dimension, norms)

    def probability(self, log_k, log_w):
        conditional = (2 ** log_k) * self.log_failure
        exponents = log_w * LN2 + log_negative_log_success(conditional)
        return float(np.exp(-np.exp(np.clip(exponents, -745, 100))) @ self.weights)


class FalseSolutions:
    def __init__(self, n, nlat, eta, bkz, sieve, gamma, length, epsilon):
        dimension = n + nlat
        self.sieve = sieve
        self.length = length
        self.epsilon = epsilon
        self.log_candidates = (n - nlat - gamma) * math.log2(2 * eta + 1)
        log_det = (sieve * nlat / dimension * LOG_Q +
                   sieve * (dimension - sieve) * log_delta(bkz))
        self.log_volume_coefficient = (sieve / 2 * math.log2(math.pi) -
                                       gammaln(sieve / 2 + 1) / LN2 +
                                       log_det - sieve * LOG_Q)
        self.log_lambda = np.log(-np.expm1(-RADII * LN2))
        self.p_at_radius = np.asarray(modular_zero(self._radius_sigma(RADII)))
        with np.errstate(divide='ignore'):
            self.log_failure = np.log1p(-self.p_at_radius)

    def _radius_sigma(self, radius_parameter):
        log_b = 2 * (math.log2(self.epsilon) - radius_parameter -
                     self.log_candidates - self.log_volume_coefficient) / self.sieve
        return self.length * np.exp2(log_b / 2) / math.sqrt(self.sieve)

    def _required_log_w(self, log_k, failure, log_lambda):
        conditional = (2 ** log_k) * failure
        log_scale = log_negative_log_success(conditional)
        remaining = self.log_candidates * LN2 - math.log(self.epsilon) - log_lambda
        return np.maximum(0, (np.log(remaining) - log_scale) / LN2)

    def min_w(self, log_k, precise=False):
        options = self._required_log_w(log_k, self.log_failure, self.log_lambda)
        best = int(np.argmin(options))
        if not precise:
            return float(options[best]), float(RADII[best])
        lower = max(1e-9, float(RADII[max(0, best - 1)]))
        upper = float(RADII[min(len(RADII) - 1, best + 1)])
        if lower >= upper:
            return float(options[best]), float(RADII[best])
        result = minimize_scalar(lambda t: self.at_radius(log_k, t)[0],
                                 bounds=(lower, upper), method='bounded',
                                 options={'xatol': 1e-8})
        candidate = self.at_radius(log_k, float(result.x))
        return min((float(options[best]), float(RADII[best])), candidate)

    def at_radius(self, log_k, t):
        p_zero = modular_zero(float(self._radius_sigma(t)))
        log_lambda = math.log(-math.expm1(-t * LN2))
        required = self._required_log_w(log_k, np.array([(-math.inf if p_zero >= 1 else math.log1p(-p_zero))]),
                                        np.array([log_lambda]))
        return float(required[0]), t

    def expected_wrong_log(self, log_k, log_w, radius):
        p_zero = modular_zero(float(self._radius_sigma(radius)))
        conditional = (2 ** log_k) * (-math.inf if p_zero >= 1 else math.log1p(-p_zero))
        log_scale = float(log_negative_log_success(np.array([conditional]))[0])
        exponent = log_w * LN2 + log_scale
        log_false = (self.log_candidates * LN2 -
                     math.exp(exponent) if exponent < 700 else -math.inf)
        log_volume = math.log(self.epsilon) - radius * LN2
        return float(np.logaddexp(log_volume, log_false))


def find_parameters(model, false_bound, budget, theta, k_min=0, k_max=24,
                    precise=False):
    low = max(0.0, k_min)
    high = k_max

    def minimum(x, refined=False):
        return false_bound.min_w(x, refined)

    def locate(refined, start, stop, points):
        previous_x = None
        previous_p = None
        for x in np.linspace(start, stop, points):
            x = float(x)
            w = minimum(x, refined)[0]
            p = model.probability(x, w)
            if p >= theta:
                if previous_x is None or previous_p >= theta:
                    return x
                return brentq(lambda k: model.probability(k, minimum(k, refined)[0]) - theta,
                              previous_x, x, xtol=1e-10)
            previous_x, previous_p = x, p
        return None

    if high <= low or low + minimum(low)[0] >= budget:
        return None
    if high + minimum(high)[0] >= budget:
        high = brentq(lambda x: x + minimum(x)[0] - budget, low, high, xtol=1e-9) - 1e-8
    if high <= low:
        return None

    first = locate(False, low, high, 32)
    if precise:
        if first is None:
            first = locate(True, low, high, 32)
        else:
            lower = max(low, first - 0.5)
            while lower > low and model.probability(lower, minimum(lower, True)[0]) >= theta:
                lower = max(low, lower - 0.5)
            if model.probability(lower, minimum(lower, True)[0]) >= theta:
                first = lower
            else:
                right = first
                while right < high and model.probability(right, minimum(right, True)[0]) < theta:
                    right = min(high, right + 1e-4)
                if model.probability(right, minimum(right, True)[0]) >= theta:
                    first = brentq(lambda k: model.probability(k, minimum(k, True)[0]) - theta,
                                   lower, right, xtol=1e-10)
                else:
                    first = None
    if first is None:
        return None

    k = math.ceil(2 ** first)
    for _ in range(128):
        log_k = math.log2(k)
        log_w, radius = minimum(log_k, precise)
        w = math.ceil(math.nextafter(2 ** log_w, math.inf))
        log_w = math.log2(max(1, w))
        if log_k + log_w >= budget:
            return None
        probability = model.probability(log_k, log_w)
        if probability + 1e-12 >= theta:
            if false_bound.expected_wrong_log(log_k, log_w, radius) > math.log(false_bound.epsilon) + 1e-9:
                return None
            return log_k, log_w, probability, radius, k, w
        k += 1
    return None


def operation_costs(n, eta, nlat, gamma, log_k, log_w, stage1):
    variables = n - nlat - gamma
    degree = 2 * eta + 1
    base = 2 * degree - 1
    log_v = (variables - 1) * math.log2(base)
    log_poly_multiply = log_v + math.log2(log_v) + math.log2(math.log2(log_v))
    log_reduction = variables * math.log2(base)
    log_resultant = (log_w + math.log2(MULTIPLICATIONS[n]) +
                     logsum(log_poly_multiply, log_reduction))
    log_generation = log_k + log_w + (variables + 1) * math.log2(degree + 1)
    field_cost = 2 * math.log2(math.ceil(LOG_Q))
    classical = field_cost + logsum(log_resultant, log_generation)
    quantum_query = (log_k + log_w + variables * math.log2(degree)) / 2
    quantum_generation = log_k + log_w + math.log2(variables)
    quantum = logsum(quantum_query, quantum_generation)
    return {
        'stage1_bits': stage1,
        'classical_stage2_bits': classical,
        'classical_total_bits': logsum(stage1, classical),
        'quantum_stage2_bits': quantum,
        'quantum_total_bits': logsum(stage1, quantum),
        'generation_bits': log_generation,
        'resultant_bits': log_resultant,
        'quantum_query_bits': quantum_query
    }


def direct_resultant_cost(n):
    eta = SCHEMES[n]
    base = 4 * eta + 1
    log_v = (n - 1) * math.log2(base)
    log_poly = log_v + math.log2(log_v) + math.log2(math.log2(log_v))
    log_terms = logsum(log_poly, n * math.log2(base))
    return 2 * math.log2(math.ceil(LOG_Q)) + math.log2(n) + math.log2(MULTIPLICATIONS[n]) + log_terms


def evaluate(n, mode, nlat, bkz, sieve, gamma, theta, epsilon,
             exact=False, k_min=0, k_max=24, prepared=None):
    if (mode == 'table2' and (bkz >= LIMITS[n] or sieve >= LIMITS[n])) or n - nlat - gamma < 2:
        return None
    lattice = prepared[0] if prepared is not None else lattice_cost(n, nlat, bkz, sieve)
    if lattice is None:
        return None
    stage1, log_samples, length = lattice
    bound = FalseSolutions(n, nlat, SCHEMES[n], bkz, sieve, gamma, length, epsilon)
    model = prepared[1] if prepared is not None else TrueSurvival(n, nlat, SCHEMES[n], length, exact=exact)
    budget = log_samples - gamma * LOG_Q
    solution = find_parameters(model, bound, budget, theta, k_min, k_max, exact)
    if solution is None:
        return None
    log_k, log_w, survival, radius, k, w = solution
    values = operation_costs(n, SCHEMES[n], nlat, gamma, log_k, log_w, stage1)
    target = 'quantum_total_bits' if mode == 'quantum' else 'classical_total_bits'
    return {
        'mode': mode, 'scheme': f'KYBER-{n}', 'nlat': nlat, 'bkz': bkz,
        'sieve': sieve, 'gamma': gamma, 'variables': n - nlat - gamma,
        'log2_k': log_k, 'log2_w': log_w, 'p_true_target': theta, 'p_true': survival,
        'p_false_bound': math.exp(bound.expected_wrong_log(log_k, log_w, radius)),
        'log2_samples': log_samples, 'sample_margin_bits': budget - log_k - log_w,
        'radius_parameter': radius, 'total_bits': values[target], **values
    }

