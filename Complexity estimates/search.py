import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from estimator import (
    LIMITS, LOG_Q, SAMPLE_RATE, SCHEMES, TrueSurvival, automatic_sieve,
    direct_resultant_cost, evaluate, lattice_cost,
    modular_zero, norm_distribution, operation_costs
)


PROFILES = {
    'quick': {'nlat': 18, 'bkz': 14, 'gamma': 6, 'basins': (6, 4, 1), 'sweeps': 3, 'radius': (2, 4, 4, 1)},
    'balanced': {'nlat': 32, 'bkz': 20, 'gamma': 9, 'basins': (10, 6, 2), 'sweeps': 6, 'radius': (3, 6, 6, 2)},
    'thorough': {'nlat': 48, 'bkz': 30, 'gamma': 13, 'basins': (16, 10, 5), 'sweeps': 12, 'radius': (4, 9, 9, 2)}
}



def threshold(mode, override):
    return override if override is not None else 0.8


def admissible(n, mode, parameters):
    nlat, bkz, sieve, gamma = parameters
    dimension = n + nlat
    upper = min(dimension, LIMITS[n] - 1) if mode == 'table2' else dimension
    return (1 <= nlat <= n - 2 and 50 <= bkz <= upper and
            50 <= sieve <= upper and 0 <= gamma <= n - nlat - 2 and
            sieve * SAMPLE_RATE - gamma * LOG_Q > 0)


def limits(n, mode, nlat):
    dimension = n + nlat
    return min(dimension, LIMITS[n] - 1) if mode == 'table2' else dimension


def integer_grid(lower, upper, count):
    return sorted(set(int(round(x)) for x in np.linspace(lower, upper, count)))


def coarse_sieves(n, mode, nlat, bkz):
    upper = limits(n, mode, nlat)
    center = automatic_sieve(bkz, n + nlat)
    interval = max(5, upper // 24)
    values = {50, upper, center}
    values.update(integer_grid(50, upper, 6))
    values.update(center + interval * offset for offset in (-2, -1, 1, 2))
    return sorted(x for x in values if 50 <= x <= upper)


def coarse_gammas(n, nlat, sieve, count):
    cap = min(n - nlat - 2, int(math.ceil((sieve * SAMPLE_RATE) / LOG_Q)) - 1)
    if cap < 0:
        return []
    return sorted(set(integer_grid(0, cap, count) + list(range(min(cap, 3) + 1))))


def point_key(row):
    return row['nlat'], row['bkz'], row['sieve'], row['gamma']


def score_bound(n, mode, parameters, lattice=None):
    nlat, bkz, sieve, gamma = parameters
    stage1 = lattice[0] if lattice is not None else lattice_cost(n, nlat, bkz, sieve)[0]
    lower = operation_costs(n, SCHEMES[n], nlat, gamma, 0, 0, stage1)
    return lower['quantum_total_bits' if mode == 'quantum' else 'classical_total_bits']


def collect_jobs(n, mode, points, ceiling):
    buckets = defaultdict(lambda: defaultdict(set))
    tested = set()
    lattice_cache = {}
    for parameters in points:
        if parameters in tested:
            continue
        tested.add(parameters)
        if not admissible(n, mode, parameters):
            continue
        nlat, bkz, sieve, gamma = parameters
        key = nlat, bkz, sieve
        if key not in lattice_cache:
            lattice_cache[key] = lattice_cost(n, nlat, bkz, sieve)
        lattice = lattice_cache[key]
        if lattice is None or score_bound(n, mode, parameters, lattice) > ceiling + 1e-9:
            continue
        buckets[(nlat, bkz)][sieve].add(gamma)
    return [(pair, [(sieve, tuple(sorted(gammas))) for sieve, gammas in sorted(buckets[pair].items())])
            for pair in sorted(buckets)]


def initial_points(n, mode):
    fractions = {
        'table2': (0.52, 0.60, 0.66, 0.73, 0.80),
        'table3': (0.75, 0.82, 0.88, 0.92, 0.96),
        'quantum': (0.60, 0.68, 0.76, 0.84, 0.90)
    }[mode]
    for fraction in fractions:
        nlat = max(1, min(n - 2, round(fraction * n)))
        upper = limits(n, mode, nlat)
        bkz_fractions = (0.84, 0.94, 1.0) if mode == 'table2' else (0.55, 0.62, 0.68, 0.74, 0.80)
        for factor in bkz_fractions:
            bkz = min(upper, max(50, round((upper if mode == 'table2' else n + nlat) * factor)))
            sieves = {
                min(upper, automatic_sieve(bkz, n + nlat)),
                min(upper, bkz),
                min(upper, max(50, bkz - 24))
            }
            for sieve in sorted(sieves):
                for gamma in (0, 2, 4, 6, 8, 10, 12, 16):
                    point = (nlat, bkz, sieve, gamma)
                    if admissible(n, mode, point):
                        yield point


def coarse_points(n, mode, profile):
    for nlat in integer_grid(1, n - 2, profile['nlat']):
        upper = limits(n, mode, nlat)
        for bkz in integer_grid(50, upper, profile['bkz']):
            for sieve in coarse_sieves(n, mode, nlat, bkz):
                for gamma in coarse_gammas(n, nlat, sieve, profile['gamma']):
                    yield nlat, bkz, sieve, gamma


def offsets(radius, dense=False):
    if dense:
        return range(-radius, radius + 1)
    return sorted({-radius, -max(1, radius // 2), 0, max(1, radius // 2), radius})


def neighboring_points(seeds, radii, dense=False):
    a, b, c, d = [tuple(offsets(radius, dense)) for radius in radii]
    for seed in seeds:
        nlat, bkz, sieve, gamma = point_key(seed)
        for x in a:
            for y in b:
                for z in c:
                    for g in d:
                        yield nlat + x, bkz + y, sieve + z, gamma + g


def best_rows(rows, number, n):
    unique = {}
    for row in rows:
        key = point_key(row)
        if key not in unique or row['total_bits'] < unique[key]['total_bits']:
            unique[key] = row
    ordered = sorted(unique.values(), key=lambda row: (row['total_bits'], point_key(row)))
    if not ordered:
        return []
    selected = []
    scale = (max(2, n // 80), max(3, n // 45), max(3, n // 45), 2)
    for row in ordered:
        if all(max(abs(x - y) / s for x, y, s in zip(point_key(row), point_key(chosen), scale)) >= 1.0
               for chosen in selected):
            selected.append(row)
            if len(selected) == number:
                break
    return selected


def candidate_job(task):
    job_id, mode, n, pair, sieve_groups, theta, epsilon, k_min, k_max, precise = task
    nlat, bkz = pair
    results = []
    for sieve, gammas in sieve_groups:
        lattice = lattice_cost(n, nlat, bkz, sieve)
        if lattice is None:
            continue
        model = TrueSurvival(n, nlat, SCHEMES[n], lattice[2], exact=precise)
        for gamma in gammas:
            row = evaluate(n, mode, nlat, bkz, sieve, gamma, theta, epsilon,
                           exact=precise, k_min=k_min, k_max=k_max,
                           prepared=(lattice, model))
            if row is not None:
                results.append(row)
    return job_id, sorted(results, key=lambda row: (row['total_bits'], point_key(row)))[:12]


def exact_job(task):
    mode, n, parameters, theta, epsilon, k_min, k_max = task
    return evaluate(n, mode, *parameters, theta, epsilon, exact=True, k_min=k_min, k_max=k_max)


def checkpoint_signature(mode, n, args):
    settings = (mode, n, 'adaptive-v2-automatic-seeding', args.profile, args.theta, args.epsilon_sol,
                args.k_min, args.k_max, args.sieve_step)
    return hashlib.sha256(json.dumps(settings).encode()).hexdigest()


def load_checkpoint(path, signature, resume):
    if not resume or not path.exists():
        return {}, False
    records = {}
    with path.open('r+b') as handle:
        header = handle.readline()
        if json.loads(header).get('signature') != signature:
            raise ValueError('Checkpoint settings differ from this search')
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                handle.truncate(offset)
                break
            records[(record['stage'], record['index'])] = record['rows']
            if not line.endswith(b'\n'):
                handle.write(b'\n')
    return records, True


def run_stage(pool, path, records, stage, jobs, mode, n, args):
    existing = {idx: rows for (name, idx), rows in records.items() if name == stage}
    pending = [
        (idx, mode, n, pair, groups, threshold(mode, args.theta),
         args.epsilon_sol, args.k_min, args.k_max, False)
        for idx, (pair, groups) in enumerate(jobs) if idx not in existing
    ]
    print(f'{mode} KYBER-{n} {stage}: {len(jobs)} jobs, {len(pending)} remaining', flush=True)
    if pending:
        iterator = map(candidate_job, pending) if pool is None else pool.imap_unordered(candidate_job, pending, chunksize=1)
        handle = path.open('a', encoding='utf-8') if path is not None else None
        try:
            for completed, (idx, rows) in enumerate(iterator, 1):
                records[(stage, idx)] = rows
                if handle is not None:
                    handle.write(json.dumps({'stage': stage, 'index': idx, 'rows': rows}) + '\n')
                if completed % 50 == 0 or completed == len(pending):
                    if handle is not None:
                        handle.flush()
                    print(f'{mode} KYBER-{n} {stage}: {completed}/{len(pending)}', flush=True)
        finally:
            if handle is not None:
                handle.close()
    return [row for idx in range(len(jobs)) for row in records[(stage, idx)]]


def auto_search(pool, mode, n, args):
    checkpoint = None
    records = {}
    if args.checkpoint_dir is not None:
        directory = Path(args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / f'checkpoint_{mode}_{n}.jsonl'
        signature = checkpoint_signature(mode, n, args)
        records, resumed = load_checkpoint(checkpoint, signature, args.resume)
        if not resumed:
            checkpoint.write_text(json.dumps({'signature': signature}) + '\n', encoding='utf-8')

    profile = PROFILES[args.profile]
    starting = []
    for point in dict.fromkeys(initial_points(n, mode)):
        candidate = evaluate(n, mode, *point, threshold(mode, args.theta),
                             args.epsilon_sol, k_min=args.k_min, k_max=args.k_max)
        if candidate is not None:
            starting.append(candidate)
    if not starting:
        raise ValueError(f'No feasible initial parameters for {mode} KYBER-{n}')

    history = best_rows(starting, max(6, profile['basins'][0]), n)
    print(f'{mode} KYBER-{n} initial best: {min(row["total_bits"] for row in history):.4f} bits', flush=True)

    def step(name, points, allowance):
        ceiling = min(row['total_bits'] for row in history) + allowance
        jobs = collect_jobs(n, mode, points, ceiling)
        rows = run_stage(pool, checkpoint, records, name, jobs, mode, n, args)
        history.extend(rows)
        best = min(history, key=lambda row: (row['total_bits'], point_key(row)))
        print(f'{mode} KYBER-{n} {name} best: {best["total_bits"]:.4f} bits', flush=True)

    step('coarse', coarse_points(n, mode, profile), 8)
    first = best_rows(history, profile['basins'][0], n)
    step('zoom', neighboring_points(first,
         (max(3, n // 32), max(6, n // 18), max(6, n // 18), 2)), 5)
    second = best_rows(history, profile['basins'][1], n)
    step('refine', neighboring_points(second,
         (max(3, n // 90), max(5, n // 65), max(5, n // 65), 2)), 3)
    third = best_rows(history, profile['basins'][2], n)
    step('fine', neighboring_points(third, profile['radius'], True), 2)

    selected = best_rows(history, profile['sweeps'], n)
    def sweep():
        for row in selected:
            nlat, bkz, _, gamma = point_key(row)
            for sieve in range(50, limits(n, mode, nlat) + 1, args.sieve_step):
                yield nlat, bkz, sieve, gamma
    step('sieve', sweep(), 2)
    selected = best_rows(history, min(3, profile['basins'][2]), n)
    step('polish', neighboring_points(selected, (2, 3, 3, 1), True), 1)

    unique = {}
    for row in history:
        key = point_key(row)
        if key not in unique or row['total_bits'] < unique[key]['total_bits']:
            unique[key] = row
    top = sorted(unique.values(), key=lambda row: (row['total_bits'], point_key(row)))
    selected = top[:args.refine]
    selected += best_rows(top, min(args.refine // 4, 16), n)
    selected.append(min(starting, key=lambda row: (row['total_bits'], point_key(row))))
    points = sorted({point_key(row) for row in selected})
    tasks = [(mode, n, point, threshold(mode, args.theta), args.epsilon_sol,
              args.k_min, args.k_max) for point in points]
    candidates = list(map(exact_job, tasks)) if pool is None else pool.map(exact_job, tasks, chunksize=1)
    candidates = [row for row in candidates if row is not None]
    if not candidates:
        raise ValueError(f'No feasible precisely verified candidates for {mode} KYBER-{n}')
    return min(candidates, key=lambda row: (row['total_bits'], point_key(row)))


def print_summary(rows):
    labels = {
        'table2': 'Table 2 - Classical',
        'table3': 'Table 3 - Classical',
        'quantum': 'Remark 3 - Quantum',
        'direct': 'Table 4 - Direct resultant'
    }
    print()
    print('Final complexity estimates')
    print('=' * 72)
    for mode in dict.fromkeys(row['mode'] for row in rows):
        print()
        print(labels[mode])
        for row in (item for item in rows if item['mode'] == mode):
            bits = f'{row["total_bits"]:.4f} bits'
            if mode in ('table2', 'direct'):
                bits += f' (ceiling: {math.ceil(row["total_bits"])})'
            print(f'  {row["scheme"]}: {bits}')
            if mode != 'direct':
                print(f'    n_lat={row["nlat"]}  beta_bkz={row["bkz"]}  '
                      f'beta_sieve={row["sieve"]}  gamma={row["gamma"]}')
                print(f'    log2(K)={row["log2_k"]:.4f}  log2(W)={row["log2_w"]:.4f}')
                print(f'    p_true_target={row["p_true_target"]:.4f}  '
                      f'p_true_achieved={row["p_true"]:.6f}')


def main():
    parser = argparse.ArgumentParser(description='Automatic multistage LWE complexity search')
    parser.add_argument('--mode', choices=['all', 'table2', 'table3', 'quantum', 'direct'], default='all')
    parser.add_argument('--scheme', type=int, nargs='+', choices=tuple(SCHEMES), default=list(SCHEMES))
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--workers', type=int, default=36)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='balanced')
    parser.add_argument('--refine', type=int, default=64)
    parser.add_argument('--sieve-step', type=int, default=1)
    parser.add_argument('--theta', '--p-true', dest='theta', type=float)
    parser.add_argument('--epsilon-sol', type=float, default=0.01)
    parser.add_argument('--k-min', type=float, default=0)
    parser.add_argument('--k-max', type=float, default=24)
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.workers < 1 or args.refine < 1 or args.sieve_step < 1:
        parser.error('Workers, finalists and sieve step must be positive')
    if not 0 < args.epsilon_sol < 1 or (args.theta is not None and not 0 < args.theta < 1):
        parser.error('Probability parameters must lie strictly between zero and one')
    if args.k_min < 0 or args.k_max <= args.k_min:
        parser.error('Invalid log2(K) interval')
    if args.resume and args.checkpoint_dir is None:
        parser.error('--resume requires --checkpoint-dir')

    modes = ['table2', 'table3', 'quantum', 'direct'] if args.mode == 'all' else [args.mode]
    if not args.search and any(mode != 'direct' for mode in modes):
        parser.error('Use --search for Table 2, Table 3 and quantum complexity estimation')
    pool_size = args.workers if args.search and any(mode != 'direct' for mode in modes) else 0
    pool = mp.get_context('spawn').Pool(processes=pool_size) if pool_size else None
    rows = []
    try:
        for mode in modes:
            for n in args.scheme:
                if mode == 'direct':
                    bits = direct_resultant_cost(n)
                    row = {'mode': mode, 'scheme': f'KYBER-{n}', 'total_bits': bits}
                else:
                    row = auto_search(pool, mode, n, args)
                rows.append(row)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print_summary(rows)


if __name__ == '__main__':
    main()
