import argparse
from functools import partial
from math import ceil, comb, cos, erf, exp, pi, prod, sin, sqrt
from multiprocessing import cpu_count, Pool
from random import random, randint, shuffle
from statistics import median
from sys import stdout
import matplotlib.pyplot as plt
import numpy as np

# FPLLL, G6K imports
from fpylll import IntegerMatrix, BKZ
from fpylll.algorithms.bkz2 import BKZReduction
from g6k import SieverParams
from g6k.siever import Siever

import os

def handle_array(a,q):
    a = list(a)  
    for i in range(len(a)):
        a[i] = a[i] % q
        if a[i] > q/2:
            a[i] = a[i] - q
    return a

def handle_num(a,q):
    a =a % q
    if a > q/2:
        a = a-q
    return a

class CentredBinomial:
    """
    Generate vector from the centered binomial distribution.
    """
    def __init__(self, eta=3):
        self.eta = eta

    def support(self):
        return range(-self.eta, self.eta + 1)

    def PDF(self, outcome):
        return 0.25**(self.eta) * comb(2*self.eta, outcome + self.eta)

    def __call__(self):
        return sum(randint(0, 1) for i in range(2*self.eta)) - self.eta

def generate_LWE_lattice(m, n, q):
    """
    Generate a basis for a random `q`-ary latice with `n` secret coefficients and `m` samples,
    i.e., it generates a matrix B of the form

        I_n A
        0   q I_{m-n},

    where I_k is the k x k identity matrix and A is a n x (m-n) matrix with
    entries uniformly sampled from {0, 1, ..., q-1}.

    :param m: the dimension of the final lattice
    :param n: the number of secret coordinates.
    :param q: the modulus to use with LWE

    :returns: The matrix A and B from above
    """
    B = IntegerMatrix.random(m, "qary", k=m-n, q=q)
    A = B.submatrix(0, n, n, m)
    return A, B

def progressive_BKZ(B: IntegerMatrix, beta, params: SieverParams, verbose=False):
    g6k = Siever(B, params)
    bkz = BKZReduction(g6k.M)

    # Run BKZ up to blocksize `beta`:
    for _beta in range(2, beta + 1):
        if verbose:
            print(f"\rBKZ_{_beta}", end="")
            stdout.flush()
        bkz(BKZ.Param(_beta, max_loops=2))
    if verbose:
        stdout.flush()
    print("finish bkz")
    return g6k

def progressive_sieve(g6k, l, r, verbose=False):
    if verbose:
        print("\rSieving", end="")
        stdout.flush()
    g6k.initialize_local(l, max(l, r - 20), r)
    g6k(alg="gauss")
    while g6k.l > l:
        if verbose:
            print("\rSieving [%3d, %3d]..." % (g6k.l, g6k.r), end="")
            stdout.flush()
        g6k.extend_left()
        g6k("bgj1" if g6k.r - g6k.l >= 45 else "gauss")
    with g6k.temp_params(saturation_ratio=0.9, db_size_factor=6):
        g6k(alg="hk3")
    g6k.resize_db(ceil(1.0 * (4 / 3)**((r - l) / 2)))
    if verbose:
        stdout.flush()
    return g6k

def change_basis(basis, vector):
    """
    calculate `vector * basis`.
    """
    return basis.multiply_left(vector)

def inner_vertor(s_lat,error,vec):
    '''
    calculate <x,e>+<y,s_lat> mod q
    '''
    return (handle_num(np.inner(vec[:n],error) + np.inner(vec[n:],s_lat) , q))

def short_vectors_sampling(basis, threads,k_lat, verbose):
    '''
    sample short vectors and return the list of vectors
    '''
    print(f"Using {threads} threads")
    sieve_params = SieverParams(threads=threads, dual_mode=False)
    g6k = progressive_BKZ(B_dual, 40, sieve_params, verbose=True)
    progressive_sieve(g6k, 0, 80, verbose=verbose >= 0)
    with Pool(threads) as pool:
        database = pool.map(partial(change_basis, g6k.M.B), g6k.itervalues())
    return [w[:n+ k_lat] for w in database]

def calculate_average_2norm(vector_list):
    total_norm = 0.0
    count = len(vector_list)
    if count == 0:
        return 0.0  
    
    for vec in vector_list:
        norm = np.linalg.norm(vec)
        total_norm += norm
    
    return total_norm / count

def cal_new_lwe(target,error,s_lat,A_g,vec):
    x = vec[:n]
    y = vec[n:]
    new_target = handle_num(np.inner(x,target) , q)
    new_a = handle_array(change_basis(A_g,x) , q)
    new_e = handle_num(np.inner(x,error) + np.inner(y,s_lat) , q)

    return  np.concatenate([
        new_a,  
        np.array([new_target]),                  
        np.array([new_e])       
    ])


###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', action='count', default=0, help='verbosity level')
    parser.add_argument('-j', type=int, default=64, help='number of threads')
    parser.add_argument('--solve', type=int, default=10, help='Dimension of the solving stage')
    parser.add_argument('--lat', type=int, default=50, help='Dimension of the dual lattice reduction stage')
    parser.add_argument('-q', type=int, default=3329, help='Prime to use in the q-ary lattice')
    parser.add_argument('--eta', type=int, default=1, help='Eta for binomial distribution')
    args = parser.parse_args()
    threads = min(cpu_count(), args.j)
    n = args.solve + args.lat # dimension of the secret s, which is n_solve+n_lat
    q = args.q

    A, B_primal = generate_LWE_lattice(2 * n, n, q)
    k_solve, k_lat = args.solve, args.lat
    assert k_solve >= 0 and k_lat >= 0
    A_solve, A_lat = A[:n - k_lat], A[n - k_lat:]
    A_solve.transpose()
    A_lat.transpose()

    print(f"Generate s, e from the centered binomial distribution")
    secret_dist = CentredBinomial(args.eta) 
    secret = [secret_dist() for i in range(n)]#generate secret s from CBD
    s_lat = secret[n - k_lat:]
    error = [secret_dist() for i in range(n)] #generate error e from CBD
    target = np.add(A.multiply_left(secret), error) 
    print("------------------------------------------------------")
    print("sample short vectors")
    B_dual = IntegerMatrix.identity(n + k_lat)
    for i in range(n, n + k_lat):
        B_dual[i, i] *= q
    for i in range(0, n):
        for j in range(0, k_lat):
            B_dual[i, n + j] = A_lat[i, j] % q
    L = short_vectors_sampling(B_dual, threads, k_lat,args.v) 
    average_norm = calculate_average_2norm(L)
    print(f"\nAverage Euclidean norm of vectors in L: {average_norm:.6f}")
    N = len(L)
    print(f"Database contains {N} dual vectors")
    print("------------------------------------------------------")
    print("calculate new lwe samples")
    # Parallel computation of new LWE samples
    with Pool(threads) as pool:
        error_new = pool.map(partial(inner_vertor, s_lat,error), L)
        new_lwe_sample = pool.map(partial(cal_new_lwe, target,error,s_lat,A_solve), L)
    

    mean_val = np.mean(error_new)  
    std_val = np.std(error_new)    
    error_arr = np.array(error_new, dtype=np.int64)  
    min_val = error_arr.min()
    max_val = error_arr.max()
    print(f"new_error statistics:")
    print(f"Mean: {mean_val:.6f}")
    print(f"Standard deviation: {std_val:.6f}")
    print("Maximum and minimum values:", max_val, min_val)

    flag = 1
    s_solve = secret[:k_solve]
    ct = 0
    print("correct solution s_solve:",s_solve)
    for ve in new_lwe_sample:
        new_a = ve[:k_solve]
        new_b = ve[k_solve]
        new_e = ve[-1]
        if new_e == 0: # Count the number of zeros in new_error
            ct = ct + 1
        if (new_b - np.inner(new_a,s_solve)) % q != new_e % q: #Verify that the newly generated LWE samples satisfy new_b-<new_a,s_solve>=new_e mod q
            flag = 0
            break
    if flag == 1:
        print("Verification passed")
        print("Number of zeros in new_e:", ct)
        output_path = "Results/new_lwe_samples(a,b,e).txt"  
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in new_lwe_sample:
                if isinstance(sample, (list, np.ndarray)):
                    elements = sample.tolist() if isinstance(sample, np.ndarray) else sample
                    line = ",".join(map(str, elements)) + "\n"
                else:
                    line = f"{sample}\n"
                f.write(line)
        print(f"✅ new_lwe_sample successfully saved to: {os.path.abspath(output_path)}")
    if flag == 0:
        print("Verification of b - a*s = e failed!")



    

   