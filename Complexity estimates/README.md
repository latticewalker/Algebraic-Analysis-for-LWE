# LWE Complexity Search

Complexity estimation for the three KYBER parameter sets using the models in Tables 2 and 3, Remark 3 (quantum), and Table 4 of *A Dual-Enhanced Two-Stage Algebraic Algorithm for LWE: New Polynomial Construction and Resultant-Based Solving*.

## Requirements

Python 3.10+ with NumPy and SciPy.

```bash
python -m pip install numpy scipy
```

## Parameter search

The following commands search KYBER-512, KYBER-768, and KYBER-1024 using 36 worker processes. Parameter ranges are selected automatically.

**Table 2**

```bash
python search.py --search --mode table2 --scheme 512 768 1024 --workers 36 --profile thorough
```

**Table 3**

```bash
python search.py --search --mode table3 --scheme 512 768 1024 --workers 36 --profile thorough
```

**Remark 3 — quantum complexity**

```bash
python search.py --search --mode quantum --scheme 512 768 1024 --workers 36 --profile thorough
```

**All models**

```bash
python search.py --search --mode all --scheme 512 768 1024 --workers 36 --profile thorough
```

Search profiles: `quick`, `balanced` (default), and `thorough`. Coarse search, automatic local refinement, and finalist sieve scans are performed without manually specifying parameter bounds. The search is heuristic and does not prove global optimality.

For a small local test:

```bash
python search.py --search --mode table3 --scheme 512 --workers 2 --profile quick
```

## Table 4

Table 4 is evaluated directly, without an optimization search.

```bash
python search.py --mode direct --scheme 512 768 1024
```

## Output and options

All results are printed to the terminal. Each searched model reports the final complexity, `n_lat`, `beta_bkz`, `beta_sieve`, `gamma`, `log2(K)`, `log2(W)`, the configured success threshold, and the achieved success probability. 


