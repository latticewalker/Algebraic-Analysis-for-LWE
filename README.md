
1. `generate_new_lwe_samples.py`  
   This script requires the installation of G6K and FPyLLL.  
   It generates new LWE samples.

2. `verify_candidate_count.py`  
   Verifies the expected number of candidate solutions (Lemma 6).  
   **Input**: New LWE samples, sorted as `(a, b, e)`, and the correct secret `s_solve`.  
   **Output**: Statistics on solution counts, including the number of systems that contain the correct secret.

3. `verify_true_secret_probability.py`  
   Verifies the probability that the true secret is contained in the system.  
   **Input**: Observed zero counts in the amplified error, along with the system parameters `(K, W, N)`.  
   **Output**: Mean simulated success probability and its standard deviation across multiple LWE instances.
4. `s_6.m`
   Magma code. Run this code using `load "s_6.m";`
   First select appropriate parameters K, W, then construct the corresponding system of equations, and finally solve it using resultants. The procedure corresponds to the proof part of Theorem 2.
   
   **Input**:New LWE samples, sorted as `(a, b, e)`(e.g. new_lwe_samples(a,b,e).m)
   
   **Output**:The obtained solution (which may be non-existent or incorrect with some probability)

5. `verify_lemma6.py`
   This code is used to verify the probablity 1/q of Lemma6.
**Example usage:**  
You can first generate new LWE samples with `generate_new_lwe_samples.py` and then pass them to `verify_candidate_count.py`.  
We also provide a pre‑generated LWE instance: `new_lwe_sample_n50_nlat40.txt`, created using the Set (1) parameters from the paper. The corresponding true secret is `[-1, 0, 0, -1, -1, 0, 1, 0, 1, 1]`.  
Simply run `verify_candidate_count.py` with this input file to directly obtain the output file `the_number_of_solutions.txt`.
