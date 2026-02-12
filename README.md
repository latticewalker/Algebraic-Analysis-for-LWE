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

**Example usage:**  
You can first generate new LWE samples with `generate_new_lwe_samples.py` and then pass them to `verify_candidate_count.py`.  
We also provide a pre‑generated LWE instance: `new_lwe_sample_n50_nlat40.txt`, created using the Set (1) parameters from the paper. The corresponding true secret is `[-1, 0, 0, -1, -1, 0, 1, 0, 1, 1]`.  
Simply run `verify_candidate_count.py` with this input file to directly obtain the output file `the_number_of_solutions.txt`..
