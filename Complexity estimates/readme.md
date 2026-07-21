run_schemes.py evaluates 9 parameter regimes:

* 3 HES parameter sets

* 3 NTRU-HPS parameter sets

* 1 NTRU-HRSS parameter set

* Concrete-TFHE512

* TFHE16-1024

complexity_kyber.py estimates the complexity of Kyber under the removal of the BKZ block‑size constraint, aiming to achieve a balance between the complexities of the two stages.

Run:
python3 run_schemes.py
python3 complexity_kyber.py
