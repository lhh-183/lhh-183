# EoTMP experiment code

`eotmp.py` is a self-contained reference experiment for the paper
“EoTMP: Efficient Over-Threshold Multi-Party Private Set Intersection”. It
implements the protocol data path used in the experiments:

1. map every participant's set to a Bloom filter using MurmurHash3 and double
   hashing;
2. add the filters position by position to obtain item frequencies;
3. model the over-threshold polynomial
   `f(x) = r * product(x - i)` for `i = T .. N` (or the paper's complementary
   roots `0 .. T-1` when `T < N/2`), where non-root positions are randomly
   masked;
4. let the receiver test its own items and report exact hits, Bloom-filter
   false positives, and false negatives;
5. report the three protocol rounds, SIMD batches, Paterson–Stockmeyer
   multiplication estimate, and communication estimates.

The vector arithmetic is intentionally plaintext. It models the BFV/MHE
operations and lets the experiments run without a cryptography dependency;
it does not provide confidentiality or replace the paper's Go/Lattigo
implementation for deployment.

## Run it

From this directory:

```bash
python3 eotmp.py demo --participants 8 --set-size 64 --threshold 6
python3 eotmp.py fpp --set-size 64 --bits 4096 --hashes 10
python3 eotmp.py benchmark --participants 8 16 32 --set-sizes 64 256 \
  --threshold-gap 3 --ciphertext-bytes 660602 --output results.csv
python3 -m unittest -v
```

The benchmark's `threshold-gap` is the paper's `alpha = N - T + 1`: the
threshold is computed as `T = N - threshold_gap + 1`. Use `--bits` and `--hashes` to study
the Bloom-filter false-positive trade-off. `--slots` models the BFV batching
width; the default is 8192.

The `run_eotmp` function also accepts `online=[...]` and requires the receiver
to be online. This exercises the protocol's t-out-of-N availability path at
the set-processing level. A full threshold key-generation and key-switching
implementation still requires the Lattigo MHE backend described in the paper.

`--ciphertext-bytes` is optional and defaults to zero. Set it to the measured
BFV ciphertext size for the chosen Lattigo parameters (the paper reports about
0.63 MiB for `log(n)=13`) to populate the three-round communication
estimates.
