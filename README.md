# ANQI: Minimal Reproducible Prototype

This repository contains a small, self-contained correctness prototype of the
final ANQI reverse-kNN design. It generates a deterministic toy data set, so it
does not require downloading SIFT/GIST/Deep data or any external library.

The prototype is intentionally not a million-scale benchmark implementation.
Its purpose is to make the final algorithmic path easy to compile, inspect,
and reproduce. The full benchmark numbers require the research runner, exact
base-to-base source tables, and the 64-thread warmed timing protocol; those
large artifacts are not included here.

## Final configuration represented

- exact base-to-base kNN radii with self-match excluded;
- one shared Any-K Lift graph built at the `Kmax` horizon;
- fixed graph degree `G48`;
- `graph_rank_M=0`: Rank-M is not part of graph geometry;
- independent verifier: Rank-M `M=8` plus packed `u8` residual LRQ with floor decoding;
- `slack=0`;
- no exact-radius recheck, RNG/Vamana, or bubble search in the default path;
- query-side verification uses the original-space distance against the decoded
  radius threshold, matching the production postfilter semantics.

The production experiments use `Kpivot=100`, `K={10,50,100}`, 64-thread
warmed batch-wall timing, and the same shared Any-K graph for all query k. The
toy sample uses `Kmax=10` only to keep the executable tiny.

## Build and run

Requirements: a C++17 compiler and `make`.

```bash
git clone https://github.com/Yujian-FU0425/anqi-rknn-prototype.git
cd anqi-rknn-prototype
./scripts/run_sample.sh
```

Or run the two steps explicitly:

```bash
make
./anqi_sample
```

The output is CSV-like. It reports `recall`, `precision`, candidate recall,
and average lifted-graph distance evaluations for the final
`rank-M8+u8-LRQ-floor` verifier at `k=1,5,10`. Exact answers are used only as
the evaluation reference.

## What the sample does

1. Generate deterministic clustered base/query vectors.
2. Compute exact base-to-base `r_k(o)^2` with diagonal exclusion.
3. Build the shared spherical Any-K Lift at the largest sample horizon.
4. Build a small fixed-degree lifted graph and run best-first range search.
5. Fit an `M=8` low-rank radius model with the same column-scaled Gram/SVD
   construction used by the final rank-M pipeline.
6. Quantize the residual per k with packed 8-bit floor codes.
7. Apply the decoded radius as the verifier predicate and compare with exact
   reverse-kNN answers.

The toy graph uses exact lifted kNN edges to keep the sample deterministic. It
does not include any alternate graph policy or fallback strategy.
