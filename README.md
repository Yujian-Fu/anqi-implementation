# ANQI Implementation

This repository contains the million-scale ANQI approximate reverse-kNN
implementation used in the paper. The sample runs the same AKNN source builder,
Lift graph builder and searcher, and Rank-M/LRQ verifier as the reported
million-scale experiments.

This release exposes the shared million-scale configuration below. Run it
through `scripts/run_final.sh`, which pins the construction and query settings
instead of inheriting experimental overrides from the shell.

## Published million-scale configuration

The supported path is:

- one approximate base-space top-100 pass with self exclusion;
- the same AKNN IDs as the graph warm start and the same AKNN distances as the
  cutoff matrix;
- one shared construction-rank Lift graph at `K_pivot=100` and degree `G48`;
- six LocalJoin refinement passes for the lifted graph;
- independent verifier: Rank-M `M=8` + learned packed `u8` residual LRQ;
- floor decoding and radius-reconstruction objective;
- query `k` selected from `{10, 50, 100}` while reusing the same graph;
- 64-thread warmed batch timing for reported runs.

Exact base-to-base thresholds are evaluation inputs only. They are not consumed
by graph construction, LRQ fitting, or online verification. The benchmark
wrapper pins these roles and values for every run.

## Technical report

The full technical report, including the mathematical derivations and extended
appendix material omitted from the submission version, is available here:

- [ANQI technical report](technical_report.pdf)

## Platform requirements

The implementation uses AVX-512, OpenMP, C++17, and Linux system interfaces.
Run the sample and benchmark on a Linux machine with AVX-512 support.

Install a C++17 compiler, OpenMP, Python 3, and NumPy. Then build:

```bash
make -j
```

## End-to-end sample

The sample creates a deterministic 256-vector base and 32 queries, builds one
approximate top-100 source, derives exact reverse-kNN ground truth separately
for `k=10,50,100`, trains the M8/u8 LRQ artifacts from the approximate source,
builds the shared G48 Lift graph, and runs the benchmark query executable. The
small input keeps this correctness test quick.

```bash
./scripts/run_sample.sh
```

The sample uses four OpenMP threads. To reproduce the reported measurements,
use 64 threads and the warmed timing protocol.

## Reproduce with SIFT1M

Use the canonical ANN SIFT1M release from the INRIA TEXMEX corpus:

```text
https://corpus-texmex.irisa.fr/
ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
```

The archive contains `sift_base.fvecs` (1,000,000 x 128),
`sift_query.fvecs` (10,000 x 128), `sift_learn.fvecs`, and the original ANN
ground-truth file. The preparation step builds the approximate top-100 source
used by ANQI. It separately generates exact reverse-kNN answers for evaluation.

From the repository root, run:

```bash
make -j
./scripts/prepare_sift1m.sh
```

The default data location is:

```text
data/sift1m/sift.tar.gz      downloaded archive
data/sift1m/raw/             extracted fvecs files
data/sift1m/SIFT1M_*        converted ANQI files and generated artifacts
```

For a manual download, place `sift.tar.gz` at
`data/sift1m/sift.tar.gz`. The preparation script uses this archive, extracts
the source vectors into `data/sift1m/raw/`, and writes the converted dataset
prefix as `data/sift1m/SIFT1M`.

This creates the following reproducible prefix and keeps all generated files
under `data/sift1m/`:

```text
data/sift1m/raw/sift_*.fvecs       downloaded source files
data/sift1m/SIFT1M_base.bin       float32 fbin, [uint32 n][uint32 d][values]
data/sift1m/SIFT1M_query.bin      float32 fbin, same dimension
data/sift1m/SIFT1M_baseknn_aknn.rec interleaved AKNN IDs/cutoffs shared by graph and LRQ
data/sift1m/SIFT1M_rknn_gt_k10.bin
data/sift1m/SIFT1M_rknn_gt_k50.bin exact RkNN CSR tables
data/sift1m/SIFT1M_rknn_gt_k100.bin
data/sift1m/SIFT1M_aknn_build_time.meta
data/sift1m/SIFT1M_radbasis_M8.bin
data/sift1m/SIFT1M_learned_rq_codebook_M8_u8.bin
data/sift1m/SIFT1M_verifier_train_time.meta
```

On a fresh cache, `run_final.sh` reports an end-to-end construction time that
includes the measured approximate-AKNN pass, verifier training and encoding,
lifting, graph construction, and index persistence. The exact all-pairs scan
used to derive evaluation ground truth is excluded from index construction.

After preparation, run the same shared graph for each query k:

```bash
./scripts/run_final.sh "$PWD/data/sift1m/SIFT1M" 10
./scripts/run_final.sh "$PWD/data/sift1m/SIFT1M" 50
./scripts/run_final.sh "$PWD/data/sift1m/SIFT1M" 100
```

The first fresh-cache command prints the end-to-end construction summary and
the measured recall/precision/QPS/distance results. By default the commands use
64 OpenMP threads, the shared G48 construction-rank Lift graph, and the M8+u8
LRQ-floor postfilter.

## Running a prepared benchmark prefix

The prefix must contain:

```text
<PFX>_base.bin                 float32 fbin: [uint32 n][uint32 d][n*d values]
<PFX>_query.bin                float32 fbin with the same dimension
<PFX>_baseknn_aknn.rec         approximate top-100 (ID, cutoff) records
<PFX>_rknn_gt_k10.bin          exact reverse-kNN CSR for k=10
<PFX>_rknn_gt_k50.bin          exact reverse-kNN CSR for k=50
<PFX>_rknn_gt_k100.bin         exact reverse-kNN CSR for k=100
<PFX>_lifted.meta              lift metadata: M*2, dimension, horizon, metric, scale
<PFX>_aknn_build_time.meta     measured AKNN source-generation time
<PFX>_radbasis_M8.bin          generated by train_final_m8_u8.py
<PFX>_learned_rq_codebook_M8_u8.bin
<PFX>_verifier_train_time.meta measured verifier-training time
```

Generate the shared approximate source and train the verifier from its cutoff
matrix:

```bash
bin/build_aknn_source DATA/base.bin DATA/run 100 500 64 600 64 10 l2
python3 tools/train_final_m8_u8.py --pfx DATA/run \
  --threshold-source DATA/run_baseknn_aknn.rec
```

The trainer matches the reported verifier fit by default: a fixed 120,000-row
PCA initialization, a seed-10 sample of the same size, three recursive
basis/codebook rounds, and eight Lloyd iterations per codebook. It also extends
the endpoint codewords to the full residual range so floor decoding remains
one-sided outside the training sample.

Generate an exact threshold table under a temporary evaluation prefix, then use
it to derive reverse-kNN ground truth:

```bash
bin/dataset_gt --base DATA/base.bin --K 100 --rk_k 100 --threads 64 \
  --mode thresholds --out_prefix DATA/eval_exact
bin/dataset_gt --base DATA/base.bin --query DATA/query.bin \
  --K 100 --rk_k 10 --threads 64 --mode ground-truth \
  --threshold-table DATA/eval_exact_baseknn_gt.bin --out_prefix DATA/run_k10
```

Repeat the ground-truth-only command for ranks 50 and 100, then install the
resulting `_rknn_gt.bin` files as `run_rknn_gt_k10.bin`,
`run_rknn_gt_k50.bin`, and `run_rknn_gt_k100.bin`. Delete the temporary exact
table after this step. Neither it nor the query-to-base scans enter the index's
construction time.

Run the final search only through the pinned wrapper:

```bash
./scripts/run_final.sh DATA/run 10
./scripts/run_final.sh DATA/run 50
./scripts/run_final.sh DATA/run 100
```

The first argument is a data prefix and the second is the query k. The wrapper
uses the reported defaults `leaf=500`, `trees=64`, `ef=600`, graph degree
`48`, and a representative `L_descent` sweep. Cache files are placed beside
the prefix unless `ANQI_RUN_ROOT` is supplied by the wrapper.

## Ground-truth format

In `--mode thresholds`, `dataset_gt` performs exact all-pairs base-to-base
search with diagonal exclusion and records its complete elapsed time in a
metadata sidecar. In `--mode ground-truth`, it reads that table and computes
exact query-to-base membership against each object's `r_k`. The base table is:

```text
[uint32 n][uint32 K][n*K uint32 neighbor ids][n*K float32 keys]
```

The reverse-kNN output is CSR:

```text
[uint32 nq][(nq+1) uint32 offsets][total uint32 ids][total float32 keys]
```

For L2, keys are squared distances. For inner product, keys are negated inner
products so that the same ascending comparison is used.

## Reproducibility notes

The sample and benchmark invoke the same AKNN, graph, verifier, and query
binaries. Keep datasets
and generated indexes in an isolated run directory, and record the compiler,
OpenMP thread count, machine, input checksums, construction summary, and
warmed-search summary alongside the results.
