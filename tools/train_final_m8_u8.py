#!/usr/bin/env python3
"""Train the only supported verifier artifact: Rank-M=8, packed u8 LRQ-floor.

The input is the exact base-to-base top-100 table produced by dataset_gt.
The output binary formats are read directly by src/rknn_query.cpp.
"""

from pathlib import Path
import struct
import sys

import numpy as np


RANK_M = 8
K_MAX = 100
BITS = 8
CODE_SIZE = 1 << BITS
CHUNK = 65_536
TRAIN_ROWS = 200_000
MAGIC = b"ANQIRQ1\0"


def read_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        raw = f.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"truncated baseknn header: {path}")
    n, k = struct.unpack("<II", raw)
    if k != K_MAX:
        raise RuntimeError(f"expected K=100 in {path}, got K={k}")
    if n < CODE_SIZE:
        raise RuntimeError(f"base table is too small for final u8 training: n={n}")
    return n, k


def radius_memmap(path: Path, n: int, k: int) -> np.memmap:
    offset = 8 + n * k * 4
    expected = offset + n * k * 4
    if path.stat().st_size != expected:
        raise RuntimeError(
            f"bad exact baseknn size for {path}: {path.stat().st_size} != {expected}"
        )
    return np.memmap(path, dtype=np.float32, mode="r", offset=offset, shape=(n, k))


def fit_codebook(values: np.ndarray) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.float64))
    positions = np.linspace(0, values.size - 1, CODE_SIZE).round().astype(np.int64)
    centers = values[positions].copy()
    for _ in range(12):
        cuts = (centers[:-1] + centers[1:]) * 0.5
        labels = np.searchsorted(cuts, values, side="right")
        sums = np.bincount(labels, weights=values, minlength=CODE_SIZE)
        counts = np.bincount(labels, minlength=CODE_SIZE)
        nonempty = counts != 0
        centers[nonempty] = sums[nonempty] / counts[nonempty]
        centers = np.maximum.accumulate(centers)
    centers[0] = values[0]
    centers[-1] = values[-1]
    return centers.astype(np.float32)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--pfx":
        print(f"usage: {sys.argv[0]} --pfx PREFIX", file=sys.stderr)
        return 2
    prefix = Path(sys.argv[2])
    source = Path(f"{prefix}_baseknn_gt.bin")
    n, k = read_shape(source)
    radii = radius_memmap(source, n, k)

    mean = np.zeros(k, dtype=np.float64)
    for start in range(0, n, CHUNK):
        mean += np.asarray(radii[start : start + CHUNK], dtype=np.float64).sum(axis=0)
    mean /= n

    gram = np.zeros((k, k), dtype=np.float64)
    for start in range(0, n, CHUNK):
        x = np.asarray(radii[start : start + CHUNK], dtype=np.float64) - mean
        gram += np.einsum("ni,nj->ij", x, x, optimize=True)
    _, vectors = np.linalg.eigh(gram)
    basis = vectors[:, -RANK_M:].T.astype(np.float32)

    train_n = min(n, TRAIN_ROWS)
    train = np.asarray(radii[:train_n], dtype=np.float64)
    centered = train - mean
    basis64 = basis.astype(np.float64)
    coeff = np.einsum("nk,mk->nm", centered, basis64, optimize=True)
    reconstruction = mean + np.einsum("nm,mk->nk", coeff, basis64, optimize=True)
    residual = train - reconstruction
    codebook = np.empty((k, CODE_SIZE), dtype=np.float32)
    for col in range(k):
        codebook[col] = fit_codebook(residual[:, col])

    basis_path = Path(f"{prefix}_radbasis_M{RANK_M}.bin")
    with basis_path.open("wb") as f:
        f.write(struct.pack("<ii", K_MAX, RANK_M))
        f.write(np.asarray(mean, dtype=np.float32).tobytes(order="C"))
        f.write(basis.tobytes(order="C"))

    codebook_path = Path(f"{prefix}_learned_rq_codebook_M{RANK_M}_u{BITS}.bin")
    with codebook_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<iiii", K_MAX, RANK_M, BITS, CODE_SIZE))
        f.write(codebook.tobytes(order="C"))

    print(
        f"[final-verifier] n={n} K={k} train_rows={train_n} "
        f"rank_m={RANK_M} bits={BITS} basis={basis_path} codebook={codebook_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
