#!/usr/bin/env python3
"""Train the M8+u8 verifier used by the reported ANQI configuration."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import struct

import numpy as np


RANK_M = 8
K_MAX = 100
BITS = 8
CODE_SIZE = 1 << BITS
TRAIN_ROWS = 120_000
RECURSIVE_ITERS = 3
LLOYD_ITERS = 8
SEED = 10
CHUNK = 65_536
MAGIC = b"ANQIRQ1\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the recursive M8+u8 LRQ verifier from exact top-100 thresholds."
    )
    parser.add_argument("--pfx", required=True, type=Path, help="dataset prefix")
    parser.add_argument("--train-sample", type=int, default=TRAIN_ROWS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--recursive-iters", type=int, default=RECURSIVE_ITERS)
    parser.add_argument("--lloyd-iters", type=int, default=LLOYD_ITERS)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument(
        "--codebook-workers",
        type=int,
        default=int(os.environ.get("ANQI_CODEBOOK_WORKERS", os.environ.get("OMP_NUM_THREADS", "1"))),
    )
    args = parser.parse_args()
    for name in ("train_sample", "recursive_iters", "lloyd_iters", "chunk", "codebook_workers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def read_threshold_matrix(path: Path) -> tuple[np.memmap, int]:
    with path.open("rb") as stream:
        raw = stream.read(8)
    if len(raw) != 8:
        raise RuntimeError(f"truncated baseknn header: {path}")
    n, k = struct.unpack("<II", raw)
    if k != K_MAX:
        raise RuntimeError(f"expected K={K_MAX} in {path}, got K={k}")
    if n < CODE_SIZE:
        raise RuntimeError(f"at least {CODE_SIZE} rows are required for u8 training, got {n}")

    offset = 8 + n * k * 4
    expected = offset + n * k * 4
    if path.stat().st_size != expected:
        raise RuntimeError(f"bad baseknn size for {path}: {path.stat().st_size} != {expected}")
    values = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(n, k))
    return values, n


def fit_basis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    covariance = centered.T @ centered / max(1, matrix.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = np.ascontiguousarray(eigenvectors[:, order[:RANK_M]].T)
    explained = float(eigenvalues[order[:RANK_M]].sum() / max(eigenvalues.sum(), 1e-30))
    return mean, basis, explained


def reconstruct(matrix: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    coefficients = (matrix - mean) @ basis.T
    return mean + coefficients @ basis


def train_codebook(values: np.ndarray, iterations: int) -> np.ndarray:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    if not np.isfinite(sorted_values).all():
        raise RuntimeError("threshold residuals contain non-finite values")
    if sorted_values[0] == sorted_values[-1]:
        return np.full(CODE_SIZE, sorted_values[0], dtype=np.float64)

    quantiles = (np.arange(CODE_SIZE, dtype=np.float64) + 0.5) / CODE_SIZE
    centers = np.quantile(sorted_values, quantiles).astype(np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(sorted_values)))
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) * 0.5
        cuts = np.searchsorted(sorted_values, boundaries, side="right")
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [sorted_values.size]))
        counts = ends - starts
        updated = centers.copy()
        nonempty = counts > 0
        sums = prefix[ends] - prefix[starts]
        updated[nonempty] = sums[nonempty] / counts[nonempty]
        updated = np.maximum.accumulate(updated)
        if np.max(np.abs(updated - centers)) < 1e-12:
            centers = updated
            break
        centers = updated
    return centers


def train_codebooks(residual: np.ndarray, iterations: int, workers: int) -> np.ndarray:
    def fit_column(column: int) -> tuple[int, np.ndarray]:
        return column, train_codebook(residual[:, column], iterations)

    codebooks = np.empty((K_MAX, CODE_SIZE), dtype=np.float64)
    worker_count = min(workers, K_MAX)
    if worker_count == 1:
        trained = map(fit_column, range(K_MAX))
        pool = None
    else:
        pool = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="lrq")
        trained = pool.map(fit_column, range(K_MAX))
    try:
        for column, centers in trained:
            codebooks[column] = centers
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    return codebooks


def decode_nearest(residual: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    decoded = np.empty_like(residual, dtype=np.float64)
    for column in range(K_MAX):
        centers = codebooks[column]
        values = residual[:, column]
        insertion = np.searchsorted(centers, values, side="left")
        high = np.clip(insertion, 0, CODE_SIZE - 1)
        low = np.clip(insertion - 1, 0, CODE_SIZE - 1)
        take_high = np.abs(centers[high] - values) < np.abs(values - centers[low])
        decoded[:, column] = np.where(take_high, centers[high], centers[low])
    return decoded


def full_residual_range(
    thresholds: np.memmap,
    n: int,
    mean: np.ndarray,
    basis: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(K_MAX, np.inf, dtype=np.float64)
    maximum = np.full(K_MAX, -np.inf, dtype=np.float64)
    for start in range(0, n, chunk):
        block = np.asarray(thresholds[start : start + chunk], dtype=np.float64)
        residual = block - reconstruct(block, mean, basis)
        minimum = np.minimum(minimum, residual.min(axis=0))
        maximum = np.maximum(maximum, residual.max(axis=0))
    return minimum, maximum


def write_basis(path: Path, mean: np.ndarray, basis: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<ii", K_MAX, RANK_M))
        stream.write(np.asarray(mean, dtype="<f4").tobytes(order="C"))
        stream.write(np.asarray(basis, dtype="<f4").tobytes(order="C"))


def write_codebooks(path: Path, codebooks: np.ndarray) -> None:
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<iiii", K_MAX, RANK_M, BITS, CODE_SIZE))
        stream.write(np.asarray(codebooks, dtype="<f4").tobytes(order="C"))


def main() -> int:
    args = parse_args()
    source = Path(f"{args.pfx}_baseknn_gt.bin")
    thresholds, n = read_threshold_matrix(source)

    sample_n = min(args.train_sample, n)
    seed_rows = np.asarray(thresholds[:sample_n], dtype=np.float64)
    mean, basis, explained = fit_basis(seed_rows)

    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(n, size=sample_n, replace=False))
    training_rows = np.asarray(thresholds[sample_ids], dtype=np.float64)

    codebooks = None
    for iteration in range(args.recursive_iters):
        rank_reconstruction = reconstruct(training_rows, mean, basis)
        residual = training_rows - rank_reconstruction
        codebooks = train_codebooks(residual, args.lloyd_iters, args.codebook_workers)
        decoded = decode_nearest(residual, codebooks)
        rmse = float(np.sqrt(np.mean((rank_reconstruction + decoded - training_rows) ** 2)))
        print(
            f"[verifier-train] round={iteration + 1}/{args.recursive_iters} "
            f"sample_rmse={rmse:.6g} explained={explained:.6f}",
            flush=True,
        )
        if iteration + 1 < args.recursive_iters:
            mean, basis, explained = fit_basis(training_rows - decoded)

    if codebooks is None:
        raise RuntimeError("no codebook was trained")

    minimum, maximum = full_residual_range(thresholds, n, mean, basis, args.chunk)
    codebooks[:, 0] = np.minimum(codebooks[:, 0], minimum)
    codebooks[:, -1] = np.maximum(codebooks[:, -1], maximum)
    codebooks = np.maximum.accumulate(codebooks, axis=1)

    if not np.isfinite(mean).all() or not np.isfinite(basis).all() or not np.isfinite(codebooks).all():
        raise RuntimeError("trained verifier contains non-finite values")
    if np.any(np.diff(codebooks, axis=1) < 0):
        raise RuntimeError("trained codebook is not sorted")

    basis_path = Path(f"{args.pfx}_radbasis_M{RANK_M}.bin")
    codebook_path = Path(f"{args.pfx}_learned_rq_codebook_M{RANK_M}_u{BITS}.bin")
    write_basis(basis_path, mean, basis)
    write_codebooks(codebook_path, codebooks)

    expected_basis_bytes = 8 + (K_MAX + RANK_M * K_MAX) * 4
    expected_codebook_bytes = 8 + 16 + K_MAX * CODE_SIZE * 4
    if basis_path.stat().st_size != expected_basis_bytes:
        raise RuntimeError(f"bad basis output size: {basis_path}")
    if codebook_path.stat().st_size != expected_codebook_bytes:
        raise RuntimeError(f"bad codebook output size: {codebook_path}")

    print(
        f"[verifier-train] n={n} sample={sample_n} seed={args.seed} "
        f"rank_m={RANK_M} bits={BITS} rounds={args.recursive_iters} "
        f"basis={basis_path} codebook={codebook_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
