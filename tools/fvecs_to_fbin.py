#!/usr/bin/env python3
"""Convert fixed-dimension ANN fvecs into ANQI's single-header fbin format."""

from pathlib import Path
import struct
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} INPUT.fvecs OUTPUT.bin", file=sys.stderr)
        return 2
    source, target = map(Path, sys.argv[1:])
    raw = np.fromfile(source, dtype=np.uint32)
    if raw.size == 0:
        raise RuntimeError(f"empty fvecs file: {source}")
    dimension = int(raw[0])
    stride = dimension + 1
    if dimension <= 0 or raw.size % stride != 0:
        raise RuntimeError(f"malformed fvecs file: {source}")
    records = raw.reshape(-1, stride)
    if not np.all(records[:, 0] == dimension):
        raise RuntimeError(f"mixed vector dimensions in: {source}")
    vectors = records[:, 1:].view(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(struct.pack("<II", vectors.shape[0], dimension))
        np.asarray(vectors, dtype=np.float32, order="C").tofile(f)
    print(f"[fvecs-to-fbin] {source} -> {target}: n={vectors.shape[0]} d={dimension}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
