#!/usr/bin/env python3
"""Create the deterministic float32 fbin input used by run_sample.sh."""

from pathlib import Path
import struct
import sys

import numpy as np


def write_fbin(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32, order="C")
    with path.open("wb") as f:
        f.write(struct.pack("<II", values.shape[0], values.shape[1]))
        values.tofile(f)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} PREFIX", file=sys.stderr)
        return 2
    prefix = Path(sys.argv[1])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260819)
    base = rng.normal(0.0, 1.0, size=(256, 8)).astype(np.float32)
    query = rng.normal(0.0, 1.0, size=(32, 8)).astype(np.float32)
    write_fbin(Path(f"{prefix}_base.bin"), base)
    write_fbin(Path(f"{prefix}_query.bin"), query)
    print(f"[sample-data] base={base.shape} query={query.shape} prefix={prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
