#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${ROOT}/sample/run"
PFX="${RUN}/sample"

cd "${ROOT}"
make -j
rm -rf "${RUN}"
mkdir -p "${RUN}/tmp"

python3 scripts/make_sample_dataset.py "${PFX}"
printf '0 8 100 l2 1.0\n' > "${PFX}_lifted.meta"

OMP_NUM_THREADS=4 bin/build_aknn_source \
  "${PFX}_base.bin" "${PFX}" 100 64 8 64 4 10 l2

threshold_tmp="${RUN}/tmp/sample_exact_thresholds"
OMP_NUM_THREADS=4 bin/dataset_gt \
  --base "${PFX}_base.bin" --K 100 --rk_k 100 --threads 4 --tile 64 \
  --mode thresholds --out_prefix "${threshold_tmp}"

for k in 10 50 100; do
  tmp="${RUN}/tmp/sample_k${k}"
  OMP_NUM_THREADS=4 bin/dataset_gt \
    --base "${PFX}_base.bin" --query "${PFX}_query.bin" \
    --K 100 --rk_k "${k}" --threads 4 --tile 64 \
    --mode ground-truth --threshold-table "${threshold_tmp}_baseknn_gt.bin" \
    --out_prefix "${tmp}"
  cp "${tmp}_rknn_gt.bin" "${PFX}_rknn_gt_k${k}.bin"
done

python3 tools/train_final_m8_u8.py \
  --pfx "${PFX}" --threshold-source "${PFX}_baseknn_aknn.rec"

for k in 10 50 100; do
  ANQI_RUN_ROOT="${RUN}/cache" \
    OMP_NUM_THREADS=4 \
    ANQI_BENCH_CALIBRATED_TIMING=0 \
    ./scripts/run_final.sh "${PFX}" "${k}" "16,32,64,128"
done

echo "[sample] complete: ${RUN}"
