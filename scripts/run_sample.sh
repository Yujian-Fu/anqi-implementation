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

for k in 10 50 100; do
  tmp="${RUN}/tmp/sample_k${k}"
  cp "${PFX}_base.bin" "${tmp}_base.bin"
  cp "${PFX}_query.bin" "${tmp}_query.bin"
  OMP_NUM_THREADS=4 bin/dataset_gt \
    --base "${tmp}_base.bin" --query "${tmp}_query.bin" \
    --K 100 --rk_k "${k}" --threads 4 --tile 64 --out_prefix "${tmp}"
  cp "${tmp}_rknn_gt.bin" "${PFX}_rknn_gt_k${k}.bin"
  if [[ "${k}" == 100 ]]; then
    cp "${tmp}_baseknn_gt.bin" "${PFX}_baseknn_gt.bin"
    cp "${tmp}_rknn_gt.bin.rk" "${PFX}_rknn_gt.bin.rk"
  fi
done

python3 tools/train_final_m8_u8.py --pfx "${PFX}"

for k in 10 50 100; do
  ANQI_RUN_ROOT="${RUN}/cache/k${k}" \
    OMP_NUM_THREADS=4 \
    ANQI_BENCH_CALIBRATED_TIMING=0 \
    ./scripts/run_final.sh "${PFX}" "${k}" "16,32,64,128"
done

echo "[sample] complete: ${RUN}"
