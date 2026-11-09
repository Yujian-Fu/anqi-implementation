#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${1:-${ROOT}/data/sift1m}"
RAW="${DATA_ROOT}/raw"
PFX="${DATA_ROOT}/SIFT1M"
ARCHIVE="${DATA_ROOT}/sift.tar.gz"

mkdir -p "${DATA_ROOT}" "${RAW}"
if [[ ! -f "${RAW}/sift_base.fvecs" ]]; then
  if [[ ! -f "${ARCHIVE}" ]]; then
    curl -fL --retry 3 \
      ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz \
      -o "${ARCHIVE}"
  fi
  tar -xzf "${ARCHIVE}" -C "${RAW}" --strip-components=1
fi

for file in sift_base.fvecs sift_query.fvecs; do
  if [[ ! -f "${RAW}/${file}" ]]; then
    echo "missing ${RAW}/${file}; check the TexMex archive layout" >&2
    exit 3
  fi
done

python3 "${ROOT}/tools/fvecs_to_fbin.py" \
  "${RAW}/sift_base.fvecs" "${PFX}_base.bin"
python3 "${ROOT}/tools/fvecs_to_fbin.py" \
  "${RAW}/sift_query.fvecs" "${PFX}_query.bin"
printf '0 128 100 l2 1.0\n' > "${PFX}_lifted.meta"

TMP="${DATA_ROOT}/.gt_tmp"
rm -rf "${TMP}"
mkdir -p "${TMP}"

OMP_NUM_THREADS=64 "${ROOT}/bin/build_aknn_source" \
  "${PFX}_base.bin" "${PFX}" 100 500 64 600 64 10 l2

# Exact base thresholds are generated only to score the returned answers. They
# are kept outside the construction prefix and deleted after the RkNN GT files
# have been produced.
T_THRESH="${TMP}/SIFT1M_exact_thresholds"
OMP_NUM_THREADS=64 "${ROOT}/bin/dataset_gt" \
  --base "${PFX}_base.bin" --K 100 --rk_k 100 --threads 64 --tile 512 \
  --mode thresholds --out_prefix "${T_THRESH}"

for k in 10 50 100; do
  T="${TMP}/SIFT1M_k${k}"
  OMP_NUM_THREADS=64 "${ROOT}/bin/dataset_gt" \
    --base "${PFX}_base.bin" --query "${PFX}_query.bin" \
    --K 100 --rk_k "${k}" --threads 64 --tile 512 \
    --mode ground-truth --threshold-table "${T_THRESH}_baseknn_gt.bin" \
    --out_prefix "${T}"
  cp "${T}_rknn_gt.bin" "${PFX}_rknn_gt_k${k}.bin"
done
rm -rf "${TMP}"

python3 "${ROOT}/tools/train_final_m8_u8.py" \
  --pfx "${PFX}" --threshold-source "${PFX}_baseknn_aknn.rec"
echo "[prepare-sift1m] ready: ${PFX}"
echo "[prepare-sift1m] run: ${ROOT}/scripts/run_final.sh ${PFX} 10"
