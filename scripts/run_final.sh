#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PREFIX QUERY_K [L_DESCENT_LIST]" >&2
  exit 2
fi

PFX="$1"
KQ="$2"
LDS="${3:-128,256,512,1024}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${KQ}" in
  10|50|100) ;;
  *) echo "query k must be one of 10, 50, 100" >&2; exit 2 ;;
esac

for path in "${PFX}_base.bin" "${PFX}_query.bin" \
            "${PFX}_lifted.meta" \
            "${PFX}_baseknn_aknn.rec" \
            "${PFX}_rknn_gt_k${KQ}.bin" \
            "${PFX}_radbasis_M8.bin" \
            "${PFX}_learned_rq_codebook_M8_u8.bin" \
            "${PFX}_aknn_build_time.meta" \
            "${PFX}_verifier_train_time.meta"; do
  [[ -f "${path}" ]] || { echo "missing required artifact: ${path}" >&2; exit 3; }
done

RUN_ROOT="${ANQI_RUN_ROOT:-${PFX}_anqi_run}"
mkdir -p "${RUN_ROOT}/graph" "${RUN_ROOT}/verifier"

read_seconds() {
  local path="$1" expected_header="$2" value
  [[ "$(sed -n '1p' "${path}")" == "${expected_header}" ]] || {
    echo "invalid timing metadata header: ${path}" >&2
    exit 3
  }
  value="$(awk '$1 == "seconds" { print $2; exit }' "${path}")"
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || {
    echo "invalid timing value in ${path}" >&2
    exit 3
  }
  printf '%s' "${value}"
}

export ANQI_AKNN_BUILD_SECONDS="$(
  read_seconds "${PFX}_aknn_build_time.meta" ANQI_AKNN_BUILD_V1
)"
export ANQI_VERIFIER_TRAIN_SECONDS="$(
  read_seconds "${PFX}_verifier_train_time.meta" ANQI_VERIFIER_TRAIN_V1
)"

# Pin the reported configuration for every run.
unset ANQI_GRAPH_GEOMETRY ANQI_GRAPH_RANKM ANQI_RADIUS_MODE \
  ANQI_VERIFIER_PLACEMENT ANQI_RANKM ANQI_NK ANQI_GRAPH_NK \
  ANQI_RANKM_SLACK ANQI_RECHECK ANQI_VAMANA ANQI_EDGE_POLICY \
  ANQI_SPECIFIC ANQI_CANDIDATE_ENVELOPE ANQI_RQ_OBJECTIVE \
  ANQI_PREBUILT_TOPK_REC ANQI_PREBUILT_TOPK_K ANQI_RADIUS_RELAX ANQI_RQ_CODEBOOK \
  ANQI_RQ_CODEBOOK_ID ANQI_RQ_BASIS_ID ANQI_RADIUS_SOURCE_ID \
  ANQI_GRAPH_RADIUS_SOURCE_PFX ANQI_GRAPH_RADIUS_ID \
  ANQI_RELEASE_PREBUILT_TOPK_AFTER_PREP \
  ANQI_COMPACT_PREBUILT_TOPK_IDS_INPLACE \
  ANQI_RELEASE_VERIFIER_DURING_GRAPH_INDEX \
  ANQI_RELEASE_BASE_DURING_GRAPH_INDEX \
  ANQI_QUERY_VISITED ANQI_TIME_BREAKDOWN

export ANQI_GRAPH_GEOMETRY=anyk_lift
export ANQI_RADIUS_MODE=rankm_resid_u8_lrq_floor
export ANQI_VERIFIER_PLACEMENT=postfilter
export ANQI_RANKM=8
export ANQI_NK=100
export ANQI_GRAPH_NK=100
export ANQI_RANKM_SLACK=0
export ANQI_RECHECK=0
export ANQI_RQ_OBJECTIVE=radius_reconstruction
export ANQI_ALPHA=1.2
export ANQI_LIFTNNK=50
export ANQI_GRAPHNND=6
export ANQI_BUILD_BREAKDOWN=1
export ANQI_PREBUILT_TOPK_REC="${PFX}_baseknn_aknn.rec"
export ANQI_PREBUILT_TOPK_K=100
export ANQI_GRAPH_CACHE_DIR="${RUN_ROOT}/graph"
export ANQI_VERIFIER_CACHE_DIR="${RUN_ROOT}/verifier"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-64}"

printf '%s\n' \
  "[anqi-run] profile=million-v1 K_pivot=100 G=48 LocalJoin=6 M=8 residual=u8 ranks=10,50,100 query_k=${KQ} threads=${OMP_NUM_THREADS}"

if [[ "${ANQI_BENCH_CALIBRATED_TIMING:-1}" == 1 ]]; then
  export ANQI_BENCH_CALIBRATED_TIMING=1
  export ANQI_BENCH_MIN_TOTAL_S=30
  export ANQI_BENCH_MIN_REPEAT_S=10
  export ANQI_BENCH_REPEATS=3
fi

exec "${ROOT}/bin/rknn_query" "${LDS}" "${PFX}" 48 500 64 600 "${KQ}"
