#!/usr/bin/env bash
# Trains MNIST configs and evaluates them with the protocol of Sec. 4.1
# (512 generated vs. 512 test clouds, 5 repeats, 5/25/125 Euler steps).
# Usage: bash scripts/run_mnist.sh mnist_llw_llw [mnist_w_w ...]
#        bash scripts/run_mnist.sh table1 | puretable | batch64 | nsot
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
N_GEN="${N_GEN:-512}"
REPEATS="${REPEATS:-5}"
STEPS="${STEPS:-5 25 125}"

TABLE1="mnist_ind_ind mnist_ind_w mnist_ind_sw mnist_ind_llw mnist_w_w mnist_bw_w mnist_euclidean_w mnist_sw_sw mnist_llw_llw"
PURETABLE="mnist_pure_ind_ind mnist_pure_w_w mnist_pure_sw_sw mnist_pure_llw_llw"
BATCH64="mnist_batch64_llw_llw mnist_batch64_euclidean_w"
NSOT="mnist_nsot mnist_nsot_beta0.2"

[ "$#" -ge 1 ] || { echo "usage: $0 <config name>... | table1 | puretable | batch64 | nsot" >&2; exit 2; }
NAMES=""
for ARG in "$@"; do
  case "$ARG" in
    table1) NAMES="$NAMES $TABLE1" ;;
    puretable) NAMES="$NAMES $PURETABLE" ;;
    batch64) NAMES="$NAMES $BATCH64" ;;
    nsot) NAMES="$NAMES $NSOT" ;;
    *) NAMES="$NAMES $ARG" ;;
  esac
done

for NAME in $NAMES; do
  RUN_DIR="runs/${NAME}"
  if [ -f "${RUN_DIR}/checkpoints/model_final.pt" ]; then
    echo "${NAME}: checkpoint exists, skipping training"
  else
    "$PYTHON" -m src.train --config "configs/mnist/${NAME}.yaml" --run_dir "$RUN_DIR"
  fi
  # shellcheck disable=SC2086
  "$PYTHON" -m src.evaluate --run_dir "$RUN_DIR" --n_gen "$N_GEN" --repeats "$REPEATS" --steps $STEPS
done
