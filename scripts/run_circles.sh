#!/usr/bin/env bash
# Circle experiment (Sec. 4.2, Fig. 1) and its ablations (App. B.2-B.4).
# Usage: bash scripts/run_circles.sh [fig1|ablations|all]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"

FIG1="circles_ind_ind circles_ind_w circles_w_w"
ABLATIONS="circles_w_w_batch4 circles_w_w_batch16
           circles_sw_sw_slices1 circles_sw_sw_slices8 circles_sw_sw_slices32
           circles_w_w_sinkhorn0.01 circles_w_w_sinkhorn0.1 circles_w_w_sinkhorn1"

case "${1:-fig1}" in
  fig1) NAMES="$FIG1" ;;
  ablations) NAMES="$ABLATIONS" ;;
  all) NAMES="$FIG1 $ABLATIONS" ;;
  *) echo "usage: $0 [fig1|ablations|all]" >&2; exit 2 ;;
esac

for NAME in $NAMES; do
  "$PYTHON" -m src.train --config "configs/circles/${NAME}.yaml" --run_dir "runs/${NAME}"
  echo "${NAME}: runs/${NAME}/viz/epoch_0000/trajectories.png"
done
