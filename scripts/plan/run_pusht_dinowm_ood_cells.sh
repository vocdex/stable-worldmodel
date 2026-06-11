#!/bin/bash
# Legacy DINO-WM (+proprio) on the dynamic-distractor PushT cells, run
# locally like the original ood_matrix_n300_eps50_s0 (config pusht_dinowm
# points at the local ~/Desktop/dino_wm checkout/ckpt).
#
# Cells: baseline (re-run per seed) + the 3 cells from
# docs/dynamic_ood_cells.md + the legacy static-distractor cell for the
# static-vs-moving ablation. Idempotent via done.flags.
#
# NOTE: each cell's eval.variation_overrides REPLACES the config's baked
# {block.scale:30, goal.scale:30} default — that's fine today: block.scale
# init IS 30 (a1437f7) and goal.scale is unused in rendering (pinned by the
# T6 sweep xfail), so appearance stays consistent with the H5.
#
# Usage:
#   SEED=0 bash scripts/plan/run_pusht_dinowm_ood_cells.sh
#   SEED=1 NUM_EVAL=50 bash scripts/plan/run_pusht_dinowm_ood_cells.sh

set -u

SEED=${SEED:-0}
NUM_EVAL=${NUM_EVAL:-50}
NUM_SAMPLES=${NUM_SAMPLES:-300}
BATCH_SIZE=${BATCH_SIZE:-4}

export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1
# fail loudly if the eval trust chain is broken (goal/start frame checks)
export SWM_EVAL_TRUST_CHECKS=1
# bg_* cells need textures: python scripts/data/fetch_textures.py once
export SWM_TEXTURE_DIR=${SWM_TEXTURE_DIR:-$HOME/.stable_worldmodel/textures}

CELLS=(
  "baseline|null"
  "distractor|{variation:[distractor.color,distractor.scale,distractor.position],variation_values:{distractor.color:[255,0,255],distractor.scale:25,distractor.position:[80,80]}}"
  "distractor_moving|{variation:[distractor.motion],variation_values:{distractor.motion:[40,20]}}"
  "bg_natural_static|{variation:[background.texture_id],variation_values:{background.texture_id:1}}"
  "bg_video_dynamic|{variation:[background.texture_id],variation_values:{background.texture_id:6}}"
  "bg_video_camouflage|{variation:[background.texture_id],variation_values:{background.texture_id:8}}"
)

RESULTS_DIR=checkpoints/dino_wm_legacy_pusht/dyn_cells_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
CSV="$RESULTS_DIR/dyn_cells.csv"
mkdir -p "$RESULTS_DIR"

for ENTRY in "${CELLS[@]}"; do
    LABEL="${ENTRY%%|*}"
    OVERRIDE_VALUE="${ENTRY##*|}"
    CELL_DIR="$RESULTS_DIR/cells/$LABEL"
    CELL_LOG="$CELL_DIR/eval.log"
    mkdir -p "$CELL_DIR"

    if [ -f "$CELL_DIR/done.flag" ]; then
        echo "== $LABEL: done, skipping"
        continue
    fi
    rm -f "$CELL_DIR/failed.flag"

    HYDRA_OVERRIDES=(
        --config-name pusht_dinowm
        eval.num_eval="$NUM_EVAL"
        solver.batch_size="$BATCH_SIZE"
        solver.num_samples="$NUM_SAMPLES"
        seed="$SEED"
        "+output.video_path=$CELL_DIR/videos"
    )
    if [ "$OVERRIDE_VALUE" != "null" ]; then
        HYDRA_OVERRIDES+=("eval.variation_overrides=${OVERRIDE_VALUE}")
    fi

    echo "== $LABEL: launching ($(date))"
    START_TIME=$(date +%s)
    uv run python scripts/plan/eval_wm.py "${HYDRA_OVERRIDES[@]}" 2>&1 | tee "$CELL_LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START_TIME ))

    SR="NA"
    if [ "$EXIT_CODE" -eq 0 ]; then
        SR=$(grep -oP "'success_rate':\s*\K[0-9.]+" "$CELL_LOG" | tail -1)
        SR=${SR:-NA}
    fi
    if [ "$SR" = "NA" ]; then
        touch "$CELL_DIR/failed.flag"
    else
        touch "$CELL_DIR/done.flag"
    fi

    if [ ! -f "$CSV" ]; then
        echo "label,SR,elapsed_s" > "$CSV"
    fi
    printf '%s,%s,%s\n' "$LABEL" "$SR" "$ELAPSED" >> "$CSV"
    echo "== $LABEL: SR=$SR elapsed=${ELAPSED}s"
done

echo "All cells done. Results: $CSV"
