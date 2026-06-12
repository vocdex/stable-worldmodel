#!/bin/bash
# Legacy DINO-WM (+proprio) on the dynamic-distractor PushT cells, run
# locally like the original ood_matrix_n300_eps50_s0 (config pusht_dinowm
# points at the local ~/Desktop/dino_wm checkout/ckpt).
#
# Cells: baseline (re-run per seed) + the 3 cells from
# docs/dynamic_ood_cells.md + the legacy static-distractor cell for the
# static-vs-moving ablation. Idempotent via done.flags.
#
# CELL-MAJOR ordering (2026-06-12): all seeds of a cell run before the next
# cell starts, so each cell's verdict lands as early as possible. Set
# SEEDS="0 1 2" (default) or legacy single-seed SEED=1.
#
# The pusht_dinowm config no longer bakes a variation_overrides default
# (it was redundant and hydra struct-mode merge broke `+` appends), so
# cells append their override with `+eval.variation_overrides=`.
#
# Usage:
#   bash scripts/plan/run_pusht_dinowm_ood_cells.sh             # seeds 0 1 2
#   SEEDS="0 1" NUM_EVAL=50 bash scripts/plan/run_pusht_dinowm_ood_cells.sh
#   SEED=2 bash scripts/plan/run_pusht_dinowm_ood_cells.sh      # one seed

set -u

SEEDS=${SEEDS:-${SEED:-"0 1 2"}}
NUM_EVAL=${NUM_EVAL:-50}
NUM_SAMPLES=${NUM_SAMPLES:-300}
BATCH_SIZE=${BATCH_SIZE:-4}
ROLLOUT_CHUNK=${ROLLOUT_CHUNK:-64}   # adapter VRAM knob; 4080: 192 ≈ 13 GB
# proprio weight in the planning cost (visual + alpha*proprio). 0 = visual-only
# planning signal, matching SC/LeWM which use no proprio anywhere. NOTE: the
# dynamics model still consumes proprio tokens internally (trained with them).
ALPHA=${ALPHA:-0}
# BLIND=1: also blind the proprio INPUT (training-mean token) — no privileged
# state anywhere. Default off; decision 2026-06-11: alpha=0 unblinded first,
# blinded comparison later. Caveat: ckpt was trained WITH informative proprio,
# so blinded SR may understate a pixels-only-trained DINO-WM.
BLIND=${BLIND:-0}

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
  "goal_marker_removed|{variation:[rendering.render_goal],variation_values:{rendering.render_goal:0}}"
  "distractor_tee|{variation:[distractor.shape],variation_values:{distractor.shape:3}}"
  "distractor_static_orange|{variation:[distractor.motion],variation_values:{distractor.motion:[0,20]}}"
  # distractor_static_square CUT (2026-06-12): the shape axis is already pinned
  # by May same-ckpt data (gray/magenta SQUARE cells) + the binding metric;
  # re-running it under this protocol confirms nothing decision-relevant.
)

BLIND_TAG=$([ "$BLIND" = 1 ] && echo "_blind" || echo "")

for ENTRY in "${CELLS[@]}"; do
  for SEED in $SEEDS; do
    LABEL="${ENTRY%%|*}"
    OVERRIDE_VALUE="${ENTRY##*|}"
    RESULTS_DIR=checkpoints/dino_wm_legacy_pusht/dyn_cells_a${ALPHA}${BLIND_TAG}_n${NUM_SAMPLES}_eps${NUM_EVAL}_s${SEED}
    CSV="$RESULTS_DIR/dyn_cells.csv"
    CELL_DIR="$RESULTS_DIR/cells/$LABEL"
    CELL_LOG="$CELL_DIR/eval.log"
    mkdir -p "$CELL_DIR"

    if [ -f "$CELL_DIR/done.flag" ]; then
        echo "== $LABEL s$SEED: done, skipping"
        continue
    fi
    rm -f "$CELL_DIR/failed.flag"

    HYDRA_OVERRIDES=(
        --config-name pusht_dinowm
        eval.num_eval="$NUM_EVAL"
        solver.batch_size="$BATCH_SIZE"
        dino_wm_rollout_chunk="$ROLLOUT_CHUNK"
        dino_wm_alpha="$ALPHA"
        dino_wm_blind_proprio=$([ "$BLIND" = 1 ] && echo true || echo false)
        solver.num_samples="$NUM_SAMPLES"
        seed="$SEED"
        "+output.video_path=$CELL_DIR/videos"
    )
    if [ "$OVERRIDE_VALUE" != "null" ]; then
        HYDRA_OVERRIDES+=("+eval.variation_overrides=${OVERRIDE_VALUE}")
    fi

    echo "== $LABEL s$SEED: launching ($(date))"
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
    echo "== $LABEL s$SEED: SR=$SR elapsed=${ELAPSED}s"
  done
done

echo "All cells done."
