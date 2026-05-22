#!/bin/bash
# Re-evaluate a single (iteration, step) of the 2026-05-06_08-47-29 ASTEROID run.
# Usage: _reeval.sh ITER STEP
# Args:
#   ITER: 1, 2, or 3
#   STEP: 30000, 40000, or 50000 (matches step_NNNNNNN.ckpt filename)
set -euo pipefail
ITER="$1"
STEP="$2"

RUN_DIR=/mnt/storage/lti/UWLab/logs/incontext_exploration_tactile/incontext_tactile_peg_small_512envs_20kdemos_eval64/2026-05-06_08-47-29
CKPT="${RUN_DIR}/iteration_${ITER}/checkpoints/step_$(printf '%07d' "$STEP").ckpt"
if [ ! -f "$CKPT" ]; then
    echo "Missing checkpoint: $CKPT" >&2
    exit 1
fi

STEP_K=$((STEP/1000))
LOG=/mnt/storage/lti/UWLab/.claude_logs/reeval_iter${ITER}_step${STEP_K}k.log
: > "$LOG"
cd /mnt/storage/lti/UWLab
bash .claude_run.sh python scripts_v2/tools/eval_distilled_policy.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0 \
    --seed 0 \
    --num_trajectories 64 \
    --num_envs 16 \
    --headless \
    env.scene.insertive_object=peg \
    --exp_name incontext_tactile_peg_small_reeval_iter${ITER}_step${STEP_K}k \
    --wandb_project incontext_exploration \
    --wandb_group eval_reeval_step${STEP_K}k \
    --checkpoint "$CKPT" \
    --episode_length_s 10.0 \
    --iteration "$ITER" \
    --enable_cameras \
    >> "$LOG" 2>&1
