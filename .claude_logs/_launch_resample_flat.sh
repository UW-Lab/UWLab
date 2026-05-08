#!/bin/bash
cd /mnt/storage/lti/UWLab
LOG=/mnt/storage/lti/UWLab/.claude_logs/resample_small_flat.log
: > "$LOG"
bash .claude_run.sh python scripts_v2/tools/record_reset_states.py \
    --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 32 \
    --num_reset_states 10 \
    --dataset_dir reset_states_dataset_small \
    --headless \
    >> "$LOG" 2>&1
