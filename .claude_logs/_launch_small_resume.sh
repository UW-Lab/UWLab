#!/bin/bash
cd /mnt/storage/lti/UWLab
LOG=/mnt/storage/lti/UWLab/.claude_logs/incontext_tactile_peg_small_512envs_20kdemos_eval64_resume.log
: > "$LOG"
bash .claude_run.sh python run_incontext_exploration.py \
    --num_demos 20000 \
    --num_data_envs 512 \
    --num_eval_envs 16 \
    --num_eval_episodes 64 \
    --max_iterations 4 \
    --schedule fixed \
    --no_video \
    --start_iteration 1 \
    --checkpoint_dir logs/incontext_exploration_tactile/incontext_tactile_peg_small_512envs_20kdemos_eval64/2026-05-05_23-46-00 \
    --config_name in_context_exploration_tactile_base.yaml \
    --output_dir logs/incontext_exploration_tactile \
    --insertive_object peg \
    --expert_policy_checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/2026-05-02_01-43-39/exported/policy.pt \
    --exp_name incontext_tactile_peg_small_512envs_20kdemos_eval64 \
    >> "$LOG" 2>&1
