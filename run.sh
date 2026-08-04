#!/bin/sh

exp_name="real_final2"
sched="fixed"
mkdir -p logs/$exp_name
# 16384
for seed in 1; do
    python run_incontext_exploration.py \
        --data_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-DataCollection-v0 \
        --eval_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-Play-v0 \
        --expert_policy_checkpoint logs/model_voff_curr/exported/policy.pt \
        --num_demos 32768 \
        --num_data_envs 512 \
        --num_eval_envs 1 \
        --num_eval_episodes 20 \
        --config_dir diffusion_policy/diffusion_policy/config \
        --config_name in_context_exploration_tactile_base.yaml \
        --output_dir logs/$exp_name \
        --exp_name $exp_name \
        --insertive_object cube \
        --expert_noise 0.00 \
        --schedule $sched \
        --max_iterations 6 \
        --no_video \
        --seed $seed
done


# # - fast env: OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0

# ckpts=(
#     # model_voff_curr
#     model_voff_nocurr
#     model_von_curr
#     model_von_nocurr
# )

# for ckpt in "${ckpts[@]}"; do
#     python scripts/reinforcement_learning/rsl_rl/play.py \
#         --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0 \
#         --num_envs 1 \
#         --checkpoint logs/$ckpt/$ckpt.pt \
#         --headless \
#         env.scene.insertive_object=cube --max_episodes 1
# done