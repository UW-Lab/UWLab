#!/bin/sh

exp_name="pomdp_tactile_cube_pick"
sched="fixed"
mkdir -p logs/$exp_name

for seed in 0; do
    python run_incontext_exploration.py \
        --data_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0 \
        --eval_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0 \
        --expert_policy_checkpoint logs/exported/policy.pt \
        --num_demos 10000 \
        --num_data_envs 512 \
        --num_eval_envs 1 \
        --num_eval_episodes 50 \
        --config_dir diffusion_policy/diffusion_policy/config \
        --config_name in_context_exploration_tactile_base.yaml \
        --output_dir logs/$exp_name \
        --exp_name $exp_name \
        --insertive_object cube \
        --expert_noise 0.05 \
        --schedule $sched \
        --max_iterations 6 \
        --no_video \
        --seed $seed
done
