#!/bin/sh

ckpts=(
    model_voff_curr
    # model_voff_nocurr
    model_von_curr
    # model_von_nocurr
)

envs=(
    Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-DataCollection-v0
    # Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0
    Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-DataCollection-v0
    # Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0
)

for i in "${!ckpts[@]}"; do
    ckpt="${ckpts[$i]}"
    env="${envs[$i]}"
    python scripts/ASTEROID/collect_demos_asteroid.py \
        --task $env \
        --dataset_file "logs/debug/dataset-iteration-0-$ckpt/data.zarr" \
        --num_envs 1 \
        --num_demos 10 \
        --headless \
        --seed 0 \
        --min_exploration_horizon 0.0 \
        --max_exploration_horizon 0.0 \
        --episode_length_s 10.0 \
        --expert_noise 0.0 \
        --video \
        --video_length 2000 \
        --video_dir "logs/$ckpt/debug_video" \
        env.scene.insertive_object=cube \
        agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["logs/$ckpt/exported/policy.pt"]
done
# python scripts/ASTEROID/collect_demos_asteroid.py \
#     --task Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0 \
#     --dataset_file "logs/dataset-iteration-0-debug/data.zarr" \
#     --num_envs 1 \
#     --num_demos 5 \
#     --headless \
#     --seed 0 \
#     --min_exploration_horizon 0.0 \
#     --max_exploration_horizon 0.0 \
#     --episode_length_s 10.0 \
#     --expert_noise 0.0 \
#     --video \
#     --video_length 2000 \
#     --video_dir "logs/debug_videos" \
#     env.scene.insertive_object=cube \
#     'agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["logs/exported/policy.pt"]'