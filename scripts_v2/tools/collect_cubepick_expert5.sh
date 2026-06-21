#!/bin/sh

python scripts_v2/tools/collect_demos.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0 \
    --dataset_file "logs/dataset-iteration-0-debug/data.zarr" \
    --num_envs 1 \
    --num_demos 5 \
    --headless \
    --seed 0 \
    --min_exploration_horizon 0.0 \
    --max_exploration_horizon 0.0 \
    --episode_length_s 10.0 \
    --expert_noise 0.0 \
    --video \
    --video_length 2000 \
    --video_dir "logs/debug_videos" \
    env.scene.insertive_object=cube \
    'agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["logs/cubepick.pt"]'