#!/bin/sh

python scripts_v2/tools/record_reset_states.py \
    --task Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
    --dataset_dir ./Datasets/CubePick \
    --reset_type ObjectAnywhereEEAnywhere \
    --num_envs 64 \
    --num_reset_states 1000 \
    --headless \
    env.scene.insertive_object=cube
