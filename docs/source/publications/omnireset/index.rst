OmniReset
=========

**OmniReset** is a robotic manipulation framework using RL to solve dexterous, contact-rich manipulation tasks without reward engineering or demos.

.. note::
   Detailed documentation will be updated following the public release of the paper.

----

.. _quick-start:

Quick Start (Try in 2 Minutes)
------------------------------

.. important::

   Make sure you have completed the `installation <https://uw-lab.github.io/UWLab/main/source/setup/installation/pip_installation.html>`_ before running these commands.

Download our pretrained checkpoint and run evaluation.

.. tab-set::

   .. tab-item:: Leg Twisting

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/leg.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/fbleg_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint fbleg_state_rl_expert.pt \
             env.scene.insertive_object=fbleg \
             env.scene.receptive_object=fbtabletop

   .. tab-item:: Drawer Assembly

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/drawer.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/fbdrawerbottom_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint fbdrawerbottom_state_rl_expert.pt \
             env.scene.insertive_object=fbdrawerbottom \
             env.scene.receptive_object=fbdrawerbox

   .. tab-item:: Peg Insertion

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/peg.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/peg_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint peg_state_rl_expert.pt \
             env.scene.insertive_object=peg \
             env.scene.receptive_object=peghole

   .. tab-item:: Rectangle on Wall

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/rectangle.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/rectangle_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint rectangle_state_rl_expert.pt \
             env.scene.insertive_object=rectangle \
             env.scene.receptive_object=wall

   .. tab-item:: Cube Stacking

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/cube.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cube_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint cube_state_rl_expert.pt \
             env.scene.insertive_object=cube \
             env.scene.receptive_object=cube

   .. tab-item:: Cupcake on Plate

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://s3.us-west-004.backblazeb2.com/uwlab-assets/Media/OmniReset/cupcake.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cupcake_state_rl_expert.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint cupcake_state_rl_expert.pt \
             env.scene.insertive_object=cupcake \
             env.scene.receptive_object=plate

----

.. _reproduce-training:

Reproduce Our Training
----------------------

Reproduce our training results from scratch.

.. tip::

   **Want to try it quickly?** Start with **Cube Stacking** or **Peg Insertion**. They have the fastest reset state collection times and converge within ~8 hours on 4×L40S GPUs.

.. tab-set::

   .. tab-item:: Leg Twisting

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=fbleg

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=fbleg env.scene.receptive_object=fbtabletop

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=fbleg \
             env.scene.receptive_object=fbtabletop

   .. tab-item:: Drawer Assembly

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=fbdrawerbottom

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=fbdrawerbottom env.scene.receptive_object=fbdrawerbox

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=fbdrawerbottom \
             env.scene.receptive_object=fbdrawerbox

   .. tab-item:: Peg Insertion

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=peg env.scene.receptive_object=peghole

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=peg

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=peg env.scene.receptive_object=peghole

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=peg env.scene.receptive_object=peghole

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=peg env.scene.receptive_object=peghole

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=peg env.scene.receptive_object=peghole

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=peg \
             env.scene.receptive_object=peghole

   .. tab-item:: Rectangle on Wall

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=rectangle env.scene.receptive_object=wall

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=rectangle

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=rectangle env.scene.receptive_object=wall

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=rectangle env.scene.receptive_object=wall

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=rectangle env.scene.receptive_object=wall

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=rectangle env.scene.receptive_object=wall

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=rectangle \
             env.scene.receptive_object=wall

   .. tab-item:: Cube Stacking

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=cube env.scene.receptive_object=cube

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=cube

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=cube env.scene.receptive_object=cube

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=cube env.scene.receptive_object=cube

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=cube env.scene.receptive_object=cube

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=cube env.scene.receptive_object=cube

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=cube \
             env.scene.receptive_object=cube

   .. tab-item:: Cupcake on Plate

      **Step 1: Collect Partial Assemblies** (~30 seconds)

      .. code:: bash

         python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=cupcake env.scene.receptive_object=plate

      **Step 2: Sample Grasp Poses** (~1 minute)

      .. code:: bash

         python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=cupcake

      **Step 3: Generate Reset State Datasets** (~1 min to 1 hour depending on the reset)

      .. important::

         Before running, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

      .. code:: bash

         # Object Anywhere, End-Effector Anywhere (Reaching)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=cupcake env.scene.receptive_object=plate

         # Object Resting, End-Effector Grasped (Near Object)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=cupcake env.scene.receptive_object=plate

         # Object Anywhere, End-Effector Grasped (Grasped)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=cupcake env.scene.receptive_object=plate

         # Object Partially Assembled, End-Effector Grasped (Near Goal)
         python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=cupcake env.scene.receptive_object=plate

      **Step 4: Train RL Policy**

      .. code:: bash

         python -m torch.distributed.run \
             --nnodes 1 \
             --nproc_per_node 4 \
             scripts/reinforcement_learning/rsl_rl/train.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
             --num_envs 16384 \
             --logger wandb \
             --headless \
             --distributed \
             env.scene.insertive_object=cupcake \
             env.scene.receptive_object=plate

Training Curves
^^^^^^^^^^^^^^^

Below are success rate curves for each task plotting over number of training iterations and wall clock time when training on 4xL40S GPUs.
Insertion, twisting, cube stacking, and rectangle orientation on wall tasks converge within **8 hours**, while drawer assembly and cupcake on plate tasks take **1 day**.

.. list-table::
   :widths: 50 50
   :class: borderless

   * - .. figure:: ../../../source/_static/publications/omnireset/success_rate_over_steps.jpg
          :width: 100%
          :alt: Training curve over steps

          Success Rate of 6 Tasks Over Number of Training Iterations

     - .. figure:: ../../../source/_static/publications/omnireset/success_rate_over_wall_clock.jpg
          :width: 100%
          :alt: Training curve over wall clock time

          Success Rate of 6 Tasks Over Wall Clock Time

----
