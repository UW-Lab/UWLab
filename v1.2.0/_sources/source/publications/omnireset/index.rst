OmniReset
=========

| **Paper:** `Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning (ICLR 2026) <https://arxiv.org/abs/2603.15789>`_
| **Project website:** `omnireset.github.io <https://omnireset.github.io/>`_

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
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/leg.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. tab-set::

         .. tab-item:: Seed 42

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/leg_state_rl_expert_seed42.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint leg_state_rl_expert_seed42.pt \
                   env.scene.insertive_object=fbleg \
                   env.scene.receptive_object=fbtabletop

         .. tab-item:: Seed 0

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/leg_state_rl_expert_seed0.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint leg_state_rl_expert_seed0.pt \
                   env.scene.insertive_object=fbleg \
                   env.scene.receptive_object=fbtabletop

         .. tab-item:: Seed 1

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/leg_state_rl_expert_seed1.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint leg_state_rl_expert_seed1.pt \
                   env.scene.insertive_object=fbleg \
                   env.scene.receptive_object=fbtabletop

   .. tab-item:: Drawer Assembly

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/drawer.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. tab-set::

         .. tab-item:: Seed 42

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/drawer_state_rl_expert_seed42.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint drawer_state_rl_expert_seed42.pt \
                   env.scene.insertive_object=fbdrawerbottom \
                   env.scene.receptive_object=fbdrawerbox

         .. tab-item:: Seed 0

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/drawer_state_rl_expert_seed0.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint drawer_state_rl_expert_seed0.pt \
                   env.scene.insertive_object=fbdrawerbottom \
                   env.scene.receptive_object=fbdrawerbox

         .. tab-item:: Seed 1

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/drawer_state_rl_expert_seed1.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint drawer_state_rl_expert_seed1.pt \
                   env.scene.insertive_object=fbdrawerbottom \
                   env.scene.receptive_object=fbdrawerbox

   .. tab-item:: Peg Insertion

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/peg.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. tab-set::

         .. tab-item:: Seed 42

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed42.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint peg_state_rl_expert_seed42.pt \
                   env.scene.insertive_object=peg \
                   env.scene.receptive_object=peghole

         .. tab-item:: Seed 0

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed0.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint peg_state_rl_expert_seed0.pt \
                   env.scene.insertive_object=peg \
                   env.scene.receptive_object=peghole

         .. tab-item:: Seed 1

            .. code:: bash

               wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/peg_state_rl_expert_seed1.pt

               python scripts/reinforcement_learning/rsl_rl/play.py \
                   --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
                   --num_envs 1 \
                   --checkpoint peg_state_rl_expert_seed1.pt \
                   env.scene.insertive_object=peg \
                   env.scene.receptive_object=peghole

   .. tab-item:: Rectangle on Wall

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/rectangle.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/rectangle_state_rl_expert_seed0.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint rectangle_state_rl_expert_seed0.pt \
             env.scene.insertive_object=rectangle \
             env.scene.receptive_object=wall

   .. tab-item:: Cube Stacking

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/cube.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/cube_state_rl_expert_seed42.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint cube_state_rl_expert_seed42.pt \
             env.scene.insertive_object=cube \
             env.scene.receptive_object=cube

   .. tab-item:: Cupcake on Plate

      .. raw:: html

         <div style="text-align: center; margin-bottom: 20px;">
           <video width="400" height="300" controls>
             <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/cupcake.mp4" type="video/mp4">
             Your browser does not support the video tag.
           </video>
         </div>

      .. code:: bash

         # Download checkpoint
         wget https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Policies/OmniReset/state_based_experts/cupcake_state_rl_expert_seed42.pt

         # Run evaluation
         python scripts/reinforcement_learning/rsl_rl/play.py \
             --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
             --num_envs 1 \
             --checkpoint cupcake_state_rl_expert_seed42.pt \
             env.scene.insertive_object=cupcake \
             env.scene.receptive_object=plate

----

.. _full-pipeline:

Full Pipeline
-------------

The full OmniReset pipeline from custom task creation to real-robot deployment:

.. raw:: html

   <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin: 24px 0; flex-wrap: nowrap; font-size: 0.85em;">
     <div style="background: #e3f2fd; border: 2px solid #1976d2; border-radius: 8px; padding: 8px 12px; text-align: center; white-space: nowrap;">
       <strong>1. Create New Task</strong><br><span style="font-size: 0.85em; color: #555;">assets &amp; variants</span>
     </div>
     <div style="font-size: 1.3em; color: #888;">&rarr;</div>
     <div style="background: #e8f5e9; border: 3px solid #2e7d32; border-radius: 8px; padding: 8px 12px; text-align: center; white-space: nowrap; box-shadow: 0 0 8px rgba(46,125,50,0.3);">
       <strong style="font-size: 1.1em;">2. Train RL Policy</strong><br><span style="font-size: 0.85em; color: #555;">resets &amp; training</span><br><span style="font-size: 0.75em; color: #2e7d32;">&#9733; most users start here</span>
     </div>
     <div style="font-size: 1.3em; color: #888;">&rarr;</div>
     <div style="background: #fff3e0; border: 2px solid #f57c00; border-radius: 8px; padding: 8px 12px; text-align: center; white-space: nowrap;">
       <strong>3. Sys-ID &amp; Finetune</strong><br><span style="font-size: 0.85em; color: #555;">sim2real alignment</span>
     </div>
     <div style="font-size: 1.3em; color: #888;">&rarr;</div>
     <div style="background: #f3e5f5; border: 2px solid #7b1fa2; border-radius: 8px; padding: 8px 12px; text-align: center; white-space: nowrap;">
       <strong>4. Distill &amp; Deploy</strong><br><span style="font-size: 0.85em; color: #555;">vision policy &amp; real robot</span>
     </div>
   </div>

.. tip::

   **Most users only need step 2.** If you're training on one of our 6 existing tasks, jump straight to :doc:`rl_training`.

- :doc:`new_task` -- Prepare USD assets, register object variants, verify in sim.
- :doc:`rl_training` -- Collect reset states and train an RL policy from scratch. **Start here for most use cases.**
- :doc:`sim2real` -- Robot calibration & USD, system identification, camera calibration, then ADR finetuning, or use our pre-finetuned checkpoints.
- :doc:`distillation` -- Evaluate pretrained RGB checkpoints, or collect demos and train your own ResNet18-MLP vision policy. Deploy on real robot.

.. toctree::
   :maxdepth: 1
   :caption: Pipeline

   new_task
   rl_training
   sim2real
   distillation

----

Compute & Hardware Requirements
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Stage
     - Requirements
   * - Policy evaluation
     - 1 GPU.
   * - RL training
     - 4 GPUs, 24+ GB VRAM each (e.g. L40S, 4090). Cube/Peg converge in ~8 hours on 4x L40S.
   * - RL finetuning
     - 1--4 GPUs depending on task (see :doc:`sim2real` for per-task env counts). Peg converges in ~8 hours on 1x L40S.
   * - Demo collection
     - 1 RTX GPU, 24+ GB VRAM (32 envs fit on an RTX 4090). 10K demos ~2 hours.
   * - Vision policy training
     - 1 GPU. ~2 days of training on a H200 for transfer. ~1 day of training on a H200 for sim-only distillation.
   * - Real-robot deploy
     - UR5e/UR7e + Robotiq 2F-85 + 3x Intel RealSense (D415/D435/D455).

----

BibTeX
------
.. code:: bibtex

   @inproceedings{
      yin2026omnireset,
      title={Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning},
      author={Patrick Yin and Tyler Westenbroek and Zhengyu Zhang and Joshua Tran and Ignacio Dagnino and Eeshani Shilamkar and Numfor Mbiziwo-Tiapo and Simran Bagaria and Xinlei Liu and Galen Mullins and Andrey Kolobov and Abhishek Gupta},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://arxiv.org/abs/2603.15789}
   }
