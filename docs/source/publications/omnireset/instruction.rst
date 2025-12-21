Instructions
============

This guide provides step-by-step instructions for using the OmniReset framework.
Choose your path: :ref:`evaluate our pre-trained checkpoints <evaluate-checkpoints>` or :ref:`reproduce training from scratch <reproduce-training>`.

.. note::

   For all commands below, replace ``insertive_object`` and ``receptive_object`` with one of the following:

   * **Drawer Assembly:** ``fbdrawerbottom`` / ``fbdrawerbox``
   * **Twisting:** ``fbleg`` / ``fbtabletop``
   * **Insertion:** ``peg`` / ``peghole``
   * **Rectangle Reorientation on Wall:** ``rectangle`` / ``wall``
   * **Cupcake on Plate:** ``cupcake`` / ``plate``
   * **Cube Stacking:** ``cube`` / ``cube``

   For grasp sampling, replace ``object`` with ``fbleg``, ``fbdrawerbottom``, ``peg``, ``rectangle``, ``cupcake``, or ``cube``.

----

.. _evaluate-checkpoints:

Download and Evaluate Pre-trained Checkpoints
---------------------------------------------

We provide trained RL checkpoints for all three tasks. Download and evaluate them immediately!

Download Checkpoints
^^^^^^^^^^^^^^^^^^^^

Download the pre-trained checkpoints from our Backblaze B2 storage (drawer assembly, leg twisting, peg insertion):

.. code:: bash

   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/fbdrawerbottom_state_rl_expert.pt
   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/fbleg_state_rl_expert.pt
   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/peg_state_rl_expert.pt
   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/rectangle_state_rl_expert.pt
   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cupcake_state_rl_expert.pt
   wget https://s3.us-west-004.backblazeb2.com/uwlab-assets/Policies/OmniReset/cube_state_rl_expert.pt

Evaluate Checkpoints
^^^^^^^^^^^^^^^^^^^^

Run evaluation on the downloaded checkpoints:

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/play.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 1 --checkpoint /path/to/checkpoint.pt env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object


.. _reproduce-training:

Reproduce Our Training
----------------------

Follow these steps to reproduce our training results from scratch. This involves collecting reset state datasets and training RL policies.

Collect Partial Assemblies
^^^^^^^^^^^^^^^^^^^^^^^^^^

Collect partial assembly datasets that will be used for generating reset states.
You can either use existing datasets from Backblaze or collect new ones.

.. code:: bash

   python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 --num_envs 10 --num_trajectories 10 --dataset_dir ./partial_assembly_datasets --headless env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object

.. note::

   This step should take approximately 30 seconds.


Sample Grasp Poses
^^^^^^^^^^^^^^^^^^

Sample grasp poses for the objects. You can either use existing datasets from Backblaze or collect new ones.

.. code:: bash

   python scripts_v2/tools/record_grasps.py --task OmniReset-Robotiq2f85-GraspSampling-v0 --num_envs 8192 --num_grasps 1000 --dataset_dir ./grasp_datasets --headless env.scene.object=object

.. note::

   This step should take approximately 1 minute.


Generate Reset State Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate reset state datasets for different configurations. You can either use existing datasets from Backblaze or collect new ones.

.. important::

   Before running these scripts, make sure ``base_path`` and ``base_paths`` in ``reset_states_cfg.py`` are set appropriately.

Object Anywhere, End-Effector Anywhere (Reaching)
"""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: bash

   python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEAnywhere env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object

Object Resting, End-Effector Grasped (Near Object)
""""""""""""""""""""""""""""""""""""""""""""""""""

.. warning::

   This task depends on reset states from **Object Anywhere, End-Effector Anywhere**. If you are generating your own reset states, make sure to set ``base_paths`` in ``reset_states_cfg.py`` to point to your generated ``ObjectAnywhereEEAnywhere`` dataset directory.

.. code:: bash

   python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectRestingEEGrasped env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object

Object Anywhere, End-Effector Grasped (Grasped)
"""""""""""""""""""""""""""""""""""""""""""""""

.. code:: bash

   python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectAnywhereEEGrasped env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object

Object Partially Assembled, End-Effector Grasped (Near Goal)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code:: bash

   python scripts_v2/tools/record_reset_states.py --task OmniReset-UR5eRobotiq2f85-ObjectPartiallyAssembledEEGrasped-v0 --num_envs 4096 --num_reset_states 10000 --headless --dataset_dir ./reset_state_datasets/ObjectPartiallyAssembledEEGrasped env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object

.. note::

   Each of these steps should take anywhere between 1 minute and 1 hour depending on the task and reset configuration.


Visualize Reset States (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Visualize the generated reset states to verify they are correct.

.. code:: bash

   python scripts_v2/tools/visualize_reset_states.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 4 --dataset_dir /path/to/dataset env.scene.insertive_object=insertive_object env.scene.receptive_object=receptive_object


Train RL Policy
^^^^^^^^^^^^^^^

Train reinforcement learning policies using the generated reset states.

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
       env.scene.insertive_object=insertive_object \
       env.scene.receptive_object=receptive_object

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

Known Issues and Solutions
--------------------------

GLIBCXX Version Error
^^^^^^^^^^^^^^^^^^^^^

If you encounter this error:

.. code-block:: text

   OSError: version `GLIBCXX_3.4.30' not found (required by /path/to/omni/libcarb.so)

Try exporting the system's ``libstdc++`` library:

.. code:: bash

   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
