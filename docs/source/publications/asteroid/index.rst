ASTEROID
========

| **Code:** ``scripts/ASTEROID`` and ``source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/asteroid``

ASTEROID trains a proprioceptive student policy for cube pick-up by iterating
**in-context exploration** and **distillation**: a state-based RL expert (trained with
the OmniReset recipe) supervises a diffusion-policy student, and from the second
iteration on the previous student acts as an *explorer* for the first part of each
data-collection episode before the expert takes over.

Every iteration runs three stages:

1. **collect** -- roll out the expert (plus the previous student as explorer) in the
   data-collection env and record proprioceptive observations, actions and a per-step
   expert mask (``scripts/ASTEROID/collect_demos_asteroid.py``).
2. **train** -- fit a diffusion-policy student on every dataset collected so far, with
   a per-iteration sampling curriculum (``diffusion_policy/train.py``).
3. **eval** -- roll out the student in the eval env
   (``scripts/ASTEROID/eval_asteroid_policy.py``).

The orchestrator ``scripts/ASTEROID/run_asteroid.py`` organises hyperparameters as a
hierarchy of dataclasses: a ``RunCfg`` holds the run-level settings plus an ordered list
of ``IterationCfg``, each owning the ``CollectCfg`` / ``TrainCfg`` / ``EvalCfg`` for that
iteration. Curricula are functions that build the iteration list (``--schedule``).

----

.. _asteroid-quick-start:

Quick Start
-----------

.. important::

   Make sure you have completed the `installation <https://uw-lab.github.io/UWLab/main/source/setup/installation/pip_installation.html>`_
   before running these commands. The distillation stages additionally need the
   ``diffusion_policy`` submodule.

Environments
^^^^^^^^^^^^

All ASTEROID environments are pick-only variants of the OmniReset UR5e + Robotiq 2F-85
tasks (no receptive object; success = object lifted with the gripper pointing down):

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Task
     - Purpose
   * - ``Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0``
     - Record reset states: object on the table, EE above it
   * - ``Asteroid-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0``
     - Record reset states: object resting, EE grasping it
   * - ``Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0``
     - Record reset states: object anywhere, EE grasping it
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-v0``
     - Train the state expert (Stage 1)
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0``
     - Finetune the expert with sysid / gain curriculum (Stage 2)
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0``
     - Evaluate a Stage 1 expert
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0``
     - Evaluate a Stage 2 expert
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0``
     - Collect student demos with a Stage 1 expert
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-DataCollection-v0``
     - Collect student demos with a Stage 2 expert
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0``
     - Evaluate a student (Stage 1 gains, front camera video)
   * - ``Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-Play-v0``
     - Evaluate a student (Stage 2 gains, front camera video)

Reset-state and grasp datasets are read from a local directory (default ``Datasets/CubePick``,
override with the ``ASTEROID_DATASETS_DIR`` environment variable) keyed by the insertive
object only::

   Datasets/CubePick/Resets/InsertiveCube/resets_ObjectAnywhereEEAnywhere.pt
   Datasets/CubePick/Resets/InsertiveCube/resets_ObjectRestingEEGrasped.pt
   Datasets/CubePick/Resets/InsertiveCube/resets_ObjectAnywhereEEGrasped.pt
   Datasets/CubePick/Grasps/InsertiveCube/grasps.pt

1. Record reset states
^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python scripts_v2/tools/record_reset_states.py \
       --task Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
       --dataset_dir Datasets/CubePick \
       --num_envs 64 --num_reset_states 1000 --headless \
       env.scene.insertive_object=cube

Repeat for ``ObjectRestingEEGrasped`` and ``ObjectAnywhereEEGrasped`` (these two need the
``ObjectAnywhereEEAnywhere`` resets and a grasp dataset; see the OmniReset
:doc:`../omnireset/rl_training` page for grasp sampling).

2. Train the state expert
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/train.py \
       --task Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 \
       --num_envs 4096 --headless \
       env.scene.insertive_object=cube

   # evaluate
   python scripts/reinforcement_learning/rsl_rl/play.py \
       --task Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
       --num_envs 1 --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_asteroid_agent/<run>/model_<n>.pt \
       env.scene.insertive_object=cube

Export the expert to TorchScript (``logs/rsl_rl/.../exported/policy.pt``) as in the OmniReset
:doc:`../omnireset/distillation` page; the data-collection stage loads it with ``torch.jit.load``.

3. Run ASTEROID
^^^^^^^^^^^^^^^

.. code:: bash

   python scripts/ASTEROID/run_asteroid.py \
       --data_task Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0 \
       --eval_task Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0 \
       --expert_policy_checkpoint logs/rsl_rl/ur5e_robotiq_2f85_asteroid_agent/<run>/exported/policy.pt \
       --config_name in_context_exploration_tactile_base.yaml \
       --num_demos 32768 --num_data_envs 512 \
       --num_eval_envs 32 --num_eval_episodes 100 \
       --max_iterations 4 --exp_name cube_asteroid --no_video

Useful flags:

- ``--dry_run`` prints the stage commands without launching Isaac Sim.
- ``--schedule`` selects a curriculum from ``CURRICULA`` in ``run_asteroid.py``.
- ``--start_iteration N --checkpoint_dir <run dir>`` resumes an interrupted run.
- ``--initial_dataset_path`` reuses an existing iteration-0 dataset.

Each run writes ``run_cfg.json`` (the full hyperparameter tree), one
``dataset-iteration-{i}/`` and one ``iteration_{i}/`` (student checkpoints) per iteration.

----

Package layout
--------------

``uwlab_tasks.manager_based.manipulation.asteroid`` mirrors ``omnireset`` and subclasses
it; only the pick-specific deltas live here:

- ``mdp/commands*.py`` -- ``PickTaskCommand``: task command without a receptive object.
- ``mdp/rewards.py`` -- ``ProgressContextPickOnly`` (lift height + gripper-down success) and
  the matching dense / sparse rewards.
- ``mdp/events.py`` -- ``SingleObjectMultiResetManager`` (resets keyed by one object),
  ``randomize_env_cfg_unified`` (coupled sysid / OSC-gain / action-scale DR),
  ``randomize_gripper_pos_affine`` (gripper-reading calibration drift),
  ``reset_root_states_discrete_grid``.
- ``mdp/observations.py`` -- ``gripper_pos_normalized`` (real-robot POS register analogue),
  ``fingertip_contact_force_b``.
- ``mdp/recorders/`` -- per-step expert mask recorder for DAgger-style datasets.
- ``mdp/actions/`` -- position-only (3-DOF + gripper) Cartesian OSC action.
- ``config/ur5e_robotiq_2f85/`` -- reset-state, RL-state and tactile data-collection configs.
