Changelog
---------

0.13.8 (2025-10-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Moved hydra utilities from uwlab apps to tasks



0.13.7 (2025-10-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Removed Leap6D robot and it related track environment as it is not used any more



0.13.6 (2025-10-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed the terrain position patching style to be minimal


0.13.5 (2025-09-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed spot augmentation function not working with the most up to date rsl-rl-lib==3.0.1


0.13.4 (2025-03-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* cleaned up spot related envs in velocity tasks


0.13.3 (2025-03-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* fixed the contains relationship does not work for x in env.scene because it doesn't have __contains__
  property, the solution is to use x in env.scene.keys() with # noqa: SIM118 to suppress flake8 complaint


0.13.2 (2025-03-23)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* the named mdp in accordance made in uwlab.envs.mdp for clarity

0.13.1 (2025-01-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Fixed LiftHammer environment to use isaac lab native multi-asset spawning configuration
* Fixed LiftHammer environment to updated tiled camera configuration


0.13.0 (2024-11-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* introducing storage manager as a module in lab_task under :class:`uwlab_apps.utils.storage_manager`


0.12.1 (2024-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* added skrl ppo config for single cake decoration environment

0.12.0 (2024-10-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introducing stateful configuratble skrl training workflow pipeline
* tested to be compatible with current evolution branch

0.11.0 (2024-10-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* merged franka workshop environment, frank multi cake environment into uwlab, Thanks Yufeng!

0.10.0 (2024-10-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* merged skrl workflow pipeline into uwlab

0.9.6 (2024-09-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* The agent experiment name for rsl_rl gym registration was not correct was "rough" now is "terrain_gen"
  changes at :func:`uwlab_tasks.tasks.locomotion.fetching.config.a1.__init__.py`

0.9.5 (2024-09-02)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* in favoring uwlab having reset_from_demostration and record_state_configuration functions
* remove functions :func:`uwlab_tasks.tasks.manipulation.cake_decoration.mdp.reset_from_demostration`
* remove functions :func:`uwlab_tasks.tasks.manipulation.cake_decoration.mdp.record_state_configuration`
* remove functions :func:`uwlab_tasks.tasks.manipulation.clockHand.mdp.reset_from_demostration`
* remove functions :func:`uwlab_tasks.tasks.manipulation.clockHand.mdp.record_state_configuration`


0.9.4 (2024-08-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* adding functions at ``uwlab_tasks.uwlab_apps.utils.cfg_utils.py`` and enable easy rotation
  modification to environments


0.9.3 (2024-08-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* separated out lift objects environment from lift hammer environment at tasks.manipulation.lift_objects

0.9.2 (2024-08-19)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* fixed problem where the order of tycho gripper joint action idex and body joint pos are reversed
  :class:`uwlab_tasks.tasks.manipulation.cake_decoration.config.hebi.tycho_joint_pos.IkdeltaAction`
  and :class:`uwlab_tasks.tasks.manipulation.cake_decoration.config.hebi.tycho_joint_pos.IkabsoluteAction`

0.9.1 (2024-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^^^
* Added necessary mdps for :mod:`uwlab_tasks.tasks.locomotion` tasks

0.9.0 (2024-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* rename unitree_a1, unitree_go1, unitree_go2 to a1, a2, a3 under
  :file:`uwlab_tasks.tasks.locomotion`


0.8.3 (2024-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^
* added terrain_gen environment as separate task in
  :file:`uwlab_tasks.tasks.locomotion.fetching.fetching_terrain_gen_env`

Changed
^^^^^^^
* renamed ``uwlab_tasks.tasks.locomotion.fetching.rough_env_cfg`` to
  ``fetching_env_cfg`` to show its difference from locomotion Velocity tasks


0.8.2 (2024-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^
* added coefficient as input argument in
  :func:`uwlab_tasks.tasks.locomotion.fetching.mdp.rewards.track_interpolated_lin_vel_xy_exp`


0.8.1 (2024-08-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^
* ui_extension is deleted to prevent the buggy import
* :file:`uwlab_tasks.uwlab_tasks.__init__.py` does not import ui_extension


0.8.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fixed
^^^^^
* :file:`uwlab_tasks.uwlab_tasks.__init__.py` did not import tasks folder
  now it is imported


0.8.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^
* updated dependency and meta information to isaac sim 4.1.0



0.7.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^
* added Unitree Go1 Go2 and spot for Fetching task at
  :mod:`uwlab_tasks.tasks.locomotion.fetching`


0.6.1 (2024-07-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* bug fix in logging name unitree a1 agent, flat config should log flat instead of rough at
  at :class:`uwlab_tasks.tasks.locomotion.fetching.config.unitree_a1.agents.rsl_rl_cfg.UnitreeA1FlatPPORunnerCfg`


0.6.0 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* restructured fetching task to new architecture and added Unitree A1
  for fetching task


0.5.2 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* merge all gym registering tasks to one whole name unseparated by "-"
  what used to be 'UW-Lift-Objects-XarmLeap-IkDel-v0' now becomes
  'UW-LiftObjects-XarmLeap-IkDel-v0'

0.5.1 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* support IkDelta action for environment LiftObjectsXarmLeap at
  :mod:`uwlab_tasks.tasks.manipulation.lift_objects`


0.5.0 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* adopting new environment structure for task track_goal


0.4.3 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* fix several minor bugs that introduced when migrating for new environment structure for tasks lift_objects


0.4.2 (2024-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* added fetching task specific reward at :func:`uwlab_tasks.locomotion.fetching.mdp.track_interpolated_lin_vel_xy_exp`
  and :func:`uwlab_tasks.locomotion.fetching.mdp.track_interpolated_ang_vel_z_exp`


0.4.1 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* update track_goal tasks under folder :mod:`uwlab_tasks.tasks.manipulation.track_goal`


0.4.0 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* renaming :mod:`uwlab_tasks.tasks.manipulation.lift_cube` as
  :mod:`uwlab_tasks.tasks.manipulation.lift_objects`
* separates lift_cube and lift_multiobjects as two different environments

* adopting new environment structure for task lift_objects


0.3.0 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* renaming :mod:`uwlab_tasks.tasks.manipulation.craneberryLavaChocoCake` as
  :mod:`uwlab_tasks.tasks.manipulation.cake_decoration`

* adopting new environment structure for task cake_decoration


0.2.3 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* sketched Fetching as a separate locomotion task, instead of being a part of
  :mod:`uwlab_tasks.tasks.locomotion.velocity`


0.2.2 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* dropped dependency of :mod:`uwlab_tasks.cfg` in favor of extension ``uwlab_assets``



0.2.1 (2024-07-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added UW as author and maintainer to :file:`uwlab_tasks.setup.py`

0.2.0 (2024-07-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added support for register gym environment with MultiConstraintDifferentialIKController for leap_hand_xarm at
  :file:`uwlab_tasks.tasks.maniputation.lift_cube.config.leap_hand_xarm.__init__`


0.2.0 (2024-07-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm reward :func:`uwlab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_fingers_object_distance`
* tuned liftCube environment reward function for LeapHandXarm environments
  reward_fingers_object_distance scale was 1.5, now 5
  reward_object_ee_distance scale was 1, now 3
  reward_fingers_object_distance tanh return std was 0.1 now 0.2

0.1.9 (2024-07-13)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm reward :func:`uwlab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_cross_finger_similarity`
* added leap hand xarm reward :func:`uwlab_tasks.cfgs.robots.leap_hand_xarm.mdp.rewards.reward_intra_finger_similarity`
* added leap hand xarm event :func:`uwlab_tasks.cfgs.robots.leap_hand_xarm.mdp.events.reset_joints_by_offset` which accepts
  additional joint ids
* changed cube lift environment cube size to be a bit larger
* added mass randomization cfg in cube lift environment ``uwlab_tasks.tasks.manipulation.lift_cube``


0.1.8 (2024-07-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* added leap hand xarm robot cfg and dynamic at :file:`uwlab_tasks.cfgs.robots.leap_hand.robot_cfg.py` and
  :file:`uwlab_tasks.cfgs.robots.leap_hand_xarm.robot_dynamics.py`
* added environment :file:`uwlab_tasks.tasks.manipulation.lift_cube.track_goal.config.leap_hand_xarm.LeapHandXarm_JointPos_GoalTracking_Env.py`
* added environment :file:`uwlab_tasks.tasks.manipulation.lift_cube.lift_cube.config.leap_hand_xarm.LeapHandXarm_JointPos_LiftCube_Env.py`


0.1.7 (2024-07-08)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Hebi Gravity Enabled now becomes default
* orbid_mdp changed to lab_mdp in :file:`uwlab_tasks.cfgs.robots.leap_hand.robot_dynamics.py`
* Removed Leap hand standard ik absolute and ik delta in :file:`uwlab_tasks.cfgs.robots.leap_hand.robot_dynamics.py`
* Reflect support of RokokoGloveKeyboard in :func:`workflows.teleoperation.teleop_se3_agent_absolute.main`


Added
^^^^^
* Added experiments run script :file:`workflows.experiments.idealpd_experiments.py`
* Added experiments :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.idealpd_scale_experiments.py`


0.1.6 (2024-07-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

memo:
^^^^^

* Termination term should be carefully considered along with the punishment reward functions.
  When there are too many negative reward in the beginning, agent would prefer to die sooner by
  exploiting the termination condition, and this would lead to the agent not learning the task.

* tips:
  When designing the reward function, try be incentive than punishment.

Changed
^^^^^^^

* Changed :class:`uwlab_tasks.cfgs.robots.hebi.robot_dynamics.RobotTerminationsCfg` to include DoneTerm: robot_extremely_bad_posture
* Changed :func:`uwlab_tasks.cfgs.robots.hebi.mdp.terminations.terminate_extremely_bad_posture` to be probabilistic
* Changed :attr:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_position_tracking`
  and :attr:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.Hebi_JointPos_GoalTracking_Env.RewardsCfg.end_effector_orientation_tracking`
  to be incentive reward instead of punishment reward.
* Renamed orbit_mdp to lab_mdp in :file:`uwlab_tasks.tasks.manipulation.track_goal.config.Hebi_JointPos_GoalTracking_Env`

Added
^^^^^

* Added hebi reward term :func:`uwlab_tasks.cfgs.robots.hebi.mdp.rewards.link_orientation_command_error_tanh`
* Added experiments run script :file:`workflows.experiments.strategy4_scale_experiments.py`
* Added experiments :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.strategy4_scale_experiments.py`

0.1.5 (2024-07-06)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Added
^^^^^

* Added experiments run script :file:`workflows.experiments.actuator_experiments.py`
* Added experiments run script :file:`workflows.experiments.agent_update_frequency_experiments.py`
* Added experiments run script :file:`workflows.experiments.decimation_experiments.py`
* Added experiments run script :file:`workflows.experiments.strategy3_scale_experiments.py`
* Added experiments :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.agent_update_rate_experiments.py`
* Added experiments :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.decimation_experiments.py`
* Added experiments :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.strategy3_scale_experiments.py`
* Modified :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.agents.rsl_rl_agent_cfg`, and
  :file:`uwlab_tasks.tasks.manipulation.track_goal.config.hebi.__init__` with logging name consistent to experiments


0.1.4 (2024-07-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* :data:`uwlab_tasks.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY3_CFG`
  :data:`uwlab_tasks.cfgs.robots.hebi.robot_cfg.HEBI_STRATEGY4_CFG`
  changed from manually editing scaling factor to cfg specifying scaling factor.
* :data:`uwlab_tasks.cfgs.robots.hebi.robot_cfg.robot_dynamic`
* :func:`workflows.teleoperation.teleop_se3_agent_absolute.main` added visualization for full gloves data

0.1.3 (2024-06-29)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* updated :func:`workflows.teleoperation.teleop_se3_agent_absolute.main` gloves device to match updated
  requirement needed for rokoko gloves. New version can define port usage, output parts




0.1.2 (2024-06-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Restructured lab to accommodate new extension lab environments
* renamed the repository from lab.tycho to lab.envs
* removed :func:`workflows.teleoperation.teleop_se3_agent_absolute_leap.main` as it has been integrated
  into :func:`workflows.teleoperation.teleop_se3_agent_absolute.main`


0.1.1 (2024-06-27)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* teleoperation absolute ik control for leap hand at :func:`workflows.teleoperation.teleop_se3_agent_absolute_leap.main`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed tycho migration. Done with Tasks: cake, liftcube, clock, meat, Goal Tracking
* Need to check: meat seems to have a bit of issue
* Plan to do: Learn a mujoco motor model, test out dreamerv3, refactorization continue
