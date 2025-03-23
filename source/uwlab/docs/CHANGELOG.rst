Changelog
---------

0.8.0 (2024-11-14)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* renamed and standardized genome, gene in :file:`uwlab.genes.genome`


0.7.2 (2024-11-10)
~~~~~~~~~~~~~~~~~~

Removed
^^^^^^^

* Deprecating :folder:`uwlab.envs.assets.deformable` related deformable modules


0.7.1 (2024-10-24)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* fixed :class:`uwlab.device.RokokoGloveT265` to support up to date teleoperation pipeline
* fixed :class:`uwlab.device.RokokoGloveKeyboard` to support up to date teleoperation pipeline

0.7.0 (2024-10-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* added :file:`uwlab.scene.large_scene_cfg` to support importing prebuilt scene in usd that has
* configured articulation, rigidbody, deformable, and so on.


0.6.1 (2024-10-20)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Dropping DeformableInteractiveScene from :file:`uwlab.scene.deformable_interactive_scene_cfg` as
official deformable has been added to Isaac Lab


0.6.0 (2024-10-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* support encodable terrain in :file:`uwlab.lab.terrains.descriptive_terrain`
* with current encoding to be behavior and terrain type with an example located at
* :file:`uwlab.lab.terrains.config.descriptive_terrain`

0.5.5 (2024-09-09)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* adding property is_closed to :class:`uwlab.envs.UWManagerBasedRl`

0.5.4 (2024-09-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Transferred Experimental Evolution code into lab extension as :dir:`uwlab.evolution_system`

0.5.3 (2024-09-02)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adding event that reset from demonstration :func:`uwlab.envs.mdp.events.reset_from_demonstration`
* Adding event that record state of simulation:func:`uwlab.envs.mdp.events.record_state_configuration`

0.5.2 (2024-09-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Adding event that make viewport camera follows robot at :func:`uwlab.envs.mdp.events.viewport_follow_robot`


0.5.1 (2024-08-23)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bug fix for :func:`uwlab.envs.UWManagerBasedRl.step` where data manager existence is not queried correctly


0.5.0 (2024-08-06)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added features that support obj typed sub-terrain, and custom supply of the spawning locations
  please check :folder:`uwlab.lab.terrains`


0.4.3 (2024-08-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed :file:`uwlab.terrains.enhanced_terrain_importer.py` as it is ended up not being a solution


0.4.2 (2024-08-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed :func:`uwlab.envs.mdp.events.reset_root_state_uniform` instead, reset_root_state_uniform is imported
  from isaac lab


0.4.1 (2024-08-06)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bug fix for :func:`uwlab.envs.UWManagerBasedRl.close` is self.extensions not self.extension


0.4.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* updated dependency and meta information to isaac sim 4.1.0


0.3.0 (2024-07-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^
Added experiment feature categorical command type for commanding anything that can be represented
by integer at :folder:`uwlab.envs.mdp.commands`


0.2.7 (2024-07-28)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* MultiConstraintDifferentialIKController now supports multi environments parallel computes
* added reward :func:`uwlab.envs.mdp.position_command_error`
* added reward :func:`uwlab.envs.mdp.position_command_error_tanh`
* added reward :func:`uwlab.envs.mdp.orientation_command_error_tanh`
* removed :func:`uwlab.envs.mdp.track_interpolated_lin_vel_xy_exp` as this is fetching task specific
* removed :func:`uwlab.envs.mdp.track_interpolated_ang_vel_z_exp` as this is fetching task specific


0.2.6 (2024-07-27)
~~~~~~~~~~~~~~~~~~
Added
^^^^^
* Added reward term :func:`uwlab.envs.mdp.reward_body1_body2_within_distance` for reward proximity
  two objects proximity

Changed
^^^^^^^
* Updating default rough terrain tiling configuration at :class:`uwlab.terrains.config`


0.2.5 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* Removed dependency on ``import os`` to support custom extension in :class:`uwlab.actuators.EffortMotor`


0.2.4 (2024-07-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^
* Changed :class:`uwlab.actuators.EffortMotor` inherites and uses super classes stiffness,
  damping, effort limit instead of redefining a redundant field as of :class:`uwlab.actuators.HebiEffortMotor`

* Changed : :class:`uwlab.actuators.EffortMotorCfg` added to support above change

* Changed : :class:`uwlab.actuators.__init__` added to support above change


0.2.3 (2024-07-20)
~~~~~~~~~~~~~~~~~~


Added
^^^^^
* Added debug :func:`uwlab.devices.RokokoGloveKeyboard.debug_advance_all_joint_data.`
  for glove data visualization

Changed
^^^^^^^
* Changed :class:`uwlab.devices.RokokoGloveKeyboard.` class requires
  input initial command pose to correctly set robot reset command target

* Edited Thumb scaling input in :class:`uwlab.devices.RokokoGlove` that corrects
  thumb length mismatch in teleoperation


0.2.2 (2024-07-15)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^
* Changed :func:`uwlab.sim.spawners.from_files.from_files_cfg.MultiAssetCfg` to support
  multi objects scaling.
* Changed :func:`uwlab.sim.spawners.from_files.from_files.spawn_multi_object_randomly_sdf`
  to support multi objects scaling.


0.2.1 (2024-07-14)
~~~~~~~~~~~~~~~~~~


Added
^^^^^
* UW lab now support multi assets spawning
* Added :func:`uwlab.sim.spawners.from_files.from_files.spawn_multi_object_randomly_sdf`
  and :func:`uwlab.sim.spawners.from_files.from_files.spawn_multi_object_randomly`
* Added :func:`uwlab.sim.spawners.from_files.from_files_cfg.MultiAssetCfg`


0.2.0 (2024-07-10)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Added Reward Term :func:`uwlab.envs.mdp.rewards.reward_body1_frame2_distance`
* Let Keyboard device accepts initial transform pose input :class:`uwlab.devices.Se3Keyboard`


0.1.9 (2024-07-10)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Documented :class:`uwlab.controllers.MultiConstraintDifferentialIKController`,
  :class:`uwlab.controllers.MultiConstraintDifferentialIKControllerCfg`


0.1.8 (2024-07-09)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Documented :class:`uwlab.devices.RokokoGlove`,
  :class:`uwlab.devices.RokokoGloveKeyboard`, :class:`uwlab.devices.Se3Keyboard`



0.1.7 (2024-07-08)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* Added proximal distance scaling in :class:`uwlab.devices.rokoko_glove.RokokoGlove`
* Fixed the order checking for the :class:`uwlab.controllers.differential_ik.MultiConstraintDifferentialIKController`


Added
^^^^^
* Added combined control that separates pose and finger joints in
  :class:`uwlab.devices.rokoko_glove_keyboard.RokokoGloveKeyboard`


0.1.6 (2024-07-06)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^

* :class:`uwlab.actuators.actuator_cfg.HebiStrategy3ActuatorCfg` added the field that scales position_p and effort_p
* :class:`uwlab.actuators.actuator_cfg.HebiStrategy4ActuatorCfg` added the field that scales position_p and effort_p
* :class:`uwlab.actuators.actuator_pd.py.HebiStrategy3Actuator` reflected the field that scales position_p and effort_p
* :class:`uwlab.actuators.actuator_pd.py.HebiStrategy4Actuator` reflected the field that scales position_p and effort_p
* Improved Reuseability :class:`uwlab.devices.rokoko_udp_receiver.Rokoko_Glove` such that the returned joint position respects the
order user inputs. Added debug visualization. Plan to add scale by knuckle width to match the leap hand knuckle width

0.1.5 (2024-07-04)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^
* :meth:`uwlab.envs.data_manager_based_rl.step` the actual environment update rate now becomes
decimation square, as square allows a nice property that tuning decimation creates minimal effect on the learning
behavior.


0.1.4 (2024-06-29)
~~~~~~~~~~~~~~~~~~


Changed
^^^^^^^
* allow user input specific tracking name :meth:`uwlab.device.rokoko_udp_receiver.Rokoko_Glove.__init__` to address
  inefficiency when left or right has tracking is unnecessary, and future need in increasing, decreasing number of track
  parts with ease. In addition, the order which parts are outputted is now ordered by user's list input, removing the need
  of manually reorder the output when the output is fixed

0.1.3 (2024-06-28)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`uwlab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* cleaned, memory preallocated :class:`uwlab.device.rokoko_udp_receiver.Rokoko_Glove` so it is much more readable and efficient


0.1.2 (2024-06-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :class:`uwlab.envs.mdp.actions.MultiConstraintsDifferentialInverseKinematicsActionCfg`


Changed
^^^^^^^
* Removed duplicate functions in :class:`uwlab.envs.mdp.actions.actions_cfg` already defined in Isaac lab
* Removed :file:`uwlab.envs.mdp.actions.binary_joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`uwlab.envs.mdp.actions.joint_actions.py` as it completely duplicates Isaac lab implementation
* Removed :file:`uwlab.envs.mdp.actions.non_holonomic_actions.py` as it completely duplicates Isaac lab implementation
* Cleaned :class:`uwlab.controllers.differential_ik.DifferentialIKController`

0.1.1 (2024-06-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Rokoko smart glove device reading
* separation of :class:`uwlab.envs.mdp.actions.MultiConstraintDifferentialInverseKinematicsAction`
  from :class:`isaaclab.envs.mdp.actions.DifferentialInverseKinematicsAction`

* separation of :class:`uwlab.envs.mdp.actions.MultiConstraintDifferentialIKController`
  from :class:`isaaclab.envs.mdp.actions.DifferentialIKController`

* separation of :class:`uwlab.envs.mdp.actions.MultiConstraintDifferentialIKControllerCfg`
  from :class:`isaaclab.envs.mdp.actions.DifferentialIKControllerCfg`


Changed
^^^^^^^
* Changed :func:`uwlab.envs.mdp.events.reset_tycho_to_default` to :func:`uwlab.envs.mdp.events.reset_robot_to_default`
* Changed :func:`uwlab.envs.mdp.events.update_joint_positions` to :func:`uwlab.envs.mdp.events.update_joint_target_positions_to_current`
* Removed unnecessary import in :class:`uwlab.envs.mdp.events`
* Removed unnecessary import in :class:`uwlab.envs.mdp.rewards`
* Removed unnecessary import in :class:`uwlab.envs.mdp.terminations`


Updated
^^^^^^^

* Updated :meth:`uwlab.envs.DeformableBasedEnv.__init__` up to date with :meth:`isaaclab.envs.ManagerBasedEnv.__init__`
* Updated :class:`uwlab.envs.HebiRlEnvCfg` to :class:`uwlab.envs.UWManagerBasedRlCfg`
* Updated :class:`uwlab.envs.HebiRlEnv` to :class:`uwlab.envs.UWManagerBasedRl`


0.1.0 (2024-06-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Performed uwlab refactorization. Tested to work alone, and also with tycho
* Updated README Instruction
* Plan to do: check out not duplicate logic, clean up this repository.
