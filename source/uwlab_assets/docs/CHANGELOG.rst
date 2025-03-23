Changelog
---------

0.5.1 (2024-10-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* add teleoperation config to leap xarm


0.5.0 (2024-10-20)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* added custom franka setup at :folder:`uwlab_asset.franka`

0.4.1 (2024-09-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Fixed the reversed y axis in Tycho Teleop Cfg at :const:`uwlab_asset.tycho.action.TELEOP_CFG`

0.4.0 (2024-09-01)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Introduced Teleop Cfg in robot tycho and franka

0.3.3 (2024-08-24)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* fixed the device inconsistency in :func:`uwlab_asset.tycho.mdp.termination:terminate_extremely_bad_posture`
  where the device is hardcoded as "cuda" instead of "env.device"

0.3.2 (2024-08-19)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* renamed HEBI_ORIBIT_ARTICULATION, and HEBI_CUSTOM_ARTICULATION to HEBI_ARTICULATION
  at :file:`uwlab_asset.tycho.tycho.py` since they are all same.


0.3.1 (2024-08-19)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* fixed problem where the order of tycho gripper joint action idex and body joint pos are reversed
  :class:`uwlab_asset.tycho.actions.IkdeltaAction` and :class:`uwlab_asset.tycho.actions.IkabsoluteAction`

0.3.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* updated dependency and meta information to isaac sim 4.1.0


0.2.0 (2024-07-29)
~~~~~~~~~~~~~~~~~~

Added
^^^^^^^

* Created new folder storing :class:`uwlab_asset.unitree` extensions



0.1.3 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Bug fix at :const:`uwlab_assets.robots.leap.actions.LEAP_JOINT_POSITION`
  and :const:`uwlab_assets.robots.leap.actions.LEAP_JOINT_EFFORT` because
  previous version did not include all joint name. it used to be
  ``joint_names=["j.*"]`` now becomes ``joint_names=["w.*", "j.*"]``




0.1.2 (2024-07-27)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing ``uwlab_asset.anymal``
* Created new folder storing ``uwlab_asset.leap``


0.1.1 (2024-07-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing ``uwlab_asset.tycho``


0.1.0 (2024-07-25)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created new folder storing ``uwlab_asset``
