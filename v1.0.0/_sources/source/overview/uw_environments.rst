.. _environments:

Available UW Environments
===========================

The following lists comprises of all the RL tasks implementations that are available in UW Lab.
While we try to keep this list up-to-date, you can always get the latest list of environments by
running the following command:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         python scripts/environments/list_envs.py

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: batch

         python scripts\environments\list_envs.py


Single-agent
------------

Manipulation
~~~~~~~~~~~~

Environments based on fixed-arm manipulation tasks.

.. table::
    :widths: 33 37 30

    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | World                          | Environment ID                                 | Description                                                                  |
    +================================+================================================+==============================================================================+
    | |track-goal-ur5|               | |track-goal-ur5-link|                          | Goal tracking with Ur5 robot with Robotiq gripper                            |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | |track-goal-tycho|             | |track-goal-tycho-link|                        | Goal tracking with Tycho robot                                               |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | |track-goal-xarm-leap|         | |track-goal-xarm-leap-link|                    | Goal tracking with Xarm with Leap Hand robot                                 |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | |ext-nut-thread-franka|        | |ext-nut-thread-franka-link|                   | Threading nut on to bolt on nist board                                       |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | |ext-gear-mesh-franka|         | |ext-gear-mesh-franka-link|                    | Inserting gear on to gear base on nist board                                 |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+
    | |ext-peg-insert-franka|        | |ext-peg-insert-franka-link|                   | Inserting peg rod into hole on nist board                                    |
    +--------------------------------+------------------------------------------------+------------------------------------------------------------------------------+

.. |track-goal-ur5| image:: ../_static/tasks/manipulation/ur5_track_goal.jpg
.. |track-goal-tycho| image:: ../_static/tasks/manipulation/tycho_track_goal.jpg
.. |track-goal-xarm-leap| image:: ../_static/tasks/manipulation/xarm_leap_track_goal.jpg
.. |ext-nut-thread-franka| image:: ../_static/tasks/manipulation/factory_ext/nut_thread_ext.jpg
.. |ext-gear-mesh-franka| image:: ../_static/tasks/manipulation/factory_ext/gear_mesh_ext.jpg
.. |ext-peg-insert-franka| image:: ../_static/tasks/manipulation/factory_ext/peg_insert_ext.jpg

.. |track-goal-ur5-link| replace:: `UW-Track-Goal-Ur5-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/ur5/track_goal_ur5_env_cfg.py>`__
.. |track-goal-tycho-link| replace:: `UW-Track-Goal-Tycho-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/tycho/tycho_track_goal.py>`__
.. |track-goal-xarm-leap-link| replace:: `UW-Track-Goal-Xarm-Leap-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/track_goal/config/xarm_leap/track_goal_xarm_leap.py>`__
.. |ext-nut-thread-franka-link| replace:: `UW-Nut-Thread-Franka-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/factory_extension/nutthread_env_cfg.py>`__
.. |ext-gear-mesh-franka-link| replace:: `UW-Gear-Mesh-Franka-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/factory_extension/gearmesh_env_cfg.py>`__
.. |ext-peg-insert-franka-link| replace:: `UW-Peg-Insert-Franka-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/factory_extension/peginsert_env_cfg.py>`__

Locomotion
~~~~~~~~~~

Environments based on legged locomotion tasks.

.. table::
    :widths: 33 37 30

    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | World                          | Environment ID                               | Description                                                                  |
    +================================+==============================================+==============================================================================+
    | |position-gap-spot|            | |position-gap-spot-link|                     | Track a position command on gap terrain with the Spot robot                  |
    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |position-pit-spot|            | |position-pit-spot-link|                     | Track a position command on pit terrain with the Spot robot                  |
    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |position-stepping-stone-spot| | |position-stepping-stone-spot-link|          | Track a position command on stepping stone terrain with the Spot robot       |
    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |position-inv-slope-spot|      | |position-inv-slope-spot-link|               | Track a position command on inverse slope terrain with the Spot robot        |
    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+
    | |position-obstacle-spot|       | |position-obstacle-spot-link|                | Track a position command on obstacle terrain with the Spot robot             |
    +--------------------------------+----------------------------------------------+------------------------------------------------------------------------------+

.. |position-gap-spot-link| replace:: `UW-Position-Gap-Spot-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py>`__
.. |position-pit-spot-link| replace:: `UW-Position-Pit-Spot-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py>`__
.. |position-stepping-stone-spot-link| replace:: `UW-Position-Stepping-Stone-Spot-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/risky_terrains/config/spot/spot_env_cfg.py>`__
.. |position-obstacle-spot-link| replace:: `UW-Position-Obstacle-Spot-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py>`__
.. |position-inv-slope-spot-link| replace:: `UW-Position-Inv-Slope-Spot-v0 <https://github.com/UW-Lab/UWLab/blob/main/source/uwlab_tasks/uwlab_tasks/manager_based/locomotion/advance_skills/config/spot/spot_env_cfg.py>`__

.. |position-gap-spot| image:: ../_static/tasks/locomotion/spot_gap.jpg
.. |position-pit-spot| image:: ../_static/tasks/locomotion/spot_pit.jpg
.. |position-stepping-stone-spot| image:: ../_static/tasks/locomotion/spot_stepping_stone.jpg
.. |position-obstacle-spot| image:: ../_static/tasks/locomotion/spot_obstacle.jpg
.. |position-inv-slope-spot| image:: ../_static/tasks/locomotion/spot_slope.jpg


.. raw:: html

  <br/>
  <br/>
  <br/>



Comprehensive List of Environments
==================================


.. list-table::
    :widths: 33 25 19 25

    * - **Task Name**
      - **Inference Task Name**
      - **Workflow**
      - **RL Library**
    * - UW-Track-Goal-Ur5-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Track-Goal-Tycho-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Track-Goal-Xarm-Leap-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Nut-Thread-Franka-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Gear-Mesh-Franka-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Peg-Insert-Franka-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Position-Gap-Spot-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Position-Pit-Spot-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Position-Stepping-Stone-Spot-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Position-Obstacle-Spot-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)
    * - UW-Position-Inv-Slope-Spot-v0
      -
      - Manager Based
      - **rsl_rl** (PPO)