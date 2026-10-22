.. _environments:

Isaac Lab Environments
======================

Upstream Isaac Lab environments are supplied by the pinned Isaac Lab dependency,
not by ``source/uwlab_tasks``. UWLab 2.0 uses Isaac Lab 3.0 Early Access at
``ae37b028ea415c91ea2bc32609efcd759ed2b974``. Task names, configuration locations,
and backend presets differ from the older 2.x catalogue.

Use the `maintained Isaac Lab environment browser
<https://isaac-sim.github.io/IsaacLab/develop/source/setup/environments.html>`_
to explore upstream tasks. That live page follows upstream development; the
`task registrations at UWLab's pinned revision
<https://github.com/isaac-sim/IsaacLab/tree/ae37b028ea415c91ea2bc32609efcd759ed2b974/source/isaaclab_tasks/isaaclab_tasks>`_
and the `browser source at that revision
<https://github.com/isaac-sim/IsaacLab/blob/ae37b028ea415c91ea2bc32609efcd759ed2b974/docs/source/setup/environments.rst>`_
are the version-specific references for this release.

Discover installed environments
-------------------------------

After installing the pinned stack, list the environments registered by the local
checkout rather than assuming that a name from another release is available:

.. code-block:: bash

   ./uwlab.sh -p scripts/environments/list_envs.py

For UWLab's own tasks and configurations, see :doc:`uw_environments`.
Availability in the registry does not by itself establish runtime qualification
of every task, robot, or backend combination.

Legacy environments
-------------------

Use UWLab's ``isaaclab2`` branch or ``v1.3.0`` tag for the Isaac Lab 2.3.2 /
Isaac Sim 5.1 stack. The `Isaac Lab 2.x environment catalogue
<https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/docs/source/overview/environments.rst>`_
may help identify older task names, but those names and configurations must not
be mixed with the pinned 3.0 stack without migration.
