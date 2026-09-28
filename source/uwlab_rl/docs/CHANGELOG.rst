Changelog
---------

0.2.0 (2026-09-28)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Migrated to the official UW-Lab RSL-RL 5.4.1 release (``uw-v5.4.1``), using separate
  actor and critic models with Isaac Lab 3.0 Early Access, Isaac Sim 6.1 and Python 3.12.
* Updated JIT export to support heteroscedastic Gaussian policies and their distributions.

Run ``./uwlab.sh --install`` to install the pinned dependencies and use the compatible
checkpoints linked in the OmniReset quick start. Use ``isaaclab2`` / ``v1.3.0`` for
legacy RSL-RL 3.x callers, including the combined ``ActorCritic`` interface.


0.1.4 (2026-09-14)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Pinned the RSL-RL dependency to commit ``e7cd3c77bdb3c94753612f208c725e1add38a655``
  so new installations do not pick up incompatible changes from its moving main branch.


0.1.3 (2025-11-09)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Point rsl_rl installation to UWLab custom link that implements gsde.



0.1.2 (2025-03-23)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Comment out all the velocity limit for asset that has actuator type of implicit actuator.


0.1.1 (2025-03-23)
~~~~~~~~~~~~~~~~~~

Fixed
^^^^^

* Pre commit fail because of continuation line over-indented for hanging indent in on_policy_runner.py


0.1.0 (2025-03-12)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

Initial version of the extension include the wrapper scripts and extensions for the supported RL libraries.

Supported RL libraries are:

* RL Games
* RSL RL
* SKRL
* Stable Baselines3
