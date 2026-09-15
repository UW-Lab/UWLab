Changelog
---------

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
