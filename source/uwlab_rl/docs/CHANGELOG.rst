Changelog
---------

0.2.1 (2026-09-26)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added CPU conversion of OmniReset beta observation layouts to EA order, including
  actor/critic normalization and Adam state without retraining or overwriting the original.

Changed
^^^^^^^

* Updated the RSL-RL dependency to version 5.4.1 for Isaac Lab 3.0 Early Access.
  Run ``./uwlab.sh --install`` to reinstall the pinned dependencies when upgrading.


0.2.0 (2026-09-16)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Ported to rsl-rl 5.x: explicit actor/critic model configs and the runner export API; the JIT
  exporter supports heteroscedastic Gaussian actors. Requires Isaac Lab 3.0.


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
