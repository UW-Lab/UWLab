Changelog
---------

0.2.4 (2026-09-27)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the pinned RSL-RL 5.4.1 integration to target separate actor and critic models for the upgraded UWLab stack.
  The legacy combined ``ActorCritic`` interface is not provided; retain the pinned 3.x stack for old callers.
  Reinstall the pinned dependency with ``./uwlab.sh --install`` after updating.


0.2.3 (2026-09-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Updated the pinned RSL-RL 5.4.1 revision without changing its library code or runtime behavior.
  Reinstall the pinned dependency with ``./uwlab.sh --install`` after updating.


0.2.2 (2026-09-26)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Pinned the integrated RSL-RL 5.4.1 fork, retaining deprecated legacy checkpoint inference
  and distributed initialization fixes. New training uses the 5.x actor/critic configuration.
  Reinstall the pinned dependency with ``./uwlab.sh --install`` after updating.


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
