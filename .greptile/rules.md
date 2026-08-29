# UW Lab review conventions

UW Lab extends [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and follows its coding
conventions (Google Python style, PEP 8, PEP 484/585 type hints, Google docstrings). This file
is review context for Greptile. Rules that pre-commit already enforces are listed so the reviewer
does not repeat them.

## Already enforced by pre-commit (do not comment on these)

`./uwlab.sh -f` runs: black (line length 120), isort (black profile, custom sections in
`pyproject.toml`), flake8 (+simplify, +return; see `.flake8` for ignored codes), pyupgrade
(`--py310-plus`), codespell, trailing whitespace / EOF fixes, license header insertion,
debug-statement check, and a 2 MB file-size cap. Everything under `scripts/` is **excluded**
from pre-commit, so basic hygiene issues there are fair game.

## Repository layout

Four Omniverse-style extensions live under `source/`, each with the same shape:

```
source/<ext>/
├── config/extension.toml   # version (semver) + metadata + pip requirements
├── docs/CHANGELOG.rst      # one entry per version, must match extension.toml
├── <ext>/                  # the python package
├── test/                   # pytest tests
└── setup.py
```

- `uwlab` – framework core (envs, mdp terms, sensors, utilities). Depends on `isaaclab`.
- `uwlab_assets` – robot / object / sensor cfg instances. Large binaries go through git LFS.
- `uwlab_tasks` – task definitions: `manager_based/` and `direct/`, each task is a package with
  `__init__.py` (gym registration), `*_env_cfg.py`, `mdp/`, and `agents/` (RL library cfgs).
- `uwlab_rl` – RL library wrappers and runners.

Standalone entry points: `scripts/` mirrors the Isaac Lab layout (`reinforcement_learning/`,
`imitation_learning/`, `tools/`, `tutorials/`); `scripts_v2/tools/` holds newer standalone tools.

## Standalone scripts

Any script that touches Isaac Sim must create the `AppLauncher` first and import `isaaclab` /
`uwlab*` modules only after `simulation_app` exists:

```python
"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import torch
from isaaclab.envs import ManagerBasedRLEnv
```

Scripts take configuration from argparse / Hydra, never from hard-coded absolute paths.
When one script shells out to another, the flags it passes must exist in the callee.

## Python file and class structure

Within a file: imports → constants → public functions → public classes → private functions →
private classes. Within a class: constants → `ClassVar` attributes → `__init__`/`__del__` →
`__repr__`/`__str__` → properties → public instance/class/static methods → private methods,
ordered the way a user would call them (`initialize`, `reset`, `update`, `close`).

Private helpers are prefixed with `_`. Imports stay at module top; the only sanctioned
`typing.TYPE_CHECKING` use is breaking the cfg ↔ implementation cycle (a cfg's `class_type` /
`func` default referencing the implementation, or an implementation annotating its cfg type).

## Type hints and docstrings

- Type hints in the signature, PEP 604 unions (`torch.Tensor | None`), no `-> None`.
- Google docstrings. `Args:` entries are `name: description` — the type is never repeated.
- Physical quantities state SI units and shape: `"""Joint positions [rad], shape (num_envs, num_joints)."""`
- Docstrings explain *why* and non-obvious design choices, not just *what*.

## Manager-based environments

- Configs are `@configclass` dataclasses (`*Cfg`), usually alongside the implementation.
- MDP terms are plain functions with the Isaac Lab signature:

  ```python
  def term(env: ManagerBasedRLEnv, ..., asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
  ```

  Observation / reward / termination terms return a tensor of shape `(num_envs,)` or
  `(num_envs, D)`. Event terms receive `env_ids: torch.Tensor | None` and must only touch those
  environments. Stateful terms subclass `ManagerTermBase`.
- Everything is batched over the env dimension on `env.device`. No Python loops over envs, no
  `.item()` / `.cpu()` / numpy round-trips in per-step code, no CPU/CUDA tensor mixing.
- Per-env randomness (anything that differs across `env_ids`) is sampled with torch on
  `env.device` in one batched call (`torch.rand(..., device=env.device)`,
  `math_utils.sample_uniform`). Python `random` is fine for scalar scene-wide choices shared by
  every env (one HDRI, one camera jitter); the env seed seeds it too. Either way results must be
  reproducible under `env_cfg.seed`, which `test_environment_determinism.py` checks.
- Scene entities are resolved through `SceneEntityCfg` (joint/body ids are resolved once, then
  indexed), not by string lookups every step.

## Task registration

New environments are registered in the task package's `__init__.py`:

```python
gym.register(
    id="UW-<Task>-<Robot>-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.<task>_env_cfg:<Task>EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:<Task>PPORunnerCfg",
    },
)
```

Every entry-point string must resolve; `-Play-v0` variants normally reuse the training cfg with
`num_envs`, randomization, and curriculum reduced. New tasks are picked up by
`source/uwlab_tasks/test/test_environments.py`, so they must construct and step headless.

## Adding a new task or publication

- One package per task under `source/uwlab_tasks/uwlab_tasks/manager_based/<category>/<task>/`
  (`direct/` for direct-workflow envs). `manipulation/omnireset/` is the reference layout:
  `__init__.py` (gym registration), `*_env_cfg.py`, `mdp/` (terms and utils; its `__init__`
  re-exports `isaaclab.envs.mdp` and `uwlab.envs.mdp`, then the local modules), and
  `config/<robot>/` or `agents/` for robot-specific cfgs and RL library cfgs.
- By default everything the task needs - new mdp terms, recorders, helpers, cfgs - lives inside
  that package. Do not extend another task's package (e.g. add cube logic to `omnireset/mdp/`) and
  do not add flags to another task's env cfg; create a separate env cfg.
- Changes outside the task package (`uwlab` core, `uwlab_rl`, `uwlab_assets`, shared `uwlab_tasks`
  utilities, another task's package) are sometimes the right call, but they must be justified in
  the PR: why can this not live in the task package? Good reasons are code that is generic and used
  by more than one task, a framework bug fix, or a new shared asset. If the justification is
  missing, ask for it rather than rejecting the change.
- Docs: add the task id to `docs/source/overview/uw_environments.rst`. Publication-tier work also
  gets `docs/source/publications/<name>/index.rst` (Quick Start first; see `omnireset/index.rst`)
  and a bullet under **Getting Started** in `README.md`.
- Checkpoints and datasets go to the Hugging Face dataset `UW-Lab/uwlab-assets` (`Policies/` for
  checkpoints) via fork + PR, and the docs link to them. They are never committed here.
- Heavy or research-only dependencies (diffusion_policy, robomimic, ...) are not added to core
  install requirements, so a default install stays light. Use an `extras_require` group
  (`EXTRAS_REQUIRE` in `source/uwlab_rl/setup.py` is the pattern), a git submodule, or install
  steps on the publication's docs page. Nothing under `source/` may import them at module import
  time.
- Submodules point at a lab-owned repo (`github.com/UW-Lab`, `github.com/WEIRDLabUW`), are pinned
  to a branch or commit, are optional to initialize, and are documented (when to run
  `git submodule update --init`). If docs previously said to clone the repo manually, update them
  in the same PR.

## Versioning and changelog

Every PR that changes an extension's package bumps that extension's version in
`config/extension.toml` (patch for fixes, minor for features, major for breaking changes) and adds
a matching entry at the top of `docs/CHANGELOG.rst`:

```rst
0.13.9 (2026-08-28)
~~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Added :func:`~uwlab_tasks.manager_based.manipulation.omnireset.mdp.cube_reset` event for cube reset states.

Fixed
^^^^^

* Fixed ``--no_video`` being ignored in :mod:`scripts.ASTEROID.run_asteroid`.
```

Bullets are past tense, concise, and say why when it is not obvious. Sub-sections are limited to
Added / Changed / Deprecated / Removed / Fixed. Breaking changes go under Changed, prefixed with
`**Breaking:**`, with migration guidance.

## Dependencies

Prefer torch, numpy, and what is already installed. A new package must be added to the owning
extension's `setup.py` and `extension.toml`, have an allowlisted license (MIT / Apache / BSD /
ISC / zlib / PSF / BSL / MPL — CI fails otherwise unless listed in
`.github/workflows/license-exceptions.json`), and be justified in the PR description.

## Tests

pytest under `source/<ext>/test/`, run with `./uwlab.sh -t` or
`./uwlab.sh -p -m pytest <path>`. Tests that need the simulator create the `AppLauncher` at
module top (headless) and are marked `@pytest.mark.isaacsim_ci`. Tests are **not required** for a
PR — do not ask for them. Do flag changes that would break the existing suite (e.g. a new task
that cannot be constructed headless, or a determinism regression).

## Portability

No absolute paths, usernames, hard-coded conda environments, wandb entities, or local
checkpoint/dataset locations in committed code. Use CLI args, environment variables, Hydra, or
paths relative to the repo / extension data directory.
