"""ASTEROID orchestrator: iterative in-context exploration + distillation.

Each iteration runs three stages, all as subprocesses:

    1. collect  -- roll out the expert (plus, from iteration 1 on, the previous
                   student as an explorer) in the data-collection env
                   (``scripts_v2/tools/collect_demos_asteroid.py``)
    2. train    -- fit a diffusion-policy student on every dataset collected so far
                   (``diffusion_policy/train.py``)
    3. eval     -- roll out the student in the eval env
                   (``scripts_v2/tools/eval_distilled_policy.py``)

Hyperparameters form a hierarchy of dataclasses: a :class:`RunCfg` holds the
run-level settings plus an ordered list of :class:`IterationCfg`, and each
iteration owns the :class:`CollectCfg` / :class:`TrainCfg` / :class:`EvalCfg`
for that iteration.  Curricula are functions that build the iteration list; add
a new one to :data:`CURRICULA` and select it with ``--schedule``.

Run from the repository root, e.g.::

    python scripts/ASTEROID/run_asteroid.py \
        --data_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0 \
        --eval_task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0 \
        --expert_policy_checkpoint logs/exported/policy.pt \
        --max_iterations 4 --exp_name my_run
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hyperparameter hierarchy
# ---------------------------------------------------------------------------


@dataclass
class CollectCfg:
    """Data collection for one iteration (``collect_demos_asteroid.py``).

    Exploration horizons are fractions of the episode during which the explorer
    (previous iteration's student) acts before the expert takes over.  Iteration 0
    has no student yet, so its horizons must be ``0.0`` (expert only).
    """

    num_demos: int = 10
    num_envs: int = 2
    episode_length_s: float = 8.0
    min_exploration_horizon: float = 0.0
    max_exploration_horizon: float = 0.0
    expert_noise: float = 0.0


@dataclass
class TrainCfg:
    """Student training for one iteration (``diffusion_policy/train.py``)."""

    lr: float = 1e-4
    #: Sampling weight for each dataset collected so far (index 0 = iteration 0's
    #: dataset).  Must have ``iteration + 1`` entries summing to 1.
    sampling_ratios: tuple[float, ...] = (1.0,)
    #: Warm-start from the previous iteration's student checkpoint.
    init_from_previous: bool = True
    #: Training step whose checkpoint is handed to eval / the next iteration.
    checkpoint_step: int = 40_000


@dataclass
class EvalCfg:
    """Student evaluation for one iteration (``eval_distilled_policy.py``)."""

    num_trajectories: int = 10
    num_envs: int = 2
    episode_length_s: float = 10.0


@dataclass
class IterationCfg:
    """Everything that varies per iteration."""

    collect: CollectCfg = field(default_factory=CollectCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


@dataclass
class RunCfg:
    """Run-level settings shared by every iteration, plus the iteration schedule."""

    exp_name: str
    output_dir: str
    wandb_project: str
    data_task: str
    eval_task: str
    expert_policy_checkpoint: str
    config_dir: str
    config_name: str
    insertive_object: str = "cube"
    receptive_object: str | None = None
    seed: int = 0
    video: bool = True
    iterations: list[IterationCfg] = field(default_factory=list)

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)

    def validate(self) -> None:
        assert self.num_iterations > 0, "RunCfg needs at least one iteration"
        for i, it in enumerate(self.iterations):
            c, t = it.collect, it.train
            assert 0.0 <= c.min_exploration_horizon <= c.max_exploration_horizon <= 1.0, (
                f"iteration {i}: exploration horizons must satisfy 0 <= min <= max <= 1, "
                f"got ({c.min_exploration_horizon}, {c.max_exploration_horizon})"
            )
            if i == 0:
                assert c.max_exploration_horizon == 0.0, (
                    "iteration 0 has no explorer yet; its exploration horizons must be 0.0"
                )
            assert len(t.sampling_ratios) == i + 1, (
                f"iteration {i}: sampling_ratios must have {i + 1} entries, got {len(t.sampling_ratios)}"
            )
            assert abs(sum(t.sampling_ratios) - 1.0) < 1e-6, (
                f"iteration {i}: sampling_ratios must sum to 1.0, got {sum(t.sampling_ratios)}"
            )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def dump(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Curricula: functions that build the per-iteration schedule
# ---------------------------------------------------------------------------


def default_curriculum(
    num_iterations: int,
    *,
    num_demos: int,
    num_data_envs: int,
    num_eval_envs: int,
    num_eval_episodes: int,
    expert_noise: float,
    init_from_previous: bool,
) -> list[IterationCfg]:
    """Curriculum used for the cube-pick runs.

    Index ``i`` of each table is iteration ``i``.  ``HORIZONS[i]`` is the explorer
    horizon used when collecting iteration ``i``'s dataset (so ``HORIZONS[0]`` is
    expert-only); ``SAMPLING_RATIOS[i]`` weights datasets ``0..i`` when training
    iteration ``i``'s student.
    """
    COLLECT_EPISODE_LENGTH_S = 8.0
    EVAL_EPISODE_LENGTH_S = 10.0
    HORIZONS = [
        (0.00, 0.00),  # expert only
        (0.20, 0.50),  # 1.6s - 4.0s of an 8s episode
        (0.30, 0.70),
        (0.40, 0.90),
        (0.50, 0.95),
        (0.60, 0.95),
    ]
    LRS = [1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
    SAMPLING_RATIOS = [
        (1.0,),
        (0.25, 0.75),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.3, 0.4),
        (0.05, 0.1, 0.2, 0.25, 0.4),
        (0.05, 0.1, 0.15, 0.15, 0.2, 0.35),
    ]
    max_supported = min(len(HORIZONS), len(LRS), len(SAMPLING_RATIOS))
    assert 1 <= num_iterations <= max_supported, (
        f"default curriculum supports 1..{max_supported} iterations, got {num_iterations}"
    )

    iterations = []
    for i in range(num_iterations):
        iterations.append(
            IterationCfg(
                collect=CollectCfg(
                    num_demos=num_demos,
                    num_envs=num_data_envs,
                    episode_length_s=COLLECT_EPISODE_LENGTH_S,
                    min_exploration_horizon=HORIZONS[i][0],
                    max_exploration_horizon=HORIZONS[i][1],
                    expert_noise=expert_noise,
                ),
                train=TrainCfg(
                    lr=LRS[i],
                    sampling_ratios=SAMPLING_RATIOS[i],
                    init_from_previous=init_from_previous,
                ),
                eval=EvalCfg(
                    num_trajectories=num_eval_episodes,
                    num_envs=num_eval_envs,
                    episode_length_s=EVAL_EPISODE_LENGTH_S,
                ),
            )
        )
    return iterations


CURRICULA = {
    "default": default_curriculum,
}


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

_STEP_CKPT_RE = re.compile(r"step_(\d+)\.ckpt$")


def expected_train_checkpoint(train_output_dir: str, step: int) -> str:
    """Resolve the checkpoint written by a training iteration.

    Selection order:
      1. ``step_{step:07d}.ckpt`` for the requested step.
      2. The highest-numbered ``step_*.ckpt`` in the checkpoints dir.
      3. ``latest.ckpt`` -- the final-state snapshot written by the workspace.
    """
    ckpt_dir = os.path.join(train_output_dir, "checkpoints")

    preferred = os.path.join(ckpt_dir, f"step_{step:07d}.ckpt")
    if os.path.exists(preferred):
        return preferred

    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.ckpt")):
        m = _STEP_CKPT_RE.search(os.path.basename(path))
        if m is not None:
            candidates.append((int(m.group(1)), path))
    if candidates:
        best_step, best_path = max(candidates)
        print(
            f"[asteroid] step_{step:07d}.ckpt missing under {ckpt_dir}; "
            f"falling back to {os.path.basename(best_path)} (step {best_step})."
        )
        return best_path

    latest = os.path.join(ckpt_dir, "latest.ckpt")
    if os.path.exists(latest):
        print(f"[asteroid] no step_*.ckpt under {ckpt_dir}; falling back to latest.ckpt.")
        return latest

    return preferred


class AsteroidRun:
    """Executes a :class:`RunCfg` under ``base_output_dir``.

    Layout::

        base_output_dir/
            run_cfg.json
            dataset-iteration-{i}/data.zarr      collected by iteration i
            iteration_{i}/checkpoints/*.ckpt     student trained in iteration i
    """

    def __init__(self, cfg: RunCfg, base_output_dir: str, dry_run: bool = False):
        cfg.validate()
        self.cfg = cfg
        self.base_output_dir = base_output_dir
        self.dry_run = dry_run

    # -- paths -------------------------------------------------------------

    def dataset_dir(self, iteration: int) -> str:
        return os.path.join(self.base_output_dir, f"dataset-iteration-{iteration}")

    def train_output_dir(self, iteration: int) -> str:
        return os.path.join(self.base_output_dir, f"iteration_{iteration}")

    def student_checkpoint(self, iteration: int) -> str:
        step = self.cfg.iterations[iteration].train.checkpoint_step
        return expected_train_checkpoint(self.train_output_dir(iteration), step)

    # -- stages ------------------------------------------------------------

    def _run(self, stage: str, command: list[str]) -> None:
        print(f"[asteroid] {stage}: {' '.join(command)}", flush=True)
        if self.dry_run:
            return
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"[asteroid] {stage} failed with return code {result.returncode}")
            sys.exit(1)
        print(f"[asteroid] {stage} finished")

    def _scene_overrides(self) -> list[str]:
        overrides = [f"env.scene.insertive_object={self.cfg.insertive_object}"]
        if self.cfg.receptive_object is not None:
            overrides.append(f"env.scene.receptive_object={self.cfg.receptive_object}")
        return overrides

    def collect(self, iteration: int, explorer_checkpoint: str | None) -> str:
        """Collect iteration ``iteration``'s dataset; returns the dataset dir."""
        cfg, c = self.cfg, self.cfg.iterations[iteration].collect
        dataset_dir = self.dataset_dir(iteration)
        command = [
            "python", "scripts_v2/tools/collect_demos_asteroid.py",
            "--task", cfg.data_task,
            "--dataset_file", os.path.join(dataset_dir, "data.zarr"),
            "--num_envs", str(c.num_envs),
            "--num_demos", str(c.num_demos),
            "--episode_length_s", str(c.episode_length_s),
            "--min_exploration_horizon", str(c.min_exploration_horizon),
            "--max_exploration_horizon", str(c.max_exploration_horizon),
            "--expert_noise", str(c.expert_noise),
            "--seed", str(cfg.seed),
            "--headless",
            f'agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=["{cfg.expert_policy_checkpoint}"]',
            *self._scene_overrides(),
        ]
        if explorer_checkpoint is not None:
            command += ["--exploration_checkpoint", explorer_checkpoint]
        if cfg.video:
            command += ["--video", "--video_dir", os.path.join(dataset_dir, "videos")]
        self._run(f"collect[{iteration}]", command)
        return dataset_dir

    def train(self, iteration: int, dataset_dirs: list[str], pretrained_checkpoint: str | None) -> str:
        """Train iteration ``iteration``'s student; returns its checkpoint path."""
        cfg, t = self.cfg, self.cfg.iterations[iteration].train
        assert len(dataset_dirs) == len(t.sampling_ratios), (
            f"iteration {iteration}: have {len(dataset_dirs)} datasets but {len(t.sampling_ratios)} sampling ratios"
        )
        dataset_config = ",".join(
            f"{{dataset_dir: {d}, sampling_ratio: {r}}}" for d, r in zip(dataset_dirs, t.sampling_ratios)
        )
        output_dir = self.train_output_dir(iteration)
        if not self.dry_run:
            os.makedirs(output_dir, exist_ok=True)
        command = [
            "python", "diffusion_policy/train.py",
            "--config-name", cfg.config_name,
            "--config-dir", cfg.config_dir,
            f"output_dir={output_dir}",
            f"task.dataset.dataset_config=[{dataset_config}]",
            f"name={cfg.exp_name}",
            f"exp_name={cfg.exp_name}",
            f"logging.project={cfg.wandb_project}",
            "logging.group=train",
            f"optimizer.lr={t.lr}",
            f"seed={cfg.seed}",
            f"iteration={iteration}",
        ]
        if pretrained_checkpoint is not None:
            command.append(f"checkpoint.pretrained_ckpt_path={pretrained_checkpoint}")
        self._run(f"train[{iteration}]", command)
        return self.student_checkpoint(iteration)

    def eval(self, iteration: int, checkpoint: str) -> None:
        cfg, e = self.cfg, self.cfg.iterations[iteration].eval
        command = [
            "python", "scripts_v2/tools/eval_distilled_policy.py",
            "--task", cfg.eval_task,
            "--checkpoint", checkpoint,
            "--num_trajectories", str(e.num_trajectories),
            "--num_envs", str(e.num_envs),
            "--episode_length_s", str(e.episode_length_s),
            "--seed", str(cfg.seed),
            "--exp_name", cfg.exp_name,
            "--wandb_project", cfg.wandb_project,
            "--wandb_group", "eval",
            "--iteration", str(iteration),
            "--headless",
            *self._scene_overrides(),
        ]
        if cfg.video:
            command += ["--save_video", "--enable_cameras"]
        self._run(f"eval[{iteration}]", command)

    # -- driver ------------------------------------------------------------

    def run(self, start_iteration: int = 0, initial_dataset_dir: str | None = None) -> None:
        """Run iterations ``start_iteration .. num_iterations-1``.

        When ``start_iteration > 0`` the datasets and student checkpoint of the
        earlier iterations are expected to exist under ``base_output_dir``.
        ``initial_dataset_dir`` replaces iteration 0's collection with an
        existing dataset.
        """
        cfg = self.cfg
        assert 0 <= start_iteration < cfg.num_iterations
        if not self.dry_run:
            os.makedirs(self.base_output_dir, exist_ok=True)
            cfg.dump(os.path.join(self.base_output_dir, "run_cfg.json"))

        dataset_dirs = [self.dataset_dir(i) for i in range(start_iteration)]
        student_checkpoint = self.student_checkpoint(start_iteration - 1) if start_iteration > 0 else None

        for i in range(start_iteration, cfg.num_iterations):
            print(f"[asteroid] ===== iteration {i} / {cfg.num_iterations - 1} =====")
            if i == 0 and initial_dataset_dir is not None:
                dataset_dirs.append(initial_dataset_dir)
            else:
                dataset_dirs.append(self.collect(i, explorer_checkpoint=student_checkpoint))

            pretrained = student_checkpoint if cfg.iterations[i].train.init_from_previous else None
            student_checkpoint = self.train(i, dataset_dirs, pretrained_checkpoint=pretrained)
            self.eval(i, student_checkpoint)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # tasks / policies
    p.add_argument("--data_task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0")
    p.add_argument("--eval_task", default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0")
    p.add_argument("--expert_policy_checkpoint", default="logs/policy_cube_final_v4.pt")
    p.add_argument("--insertive_object", default="cube")
    p.add_argument("--receptive_object", default=None)
    # student training config
    p.add_argument("--config_dir", default="diffusion_policy/diffusion_policy/config")
    p.add_argument("--config_name", default="incontext_exploration_debug.yaml")
    # logging
    p.add_argument("--output_dir", default="logs/incontext_exploration_debug")
    p.add_argument("--exp_name", default="incontext_exploration_debug")
    p.add_argument("--wandb_project", default="incontext_exploration")
    p.add_argument("--no_video", action="store_true", help="Disable video recording in collect and eval.")
    p.add_argument("--seed", type=int, default=0)
    # curriculum
    p.add_argument("--schedule", choices=sorted(CURRICULA), default="default", help="Which curriculum to run.")
    p.add_argument("--max_iterations", type=int, default=3, help="Number of iterations to run.")
    p.add_argument("--num_demos", type=int, default=10, help="Demos collected per iteration.")
    p.add_argument("--num_data_envs", type=int, default=2)
    p.add_argument("--num_eval_envs", type=int, default=2)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--expert_noise", type=float, default=0.0)
    p.add_argument("--not_use_pretrained_checkpoint", action="store_true",
                   help="Train every iteration's student from scratch instead of warm-starting.")
    # resume
    p.add_argument("--initial_dataset_path", default=None, help="Use this dataset for iteration 0 instead of collecting.")
    p.add_argument("--start_iteration", type=int, default=None, help="Resume from this iteration (requires --checkpoint_dir).")
    p.add_argument("--checkpoint_dir", default=None, help="Existing run directory to resume from.")
    p.add_argument("--dry_run", action="store_true", help="Print the stage commands without running them.")
    return p.parse_args(argv)


def build_run_cfg(args: argparse.Namespace) -> RunCfg:
    iterations = CURRICULA[args.schedule](
        args.max_iterations,
        num_demos=args.num_demos,
        num_data_envs=args.num_data_envs,
        num_eval_envs=args.num_eval_envs,
        num_eval_episodes=args.num_eval_episodes,
        expert_noise=args.expert_noise,
        init_from_previous=not args.not_use_pretrained_checkpoint,
    )
    return RunCfg(
        exp_name=args.exp_name,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        data_task=args.data_task,
        eval_task=args.eval_task,
        expert_policy_checkpoint=args.expert_policy_checkpoint,
        config_dir=args.config_dir,
        config_name=args.config_name,
        insertive_object=args.insertive_object,
        receptive_object=args.receptive_object,
        seed=args.seed,
        video=not args.no_video,
        iterations=iterations,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = build_run_cfg(args)
    print("[asteroid] run config:\n" + json.dumps(cfg.to_dict(), indent=2))

    if args.start_iteration is not None:
        assert args.checkpoint_dir is not None, "--start_iteration requires --checkpoint_dir"
        assert args.start_iteration > 0, "--start_iteration must be > 0 (use a fresh run for iteration 0)"
        assert args.initial_dataset_path is None, "--initial_dataset_path only applies to a fresh run"
        base_output_dir = args.checkpoint_dir
        start_iteration = args.start_iteration
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_output_dir = os.path.join(cfg.output_dir, cfg.exp_name, stamp)
        start_iteration = 0

    AsteroidRun(cfg, base_output_dir, dry_run=args.dry_run).run(
        start_iteration=start_iteration, initial_dataset_dir=args.initial_dataset_path
    )


if __name__ == "__main__":
    main()
