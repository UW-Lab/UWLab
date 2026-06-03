import argparse
import datetime
import glob
import os
import re
import subprocess


_STEP_CKPT_RE = re.compile(r"step_(\d+)\.ckpt$")


def _expected_train_checkpoint(output_dir: str, step: int = 20_000) -> str:
    """Resolve the checkpoint path produced by a training iteration.

    Selection order (ported from UWLab-ICL/run_incontext_exploration.py):
      1. ``step_{step:07d}.ckpt`` for the requested step.
      2. The highest-numbered ``step_*.ckpt`` in the checkpoints dir.
      3. ``latest.ckpt`` — the final-state snapshot written by the workspace.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints")

    preferred = os.path.join(ckpt_dir, f"step_{step:07d}.ckpt")
    if os.path.exists(preferred):
        return preferred

    candidates: list[tuple[int, str]] = []
    for path in glob.glob(os.path.join(ckpt_dir, "step_*.ckpt")):
        m = _STEP_CKPT_RE.search(os.path.basename(path))
        if m is not None:
            candidates.append((int(m.group(1)), path))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_step, best_path = candidates[-1]
        print(
            f"[orchestrator] step_{step:07d}.ckpt missing under {ckpt_dir}; "
            f"falling back to {os.path.basename(best_path)} (step {best_step})."
        )
        return best_path

    latest = os.path.join(ckpt_dir, "latest.ckpt")
    if os.path.exists(latest):
        print(f"[orchestrator] no step_*.ckpt under {ckpt_dir}; falling back to latest.ckpt.")
        return latest

    return preferred

def collect_demos(
        task: str,
        dataset_file: str,
        num_envs: int,
        num_demos: int,
        min_exploration_horizon: float,
        max_exploration_horizon: float,
        episode_length_s: float,
        expert_path: str = "logs/policy_peg_final_v4.pt",
        insertive_object: str = "cube",
        receptive_object: str = None,
        exploration_checkpoint: str = None,
        no_video: bool = False,
        seed: int = 0,
        expert_noise: float = 0.0,
):
    command = [
        "python",
        "scripts_v2/tools/collect_demos.py",
        "--task",
        task,
        "--dataset_file",
        dataset_file,
        "--num_envs",
        str(num_envs),
        "--num_demos",
        str(num_demos),
        "--headless",
        "--seed",
        str(seed),
        f"--expert_noise",
        str(expert_noise),
        f"env.scene.insertive_object={insertive_object}",
        f"--max_exploration_horizon",
        str(max_exploration_horizon),
        f"--min_exploration_horizon",
        str(min_exploration_horizon),
        f"--episode_length_s",
        str(episode_length_s),
        "agent.algorithm.offline_algorithm_cfg.behavior_cloning_cfg.experts_path=[\""
        + expert_path
        + "\"]",
    ]
    if exploration_checkpoint is not None:
        command += [
            "--exploration_checkpoint",
            exploration_checkpoint,
        ]
    if receptive_object is not None:
        command += [f"env.scene.receptive_object={receptive_object}"]
    if not no_video:
        command += ["--save_video", "--enable_cameras"]

    demos = subprocess.run(command)
    if demos.returncode != 0:
        print("Demo collection process failed with return code:", demos.returncode)
        exit(1)

    print("Demo collection process finished with return code:", demos.returncode)

def visualize_demos(
    dataset_file: str,
    video_path: str,
    num_videos: int,
    wandb_project: str,
    wandb_group: str,
    exp_name: str,
    iteration: int
):
    subprocess.run(
        [
            "python",
            "scripts_v2/tools/conversions/convert_zarr_video.py",
            dataset_file,
            video_path,
            str(num_videos),
        ]
    )
    # push the video to wandb
    import wandb
    wandb.init(project=wandb_project, group=wandb_group,
               config={"iteration": iteration, "exp_name": exp_name})
    wandb.log({"demo_videos": wandb.Video(video_path)})
    wandb.finish()

def train_policy(
    config_name,
    config_dir,
    output_dir,
    dataset_config,
    wandb_project: str,
    wandb_group: str,
    exp_name: str,
    lr: float,
    pretrained_checkpoint=None,
    seed: int = 0,
    iteration: int = 0,
):
    
    dataset_str = "task.dataset.dataset_config=["
    for data in dataset_config:
        dataset_str += "{dataset_dir: " + data["dataset_dir"] + ", sampling_ratio: " + str(data["sampling_ratio"]) + "},"
    dataset_str = dataset_str[:-1] + "]"
    
    command = [
        "python",
        "diffusion_policy/train.py",
        "--config-name",
        config_name,
        "--config-dir",
        config_dir,
        "output_dir=" + output_dir,
        dataset_str,
        "name=" + exp_name,
        "exp_name=" + exp_name,
        "logging.project=" + wandb_project,
        "logging.group=" + wandb_group,
        "optimizer.lr=" + str(lr),
        "seed=" + str(seed),
        "iteration=" + str(iteration),
    ]

    if pretrained_checkpoint is not None:
        pretrained_checkpoint_str = "checkpoint.pretrained_ckpt_path=" + pretrained_checkpoint
        command.append(pretrained_checkpoint_str)
    
    # wait for the process to complete
    train_process_return = subprocess.run(command)
    if train_process_return.returncode != 0:
        print("Training process failed with return code:", train_process_return.returncode)
        exit(1)
    print("Training process finished with return code:", train_process_return.returncode)

def eval_policy(
    task: str,
    checkpoint: str,
    num_trajectories: int,
    num_envs: int,
    exp_name: str,
    wandb_project: str,
    wandb_group: str,
    episode_length_s: float = 24.0,
    insertive_object: str = "cube",
    receptive_object: str | None = None,
    no_video: bool = False,
    seed: int = 0,
    iteration: int = 0,
):
    command = [
        "python",
        "scripts_v2/tools/eval_distilled_policy.py",
        "--task",
        task,
        "--seed",
        str(seed),
        "--num_trajectories",
        str(num_trajectories),
        "--num_envs",
        str(num_envs),
        "--headless",
        f"env.scene.insertive_object={insertive_object}",
        "--exp_name",
        exp_name,
        "--wandb_project",
        wandb_project,
        "--wandb_group",
        wandb_group,
        "--checkpoint",
        checkpoint,
        "--episode_length_s",
        str(episode_length_s),
        "--iteration",
        str(iteration),
    ]
    if receptive_object is not None:
        command += [f"env.scene.receptive_object={receptive_object}"]
    # if not no_video:
    command += ["--save_video", "--enable_cameras"]

    eval_process = subprocess.run(command)
    if eval_process.returncode != 0:
        print("Evaluation process failed with return code:", eval_process.returncode)
        exit(1)
    print("Evaluation process finished with return code:", eval_process.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run in-context exploration and collect demos")

    """
    Script asks for expert policy checkpoint (with default value), num_demos, path to config, config name, output dir for dataset and checkpoints, exp_name, 
    and path to initial dataset
    """
    parser.add_argument(
        "--data_task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0", help="Data collection task name"
    )
    parser.add_argument("--eval_task", type=str, default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0", help="Evaluation task name")
    parser.add_argument(
        "--expert_policy_checkpoint", default="logs/policy_cube_final_v4.pt", help="Path to expert policy checkpoint"
    )
    parser.add_argument(
        "--num_demos", type=int, default=10, help="Number of demos to collect"
    )
    parser.add_argument(
        "--num_data_envs", type=int, default=2, help="Number of parallel environments for data collection"
    )
    parser.add_argument(
        "--num_eval_envs", type=int, default=2, help="Number of parallel environments for evaluation"
    )
    parser.add_argument(
        "--num_eval_episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="diffusion_policy/diffusion_policy/config",
        help="Path to config directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="incontext_exploration_debug.yaml",
        help="Name of the config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/incontext_exploration_debug",
        help="Directory to save output logs and models",
    )
    parser.add_argument(
        "--initial_dataset_path",
        type=str,
        default=None,
        help="Path to initial dataset for training the policy",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="incontext_exploration_debug",
        help="Experiment name for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="incontext_exploration",
        help="Wandb project name",
    )
    parser.add_argument(
        "--not_use_pretrained_checkpoint",
        action="store_true",
        help="If set, do not use pretrained checkpoint for training",
    )
    # parser.add_argument(
    #     "--episode_length_s",
    #     type=float,
    #     default=16.0,
    #     help="Episode length in seconds",
    # )
    # parser.add_argument(
    #     "--episode_length_s_increment",
    #     type=float,
    #     default=4.0,
    #     help="Episode length increment after each iteration",
    # )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="If set, do not save videos during evaluation",
    )
    parser.add_argument(
        "--insertive_object",
        type=str,
        default="cube",
        help="Insertive object type",
    )
    parser.add_argument(
        "--receptive_object",
        type=str,
        default=None,
        help="Receptive object type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environment",
    )
    parser.add_argument("--start_iteration", type=int, default=None, help="Starting iteration number")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of run to resmume from")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="Maximum number of iterations to run",
    )
    parser.add_argument(
        "--expert_noise",
        type=float,
        default=0.0,
        help="Noise level for the expert policy"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="default",
        help="Schedule for curriculum. Options: fixed, linear, exponential",
    )
    args = parser.parse_args()

    # CONSTANTS / CURRICULUM SETTINGS
    sampling_ratio_curriculum = [
        (1.0, ),
        (0.25, 0.75),
        (0.2, 0.3, 0.5),
        (0.1, 0.2, 0.3, 0.4),
        (0.05, 0.1, 0.2, 0.25, 0.4),
        (0.05, 0.1, 0.15, 0.15, 0.2, 0.45,)
        # (0.4, 0.3, 0.2, 0.1),
    ]

    lrs = [
        1e-4,
        1e-5,
        1e-5,
        1e-5,
        1e-5,
        1e-5
    ]

    horizons = [
        # (0.1, 0.3),
        (0.2, 0.5),
        # (0.15, 0.5),
        (0.3, 0.8),
        # (0.3, 0.7)
        (0.4, 0.9)
    ]
    episode_length_s = [
        # 5.0,
        7.0,
        9.0,
        11.0
    ]
    if args.schedule == "fixed":
        episode_length_s = [10.0, 10.0, 10.0, 10.0, 10.0]
        horizons = [
            (0.2, 0.5),
            (0.3, 0.7),
            (0.4, 0.9),
            (0.5, 0.95),
            (0.6, 0.95)
        ]
    elif args.schedule == "fixed2":
        episode_length_s = [8.0, 8.0, 8.0]
        horizons = [
            (0.2, 0.2),
            (0.5, 0.5),
            (0.8, 0.8)
        ]
    elif args.schedule == "fixed3":
        # episode_length_s = [8.0, 8.0, 8.0]
        horizons = [
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
        ]

    

    exp_name = args.exp_name
    wandb_project = args.wandb_project

    if args.start_iteration is not None:
        assert args.checkpoint_dir is not None, "If start_iteration is provided, checkpoint_dir must also be provided"
        assert args.start_iteration > 0, "start_iteration must be greater than 0"
        assert args.start_iteration < args.max_iterations, "start_iteration must be less than max_iterations"
        assert len(sampling_ratio_curriculum) >= args.max_iterations, "sampling_ratio_curriculum length must be >= max_iterations"
        assert len(lrs) >= args.max_iterations, "lrs length must be >= max_iterations"
        assert len(horizons) >= args.max_iterations - 1, "horizons length must be >= max_iterations - 1"
        assert len(episode_length_s) >= args.max_iterations - 1, "episode_length_s length must be >= max_iterations - 1"

        base_output_dir = args.checkpoint_dir
        args.initial_dataset_path = os.path.join(
            base_output_dir, f"dataset-iteration-{args.start_iteration}"
        )
        exploration_checkpoint = _expected_train_checkpoint(
            os.path.join(base_output_dir, f"iteration_{args.start_iteration - 1}")
        )
        # collect demos using the provided checkpoint for this iteration
        dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{args.start_iteration}")
        collect_demos(
            task=args.data_task,
            dataset_file=os.path.join(dataset_path, "data.zarr"),
            num_envs=args.num_data_envs,
            num_demos=args.num_demos,
            min_exploration_horizon=0.4,
            max_exploration_horizon=0.9,
            episode_length_s=8.0,
            expert_path=args.expert_policy_checkpoint,
            exploration_checkpoint=exploration_checkpoint,
            insertive_object=args.insertive_object,
            receptive_object=args.receptive_object,
            no_video=args.no_video,
            seed=args.seed,
            expert_noise=args.expert_noise,
        )
        dataset_paths = [f"{base_output_dir}/dataset-iteration-{i}" for i in range(args.start_iteration + 1)]
        iteration_checkpoint = exploration_checkpoint
        start_iteration = args.start_iteration
    else:
        base_output_dir = os.path.join(args.output_dir, exp_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(base_output_dir, exist_ok=True)
        # if initial dataset path is none, collect demos using expert policy
        if args.initial_dataset_path is None:
            dataset_path = os.path.join(base_output_dir, "dataset-iteration-0")
            collect_demos(
                task=args.data_task,
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_envs=args.num_data_envs,
                num_demos=args.num_demos,
                min_exploration_horizon=0.0,
                max_exploration_horizon=0.0,
                episode_length_s=10.0,
                expert_path=args.expert_policy_checkpoint,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                expert_noise=args.expert_noise,
            )
            args.initial_dataset_path = dataset_path
            # visualize the collected demos
            if not args.no_video:
                visualize_demos(
                    dataset_file=args.initial_dataset_path,
                    video_path=os.path.join(base_output_dir, "demo_videos_iteration_0.mp4"),
                    num_videos=10,
                    wandb_project=wandb_project,
                    wandb_group="data",
                    exp_name=exp_name,
                    iteration=0,
                )
        # Whether the dataset was collected fresh or pointed to with
        # --initial_dataset_path, kick off training from iter 0 with that single
        # dataset and no prior student checkpoint.
        dataset_paths = [args.initial_dataset_path]
        iteration_checkpoint = None
        start_iteration = 0

    for iteration in range(start_iteration, args.max_iterations):
        print(f"Starting iteration {iteration}...")
        # train policy on the collected demos with previous dataset as well and pretrained checkpoint
        train_output_dir = os.path.join(base_output_dir, f"iteration_{iteration}")
        os.makedirs(train_output_dir, exist_ok=True)
        train_policy(
            config_name=args.config_name,
            config_dir=args.config_dir,
            output_dir=train_output_dir,
            dataset_config=[
                {"dataset_dir": dataset_paths[i], "sampling_ratio": sampling_ratio_curriculum[iteration][i]} 
                for i in range(len(sampling_ratio_curriculum[iteration]))
            ],
            wandb_project=wandb_project,
            wandb_group="train",
            exp_name=exp_name,
            pretrained_checkpoint=iteration_checkpoint,
            lr=lrs[iteration],
            seed=args.seed,
            iteration=iteration,
        )

        # evaluate the trained policy
        iteration_checkpoint = _expected_train_checkpoint(
            os.path.join(base_output_dir, f"iteration_{iteration}")
        )
        eval_policy(
            task=args.eval_task,
            checkpoint=iteration_checkpoint,
            num_trajectories=args.num_eval_episodes,
            num_envs=args.num_eval_envs,
            exp_name=exp_name,
            wandb_project=wandb_project,
            wandb_group="eval",
            episode_length_s=10.0,
            insertive_object=args.insertive_object,
            receptive_object=args.receptive_object,
            no_video=args.no_video,
            seed=args.seed,
            iteration=iteration,
        )

        # If not the last iteration, collect demos with the trained policy as the explorer
        if iteration < args.max_iterations - 1:
            dataset_path = os.path.join(base_output_dir, f"dataset-iteration-{iteration+1}")
            collect_demos(
                task=args.data_task,
                dataset_file=os.path.join(dataset_path, "data.zarr"),
                num_envs=args.num_data_envs,
                num_demos=args.num_demos,
                min_exploration_horizon=horizons[iteration][0],
                max_exploration_horizon=horizons[iteration][1],
                episode_length_s=episode_length_s[iteration],
                expert_path=args.expert_policy_checkpoint,
                exploration_checkpoint=iteration_checkpoint,
                insertive_object=args.insertive_object,
                receptive_object=args.receptive_object,
                no_video=args.no_video,
                seed=args.seed,
                expert_noise=args.expert_noise,
            )
            if not args.no_video:
                visualize_demos(
                    dataset_file=dataset_path,
                    video_path=os.path.join(base_output_dir, f"demo_videos_iteration_{iteration+1}.mp4"),
                    num_videos=10,
                    wandb_project=wandb_project,
                    wandb_group="data",
                    exp_name=exp_name,
                    iteration=iteration + 1,
                )
            dataset_paths.append(dataset_path)







    
