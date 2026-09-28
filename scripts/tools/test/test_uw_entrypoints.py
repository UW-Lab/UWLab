# Copyright (c) 2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "keyword, expected",
    [(None, ["Isaac-Test", "UW-Test", "OmniReset-Test"]), ("UW-", ["UW-Test"]), ("OmniReset-", ["OmniReset-Test"])],
)
def test_environment_listing_includes_uw_tasks(monkeypatch, capsys, keyword, expected):
    gym = ModuleType("gymnasium")
    gym.registry = {
        name: SimpleNamespace(
            id=name,
            entry_point="fixture:Env",
            kwargs={"env_cfg_entry_point": "fixture:Cfg", "deprecated": name == "Isaac-Old"},
        )
        for name in ("Isaac-Test", "UW-Test", "OmniReset-Test", "Isaac-Old", "Other-Test")
    }
    monkeypatch.setitem(sys.modules, "gymnasium", gym)
    for name in ("isaaclab_tasks", "isaaclab_tasks_experimental", "uwlab_tasks"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    argv = ["list_envs.py"] + (["--keyword", keyword] if keyword else [])
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "scripts/environments/list_envs.py"), run_name="__main__")
    output = capsys.readouterr().out
    for name in gym.registry:
        assert (name in output) == (name in expected)


@pytest.mark.parametrize(
    "arguments, resume",
    [([], True), (["--experiment_name", "custom"], False), (["--resume"], False), (["--device", "cpu"], False)],
)
def test_rsl_cli_overrides_only_requested_values(arguments, resume):
    module = runpy.run_path(str(ROOT / "scripts/reinforcement_learning/rsl_rl/cli_args.py"))
    parser = argparse.ArgumentParser()
    module["add_rsl_rl_args"](parser)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(arguments)
    cfg = SimpleNamespace(
        seed=42,
        resume=resume,
        load_run="saved",
        load_checkpoint="model.pt",
        experiment_name="original",
        run_name="base",
        logger="tensorboard",
        device="cuda:0",
    )
    result = module["update_rsl_rl_cfg"](cfg, args)
    assert result is cfg
    assert cfg.resume is (resume or "--resume" in arguments)
    assert cfg.experiment_name == ("custom" if "--experiment_name" in arguments else "original")
    assert cfg.device == ("cpu" if "--device" in arguments else "cuda:0")
    assert cfg.load_run == "saved" and cfg.load_checkpoint == "model.pt"


def _camera_configs():
    def load(path, names, namespace):
        tree = ast.parse((ROOT / path).read_text())
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
        return namespace

    class Sample:
        def __init__(self, value):
            self.value = value

        def sample(self):
            return self.value

    tune = SimpleNamespace(
        choice=lambda values: Sample(values[0]),
        randint=lambda low, high: Sample(low),
        sample_from=lambda function: Sample(function),
    )
    base = load("scripts/reinforcement_learning/ray/tuner.py", ["JobCfg"], {})
    util = load("scripts/reinforcement_learning/ray/util.py", ["populate_isaac_ray_cfg_args"], {})
    vision = load(
        "scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cfg.py",
        ["CameraJobCfg", "ResNetCameraJob", "TheiaCameraJob"],
        {
            "tune": tune,
            "tuner": SimpleNamespace(JobCfg=base["JobCfg"]),
            "util": SimpleNamespace(populate_isaac_ray_cfg_args=util["populate_isaac_ray_cfg_args"]),
        },
    )
    jobs = load(
        "scripts/reinforcement_learning/ray/hyperparameter_tuning/vision_cartpole_cfg.py",
        [
            "CartpoleRGBNoTuneJobCfg",
            "CartpoleRGBCNNOnlyJobCfg",
            "CartpoleRGBJobCfg",
            "CartpoleResNetJobCfg",
            "CartpoleTheiaJobCfg",
        ],
        {"tune": tune, "util": vision["util"], "vision_cfg": SimpleNamespace(**vision)},
    )
    return vision, jobs


@pytest.mark.parametrize(
    "name",
    [
        "CartpoleRGBNoTuneJobCfg",
        "CartpoleRGBCNNOnlyJobCfg",
        "CartpoleRGBJobCfg",
        "CartpoleResNetJobCfg",
        "CartpoleTheiaJobCfg",
    ],
)
def test_camera_tuning_selects_matching_agent(name):
    _, jobs = _camera_configs()
    cfg = jobs[name]({}).cfg
    assert cfg["runner_args"]["--rl_library"] == "rl_games"
    assert cfg["hydra_args"]["agent.params.config.max_epochs"] == 200
    assert all(not key.startswith("agent.") or key.startswith("agent.params.") for key in cfg["hydra_args"])


def test_camera_tuning_rejects_conflicting_agent():
    vision, _ = _camera_configs()
    with pytest.raises(ValueError, match="rl_games"):
        vision["CameraJobCfg"]({"runner_args": {"--task": "Isaac-Cartpole-Camera", "--rl_library": "rsl_rl"}})
