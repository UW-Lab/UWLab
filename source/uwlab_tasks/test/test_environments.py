# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import importlib
import math
import torch
from types import SimpleNamespace

import pytest
from env_test_utils import _run_environments, setup_environment
from isaaclab.managers import EventManager, ObservationManager, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg

import uwlab_tasks  # noqa: F401


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
@pytest.mark.isaacsim_ci
def test_environments(task_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)


@pytest.mark.parametrize("task_family", ["omnireset", "factory_extension"])
@pytest.mark.parametrize("body_ids", [slice(None), [1]])
@pytest.mark.parametrize("stationary", [True, False])
@pytest.mark.parametrize("translated", [True, False])
@pytest.mark.isaacsim_ci
def test_asset_link_velocity_frame_transform(task_family, body_ids, stationary, translated):
    mdp = importlib.import_module(f"uwlab_tasks.manager_based.manipulation.{task_family}.mdp.observations")
    positions = torch.tensor([[1.0, 2.0, 3.0], [-4.0, 7.0, 9.0], [25.0, -6.0, 0.5]])
    if not translated:
        positions.zero_()
    half_sqrt = math.sqrt(0.5)
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt], [1.0, 0.0, 0.0, 0.0]])
    linear = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]]).repeat(3, 1, 1)
    angular = torch.tensor([[[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]]).repeat(3, 1, 1)
    if stationary:
        linear.zero_()
        angular.zero_()
    target = SimpleNamespace(
        data=SimpleNamespace(
            body_lin_vel_w=SimpleNamespace(torch=linear), body_ang_vel_w=SimpleNamespace(torch=angular)
        )
    )
    root = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=positions), root_quat_w=SimpleNamespace(torch=quaternions)
        )
    )
    env = SimpleNamespace(scene={"target": target, "robot": root})
    actual = mdp.asset_link_velocity_in_root_asset_frame(env, SceneEntityCfg("target", body_ids=body_ids))
    index = 0 if isinstance(body_ids, slice) else body_ids[0]
    linear_selected, angular_selected = linear[:, index], angular[:, index]
    expected = torch.cat([linear_selected, angular_selected], dim=-1)
    expected[1] = expected[1, [1, 0, 2, 4, 3, 5]] * torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    expected[2] *= torch.tensor([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.isaacsim_ci
def test_omnireset_renderer_configuration():
    mdp = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.events")
    front = CameraCfg(prim_path="/World/Front", spawn=None, width=16, height=16)
    side = CameraCfg(prim_path="/World/Side", spawn=None, width=16, height=16)
    cfg = SimpleNamespace(scene=SimpleNamespace(front=front, side=side, robot=object()), events=SimpleNamespace())
    settings = {
        "enable_dlssg": False,
        "enable_reflections": True,
        "enable_ambient_occlusion": True,
        "enable_dl_denoiser": True,
        "antialiasing_mode": "DLAA",
    }
    mdp.configure_isaac_rtx(cfg, **settings)
    event = cfg.events.render_settings
    assert event.mode == "startup"
    assert event.func is mdp.apply_isaac_rtx_settings
    for camera in (front, side):
        assert camera.renderer_cfg.renderer_type == "isaac_rtx"
        assert camera.renderer_cfg.enable_scene_partitioning is False
        for key, value in settings.items():
            assert getattr(camera.renderer_cfg.global_settings, key) == value
    front.renderer_cfg.global_settings.enable_reflections = False
    assert side.renderer_cfg.global_settings.enable_reflections is True
    assert event.params["settings"].enable_reflections is True


@pytest.mark.isaacsim_ci
def test_omnireset_renderer_configuration_without_cameras():
    mdp = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.events")
    cfg = SimpleNamespace(scene=SimpleNamespace(robot=object()), events=SimpleNamespace())
    mdp.configure_isaac_rtx(cfg, enable_dlssg=True)
    mdp.configure_isaac_rtx(cfg, enable_dlssg=False)
    assert list(vars(cfg.events)) == ["render_settings"]
    assert cfg.events.render_settings.params["settings"].enable_dlssg is False


@pytest.mark.isaacsim_ci
def test_omnireset_renderer_configuration_with_replicated_scene():
    mdp = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.events")
    scene = SimpleNamespace(replicate_physics=True)
    cfg = SimpleNamespace(scene=scene, events=SimpleNamespace())
    mdp.configure_isaac_rtx(cfg, enable_dlssg=True)
    handle = SimpleNamespace(deregister=lambda: None)
    sim = SimpleNamespace(
        is_playing=lambda: False,
        physics_manager=SimpleNamespace(register_callback=lambda *args, **kwargs: handle),
    )
    env = SimpleNamespace(scene=SimpleNamespace(cfg=scene), sim=sim, num_envs=1, device="cpu")
    manager = EventManager(cfg.events, env)
    assert manager.available_modes == ["startup"]
    assert scene.replicate_physics is True


@pytest.mark.isaacsim_ci
def test_omnireset_checkpoint_observation_layout():
    module = importlib.import_module(
        "uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg"
    )
    names = [
        "prev_actions",
        "joint_pos",
        "end_effector_pose",
        "insertive_asset_pose",
        "receptive_asset_pose",
        "insertive_asset_in_receptive_asset_frame",
    ]
    widths = [7, 12, 6, 6, 6, 6]
    for group_type in (module.ObservationsCfg.PolicyCfg, module.ObservationsCfg.CriticCfg):
        group = group_type()
        terms = [name for name, value in vars(group).items() if isinstance(value, ObservationTermCfg)]
        assert terms[:6] == names

    def sentinel(env, width, value):
        return torch.full((env.num_envs, width), value, device=env.device)

    group = module.ObservationsCfg.PolicyCfg()
    for index, (name, width) in enumerate(zip(names, widths)):
        term = getattr(group, name)
        term.func = sentinel
        term.params = {"width": width, "value": float(index + 1)}
        term.noise = None
    env = SimpleNamespace(num_envs=2, device="cpu", sim=SimpleNamespace(is_playing=lambda: True))
    manager = ObservationManager({"policy": group}, env)
    expected = torch.cat([torch.full((2, width), float(index + 1)) for index, width in enumerate(widths)], dim=-1)
    assert manager.active_terms["policy"] == names
    torch.testing.assert_close(manager.compute()["policy"], expected, rtol=0, atol=0)
