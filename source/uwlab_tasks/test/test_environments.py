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


@pytest.mark.parametrize("env_ids, expected", [([1], [4, 0]), ([], [4, 2]), (None, [0, 0]), ([0, 1], [0, 0])])
@pytest.mark.isaacsim_ci
def test_progress_context_reset_is_per_environment(env_ids, expected):
    module = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.rewards")
    context = module.ProgressContext.__new__(module.ProgressContext)
    context.continuous_success_counter = torch.tensor([4, 2], dtype=torch.int32)
    reference = context.continuous_success_counter
    ids = None if env_ids is None else torch.tensor(env_ids, dtype=torch.long)
    context.reset(ids)
    assert context.continuous_success_counter is reference
    assert torch.equal(reference, torch.tensor(expected, dtype=torch.int32))
    context.reset(ids)
    assert torch.equal(reference, torch.tensor(expected, dtype=torch.int32))


@pytest.mark.parametrize("pattern", ["/World/envs/env_.*/Object", "/World/envs/env_[^/]+/Object"])
@pytest.mark.isaacsim_ci
def test_collision_asset_paths_and_frames(monkeypatch, pattern):
    from isaaclab.sim.utils import find_matching_prim_paths, get_all_matching_child_prims
    from pxr import Usd, UsdGeom, UsdPhysics

    module = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.rigid_object_hasher")
    stage = Usd.Stage.CreateInMemory()
    for index in (10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11):
        root = f"/World/envs/env_{index}/Object"
        UsdGeom.Xform.Define(stage, root)
        for name, y in (("a", 1.0), ("b", float(index + 2))):
            cube = UsdGeom.Cube.Define(stage, f"{root}/{name}")
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cube.AddTranslateOp().Set((0.0, y, 0.0))
    monkeypatch.setattr(module.stage_utils, "get_current_stage", lambda: stage)
    monkeypatch.setattr(module.stage_utils, "get_current_stage_id", lambda: None)
    monkeypatch.setattr(module, "HASH_STORE", {"warp_mesh_store": {}, "__stage_id__": None})
    monkeypatch.setattr(
        module,
        "find_matching_prim_paths",
        lambda expr, stage=None: find_matching_prim_paths(expr, stage=stage),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "get_all_matching_child_prims",
        lambda path, **kwargs: get_all_matching_child_prims(path, stage=stage, **kwargs),
    )
    hasher = module.RigidObjectHasher(12, pattern, device="cpu")
    paths = [str(prim.GetPath()) for prim in hasher.collider_prims]
    assert paths == [f"/World/envs/env_{index}/Object/{name}" for index in range(12) for name in ("a", "b")]
    expected_quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(24, -1)
    torch.testing.assert_close(hasher.collider_prim_relative_transforms[:, 3:7], expected_quaternions, rtol=0, atol=0)
    assert hasher.root_prim_hashes[0] != hasher.root_prim_hashes[1]


@pytest.mark.parametrize("term_name", ["check_grasp_success", "check_reset_state_success"])
@pytest.mark.parametrize("object_kind", ["rigid", "collection"])
@pytest.mark.isaacsim_ci
def test_dataset_success_requires_backend_asset_stability(term_name, object_kind):
    from unittest.mock import Mock

    from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection

    module = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.terminations")
    positions = torch.tensor([[0.0, 0.0, 0.1]]).repeat(4, 1)
    quaternions = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
    joint_velocities = torch.zeros(4, 2)
    linear_velocities = torch.zeros(4, 2, 3)
    angular_velocities = torch.zeros(4, 2, 3)
    joint_velocities[1, 0] = 6.0
    linear_velocities[2, 1, 0] = 0.2
    angular_velocities[3, 1, 0] = 2.0
    robot = Mock(spec=BaseArticulation)
    robot.initial_pos = positions.clone()
    robot.data = SimpleNamespace(
        joint_vel=SimpleNamespace(torch=joint_velocities),
        joint_vel_limits=SimpleNamespace(torch=torch.full_like(joint_velocities, 100.0)),
        root_pos_w=SimpleNamespace(torch=positions),
        root_quat_w=SimpleNamespace(torch=quaternions),
        body_link_pos_w=SimpleNamespace(torch=positions[:, None]),
        body_link_quat_w=SimpleNamespace(torch=quaternions[:, None]),
    )
    obj = Mock(spec=BaseRigidObject if object_kind == "rigid" else BaseRigidObjectCollection)
    obj.initial_pos = positions.clone()
    obj.data = SimpleNamespace(
        root_pos_w=SimpleNamespace(torch=positions),
        root_quat_w=SimpleNamespace(torch=quaternions),
        body_lin_vel_w=SimpleNamespace(torch=linear_velocities),
        body_ang_vel_w=SimpleNamespace(torch=angular_velocities),
        object_lin_vel_w=SimpleNamespace(torch=linear_velocities),
        object_ang_vel_w=SimpleNamespace(torch=angular_velocities),
    )
    env = SimpleNamespace(
        scene={"robot": robot, "object": obj},
        num_envs=4,
        device="cpu",
        episode_length_buf=torch.full((4,), 10),
        max_episode_length=10,
    )
    term_type = getattr(module, term_name)
    term = term_type.__new__(term_type)
    term.stability_counter = torch.zeros(4, dtype=torch.int32)
    term.consecutive_stability_steps = 2
    term.pos_z_threshold = 0.05

    def collision_free(env, ids):
        return torch.ones(len(ids), dtype=torch.bool)

    robot_cfg = SceneEntityCfg("robot")
    object_cfg = SceneEntityCfg("object")
    if term_name == "check_grasp_success":
        term.object_cfg = object_cfg
        term.gripper_cfg = robot_cfg
        term.max_pos_deviation = 0.05
        term.collision_analyzer = collision_free
        params = {"object_cfg": object_cfg, "gripper_cfg": robot_cfg, "collision_analyzer_cfg": None}
    else:
        term.robot_asset = robot
        term.assets_to_check = [obj, robot]
        term.ee_body_idx = 0
        term.gripper_approach_direction = (0.0, 0.0, -1.0)
        term.max_robot_pos_deviation = term.max_object_pos_deviation = 0.05
        term.collision_analyzers = [collision_free]
        term.assembly_success_prob = None
        params = {
            "object_cfgs": [object_cfg],
            "robot_cfg": robot_cfg,
            "ee_body_name": "tool",
            "collision_analyzer_cfgs": [],
        }
    assert not term(env, **params).any()
    torch.testing.assert_close(term(env, **params), torch.tensor([True, False, False, False]))
    torch.testing.assert_close(term.stability_counter, torch.tensor([2, 0, 0, 0], dtype=torch.int32))
    joint_velocities.zero_()
    linear_velocities.zero_()
    angular_velocities.zero_()
    torch.testing.assert_close(term(env, **params), torch.tensor([True, False, False, False]))
    assert term(env, **params).all()
    linear_velocities[0, 1, 0] = 0.2
    torch.testing.assert_close(term(env, **params), torch.tensor([False, True, True, True]))
    assert term.stability_counter[0] == 0


@pytest.mark.isaacsim_ci
def test_grasp_sampling_preserves_asset_masses():
    module = importlib.import_module(
        "uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.grasp_sampling_cfg"
    )
    assert module.GraspSamplingSceneCfg().object.spawn.mass_props is None
    assert set(module.variants["scene.object"]) == {"peg", "cube", "cupcake", "rectangle", "fbleg", "fbdrawerbottom"}
    for object_cfg in module.variants["scene.object"].values():
        assert object_cfg.spawn.mass_props is None
        assert object_cfg.spawn.rigid_props.disable_gravity is False


@pytest.mark.parametrize("contract", ["asset", "armature"])
@pytest.mark.isaacsim_ci
def test_omnireset_published_expert_robot_defaults(contract):
    if contract == "asset":
        module = importlib.import_module("uwlab_assets.robots.ur5e_robotiq_gripper.ur5e_robotiq_2f85_gripper")
        assert module.UR5E_ARTICULATION.spawn.usd_path.endswith(
            "/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd"
        )
    else:
        module = importlib.import_module(
            "uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg"
        )
        assert module.BaseEventCfg().robot_wrist_armature is None


@pytest.mark.isaacsim_ci
def test_sysid_armature_startup_selects_wrist_joints(monkeypatch):
    from isaaclab.managers import SceneEntityCfg

    module = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.events")
    names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    nominal = [3.0, 1.2, 1.4, 0.17, 0.08, 0.38]
    written = []
    robot = SimpleNamespace(
        cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path="fixture/robot.usd")),
        device="cpu",
        joint_names=names,
        find_joints=lambda selected: ([names.index(name) for name in selected], selected),
        write_joint_armature_to_sim_index=lambda **kwargs: written.append(kwargs),
    )
    monkeypatch.setattr(module.utils, "read_metadata_from_usd_directory", lambda path: {"sysid": {"armature": nominal}})
    env = SimpleNamespace(scene={"robot": robot}, num_envs=4)
    ids = torch.tensor([1, 3])
    module.set_armature_from_sysid(env, ids, SceneEntityCfg("robot", joint_names=names[3:]))
    assert written[0]["joint_ids"] == [3, 4, 5]
    assert torch.equal(written[0]["env_ids"], ids)
    torch.testing.assert_close(written[0]["armature"], torch.tensor([nominal[3:], nominal[3:]]), rtol=0, atol=0)


@pytest.mark.isaacsim_ci
def test_armature_curriculum_preserves_startup_baseline():
    module = importlib.import_module("uwlab_tasks.manager_based.manipulation.omnireset.mdp.events")
    initial = torch.tensor([[0.0, 0.0, 0.0, 0.17, 0.08, 0.38]]).repeat(4, 1)
    current = initial.clone()
    written = []

    def write_armature(armature, joint_ids, env_ids):
        current[env_ids[:, None], joint_ids] = armature
        written.append(armature.clone())

    robot = SimpleNamespace(
        device="cpu",
        data=SimpleNamespace(joint_armature=SimpleNamespace(torch=current)),
        actuators={"arm": SimpleNamespace()},
        write_joint_armature_to_sim_index=write_armature,
        write_joint_friction_coefficient_to_sim_index=lambda **kwargs: None,
    )
    term = module.randomize_arm_from_sysid.__new__(module.randomize_arm_from_sysid)
    term.robot = robot
    term.joint_ids = list(range(6))
    term.actuator_name = "arm"
    term.armature = [3.0, 1.2, 1.4, 0.17, 0.08, 0.38]
    term.static_friction = term.dynamic_ratio = term.viscous_friction = [1.0] * 6
    term._initial_armature = None
    ids = torch.tensor([1, 3])
    for progress in (0.0, 1.0, 0.5):
        term.scale_progress = progress
        term(None, ids, None, [], "arm", scale_range=(1.0, 1.0), delay_range=(0, 0))
        expected = initial[ids] * (1.0 - progress) + torch.tensor(term.armature).repeat(2, 1) * progress
        torch.testing.assert_close(written[-1], expected, rtol=0, atol=0)
        torch.testing.assert_close(current[[0, 2]], initial[[0, 2]], rtol=0, atol=0)
