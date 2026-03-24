# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene and env config for sim2real camera alignment.

Minimal env with robot + cameras (from data_collection_rgb_cfg) but NO
randomization.  The interactive alignment script (scripts_v2/tools/sim2real/align_cameras.py)
uses keyboard controls to move/rotate the sim camera and overlay the sim
render on a real reference image, then prints the final (pos, rot, focal_length)
to paste back into data_collection_rgb_cfg.py.

Mirrors the sysid pattern:
  sysid_cfg.py  +  scripts_v2/tools/sim2real/sysid_ur5e_osc.py
  camera_align_cfg.py  +  scripts_v2/tools/sim2real/align_cameras.py
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85SysidOSCAction
from .rl_state_cfg import RlStateSceneCfg

# Same sim dt as sysid / finetune (500 Hz)
CAMERA_ALIGN_SIM_DT = 1.0 / 500.0


@configclass
class CameraAlignSceneCfg(RlStateSceneCfg):
    """Scene for camera alignment.

    Inherits from RlStateSceneCfg (robot, table, ur5_metal_support, ground,
    sky_light, insertive/receptive objects) and adds curtains + cameras.
    Same structure as DataCollectionRGBObjectSceneCfg but with NO randomization.
    """

    # Use explicit (sysid-tuned) actuator model
    robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # --- Background curtains (match real workspace) ---
    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    # --- Cameras (initial poses from data_collection_rgb_cfg) ---
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )


# ---------------------------------------------------------------------------
# Minimal MDP (camera alignment only needs RGB obs + joint_pos action)
# ---------------------------------------------------------------------------
@configclass
class CameraAlignObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )
        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )
        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CameraAlignRewardsCfg:
    pass


@configclass
class CameraAlignTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class CameraAlignEnvCfg(ManagerBasedRLEnvCfg):
    """Env for interactive sim2real camera alignment.

    Uses the same robot/action as sysid so the robot can be positioned
    at arbitrary joint angles.  Only 1 env needed (interactive tool).
    """

    scene: CameraAlignSceneCfg = CameraAlignSceneCfg(num_envs=1, env_spacing=2.0)
    actions: Ur5eRobotiq2f85SysidOSCAction = Ur5eRobotiq2f85SysidOSCAction()
    observations: CameraAlignObservationsCfg = CameraAlignObservationsCfg()
    rewards: CameraAlignRewardsCfg = CameraAlignRewardsCfg()
    terminations: CameraAlignTerminationsCfg = CameraAlignTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1
        self.episode_length_s = 99999.0
        self.sim.dt = CAMERA_ALIGN_SIM_DT

        # Place robot at average real-world position (reset_states_cfg y avg = -0.039).
        self.scene.robot.init_state.pos = (0.0, -0.039, 0.0)
        self.scene.ur5_metal_support.init_state.pos = (0.0, -0.039, -0.013)

        # Render settings for visual fidelity
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render_interval = 1

        # rerender on reset
        self.num_rerenders_on_reset = 1
