# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm

from uwlab_tasks.manager_based.manipulation.factory_extension.factory_tasks import CtrlCfg, ObsRandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg_nist import TaskSpaceBoundedDifferentialInverseKinematicsActionNISTCfg


class TaskSpaceBoundedDifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: TaskSpaceBoundedDifferentialInverseKinematicsActionNISTCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    def __init__(self, cfg: TaskSpaceBoundedDifferentialInverseKinematicsActionNISTCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self.task_name = ""

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        self.ee_stiffness = torch.tensor(CtrlCfg.default_task_prop_gains, device=self.device).repeat((self.num_envs, 1))
        self.ee_damping = 2 * torch.sqrt(self.ee_stiffness)

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    """
    Operations.
    """
    # WE CAN MAKE ANOTHER CLASS, THE MAIN DIFFERENCE WILL BE HERE, WHAT WE CAN DO IN ADDITION
    # TO HAVING CLIP IS THE FOLLOWING
    """

    """

    def process_actions(self, actions: torch.Tensor):
        """
        Action preprocess

        1. threshold for x,y,z
        2. threshold for r,p,y
        3. modify the yaw properly
        4. larger emphasis on past actions then current actions <-- when did this happen, looking over the code I do not see this
        """

        self._processed_actions = (
            CtrlCfg.ema_factor * actions.to(self.device) + (1 - CtrlCfg.ema_factor) * self._raw_actions
        )
        self._raw_actions[:] = actions

        self._processed_actions[:, 0:3] *= torch.tensor(CtrlCfg.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        if self.cfg.task_cfg.unidirectional_rot:
            self._processed_actions[:, 5] = -(self._processed_actions[:, 5] + 1) * 0.5
        self._processed_actions[:, 3:] *= torch.tensor(CtrlCfg.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # This is not even needed since the scale is 1.0
        self._processed_actions[:] = self._processed_actions * self._scale

        """
        This is clip respect to the center of mid gripper, this will set the max and min value of the delta
        gripper pos.

        Restrictions:
        - have the actions be within 5cm of the tip of the fixed asset
        - have the roll and pitch be fixed.
            - roll = 3.141519 - pi
            - pitch = 0.0
        """
        # set some useful reference to environment assets states

        fixed_asset: Articulation = self._env.scene["fixed_asset"]
        fixed_pos = fixed_asset.data.root_link_pos_w - self._env.scene.env_origins
        fixed_quat = fixed_asset.data.root_link_quat_w
        fixed_tip_pos_local = torch.zeros_like(fixed_pos)

        # here we want to be able to get the
        fixed_tip_pos_local[:, 2] += self.cfg.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg.fixed_asset_cfg.base_height
        fixed_tip_pos, _ = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, fixed_tip_pos_local)

        fixed_asset_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(ObsRandCfg.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)

        # clip begins
        fixed_pos_action_frame = fixed_tip_pos + fixed_asset_pos_noise

        pos_actions = self._processed_actions[:, 0:3]
        rot_actions = self._processed_actions[:, 3:6]

        self.fingertip_midpoint_quat = self._asset.data.body_link_quat_w[:, self._body_idx]
        self.fingertip_midpoint_pos = self._asset.data.body_link_pos_w[:, self._body_idx] - self._env.scene.env_origins

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Make sure that the actions are within 5cm of the tip of the fixed asset
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(delta_pos, -CtrlCfg.pos_action_bounds[0], CtrlCfg.pos_action_bounds[1])
        self.ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = math_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = math_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(math_utils.euler_xyz_from_quat(self.ctrl_target_fingertip_midpoint_quat), dim=1)

        """
        Restrictions for the roll and pitch
        """
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = math_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        # THIS WE WILL BE DOING WITH THE INVERSE KINEMATICS
        self.ctrl_target_gripper_dof_pos = 0.0

    def apply_actions(self):
        joint_torque = torch.zeros((self.num_envs, self._num_joints), device=self.device)

        # to get the pos and quat for the end effector
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # """
        # Use axis angle in order to get (x,y,z) and (rx,ry,rz)

        # This is so that we can get pose
        # """
        pos_error, axis_angle_error = math_utils.compute_pose_error(
            ee_pos_curr,
            ee_quat_curr,
            self.ctrl_target_fingertip_midpoint_pos,
            self.ctrl_target_fingertip_midpoint_quat,
            rot_error_type="axis_angle",
        )
        # set command into controller

        delta_fingertip_pose = torch.cat((pos_error, axis_angle_error), dim=1)

        task_wrench = torch.zeros((self.num_envs, 6), device=self.device)

        def _apply_task_space_gains():
            task_wrench = torch.zeros_like(delta_fingertip_pose)

            # Apply gains to lin error components
            lin_error = delta_fingertip_pose[:, 0:3]

            default_gains = torch.tensor(CtrlCfg.default_task_prop_gains, device=self.device).repeat((self.num_envs, 1))

            task_deriv_gains = 2 * torch.sqrt(default_gains)

            task_wrench[:, 0:3] = default_gains[:, 0:3] * lin_error + task_deriv_gains[:, 0:3] * (
                0.0 - self._asset.data.body_lin_vel_w[..., self._body_idx, :]
            )

            # Apply gains to rot error components
            rot_error = delta_fingertip_pose[:, 3:6]
            task_wrench[:, 3:6] = default_gains[:, 3:6] * rot_error + task_deriv_gains[:, 3:6] * (
                0.0 - self._asset.data.body_ang_vel_w[..., self._body_idx, :]
            )
            return task_wrench

        task_wrench_motion = _apply_task_space_gains()

        task_wrench += task_wrench_motion

        # Set tau = J^T * tau, i.e., map tau into joint space as desired
        jacobian = self._compute_frame_jacobian()
        jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
        joint_torque[:, 0:7] = (jacobian_T @ task_wrench.unsqueeze(-1)).squeeze(-1)

        # # adapted from https://gitlab-master.nvidia.com/carbon-gym/carbgym/-/blob/b4bbc66f4e31b1a1bee61dbaafc0766bbfbf0f58/python/examples/franka_cube_ik_osc.py#L70-78
        # # roboticsproceedings.org/rss07/p31.pdf

        arm_mass_matrix = self._asset.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        # useful tensors
        arm_mass_matrix_inv = torch.inverse(arm_mass_matrix)
        arm_mass_matrix_task = torch.inverse(
            jacobian @ torch.inverse(arm_mass_matrix) @ jacobian_T
        )  # ETH eq. 3.86; geometric Jacobian is assumed
        j_eef_inv = arm_mass_matrix_task @ jacobian @ arm_mass_matrix_inv
        default_dof_pos_tensor = torch.tensor(CtrlCfg.default_dof_pos_tensor, device=self.device).repeat(
            (self.num_envs, 1)
        )
        # nullspace computation
        distance_to_default_dof_pos = default_dof_pos_tensor - self._asset.data.joint_pos[:, :7]
        distance_to_default_dof_pos = (distance_to_default_dof_pos + torch.pi) % (
            2 * torch.pi
        ) - torch.pi  # normalize to [-pi, pi]

        # null space control
        u_null = CtrlCfg.kd_null * -self._asset.data.joint_vel[:, :7] + CtrlCfg.kp_null * distance_to_default_dof_pos
        u_null = arm_mass_matrix @ u_null.unsqueeze(-1)
        torque_null = (
            torch.eye(7, device=self.device).unsqueeze(0) - torch.transpose(jacobian, 1, 2) @ j_eef_inv
        ) @ u_null
        joint_torque[:, 0:7] += torque_null.squeeze(-1)

        joint_torque = torch.clamp(joint_torque, min=-100.0, max=100.0)

        self._asset.set_joint_effort_target(joint_torque, joint_ids=self._joint_ids)

        self._asset.set_joint_position_target(torch.zeros((self.num_envs, 2), device=self.device), joint_ids=[7, 8])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """
        This is the reset portion for actions

        Code references from factory_env.py's randomize_initial_state
        """
        if self.task_name == "":
            self.task_name: str = self.cfg.task_cfg.name

        self._raw_actions[env_ids] = 0.0

        # Custom part from NIST assembly - reset
        fixed_asset: Articulation = self._env.scene["fixed_asset"]
        robot: Articulation = self._env.scene["robot"]
        fixed_pos = fixed_asset.data.root_link_pos_w - self._env.scene.env_origins
        fixed_quat = fixed_asset.data.root_link_quat_w

        fixed_tip_pos_local = torch.zeros_like(fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg.fixed_asset_cfg.height + self.cfg.fixed_asset_cfg.base_height
        fixed_tip_pos, _ = math_utils.combine_frame_transforms(fixed_pos, fixed_quat, fixed_tip_pos_local)

        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(ObsRandCfg.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise = fixed_asset_pos_noise
        fixed_pos_action_frame = fixed_tip_pos + self.init_fixed_pos_obs_noise
        fingertip_midpoint_quat = robot.data.body_link_quat_w[:, self._body_idx]

        fingertip_midpoint_pos = robot.data.body_link_pos_w[:, self._body_idx] - self._env.scene.env_origins

        pos_actions = fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(CtrlCfg.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self._raw_actions[:, 0:3] = pos_actions

        # Relative yaw to fixed asset.
        unrot_180_euler = torch.tensor([np.pi, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        unrot_quat = math_utils.quat_from_euler_xyz(
            roll=unrot_180_euler[:, 0], pitch=unrot_180_euler[:, 1], yaw=unrot_180_euler[:, 2]
        )

        fingertip_quat_rel_fixed_asset = math_utils.quat_mul(unrot_quat, fingertip_midpoint_quat)
        fingertip_yaw_fixed_asset = math_utils.euler_xyz_from_quat(fingertip_quat_rel_fixed_asset)[-1]
        fingertip_yaw_fixed_asset = torch.where(
            fingertip_yaw_fixed_asset > torch.pi / 2,
            fingertip_yaw_fixed_asset - 2 * torch.pi,
            fingertip_yaw_fixed_asset,
        )
        fingertip_yaw_fixed_asset = torch.where(
            fingertip_yaw_fixed_asset < -torch.pi, fingertip_yaw_fixed_asset + 2 * torch.pi, fingertip_yaw_fixed_asset
        )

        yaw_action = (fingertip_yaw_fixed_asset + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self._raw_actions[:, 5] = yaw_action

        self._target_joint_pos_at_reset = robot.data.joint_pos_target.clone()

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pos_w = self._asset.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_link_pos_w
        root_quat_w = self._asset.data.root_link_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        # account for the offset

        ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self.jacobian_b
        # account for the offset
        # Modify the jacobian to account for the offset
        # -- translational part
        # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
        #        = (v_J_ee + w_J_ee x r_link_ee ) * q
        #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        # -- rotational part
        # w_link = R_link_ee @ w_ee
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian
