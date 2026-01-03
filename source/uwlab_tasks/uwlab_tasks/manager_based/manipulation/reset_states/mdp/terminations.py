# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for manipulation tasks."""

import numpy as np
import torch

import isaacsim.core.utils.bounds as bounds_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import math as math_utils

from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils

from .collision_analyzer_cfg import CollisionAnalyzerCfg


def _check_obb_overlap(centroids_a, axes_a, half_extents_a, centroids_b, axes_b, half_extents_b) -> torch.Tensor:
    """
    OBB overlap check.

    Args:
        centroids_a: Centers of OBB A for all envs (num_envs, 3) - torch tensor on GPU
        axes_a: Orientation axes of OBB A for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_a: Half extents of OBB A (3,) - torch tensor on GPU
        centroids_b: Centers of OBB B for all envs (num_envs, 3) - torch tensor on GPU
        axes_b: Orientation axes of OBB B for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_b: Half extents of OBB B (3,) - torch tensor on GPU

    Returns:
        torch.Tensor: Boolean tensor (num_envs,) indicating overlap for each environment
    """
    num_envs = centroids_a.shape[0]
    device = centroids_a.device

    # Vector between centroids for all envs (num_envs, 3)
    d = centroids_b - centroids_a

    # Matrix C = A^T * B (rotation from A to B) for all envs (num_envs, 3, 3)
    C = torch.bmm(axes_a.transpose(1, 2), axes_b)
    abs_C = torch.abs(C)

    # Initialize overlap results (assume all overlap initially)
    overlap_results = torch.ones(num_envs, device=device, dtype=torch.bool)

    # Test all axes of A at once (vectorized across all 3 axes and all environments)
    # axes_a: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_a = torch.abs(torch.bmm(d.unsqueeze(1), axes_a).squeeze(1))  # (num_envs, 3)
    ra_all = half_extents_a.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    rb_all = torch.sum(half_extents_b.unsqueeze(0).unsqueeze(0) * abs_C, dim=2)  # (num_envs, 3)
    no_overlap_a = projections_on_axes_a > (ra_all + rb_all)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_a, dim=1)  # (num_envs,)

    # Test all axes of B at once (vectorized across all 3 axes and all environments)
    # axes_b: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_b = torch.abs(torch.bmm(d.unsqueeze(1), axes_b).squeeze(1))  # (num_envs, 3)
    ra_all_b = torch.sum(half_extents_a.unsqueeze(0).unsqueeze(0) * abs_C.transpose(1, 2), dim=2)  # (num_envs, 3)
    rb_all_b = half_extents_b.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    no_overlap_b = projections_on_axes_b > (ra_all_b + rb_all_b)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_b, dim=1)  # (num_envs,)

    # Test all cross products at once (9 cross products per environment)
    # Reshape axes for broadcasting: axes_a (num_envs, 3, 1, 3), axes_b (num_envs, 1, 3, 3)
    axes_a_expanded = axes_a.unsqueeze(2)  # (num_envs, 3, 1, 3)
    axes_b_expanded = axes_b.unsqueeze(1)  # (num_envs, 1, 3, 3)

    # Compute all 9 cross products at once: (num_envs, 3, 3, 3)
    cross_products = torch.cross(axes_a_expanded, axes_b_expanded, dim=3)  # (num_envs, 3, 3, 3)

    # Compute norms and filter out near-parallel axes: (num_envs, 3, 3)
    cross_norms = torch.norm(cross_products, dim=3)  # (num_envs, 3, 3)
    valid_crosses = cross_norms > 1e-6  # (num_envs, 3, 3)

    # Normalize cross products (set invalid ones to zero)
    normalized_crosses = torch.where(
        valid_crosses.unsqueeze(3),
        cross_products / cross_norms.unsqueeze(3).clamp(min=1e-6),
        torch.zeros_like(cross_products),
    )  # (num_envs, 3, 3, 3)

    # Project d onto all cross product axes: (num_envs, 3, 3)
    d_expanded = d.unsqueeze(1).unsqueeze(1)  # (num_envs, 1, 1, 3)
    projections_cross = torch.abs(torch.sum(d_expanded * normalized_crosses, dim=3))  # (num_envs, 3, 3)

    # Compute ra for all cross products: (num_envs, 3, 3)
    # half_extents_a: (3,), axes_a: (num_envs, 3, 3), normalized_crosses: (num_envs, 3, 3, 3)
    axes_a_cross_dots = torch.abs(
        torch.sum(axes_a.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(3), dim=4)
    )  # (num_envs, 3, 3, 3)
    ra_cross = torch.sum(
        half_extents_a.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_a_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Compute rb for all cross products: (num_envs, 3, 3)
    axes_b_cross_dots = torch.abs(
        torch.sum(axes_b.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(4), dim=4)
    )  # (num_envs, 3, 3, 3)
    rb_cross = torch.sum(
        half_extents_b.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_b_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Check separating condition for all cross products: (num_envs, 3, 3)
    no_overlap_cross = projections_cross > (ra_cross + rb_cross)  # (num_envs, 3, 3)
    # Only consider valid cross products
    no_overlap_cross_valid = no_overlap_cross & valid_crosses  # (num_envs, 3, 3)
    overlap_results &= ~torch.any(no_overlap_cross_valid.view(num_envs, -1), dim=1)  # (num_envs,)

    return overlap_results


class check_grasp_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfg = cfg.params.get("object_cfg")
        self.gripper_cfg = cfg.params.get("gripper_cfg")
        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)
        self.max_pos_deviation = cfg.params.get("max_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)
        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        object_asset = self._env.scene[self.object_cfg.name]
        if not hasattr(object_asset, "initial_pos"):
            object_asset.initial_pos = object_asset.data.root_pos_w.clone()
            object_asset.initial_quat = object_asset.data.root_quat_w.clone()
        else:
            object_asset.initial_pos[env_ids] = object_asset.data.root_pos_w[env_ids].clone()
            object_asset.initial_quat[env_ids] = object_asset.data.root_quat_w[env_ids].clone()

        if env_ids is None:
            self.stability_counter.zero_()
        else:
            self.stability_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        collision_analyzer_cfg: CollisionAnalyzerCfg,
        max_pos_deviation: float = 0.05,
        pos_z_threshold: float = 0.05,
        consecutive_stability_steps: int = 5,
    ) -> torch.Tensor:
        # Get object and gripper from scene
        object_asset = env.scene[self.object_cfg.name]
        gripper_asset = env.scene[self.gripper_cfg.name]

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (gripper_asset.data.joint_vel.abs() > (gripper_asset.data.joint_vel_limits * 2)).any(
            dim=1
        )

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        # Check gripper (articulation) velocities
        current_step_stable &= gripper_asset.data.joint_vel.abs().sum(dim=1) < 5.0
        # Check object (rigid object) velocities
        if isinstance(object_asset, RigidObject):
            current_step_stable &= object_asset.data.body_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.body_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0
        elif isinstance(object_asset, RigidObjectCollection):
            current_step_stable &= object_asset.data.object_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.object_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Skip if position or quaternion is NaN
        pos_is_nan = torch.isnan(object_asset.data.root_pos_w).any(dim=1)
        quat_is_nan = torch.isnan(object_asset.data.root_quat_w).any(dim=1)
        skip_check = pos_is_nan | quat_is_nan

        # Object has excessive pose deviation if position exceeds thresholds
        pos_deviation = (object_asset.data.root_pos_w - object_asset.initial_pos).norm(dim=1)
        valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
        excessive_pose_deviation = valid_pos_deviation > self.max_pos_deviation

        # Object is above ground if position is greater than z threshold
        pos_above_ground = object_asset.data.root_pos_w[:, 2] >= self.pos_z_threshold

        # Check for collisions between gripper and object
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        grasp_success = (
            (~abnormal_gripper_state)
            & stability_reached
            & (~excessive_pose_deviation)
            & pos_above_ground
            & collision_free
            & time_out
        )

        return grasp_success


class check_reset_state_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfgs = cfg.params.get("object_cfgs")
        self.robot_cfg = cfg.params.get("robot_cfg")
        self.ee_body_name = cfg.params.get("ee_body_name")
        self.collision_analyzer_cfgs = cfg.params.get("collision_analyzer_cfgs")
        self.collision_analyzers = [
            collision_analyzer_cfg.class_type(collision_analyzer_cfg, self._env)
            for collision_analyzer_cfg in self.collision_analyzer_cfgs
        ]
        self.max_robot_pos_deviation = cfg.params.get("max_robot_pos_deviation")
        self.max_object_pos_deviation = cfg.params.get("max_object_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)

        # Load gripper_approach_direction from metadata
        robot_asset = env.scene[self.robot_cfg.name]
        usd_path = robot_asset.cfg.spawn.usd_path
        metadata = utils.read_metadata_from_usd_directory(usd_path)
        self.gripper_approach_direction = tuple(metadata.get("gripper_approach_direction"))

        # Initialize stability counter for consecutive stability checking
        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

        self.object_assets = [env.scene[cfg.name] for cfg in self.object_cfgs]
        self.robot_asset = env.scene[self.robot_cfg.name]
        self.assets_to_check = self.object_assets + [self.robot_asset]
        self.ee_body_idx = self.robot_asset.data.body_names.index(self.ee_body_name)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        for asset in self.assets_to_check:
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.clone()
            if not hasattr(asset, "initial_pos"):
                asset.initial_pos = asset_pos
            else:
                asset.initial_pos[env_ids] = asset_pos[env_ids].clone()

        self.stability_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfgs: list[SceneEntityCfg],
        robot_cfg: SceneEntityCfg,
        ee_body_name: str,
        collision_analyzer_cfgs: list[CollisionAnalyzerCfg],
        max_robot_pos_deviation: float = 0.1,
        max_object_pos_deviation: float = 0.1,
        pos_z_threshold: float = -0.01,
        consecutive_stability_steps: int = 5,
    ) -> torch.Tensor:

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (
            self.robot_asset.data.joint_vel.abs() > (self.robot_asset.data.joint_vel_limits * 2)
        ).any(dim=1)

        # Check if gripper orientation is pointing downward within 60 degrees of vertical
        ee_quat = self.robot_asset.data.body_link_quat_w[:, self.ee_body_idx]
        gripper_approach_local = torch.tensor(
            self.gripper_approach_direction, device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        gripper_approach_world = math_utils.quat_apply(ee_quat, gripper_approach_local)
        gripper_orientation_within_range = (
            gripper_approach_world[:, 2] < -0.5
        )  # cos(60°) = 0.5, so z < -0.5 for 60° cone

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for asset in self.assets_to_check:
            if isinstance(asset, Articulation):
                current_step_stable &= asset.data.joint_vel.abs().sum(dim=1) < 5.0
            elif isinstance(asset, RigidObject):
                current_step_stable &= asset.data.body_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.body_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0
            elif isinstance(asset, RigidObjectCollection):
                current_step_stable &= asset.data.object_lin_vel_w.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.object_ang_vel_w.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Reset initial positions on first check or after env reset
        excessive_pose_deviation = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        pos_below_threshold = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for asset in self.assets_to_check:
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.clone()

            # Skip if position or quaternion is NaN
            pos_is_nan = torch.isnan(asset.data.root_pos_w).any(dim=1)
            quat_is_nan = torch.isnan(asset.data.root_quat_w).any(dim=1)
            skip_check = pos_is_nan | quat_is_nan

            # Asset has excessive pose deviation if position exceeds thresholds
            pos_deviation = (asset_pos - asset.initial_pos).norm(dim=1)
            valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
            if asset is self.robot_asset:
                excessive_pose_deviation |= valid_pos_deviation > self.max_robot_pos_deviation
            else:
                excessive_pose_deviation |= valid_pos_deviation > self.max_object_pos_deviation

            # Asset is above ground if position is greater than z threshold
            pos_below_threshold |= asset_pos[:, 2] < self.pos_z_threshold

        # Check for collisions between gripper and object
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = torch.all(
            torch.stack([collision_analyzer(env, all_env_ids) for collision_analyzer in self.collision_analyzers]),
            dim=0,
        )

        reset_success = (
            (~abnormal_gripper_state)
            & gripper_orientation_within_range
            & stability_reached
            & (~excessive_pose_deviation)
            & (~pos_below_threshold)
            & collision_free
            & time_out
        )

        return reset_success


class check_obb_no_overlap_termination(ManagerTermBase):
    """Termination condition that checks if OBBs of two objects no longer overlap."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

        insertive_metadata = utils.read_metadata_from_usd_directory(self.insertive_object.cfg.spawn.usd_path)

        self.insertive_target_mesh_path = insertive_metadata.get("target_mesh_path")
        self.enable_visualization = cfg.params.get("enable_visualization", False)

        # Initialize OBB computation cache and compute OBBs once
        self._bbox_cache = bounds_utils.create_bbox_cache()
        self._compute_object_obbs()

        # Initialize placeholders for initial poses (will be set in reset)
        self._insertive_initial_pos = None
        self._insertive_initial_quat = None

        # Store debug draw interface if visualization is enabled
        if self.enable_visualization:
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._omni_debug_draw = omni_debug_draw
        else:
            self._omni_debug_draw = None

    def _compute_object_obbs(self):
        """Compute OBB for insertive object and convert to body frame."""
        # Get prim path (use env 0 as template)
        insertive_base_path = self.insertive_object.cfg.prim_path.replace(".*", "0", 1)

        # Determine object prim path - use specific mesh if provided
        if self.insertive_target_mesh_path is not None:
            insertive_prim_path = f"{insertive_base_path}/{self.insertive_target_mesh_path}"
        else:
            insertive_prim_path = insertive_base_path

        # Compute OBB in world frame using Isaac Sim's built-in functions
        insertive_centroid_world, insertive_axes_world, insertive_half_extents = bounds_utils.compute_obb(
            self._bbox_cache, insertive_prim_path
        )

        # Get current world pose of object (env 0) to convert OBB to body frame
        insertive_pos_world = self.insertive_object.data.root_pos_w[0]  # (3,)
        insertive_quat_world = self.insertive_object.data.root_quat_w[0]  # (4,)

        device = self._env.device

        # Convert world frame OBB data to torch tensors
        insertive_centroid_world_tensor = torch.tensor(insertive_centroid_world, device=device, dtype=torch.float32)
        insertive_axes_world_tensor = torch.tensor(insertive_axes_world, device=device, dtype=torch.float32)

        # Convert centroid from world frame to body frame
        insertive_centroid_body = math_utils.quat_apply_inverse(
            insertive_quat_world, insertive_centroid_world_tensor - insertive_pos_world
        )

        # Convert axes from world frame to body frame
        insertive_rot_matrix_world = math_utils.matrix_from_quat(insertive_quat_world.unsqueeze(0))[0]  # (3, 3)

        # Transform axes: R_world_to_body @ world_axes = R_world^T @ world_axes
        # Note: Isaac Sim's compute_obb returns axes as column vectors, so we need to transpose
        insertive_axes_body = torch.matmul(insertive_rot_matrix_world.T, insertive_axes_world_tensor.T).T

        # Cache OBB data in body frame as torch tensors on device for fast access
        self._insertive_obb_centroid = insertive_centroid_body
        self._insertive_obb_axes = insertive_axes_body
        self._insertive_obb_half_extents = torch.tensor(insertive_half_extents, device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Store initial pose of insertive object when environments are reset."""
        super().reset(env_ids)

        insertive_pos = self.insertive_object.data.root_pos_w.clone()
        insertive_quat = self.insertive_object.data.root_quat_w.clone()

        if self._insertive_initial_pos is None or self._insertive_initial_quat is None or env_ids is None:
            # First time initialization or reset all environments
            self._insertive_initial_pos = insertive_pos
            self._insertive_initial_quat = insertive_quat
        else:
            # Update only the reset environments
            self._insertive_initial_pos[env_ids] = insertive_pos[env_ids]
            self._insertive_initial_quat[env_ids] = insertive_quat[env_ids]

    def _compute_obb_corners_batch(self, centroids, axes, half_extents):
        """
        Compute the 8 corners of Oriented Bounding Boxes for all environments using Isaac Sim's built-in function.

        Args:
            centroids: Centers of OBBs (num_envs, 3)
            axes: Orientation axes of OBBs (num_envs, 3, 3) - rows are the axes
            half_extents: Half extents of OBB along its axes (3,)

        Returns:
            corners: 8 corners of the OBBs (num_envs, 8, 3)
        """
        num_envs = centroids.shape[0]
        device = centroids.device

        # Convert torch tensors to numpy for Isaac Sim functions
        centroids_np = centroids.detach().cpu().numpy()
        axes_np = axes.detach().cpu().numpy()
        half_extents_np = half_extents.detach().cpu().numpy()

        # Compute corners for each environment using Isaac Sim's function
        all_corners = []
        for env_idx in range(num_envs):
            # Use Isaac Sim's get_obb_corners function
            corners_np = bounds_utils.get_obb_corners(
                centroids_np[env_idx], axes_np[env_idx], half_extents_np
            )  # (8, 3)
            all_corners.append(corners_np)

        # Convert back to torch tensor
        corners_tensor = torch.tensor(np.stack(all_corners), device=device, dtype=torch.float32)
        return corners_tensor  # (num_envs, 8, 3)

    def _visualize_bounding_boxes(self, env: ManagerBasedEnv):
        """Visualize oriented bounding boxes for initial and current insertive object positions using wireframe edges."""
        # Clear previous debug lines
        draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()
        draw_interface.clear_lines()

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w  # (num_envs, 4)

        # Transform current insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform current insertive object OBB orientation from body frame to world coordinates for all environments
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_obb_axes_body = self._insertive_obb_axes
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for current position visualization using Isaac Sim's built-in function
        insertive_current_corners = self._compute_obb_corners_batch(
            insertive_current_world_centroids, insertive_current_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Transform initial insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object OBB orientation from body frame to world coordinates
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)
        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for initial position visualization using Isaac Sim's built-in function
        insertive_initial_corners = self._compute_obb_corners_batch(
            insertive_initial_world_centroids, insertive_initial_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Draw wireframe boxes for each environment
        for env_idx in range(env.num_envs):
            # Draw current insertive object bounding box edges (blue)
            self._draw_obb_wireframe(
                insertive_current_corners[env_idx],  # (8, 3)
                color=(0.0, 0.5, 1.0, 1.0),  # Bright blue
                line_width=4.0,
                draw_interface=draw_interface,
            )

            # Draw initial insertive object bounding box edges (red)
            self._draw_obb_wireframe(
                insertive_initial_corners[env_idx],  # (8, 3)
                color=(1.0, 0.2, 0.0, 1.0),  # Bright red
                line_width=4.0,
                draw_interface=draw_interface,
            )

    def _draw_obb_wireframe(
        self, corners: torch.Tensor, color: tuple = (1.0, 1.0, 1.0, 1.0), line_width: float = 2.0, draw_interface=None
    ):
        """
        Draw wireframe edges of an oriented bounding box.

        Args:
            corners: 8 corners of the OBB (8, 3)
            color: RGBA color tuple for the lines
            line_width: Width of the lines
            draw_interface: Debug draw interface (optional, will acquire if not provided)
        """
        # Define the edges of a cube by connecting corner indices
        # Corners are ordered as: [0-3] bottom face, [4-7] top face
        edge_indices = [
            # Bottom face edges
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical edges connecting bottom to top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            # Diagonal X on top face (4-5-6-7)
            (4, 6),
            (5, 7),
            # Diagonal X on bottom face (0-1-2-3)
            (0, 2),
            (1, 3),
            # Diagonal X on front face (0-1-4-5)
            (0, 5),
            (1, 4),
            # Diagonal X on back face (2-3-6-7)
            (2, 7),
            (3, 6),
            # Diagonal X on left face (0-3-4-7)
            (0, 7),
            (3, 4),
            # Diagonal X on right face (1-2-5-6)
            (1, 6),
            (2, 5),
        ]

        # Create line segments for all edges
        line_starts = []
        line_ends = []

        for start_idx, end_idx in edge_indices:
            line_starts.append(corners[start_idx].cpu().numpy().tolist())
            line_ends.append(corners[end_idx].cpu().numpy().tolist())

        # Use provided interface or acquire new one
        if draw_interface is None:
            draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()

        colors = [list(color)] * len(edge_indices)
        line_thicknesses = [line_width] * len(edge_indices)

        # Draw all edges at once
        draw_interface.draw_lines(line_starts, line_ends, colors, line_thicknesses)

    def __call__(
        self,
        env: ManagerBasedEnv,
        insertive_object_cfg: SceneEntityCfg,
        enable_visualization: bool = False,
    ) -> torch.Tensor:
        """Check if OBB overlap condition is violated between initial and current insertive object positions."""

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w  # (num_envs, 4)

        # Transform current insertive object centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform OBB axes to world coordinates
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)

        insertive_obb_axes_body = self._insertive_obb_axes

        # Transform axes from body frame to world frame: R @ body_axes for all environments
        # Since axes are stored as row vectors, we need to handle the transpose properly
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Check OBB overlap between initial and current insertive object positions for all environments
        obb_overlap = _check_obb_overlap(
            insertive_current_world_centroids,
            insertive_current_world_axes,
            self._insertive_obb_half_extents,
            insertive_initial_world_centroids,
            insertive_initial_world_axes,
            self._insertive_obb_half_extents,
        )

        # Visualize bounding boxes if enabled
        if self.enable_visualization:
            self._visualize_bounding_boxes(env)

        return ~obb_overlap
