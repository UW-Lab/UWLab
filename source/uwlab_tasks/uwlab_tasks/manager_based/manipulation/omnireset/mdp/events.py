# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for manipulation tasks."""

import logging
import numpy as np
import os
import random
import scipy.stats as stats
import torch
import trimesh
import trimesh.transformations as tra
from collections.abc import Sequence

import carb
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from pxr import Gf, UsdGeom, UsdLux

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils

from ..assembly_keypoints import Offset
from .success_monitor_cfg import SuccessMonitorCfg


class grasp_sampling_event(ManagerTermBase):
    """EventTerm class for grasp sampling and positioning gripper."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Extract parameters from config
        self.object_cfg = cfg.params.get("object_cfg")
        self.gripper_cfg = cfg.params.get("gripper_cfg")
        self.num_candidates = cfg.params.get("num_candidates")
        self.num_standoff_samples = cfg.params.get("num_standoff_samples")
        self.num_orientations = cfg.params.get("num_orientations")
        self.lateral_sigma = cfg.params.get("lateral_sigma")
        self.visualize_grasps = cfg.params.get("visualize_grasps", False)
        self.visualization_scale = cfg.params.get("visualization_scale", 0.03)

        # Read parameters from object metadata
        gripper_asset = env.scene[self.gripper_cfg.name]
        usd_path = gripper_asset.cfg.spawn.usd_path
        metadata = utils.read_metadata_from_usd_directory(usd_path)

        # Extract parameters from metadata
        self.gripper_maximum_aperture = metadata.get("maximum_aperture")
        self.finger_offset = metadata.get("finger_offset")
        self.finger_clearance = metadata.get("finger_clearance")
        self.gripper_approach_direction = tuple(metadata.get("gripper_approach_direction"))
        self.grasp_align_axis = tuple(metadata.get("grasp_align_axis"))
        self.orientation_sample_axis = tuple(metadata.get("orientation_sample_axis"))
        self.gripper_joint_reset_config = {"finger_joint": metadata.get("finger_open_joint_angle")}

        # Store environment reference for later use
        self._env = env

        # Grasp candidates will be generated lazily when first called
        self.grasp_candidates = None

        # Initialize pose markers for visualization
        if self.visualize_grasps:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
            frame_marker_cfg.markers["frame"].scale = (
                self.visualization_scale,
                self.visualization_scale,
                self.visualization_scale,
            )
            self.pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/grasp_poses"))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        object_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        num_candidates: int,
        num_standoff_samples: int,
        num_orientations: int,
        lateral_sigma: float,
        visualize_grasps: bool = False,
        visualization_scale: float = 0.01,
    ) -> None:
        """Execute grasp sampling event - sample from pre-computed candidates."""
        # Generate grasp candidates if not already done
        if self.grasp_candidates is None:
            candidates_list = self._generate_grasp_candidates()
            # Convert to tensor for efficient indexing
            self.grasp_candidates = torch.stack(
                [torch.tensor(candidate, dtype=torch.float32, device=env.device) for candidate in candidates_list]
            )

            # Visualize grasp poses if requested
            if self.visualize_grasps:
                self._visualize_grasp_poses(env, self.visualization_scale)

        # Get gripper from scene
        gripper_asset = env.scene[self.gripper_cfg.name]
        # First: Check for and fix any abnormal states before positioning
        self._ensure_stable_gripper_state(env, gripper_asset, env_ids)
        # Second: Open gripper to prepare for grasping
        self._open_gripper(env, gripper_asset, env_ids)
        # Randomly sample grasp candidates for the environments being reset
        num_envs_reset = len(env_ids)
        grasp_indices = torch.randint(0, len(self.grasp_candidates), (num_envs_reset,), device=env.device)

        # Apply grasp transforms to gripper (vectorized for multiple environments)
        sampled_transforms = self.grasp_candidates[grasp_indices]
        self._apply_grasp_transforms_vectorized(env, gripper_asset, sampled_transforms, env_ids)

        # Store grasp candidates for later evaluation
        if not hasattr(env, "grasp_candidates"):
            env.grasp_candidates = self.grasp_candidates
            env.current_grasp_idx = 0
            env.grasp_results = []

    def _generate_grasp_candidates(self):
        """Generate grasp candidates using antipodal grasp sampling."""
        object_asset = self._env.scene[self.object_cfg.name]
        mesh = self._extract_mesh_from_asset(object_asset)
        grasp_transforms = self._sample_antipodal_grasps(mesh)
        return grasp_transforms

    def _extract_mesh_from_asset(self, asset):
        """Extract trimesh from IsaacLab asset."""
        # Get USD stage and prim path from the asset
        stage = omni.usd.get_context().get_stage()

        # For multi-environment setups, we need to get the first environment's path
        prim_path = asset.cfg.prim_path.replace(".*", "0", 1)

        # Get the USD prim
        prim = stage.GetPrimAtPath(prim_path)

        # Find mesh geometry in the prim hierarchy
        mesh_schema = self._find_mesh_in_prim(prim)

        # Convert USD mesh to trimesh
        return self._usd_mesh_to_trimesh(mesh_schema)

    def _find_mesh_in_prim(self, prim):
        """Find the first mesh under a prim."""
        if prim.IsA(UsdGeom.Mesh):
            return UsdGeom.Mesh(prim)

        from pxr import Usd

        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                return UsdGeom.Mesh(child)
        return None

    def _usd_mesh_to_trimesh(self, usd_mesh):
        """Convert USD mesh to trimesh for grasp sampling."""
        # Get vertices
        points_attr = usd_mesh.GetPointsAttr()
        vertices = torch.tensor(points_attr.Get(), dtype=torch.float32)
        max_distance = torch.max(torch.norm(vertices, dim=1))
        # if the max distance is greater than 1.0, then the mesh is in mm
        if max_distance > 1.0:
            vertices = vertices / 1000.0

        # Get faces
        face_indices_attr = usd_mesh.GetFaceVertexIndicesAttr()
        face_counts_attr = usd_mesh.GetFaceVertexCountsAttr()

        vertex_indices = torch.tensor(face_indices_attr.Get(), dtype=torch.long)
        vertex_counts = torch.tensor(face_counts_attr.Get(), dtype=torch.long)

        # Convert to triangles
        triangles = []
        offset = 0
        for count in vertex_counts:
            indices = vertex_indices[offset : offset + count]
            if count == 3:
                triangles.append(indices.numpy())
            elif count == 4:
                # Split quad into two triangles
                triangles.extend([indices[[0, 1, 2]].numpy(), indices[[0, 2, 3]].numpy()])
            offset += count

        faces = torch.tensor(np.array(triangles), dtype=torch.long)
        return trimesh.Trimesh(vertices=vertices.numpy(), faces=faces.numpy(), process=False)

    def _sample_antipodal_grasps(self, mesh):
        """Sample antipodal grasp poses on a mesh using proper gripper parameterization."""
        # Extract parameters with defaults
        num_surface_samples = max(1, int(self.num_candidates // (self.num_orientations * self.num_standoff_samples)))

        # Normalize input vectors using torch
        gripper_approach_direction = torch.tensor(self.gripper_approach_direction, dtype=torch.float32)
        gripper_approach_direction = gripper_approach_direction / torch.norm(gripper_approach_direction)

        grasp_align_axis = torch.tensor(self.grasp_align_axis, dtype=torch.float32)
        grasp_align_axis = grasp_align_axis / torch.norm(grasp_align_axis)

        orientation_sample_axis = torch.tensor(self.orientation_sample_axis, dtype=torch.float32)
        orientation_sample_axis = orientation_sample_axis / torch.norm(orientation_sample_axis)

        # Simple mesh-adaptive standoff: use bounding box diagonal for size-aware clearance
        mesh_extents = mesh.extents
        mesh_diagonal = np.linalg.norm(mesh_extents)

        # Handle standoff distance(s) with mesh-adaptive bonus
        standoff_distances = torch.linspace(
            self.finger_offset,
            self.finger_offset + mesh_diagonal + self.finger_clearance / 2,
            self.num_standoff_samples,
        )

        max_gripper_width = self.gripper_maximum_aperture

        # Sample more points initially to allow for top-bias filtering
        initial_sample_size = num_surface_samples * 10  # Sample 10x more for filtering
        surface_points, face_indices = mesh.sample(initial_sample_size, return_index=True)
        surface_normals = mesh.face_normals[face_indices]

        # Bias toward top surfaces: prioritize points with higher Z coordinates and upward-facing normals
        z_coords = surface_points[:, 2]
        normal_z_components = surface_normals[:, 2]  # Z component of surface normals

        # Calculate top-bias scores (higher Z + upward normal = higher score)
        z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min() + 1e-8)
        normal_score = np.maximum(normal_z_components, 0)  # Only positive Z normals
        top_bias_scores = z_normalized + normal_score

        # Select top-biased subset
        top_indices = np.argsort(top_bias_scores)[-num_surface_samples:]
        surface_points = surface_points[top_indices]
        surface_normals = surface_normals[top_indices]

        # Cast rays in opposite direction of the surface normal
        ray_directions = -surface_normals
        ray_intersections, ray_indices, _ = mesh.ray.intersects_location(
            surface_points, ray_directions, multiple_hits=True
        )

        grasp_transforms = []

        # Process each sampled point to find valid grasp candidates
        for point_idx in range(len(surface_points)):
            # Find intersection points for this ray
            ray_hits = ray_intersections[ray_indices == point_idx]

            if len(ray_hits) == 0:
                continue

            # Find the furthest intersection point for more stable grasps
            if len(ray_hits) > 1:
                distances = torch.norm(torch.tensor(ray_hits) - torch.tensor(surface_points[point_idx]), dim=1)
                valid_indices = torch.where(distances <= max_gripper_width)[0]
                if len(valid_indices) > 0:
                    furthest_idx = valid_indices[torch.argmax(distances[valid_indices])]
                    opposing_point = ray_hits[furthest_idx]
                else:
                    continue
            else:
                opposing_point = ray_hits[0]
                distance = torch.norm(torch.tensor(opposing_point) - torch.tensor(surface_points[point_idx]))
                if distance > max_gripper_width:
                    continue

            # Calculate grasp axis and distance
            grasp_axis = opposing_point - surface_points[point_idx]
            axis_length = torch.norm(torch.tensor(grasp_axis))

            if axis_length > trimesh.tol.zero and axis_length <= max_gripper_width:
                grasp_axis = grasp_axis / axis_length.numpy()

                # Calculate grasp center with optional lateral perturbation
                if self.lateral_sigma > 0:
                    midpoint_ratio = 0.5
                    sigma_ratio = self.lateral_sigma / axis_length.numpy()
                    a = (0.0 - midpoint_ratio) / sigma_ratio
                    b = (1.0 - midpoint_ratio) / sigma_ratio
                    truncated_dist = stats.truncnorm(a, b, loc=midpoint_ratio, scale=sigma_ratio)
                    center_offset_ratio = truncated_dist.rvs()
                    grasp_center = surface_points[point_idx] + grasp_axis * axis_length.numpy() * center_offset_ratio
                else:
                    grasp_center = surface_points[point_idx] + grasp_axis * axis_length.numpy() * 0.5

                # Generate different orientations around each grasp axis
                rotation_angles = torch.linspace(-torch.pi, torch.pi, self.num_orientations)

                for angle in rotation_angles:
                    # Align the gripper's grasp_align_axis with the computed grasp axis
                    align_matrix = trimesh.geometry.align_vectors(grasp_align_axis.numpy(), grasp_axis)
                    center_transform = tra.translation_matrix(grasp_center)

                    # Create orientation transformation
                    orient_tf_rot = tra.rotation_matrix(angle=angle.item(), direction=orientation_sample_axis.numpy())

                    # Generate transforms for each standoff distance
                    for standoff_dist in standoff_distances:
                        standoff_translation = gripper_approach_direction.numpy() * -float(standoff_dist)
                        standoff_transform = tra.translation_matrix(standoff_translation)

                        # Full transform: T_center * R_align * R_orient * T_standoff
                        align_mat = torch.tensor(align_matrix, dtype=torch.float32)
                        full_orientation_tf = torch.matmul(align_mat, torch.tensor(orient_tf_rot, dtype=torch.float32))
                        full_orientation_tf = torch.matmul(
                            full_orientation_tf, torch.tensor(standoff_transform, dtype=torch.float32)
                        )
                        grasp_world_tf = torch.matmul(
                            torch.tensor(center_transform, dtype=torch.float32), full_orientation_tf
                        )
                        grasp_transforms.append(grasp_world_tf.numpy())

        return grasp_transforms

    def _apply_grasp_transform_to_gripper(self, env, gripper_asset, grasp_transform, env_idx):
        """Apply grasp transform to gripper asset."""
        # Get object's current pose in world coordinates
        object_asset = env.scene[self.object_cfg.name]
        object_pos = object_asset.data.root_pos_w[env_idx]
        object_quat = object_asset.data.root_quat_w[env_idx]

        # Convert numpy transform matrix to torch tensors (object-local coordinates)
        transform_tensor = torch.tensor(grasp_transform, dtype=torch.float32, device=env.device)
        local_pos = transform_tensor[:3, 3]
        rotation_matrix = transform_tensor[:3, :3]
        local_quat = math_utils.quat_from_matrix(rotation_matrix.unsqueeze(0))[0]  # (w, x, y, z)

        # Transform from object-local to world coordinates
        world_pos, world_quat = math_utils.combine_frame_transforms(
            object_pos.unsqueeze(0), object_quat.unsqueeze(0), local_pos.unsqueeze(0), local_quat.unsqueeze(0)
        )

        # Apply world transform to gripper asset for the specific environment
        gripper_asset.data.root_pos_w[env_idx] = world_pos[0]
        gripper_asset.data.root_quat_w[env_idx] = world_quat[0]

        # Write the new pose to simulation
        indices = torch.tensor([env_idx], device=env.device)
        root_pose = torch.cat([gripper_asset.data.root_pos_w[indices], gripper_asset.data.root_quat_w[indices]], dim=-1)
        gripper_asset.write_root_pose_to_sim(root_pose, env_ids=indices)

    def _apply_grasp_transforms_vectorized(self, env, gripper_asset, grasp_transforms, env_ids):
        """Apply grasp transforms to gripper assets for multiple environments (vectorized)."""
        # Get object's current pose in world coordinates for all environments
        object_asset = env.scene[self.object_cfg.name]
        object_pos = object_asset.data.root_pos_w[env_ids]
        object_quat = object_asset.data.root_quat_w[env_ids]

        # Extract positions and quaternions from transform matrices (already tensors)
        local_positions = grasp_transforms[:, :3, 3]  # Extract translation
        rotation_matrices = grasp_transforms[:, :3, :3]  # Extract rotation
        local_quaternions = math_utils.quat_from_matrix(rotation_matrices)  # (N, 4) in (w, x, y, z)

        # Transform from object-local to world coordinates (vectorized)
        world_positions, world_quaternions = math_utils.combine_frame_transforms(
            object_pos, object_quat, local_positions, local_quaternions
        )

        # Apply world transforms to gripper assets (vectorized)
        gripper_asset.data.root_pos_w[env_ids] = world_positions
        gripper_asset.data.root_quat_w[env_ids] = world_quaternions

        # Write the new poses to simulation (single vectorized call)
        root_poses = torch.cat([world_positions, world_quaternions], dim=-1)
        gripper_asset.write_root_pose_to_sim(root_poses, env_ids=env_ids)

    def _visualize_grasp_poses(self, env, scale: float = 0.03):
        """Visualize all grasp poses using pose markers."""
        if self.grasp_candidates is None or not hasattr(self, "pose_marker"):
            return

        # Get object asset for world transformation
        object_asset = env.scene[self.object_cfg.name]

        # Get object's current pose in world coordinates
        object_pos = object_asset.data.root_pos_w[0]  # Use first environment
        object_quat = object_asset.data.root_quat_w[0]  # Use first environment

        # Convert grasp transforms to poses and transform to world coordinates
        world_positions = []
        world_orientations = []

        for transform in self.grasp_candidates:
            # Extract position and rotation from transform matrix (object-local coordinates)
            local_pos = transform[:3, 3].clone().detach().to(env.device)
            rot_mat = transform[:3, :3].clone().detach().unsqueeze(0).to(env.device)
            local_quat = math_utils.quat_from_matrix(rot_mat)[0]  # (w, x, y, z)

            # Transform from object-local to world coordinates
            world_pos, world_quat = math_utils.combine_frame_transforms(
                object_pos.unsqueeze(0), object_quat.unsqueeze(0), local_pos.unsqueeze(0), local_quat.unsqueeze(0)
            )

            world_positions.append(world_pos[0])
            world_orientations.append(world_quat[0])

        # Stack into final tensors
        world_pos_tensor = torch.stack(world_positions)  # Shape: (N, 3)
        world_quat_tensor = torch.stack(world_orientations)  # Shape: (N, 4)

        # Visualize using pose markers
        self.pose_marker.visualize(world_pos_tensor, world_quat_tensor)

    def _open_gripper(self, env, gripper_asset, env_ids):
        """Open gripper to prepare for grasping."""
        # Get current joint positions
        current_joint_pos = gripper_asset.data.joint_pos[env_ids].clone()

        # Find joint indices using configurable joint names and positions
        joint_configs = []
        for joint_name, target_position in self.gripper_joint_reset_config.items():
            if joint_name in gripper_asset.joint_names:
                joint_idx = list(gripper_asset.joint_names).index(joint_name)
                joint_configs.append((joint_idx, target_position))

        if joint_configs:
            # Set joints to their configured target positions
            for env_idx_in_batch, env_id in enumerate(env_ids):
                for joint_idx, target_position in joint_configs:
                    current_joint_pos[env_idx_in_batch, joint_idx] = target_position

            # Apply joint positions to simulation
            gripper_asset.write_joint_state_to_sim(
                position=current_joint_pos,
                velocity=torch.zeros_like(current_joint_pos),
                env_ids=env_ids,
            )

    def _ensure_stable_gripper_state(self, env, gripper_asset, env_ids):
        """Comprehensively reset gripper to stable state before positioning."""
        # Always perform comprehensive reset to ensure clean state
        # 1. Reset actuators to clear any accumulated forces/torques
        gripper_asset.reset(env_ids)

        # 2. Reset to default root state (position and velocity)
        default_root_state = gripper_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        gripper_asset.write_root_state_to_sim(default_root_state, env_ids=env_ids)

        # 3. Reset all joints to default positions with zero velocities
        default_joint_pos = gripper_asset.data.default_joint_pos[env_ids].clone()
        zero_joint_vel = torch.zeros_like(gripper_asset.data.default_joint_vel[env_ids])
        gripper_asset.write_joint_state_to_sim(default_joint_pos, zero_joint_vel, env_ids=env_ids)

        # 4. Set joint targets to default positions to prevent drift
        gripper_asset.set_joint_position_target(default_joint_pos, env_ids=env_ids)
        gripper_asset.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)


class global_physics_control_event(ManagerTermBase):
    """Event class for global gravity and force/torque control based on synchronized timesteps."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.gravity_on_interval = cfg.params.get("gravity_on_interval")
        self.gravity_on_interval_s = (
            self.gravity_on_interval[0] / env.step_dt,
            self.gravity_on_interval[1] / env.step_dt,
        )
        self.force_torque_on_interval = cfg.params.get("force_torque_on_interval")
        self.force_torque_on_interval_s = (
            self.force_torque_on_interval[0] / env.step_dt,
            self.force_torque_on_interval[1] / env.step_dt,
        )
        self.force_torque_asset_cfgs = cfg.params.get("force_torque_asset_cfgs", [])
        self.force_torque_magnitude = cfg.params.get("force_torque_magnitude", 0.005)
        self.physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Called when environments reset - disable gravity for positioning."""
        self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        self.gravity_enabled = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        gravity_on_interval: tuple[float, float],
        force_torque_on_interval: tuple[float, float],
        force_torque_asset_cfgs: list[SceneEntityCfg],
        force_torque_magnitude: float,
    ) -> None:
        """Control global gravity based on timesteps since reset."""
        should_enable_gravity = (
            (env.episode_length_buf > self.gravity_on_interval_s[0])
            & (env.episode_length_buf < self.gravity_on_interval_s[1])
        ).any()
        should_apply_force_torque = (
            (env.episode_length_buf > self.force_torque_on_interval_s[0])
            & (env.episode_length_buf < self.force_torque_on_interval_s[1])
        ).any()

        if should_enable_gravity and not self.gravity_enabled:
            self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
            self.gravity_enabled = True
        elif not should_enable_gravity and self.gravity_enabled:
            self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
            self.gravity_enabled = False
        else:
            pass

        if should_apply_force_torque:
            # resolve environment ids
            if env_ids is None:
                env_ids = torch.arange(env.scene.num_envs, device=env.device)
            for asset_cfg in self.force_torque_asset_cfgs:
                # extract the used quantities (to enable type-hinting)
                asset: RigidObject | Articulation = env.scene[asset_cfg.name]
                # resolve number of bodies
                num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

                # Generate random forces in all directions
                size = (len(env_ids), num_bodies, 3)
                force_directions = torch.randn(size, device=asset.device)
                force_directions = force_directions / torch.norm(force_directions, dim=-1, keepdim=True)
                forces = force_directions * self.force_torque_magnitude

                # Generate independent random torques (pure rotational moments)
                # These represent direct angular impulses rather than forces at lever arms
                torque_directions = torch.randn(size, device=asset.device)
                torque_directions = torque_directions / torch.norm(torque_directions, dim=-1, keepdim=True)
                torques = torque_directions * self.force_torque_magnitude

                # set the forces and torques into the buffers
                # note: these are only applied when you call: `asset.write_data_to_sim()`
                asset.permanent_wrench_composer.set_forces_and_torques(
                    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
                )


class reset_end_effector_round_fixed_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore
        self.reset_velocity = torch.zeros((env.num_envs, self.robot.data.joint_vel.shape[1]), device=env.device)
        self.reset_position = torch.zeros((env.num_envs, self.robot.data.joint_pos.shape[1]), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
    ) -> None:
        if fixed_asset_offset is None:
            fixed_tip_pos_w, fixed_tip_quat_w = (
                env.scene[fixed_asset_cfg.name].data.root_pos_w,
                env.scene[fixed_asset_cfg.name].data.root_quat_w,
            )
        else:
            fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)

        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (env.num_envs, 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w, self.robot.data.root_link_quat_w, pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )


class reset_end_effector_from_grasp_dataset(ManagerTermBase):
    """Reset end effector pose using saved grasp dataset from grasp sampling."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        self.dataset_dir: str = cfg.params.get("dataset_dir")
        self.fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))
        gripper_cfg: SceneEntityCfg = cfg.params.get(
            "gripper_cfg", SceneEntityCfg("robot", joint_names=["finger_joint"])
        )
        # Set up robot and IK solver for arm joints
        self.fixed_asset: Articulation | RigidObject = env.scene[self.fixed_asset_cfg.name]
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)

        # Pose range for sampling variations
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b", dict())
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)  # type: ignore

        # Set up gripper joint control separately
        self.gripper: Articulation = env.scene[
            gripper_cfg.name
        ]  # Should be same as robot but different joint selection
        self.gripper_joint_ids: list[int] | slice = gripper_cfg.joint_ids
        self.gripper_joint_names: list[str] = gripper_cfg.joint_names if gripper_cfg.joint_names else []

        # Compute grasp dataset path from object name
        self.grasp_dataset_path = self._compute_grasp_dataset_path()

        # Load and pre-compute grasp data for fast sampling
        self._load_and_precompute_grasps(env)

    def _compute_grasp_dataset_path(self) -> str:
        usd_path = self.fixed_asset.cfg.spawn.usd_path
        obj_name = utils.object_name_from_usd(usd_path)
        return f"{self.dataset_dir}/Grasps/{obj_name}/grasps.pt"

    def _load_and_precompute_grasps(self, env):
        """Load Torch (.pt) grasp data and convert to optimized tensors."""
        local_path = utils.safe_retrieve_file_path(self.grasp_dataset_path)
        data = torch.load(local_path, map_location="cpu")

        # TorchDatasetFileHandler stores nested dicts; grasp data likely under 'grasp_relative_pose'
        grasp_group = data.get("grasp_relative_pose", data)

        rel_pos_list = grasp_group.get("relative_position", [])
        rel_quat_list = grasp_group.get("relative_orientation", [])
        gripper_joint_positions_dict = grasp_group.get("gripper_joint_positions", {})

        num_grasps = len(rel_pos_list)
        if num_grasps == 0:
            raise ValueError(f"No grasp data found in {self.grasp_dataset_path}")

        # Convert positions and orientations to tensors on env device
        self.rel_positions = torch.stack(
            [
                (pos if isinstance(pos, torch.Tensor) else torch.as_tensor(pos, dtype=torch.float32))
                for pos in rel_pos_list
            ],
            dim=0,
        ).to(env.device, dtype=torch.float32)

        self.rel_quaternions = torch.stack(
            [
                (quat if isinstance(quat, torch.Tensor) else torch.as_tensor(quat, dtype=torch.float32))
                for quat in rel_quat_list
            ],
            dim=0,
        ).to(env.device, dtype=torch.float32)

        # Get gripper joint mapping
        if isinstance(self.gripper_joint_ids, slice):
            gripper_joint_list = list(range(self.robot.num_joints))[self.gripper_joint_ids]
        else:
            gripper_joint_list = self.gripper_joint_ids

        num_gripper_joints = len(gripper_joint_list)
        self.gripper_joint_positions = torch.zeros(
            (num_grasps, num_gripper_joints), device=env.device, dtype=torch.float32
        )

        # Build joint matrix ordered by robot joint indices per provided gripper_joint_ids
        for gripper_idx, robot_joint_idx in enumerate(gripper_joint_list):
            joint_name = self.robot.joint_names[robot_joint_idx]
            joint_series = gripper_joint_positions_dict.get(joint_name, [0.0] * num_grasps)
            joint_tensor = torch.stack(
                [(j if isinstance(j, torch.Tensor) else torch.as_tensor(j, dtype=torch.float32)) for j in joint_series],
                dim=0,
            ).to(env.device, dtype=torch.float32)
            self.gripper_joint_positions[:, gripper_idx] = joint_tensor

        print(f"Loaded and pre-computed {num_grasps} grasp tensors from Torch file: {self.grasp_dataset_path}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str,
        fixed_asset_cfg: SceneEntityCfg,
        robot_ik_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
    ) -> None:
        """Apply grasp poses to reset end effector."""
        # RigidObject asset
        object_pos_w = self.fixed_asset.data.root_pos_w[env_ids]
        object_quat_w = self.fixed_asset.data.root_quat_w[env_ids]

        # Randomly sample grasp indices for each environment
        num_envs = len(env_ids)
        grasp_indices = torch.randint(0, len(self.rel_positions), (num_envs,), device=env.device)

        # Use pre-computed tensors for sampled grasps
        sampled_rel_positions = self.rel_positions[grasp_indices]
        sampled_rel_quaternions = self.rel_quaternions[grasp_indices]

        # Vectorized transform to world coordinates: T_gripper_world = T_object_world * T_relative
        gripper_pos_w, gripper_quat_w = math_utils.combine_frame_transforms(
            object_pos_w, object_quat_w, sampled_rel_positions, sampled_rel_quaternions
        )

        # Vectorized transform to robot base coordinates
        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids],
            self.robot.data.root_link_quat_w[env_ids],
            gripper_pos_w,
            gripper_quat_w,
        )

        # Add pose variation sampling if ranges are specified (in body frame)
        if torch.any(self.ranges != 0.0):
            samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (num_envs, 6), device=env.device)
            pos_b[env_ids], quat_b[env_ids] = math_utils.combine_frame_transforms(
                pos_b[env_ids],
                quat_b[env_ids],
                samples[:, 0:3],
                math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
            )

        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Solve IK iteratively for better convergence
        for i in range(25):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )

        # Sample gripper joint positions using the same indices
        sampled_gripper_positions = self.gripper_joint_positions[grasp_indices]

        # Single vectorized write for all environments
        self.robot.write_joint_state_to_sim(
            position=sampled_gripper_positions,
            velocity=torch.zeros_like(sampled_gripper_positions),
            joint_ids=self.gripper_joint_ids,
            env_ids=env_ids,
        )


class reset_insertive_object_from_partial_assembly_dataset(ManagerTermBase):
    """EventTerm class for resetting the insertive object from a partial assembly dataset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Extract parameters from config
        self.dataset_dir: str = cfg.params.get("dataset_dir")
        self.receptive_object_cfg: SceneEntityCfg = cfg.params.get("receptive_object_cfg")
        self.receptive_object: RigidObject = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg: SceneEntityCfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object: RigidObject = env.scene[self.insertive_object_cfg.name]

        # Pose range for sampling variations
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b", dict())
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        # Compute partial assembly dataset path from object pair names
        self.partial_assembly_dataset_path = self._compute_partial_assembly_dataset_path()

        # Load and pre-compute partial assembly data for fast sampling
        self._load_and_precompute_partial_assemblies(env)

    def _compute_partial_assembly_dataset_path(self) -> str:
        insertive_usd_path = self.insertive_object.cfg.spawn.usd_path
        receptive_usd_path = self.receptive_object.cfg.spawn.usd_path
        pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)
        return f"{self.dataset_dir}/Resets/{pair}/partial_assemblies.pt"

    def _load_and_precompute_partial_assemblies(self, env):
        """Load Torch (.pt) partial assembly data and convert to optimized tensors."""
        local_path = utils.safe_retrieve_file_path(self.partial_assembly_dataset_path)
        data = torch.load(local_path, map_location="cpu")

        rel_pos = data.get("relative_position")
        rel_quat = data.get("relative_orientation")

        if rel_pos is None or rel_quat is None or len(rel_pos) == 0:
            raise ValueError(f"No partial assembly data found in {self.partial_assembly_dataset_path}")

        # Tensors were saved via torch.save; ensure proper device/dtype
        if not isinstance(rel_pos, torch.Tensor):
            rel_pos = torch.as_tensor(rel_pos, dtype=torch.float32)
        if not isinstance(rel_quat, torch.Tensor):
            rel_quat = torch.as_tensor(rel_quat, dtype=torch.float32)

        self.rel_positions = rel_pos.to(env.device, dtype=torch.float32)
        self.rel_quaternions = rel_quat.to(env.device, dtype=torch.float32)

        print(
            f"Loaded {len(self.rel_positions)} partial assembly tensors from Torch file:"
            f" {self.partial_assembly_dataset_path}"
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str,
        insertive_object_cfg: SceneEntityCfg,
        receptive_object_cfg: SceneEntityCfg,
        pose_range_b: dict[str, tuple[float, float]] = dict(),
    ) -> None:
        """Reset the insertive object from a partial assembly dataset."""
        # Get receptive object pose (world coordinates)
        receptive_pos_w = self.receptive_object.data.root_pos_w[env_ids]
        receptive_quat_w = self.receptive_object.data.root_quat_w[env_ids]

        # Randomly sample partial assembly indices for each environment
        num_envs = len(env_ids)
        assembly_indices = torch.randint(0, len(self.rel_positions), (num_envs,), device=env.device)

        # Use pre-computed tensors for sampled partial assemblies
        sampled_rel_positions = self.rel_positions[assembly_indices]
        sampled_rel_quaternions = self.rel_quaternions[assembly_indices]

        # Vectorized transform to world coordinates: T_insertive_world = T_receptive_world * T_relative
        insertive_pos_w, insertive_quat_w = math_utils.combine_frame_transforms(
            receptive_pos_w, receptive_quat_w, sampled_rel_positions, sampled_rel_quaternions
        )

        # Add pose variation sampling if ranges are specified
        if torch.any(self.ranges != 0.0):
            samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (num_envs, 6), device=env.device)
            insertive_pos_w, insertive_quat_w = math_utils.combine_frame_transforms(
                insertive_pos_w,
                insertive_quat_w,
                samples[:, 0:3],
                math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
            )

        # Set insertive object pose
        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat(
                [
                    insertive_pos_w,
                    insertive_quat_w,
                    torch.zeros((num_envs, 6), device=env.device),  # Zero linear and angular velocities
                ],
                dim=-1,
            ),
            env_ids=env_ids,
        )


class pose_logging_event(ManagerTermBase):
    """EventTerm class for logging pose data from all environments."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.receptive_object_cfg = cfg.params.get("receptive_object_cfg")
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        receptive_object_cfg: SceneEntityCfg,
        insertive_object_cfg: SceneEntityCfg,
    ) -> None:
        """Collect pose data from all environments."""

        # Get object poses for all environments
        receptive_pos = self.receptive_object.data.root_pos_w[env_ids]
        receptive_quat = self.receptive_object.data.root_quat_w[env_ids]
        insertive_pos = self.insertive_object.data.root_pos_w[env_ids]
        insertive_quat = self.insertive_object.data.root_quat_w[env_ids]

        # Calculate relative transform
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(
            receptive_pos, receptive_quat, insertive_pos, insertive_quat
        )

        # Store pose data for external access
        if "log" not in env.extras:
            env.extras["log"] = {}
        env.extras["log"]["current_pose_data"] = {
            "relative_position": relative_pos,
            "relative_orientation": relative_quat,
            "relative_pose": torch.cat([relative_pos, relative_quat], dim=-1),
            "receptive_object_pose": torch.cat([receptive_pos, receptive_quat], dim=-1),
            "insertive_object_pose": torch.cat([insertive_pos, insertive_quat], dim=-1),
        }


class assembly_sampling_event(ManagerTermBase):
    """EventTerm class for spawning insertive object at assembled offset position."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.receptive_object_cfg = cfg.params.get("receptive_object_cfg")
        self.receptive_object = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

        insertive_metadata = utils.read_metadata_from_usd_directory(self.insertive_object.cfg.spawn.usd_path)
        receptive_metadata = utils.read_metadata_from_usd_directory(self.receptive_object.cfg.spawn.usd_path)

        self.insertive_assembled_offset = Offset(
            pos=insertive_metadata.get("assembled_offset").get("pos"),
            quat=insertive_metadata.get("assembled_offset").get("quat"),
        )
        self.receptive_assembled_offset = Offset(
            pos=receptive_metadata.get("assembled_offset").get("pos"),
            quat=receptive_metadata.get("assembled_offset").get("quat"),
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        receptive_object_cfg: SceneEntityCfg,
        insertive_object_cfg: SceneEntityCfg,
    ) -> None:
        """Spawn insertive object at assembled offset position."""

        # Get receptive object poses
        receptive_pos = self.receptive_object.data.root_pos_w[env_ids]
        receptive_quat = self.receptive_object.data.root_quat_w[env_ids]

        # Apply receptive assembled offset to get target position
        target_pos, target_quat = self.receptive_assembled_offset.combine(receptive_pos, receptive_quat)

        # Handle position and orientation separately
        # Offset quat is in insertive object's frame: target_quat = insertive_quat * offset_quat
        offset_quat = (
            torch.tensor(self.insertive_assembled_offset.quat).to(target_quat.device).repeat(target_quat.shape[0], 1)
        )
        insertive_quat = math_utils.quat_mul(target_quat, math_utils.quat_inv(offset_quat))

        # Position offset is in insertive object's frame, but rotated by target_quat to keep it independent of offset_quat
        # This ensures changing offset_quat doesn't change the position offset direction
        offset_pos = (
            torch.tensor(self.insertive_assembled_offset.pos).to(target_pos.device).repeat(target_pos.shape[0], 1)
        )
        offset_pos_world = math_utils.quat_apply(target_quat, offset_pos)
        insertive_pos = target_pos - offset_pos_world

        # Set insertive object pose
        self.insertive_object.write_root_state_to_sim(
            root_state=torch.cat(
                [insertive_pos, insertive_quat, torch.zeros((len(env_ids), 6), device=env.device)],  # Zero velocities
                dim=-1,
            ),
            env_ids=env_ids,
        )


class MultiResetManager(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        dataset_dir: str = cfg.params.get("dataset_dir", "")
        reset_types: list[str] = cfg.params.get("reset_types", [])
        probabilities: list[float] = cfg.params.get("probs", [])

        if not reset_types:
            raise ValueError("No reset_types provided")
        if len(reset_types) != len(probabilities):
            raise ValueError("Number of reset_types must match number of probabilities")

        # Derive pair directory from scene objects
        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
        pair = utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

        # Generate dataset paths from pair directory and reset types
        dataset_files = []
        for rt in reset_types:
            dataset_files.append(f"{dataset_dir}/Resets/{pair}/resets_{rt}.pt")

        # Load all datasets
        self.datasets = []
        num_states = []
        for dataset_file in dataset_files:
            local_file_path = utils.safe_retrieve_file_path(dataset_file)

            # Check if local file exists (after potential download)
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")

            dataset = torch.load(local_file_path)
            num_states.append(len(dataset["initial_state"]["articulation"]["robot"]["joint_position"]))
            init_indices = torch.arange(num_states[-1], device=env.device)
            self.datasets.append(sample_state_data_set(dataset, init_indices, env.device))

        # Normalize probabilities and store dataset lengths
        self.probs = torch.tensor(probabilities, device=env.device) / sum(probabilities)
        self.num_states = torch.tensor(num_states, device=env.device)
        self.num_tasks = len(self.datasets)

        # Initialize success monitor
        if cfg.params.get("success") is not None:
            success_monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=100, num_monitored_data=self.num_tasks, device=env.device
            )
            self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        dataset_dir: str,
        reset_types: list[str],
        probs: list[float],
        success: str | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self._env.device)

        # Log current data
        if success is not None:
            success_mask = torch.where(eval(success)[env_ids], 1.0, 0.0)
            self.success_monitor.success_update(self.task_id[env_ids], success_mask)

            # Log metrics for each task
            success_rates = self.success_monitor.get_success_rate()
            if "log" not in self._env.extras:
                self._env.extras["log"] = {}
            for task_idx in range(self.num_tasks):
                self._env.extras["log"].update({
                    f"Metrics/task_{task_idx}_success_rate": success_rates[task_idx].item(),
                    f"Metrics/task_{task_idx}_prob": self.probs[task_idx].item(),
                    f"Metrics/task_{task_idx}_normalized_prob": self.probs[task_idx].item(),
                })

            # Log episode length at reset
            ep_lengths = self._env.episode_length_buf[env_ids].float()
            self._env.extras["log"]["Metrics/mean_episode_length"] = ep_lengths.mean().item()

        # Sample which dataset to use for each environment
        dataset_indices = torch.multinomial(self.probs, len(env_ids), replacement=True)
        self.task_id[env_ids] = dataset_indices

        # Process each dataset's environments
        for dataset_idx in range(self.num_tasks):
            mask = dataset_indices == dataset_idx
            if not mask.any():
                continue

            current_env_ids = env_ids[mask]
            state_indices = torch.randint(
                0, self.num_states[dataset_idx], (len(current_env_ids),), device=self._env.device
            )
            states_to_reset_from = sample_from_nested_dict(self.datasets[dataset_idx], state_indices)
            self._reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        # Reset velocities
        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)

    def _reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None = None,
        is_relative: bool = False,
    ):
        """Resets the entities in the scene to the provided state.

        Args:
            state: The state to reset the scene entities to. Please refer to :meth:`get_state` for the format.
            env_ids: The indices of the environments to reset. Defaults to None, in which case
                all environment instances are reset.
            is_relative: If set to True, the state is considered relative to the environment origins.
                Defaults to False.
        """
        # resolve env_ids
        if env_ids is None:
            env_ids = self._env.scene._ALL_INDICES
        # articulations
        for asset_name, articulation in self._env.scene._articulations.items():
            if asset_name not in state["articulation"]:
                continue
            asset_state = state["articulation"][asset_name]
            # root state
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self._env.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
            # joint state
            joint_position = asset_state["joint_position"].clone()
            joint_velocity = asset_state["joint_velocity"].clone()
            articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
            # FIXME: This is not generic as it assumes PD control over the joints.
            #   This assumption does not hold for effort controlled joints.
            articulation.set_joint_position_target(joint_position, env_ids=env_ids)
            articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)
        # deformable objects
        for asset_name, deformable_object in self._env.scene._deformable_objects.items():
            if asset_name not in state["deformable_object"]:
                continue
            asset_state = state["deformable_object"][asset_name]
            nodal_position = asset_state["nodal_position"].clone()
            if is_relative:
                nodal_position[:, :3] += self._env.scene.env_origins[env_ids]
            nodal_velocity = asset_state["nodal_velocity"].clone()
            deformable_object.write_nodal_pos_to_sim(nodal_position, env_ids=env_ids)
            deformable_object.write_nodal_velocity_to_sim(nodal_velocity, env_ids=env_ids)
        # rigid objects
        for asset_name, rigid_object in self._env.scene._rigid_objects.items():
            if asset_name not in state["rigid_object"]:
                continue
            asset_state = state["rigid_object"][asset_name]
            root_pose = asset_state["root_pose"].clone()
            if is_relative:
                root_pose[:, :3] += self._env.scene.env_origins[env_ids]
            root_velocity = asset_state["root_velocity"].clone()
            rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        # surface grippers
        for asset_name, surface_gripper in self._env.scene._surface_grippers.items():
            asset_state = state["gripper"][asset_name]
            surface_gripper.set_grippers_command(asset_state)

        # write data to simulation to make sure initial state is set
        # this propagates the joint targets to the simulation
        self._env.scene.write_data_to_sim()


def sample_state_data_set(episode_data: dict, idx: torch.Tensor, device: torch.device) -> dict:
    """Sample state from episode data and move tensors to device in one pass."""
    result = {}
    for key, value in episode_data.items():
        if isinstance(value, dict):
            result[key] = sample_state_data_set(value, idx, device)
        elif isinstance(value, list):
            result[key] = torch.stack([value[i] for i in idx.tolist()], dim=0).to(device)
        else:
            raise TypeError(f"Unsupported type in episode data: {type(value)}")
    return result


def sample_from_nested_dict(nested_dict: dict, idx) -> dict:
    """Extract elements from a nested dictionary using given indices."""
    sampled_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            sampled_dict[key] = sample_from_nested_dict(value, idx)
        elif isinstance(value, torch.Tensor):
            sampled_dict[key] = value[idx].clone()
        else:
            raise TypeError(f"Unsupported type in nested dictionary: {type(value)}")
    return sampled_dict


class reset_root_states_uniform(ManagerTermBase):
    """Reset multiple assets' root states to random positions and velocities uniformly within given ranges.

    This function randomizes the root position and velocity of multiple assets using the same random offsets.
    This keeps the relative positioning between assets intact while randomizing their global position.

    * It samples the root position from the given ranges and adds them to each asset's default root position
    * It samples the root orientation from the given ranges and sets them into the physics simulation
    * It samples the root velocity from the given ranges and sets them into the physics simulation

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Args:
        env: The environment instance
        env_ids: The environment IDs to reset
        pose_range: Dictionary of position and orientation ranges
        velocity_range: Dictionary of linear and angular velocity ranges
        asset_cfgs: List of asset configurations to reset (all receive same random offset)
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        pose_range_dict = cfg.params.get("pose_range")
        velocity_range_dict = cfg.params.get("velocity_range")

        self.pose_range = torch.tensor(
            [pose_range_dict.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]], device=env.device
        )
        self.velocity_range = torch.tensor(
            [velocity_range_dict.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
        )
        self.asset_cfgs = list(cfg.params.get("asset_cfgs", dict()).values())
        self.offset_asset_cfg = cfg.params.get("offset_asset_cfg")
        self.use_bottom_offset = cfg.params.get("use_bottom_offset", False)

        if self.use_bottom_offset:
            self.bottom_offset_positions = dict()
            for asset_cfg in self.asset_cfgs:
                asset: RigidObject | Articulation = env.scene[asset_cfg.name]
                usd_path = asset.cfg.spawn.usd_path
                metadata = utils.read_metadata_from_usd_directory(usd_path)
                bottom_offset = metadata.get("bottom_offset")
                self.bottom_offset_positions[asset_cfg.name] = (
                    torch.tensor(bottom_offset.get("pos"), device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
                )
                assert tuple(bottom_offset.get("quat")) == (
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ), "Bottom offset rotation must be (1.0, 0.0, 0.0, 0.0)"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfgs: dict[str, SceneEntityCfg] = dict(),
        offset_asset_cfg: SceneEntityCfg = None,
        use_bottom_offset: bool = False,
    ) -> None:
        # poses
        rand_pose_samples = math_utils.sample_uniform(
            self.pose_range[:, 0], self.pose_range[:, 1], (len(env_ids), 6), device=env.device
        )

        # Create orientation delta quaternion from the random Euler angles
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )

        # velocities
        rand_vel_samples = math_utils.sample_uniform(
            self.velocity_range[:, 0], self.velocity_range[:, 1], (len(env_ids), 6), device=env.device
        )

        # Apply the same random offsets to each asset
        for asset_cfg in self.asset_cfgs:
            asset: RigidObject | Articulation = env.scene[asset_cfg.name]

            # Get default root state for this asset
            root_states = asset.data.default_root_state[env_ids].clone()

            # Apply position offset
            positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]

            if self.offset_asset_cfg:
                offset_asset: RigidObject | Articulation = env.scene[self.offset_asset_cfg.name]
                offset_positions = offset_asset.data.default_root_state[env_ids].clone()
                positions += offset_positions[:, 0:3]

            if self.use_bottom_offset:
                bottom_offset_position = self.bottom_offset_positions[asset_cfg.name]
                positions -= bottom_offset_position[env_ids, 0:3]

            # Apply orientation offset
            orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

            # Apply velocity offset
            velocities = root_states[:, 7:13] + rand_vel_samples

            # Set the new pose and velocity into the physics simulation
            asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


class randomize_hdri(ManagerTermBase):
    """Randomizes the HDRI texture, intensity, and rotation.

    HDRI paths are loaded from a YAML config file once during initialization.
    Paths under 'isaac_nucleus' section are prefixed with ISAAC_NUCLEUS_DIR,
    all other paths are prefixed with NVIDIA_NUCLEUS_DIR.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term and cache HDRI paths."""
        super().__init__(cfg, env)

        hdri_config_path = cfg.params.get("hdri_config_path")

        # Load and cache HDRI paths once during init
        if hdri_config_path is not None:
            self.hdri_paths = utils.load_asset_paths_from_config(
                hdri_config_path, cache_subdir="hdris", skip_validation=False
            )
            logging.info(f"[randomize_hdri] Loaded {len(self.hdri_paths)} HDRI paths.")
        else:
            self.hdri_paths = []

        if not self.hdri_paths:
            raise RuntimeError(f"[randomize_hdri] No HDRI paths loaded. Check hdri_config_path={hdri_config_path}")
        non_local = [p for p in self.hdri_paths if not p.startswith("/")]
        if non_local:
            raise RuntimeError(
                f"[randomize_hdri] {len(non_local)} HDRI paths are non-local (Nucleus) "
                "and will silently fail if Nucleus is unreachable. "
                f"First 3: {non_local[:3]}. "
                "Use only local/cloud-cached HDRIs."
            )
        missing = [p for p in self.hdri_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"[randomize_hdri] {len(missing)}/{len(self.hdri_paths)} HDRI files missing on disk. "
                f"First 3: {missing[:3]}"
            )

        # Apply initial randomization so envs don't start with default lighting
        self(env, torch.arange(env.num_envs, device=env.device), **cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        light_path: str = "/World/skyLight",
        hdri_config_path: str | None = None,
        intensity_range: tuple = (500.0, 1000.0),
        rotation_range: tuple = (0.0, 360.0),
    ) -> None:
        stage = omni.usd.get_context().get_stage()
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            raise RuntimeError(
                f"[randomize_hdri] Light prim at '{light_path}' does not exist on the stage. "
                "This is likely because the DomeLightCfg failed to spawn (e.g. Nucleus server unreachable). "
                "Remove the texture_file from DomeLightCfg or use a local path."
            )

        dome_light = UsdLux.DomeLight(light_prim)
        if not dome_light:
            raise RuntimeError(f"[randomize_hdri] Prim at '{light_path}' is not a DomeLight.")

        random_hdri = random.choice(self.hdri_paths)
        intensity = random.randint(int(intensity_range[0]), int(intensity_range[1]))

        # Use direct attribute access (DEXTRAH-style) -- UsdLux helper methods
        # can map to the wrong schema attribute name depending on USD version.
        light_prim.GetAttribute("inputs:texture:file").Set(random_hdri)
        light_prim.GetAttribute("inputs:intensity").Set(float(intensity))

        from scipy.spatial.transform import Rotation as R

        quat = R.random().as_quat()  # [x, y, z, w] scipy convention
        xformable = UsdGeom.Xformable(light_prim)
        xformable.ClearXformOpOrder()
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quat[3]), Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2])))
        )

        logging.debug(f"[randomize_hdri] Applied: {random_hdri}, intensity={intensity}")


def randomize_tiled_cameras(
    env,
    env_ids: torch.Tensor,
    camera_path_template: str,
    base_position: tuple,
    base_rotation: tuple,
    position_deltas: dict,
    euler_deltas: dict,
) -> None:
    """Randomizes tiled cameras with XYZ and Euler angle deltas from base values."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    for env_idx in env_ids:
        env_idx_value = env_idx.item() if hasattr(env_idx, "item") else env_idx

        # Get the camera path for this environment using the template
        camera_path = camera_path_template.format(env_idx_value)

        # Get the stage
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid():
            continue

        # === Randomize Position ===
        pos_delta_x = random.uniform(*position_deltas["x"])
        pos_delta_y = random.uniform(*position_deltas["y"])
        pos_delta_z = random.uniform(*position_deltas["z"])

        new_pos = (base_position[0] + pos_delta_x, base_position[1] + pos_delta_y, base_position[2] + pos_delta_z)

        # === Randomize Rotation (Euler deltas in degrees, convert to radians) ===
        # Convert base quaternion (w, x, y, z) to GfQuatf
        base_quat = Gf.Quatf(base_rotation[0], Gf.Vec3f(base_rotation[1], base_rotation[2], base_rotation[3]))
        base_rot = Gf.Rotation(base_quat)

        # Create delta rotation from Euler angles (ZYX order: yaw, pitch, roll)
        delta_pitch = random.uniform(*euler_deltas["pitch"])
        delta_yaw = random.uniform(*euler_deltas["yaw"])
        delta_roll = random.uniform(*euler_deltas["roll"])

        delta_rot = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), delta_yaw)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), delta_pitch)
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), delta_roll)
        )

        # Apply delta rotation to base rotation
        new_rot = delta_rot * base_rot
        new_quat = new_rot.GetQuat()

        # === Apply pose to the USD prim ===
        xform = UsdGeom.Xformable(camera_prim)
        xform_ops = xform.GetOrderedXformOps()

        if not xform_ops:
            xform.AddTransformOp()

        # Set translation and orientation
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*new_pos))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(new_quat)


def randomize_camera_focal_length(
    env, env_ids: torch.Tensor, camera_path_template: str, focal_length_range: tuple = (0.8, 1.8)
) -> None:
    """Randomizes the focal length of cameras."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()

    for env_idx in env_ids:
        camera_path = camera_path_template.format(env_idx)
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            continue

        focal_length = random.uniform(focal_length_range[0], focal_length_range[1])
        focal_attr = camera_prim.GetAttribute("focalLength")
        if focal_attr.IsValid():
            focal_attr.Set(focal_length)


class randomize_arm_from_sysid(ManagerTermBase):
    """Randomize arm joint dynamics around sysid nominal values.

    Sysid parameters (armature, friction, etc.) are loaded from ``metadata.yaml``
    next to the robot USD.  ``scale_range = (lo, hi)`` scales each nominal:
    ``nominal * uniform(lo, hi)`` per env per joint.

    When used with ADR, ``scale_progress`` (0→1) linearly interpolates armature,
    friction, and motor delay from 0 to the full sysid-randomized values.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        self.joint_ids = self.robot.find_joints(cfg.params["joint_names"])[0]
        self.actuator_name: str = cfg.params["actuator_name"]

        # Load sysid from robot metadata (co-located with USD)
        metadata = utils.read_metadata_from_usd_directory(self.robot.cfg.spawn.usd_path)
        sysid = metadata["sysid"]
        self.armature = sysid["armature"]
        self.static_friction = sysid["static_friction"]
        self.dynamic_ratio = sysid["dynamic_ratio"]
        self.viscous_friction = sysid["viscous_friction"]

        # ADR progress: 0 = armature/friction are 0, 1 = full sysid randomization
        self.scale_progress: float = cfg.params.get("initial_scale_progress", 0.0)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        joint_names: list[str],
        actuator_name: str,
        scale_range: tuple[float, float] = (0.8, 1.2),
        delay_range: tuple[int, int] = (0, 2),
        initial_scale_progress: float = 0.0,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.robot.device)
        N = len(env_ids)
        n_joints = len(self.joint_ids)
        lo, hi = scale_range
        device = self.robot.device
        p = self.scale_progress

        def _scale(nominal):
            val = torch.as_tensor(nominal, device=device, dtype=torch.float32)
            return val * (lo + torch.rand(N, n_joints, device=device) * (hi - lo))

        # Armature and friction: scaled by ADR progress (0 → sysid)
        arm_vals = _scale(self.armature) * p
        sfric_vals = _scale(self.static_friction) * p
        dratio_vals = _scale(self.dynamic_ratio) * p
        dfric_vals = torch.minimum(dratio_vals * sfric_vals, sfric_vals)
        vfric_vals = _scale(self.viscous_friction) * p

        self.robot.write_joint_armature_to_sim(arm_vals, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_friction_coefficient_to_sim(
            sfric_vals,
            joint_dynamic_friction_coeff=dfric_vals,
            joint_viscous_friction_coeff=vfric_vals,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

        # Motor delay scaled by ADR progress (if actuator supports it)
        delay_lo, delay_hi = delay_range
        actuator = self.robot.actuators[self.actuator_name]
        if hasattr(actuator, "positions_delay_buffer"):
            effective_hi = int(round(p * delay_hi))
            effective_lo = min(delay_lo, effective_hi)
            delays = torch.randint(effective_lo, effective_hi + 1, (N,), device=device, dtype=torch.int)
            actuator.positions_delay_buffer.set_time_lag(delays, env_ids)
            actuator.velocities_delay_buffer.set_time_lag(delays, env_ids)
            actuator.efforts_delay_buffer.set_time_lag(delays, env_ids)


class randomize_arm_from_sysid_fixed(randomize_arm_from_sysid):
    """Same as randomize_arm_from_sysid but always applies scale_range (no curriculum)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.scale_progress = 1.0


class randomize_gripper_from_sysid(ManagerTermBase):
    """Randomize gripper dynamics around sysid nominal values.

    Each parameter is a nominal scalar.
    ``scale_range = (lo, hi)`` scales it: ``nominal * uniform(lo, hi)`` per env.

    When used with ADR, ``scale_progress`` (0→1):
    - Armature/friction: interpolate from 0 to sysid × U(scale_range).
    - Stiffness/damping: interpolate from ``initial_stiffness``/``initial_damping``
      (sim defaults) to sysid × U(scale_range).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        self.gripper_joint_ids = self.robot.find_joints(cfg.params["joint_names"])[0]
        self.actuator_name: str = cfg.params["actuator_name"]
        # ADR progress: 0 = initial (defaults), 1 = full sysid randomization
        self.scale_progress: float = 0.0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        joint_names: list[str],
        actuator_name: str,
        stiffness: float,
        damping: float,
        armature: float,
        friction: float,
        scale_range: tuple[float, float] = (0.8, 1.2),
        initial_stiffness: float | None = None,
        initial_damping: float | None = None,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.robot.device)
        N = len(env_ids)
        lo, hi = scale_range
        device = self.robot.device
        p = self.scale_progress

        def _scale(nominal):
            return nominal * (lo + torch.rand(N, 1, device=device) * (hi - lo))

        # Stiffness/damping: interpolate from initial defaults to sysid × U(scale_range)
        target_stiff = _scale(stiffness)
        target_damp = _scale(damping)
        if initial_stiffness is not None and initial_damping is not None:
            stiff_vals = initial_stiffness + p * (target_stiff - initial_stiffness)
            damp_vals = initial_damping + p * (target_damp - initial_damping)
        else:
            stiff_vals = target_stiff
            damp_vals = target_damp
        # Armature and friction: scaled by ADR progress (0 → sysid)
        arm_vals = _scale(armature) * p
        fric_vals = _scale(friction) * p

        gripper_actuator = self.robot.actuators[self.actuator_name]
        gripper_actuator.stiffness[env_ids] = stiff_vals
        gripper_actuator.damping[env_ids] = damp_vals
        self.robot.write_joint_stiffness_to_sim(stiff_vals, joint_ids=self.gripper_joint_ids, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(damp_vals, joint_ids=self.gripper_joint_ids, env_ids=env_ids)
        self.robot.write_joint_armature_to_sim(arm_vals, joint_ids=self.gripper_joint_ids, env_ids=env_ids)
        self.robot.write_joint_friction_coefficient_to_sim(fric_vals, joint_ids=self.gripper_joint_ids, env_ids=env_ids)


class randomize_rel_cartesian_osc_gains(ManagerTermBase):
    """Randomize RelCartesianOSCAction Kp/Kd gains.

    XYZ and RPY components are sampled independently (one scalar each).
    ``scale_range = (lo, hi)`` scales the target Kp: ``target_kp * uniform(lo, hi)``.

    When used with ADR, ``scale_progress`` (0→1) interpolates from the action
    config's default Kp/damping_ratio (initial) to ``terminal_kp``/
    ``terminal_damping_ratio``, with U(scale_range) randomization applied
    to the terminal values.  If no terminal params are given, randomizes
    around the action config defaults directly.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._action_name: str = cfg.params["action_name"]
        self._action_term = None
        # ADR progress: 0 = action defaults (initial), 1 = terminal gains
        self.scale_progress: float = cfg.params.get("initial_scale_progress", 0.0)

    def _resolve_action_term(self):
        if self._action_term is not None:
            return
        from .actions.task_space_actions import RelCartesianOSCAction

        action_term = self._env.action_manager._terms.get(self._action_name)
        if action_term is None or not isinstance(action_term, RelCartesianOSCAction):
            raise ValueError(f"Action term '{self._action_name}' is not a RelCartesianOSCAction.")
        self._action_term = action_term

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        action_name: str,
        scale_range: tuple[float, float] = (0.8, 1.2),
        terminal_kp: tuple[float, ...] | None = None,
        terminal_damping_ratio: tuple[float, ...] | None = None,
        initial_scale_progress: float = 0.0,
    ) -> None:
        self._resolve_action_term()

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)

        lo, hi = scale_range
        n = len(env_ids)
        p = self.scale_progress

        s_xyz = lo + torch.rand(n, 1, device=env.device) * (hi - lo)
        s_rpy = lo + torch.rand(n, 1, device=env.device) * (hi - lo)
        s_dr_xyz = lo + torch.rand(n, 1, device=env.device) * (hi - lo)
        s_dr_rpy = lo + torch.rand(n, 1, device=env.device) * (hi - lo)

        kp_default = self._action_term._kp_default  # (6,)
        dr_default = self._action_term._damping_ratio_default  # (6,)

        if terminal_kp is not None and terminal_damping_ratio is not None:
            # Terminal Kp with randomization
            kp_term = torch.tensor(terminal_kp, device=env.device, dtype=torch.float32)
            target_kp = kp_term.unsqueeze(0).repeat(n, 1)
            target_kp[:, :3] *= s_xyz
            target_kp[:, 3:] *= s_rpy

            dr_term = torch.tensor(terminal_damping_ratio, device=env.device, dtype=torch.float32)
            target_dr = dr_term.unsqueeze(0).repeat(n, 1)
            target_dr[:, :3] *= s_dr_xyz
            target_dr[:, 3:] *= s_dr_rpy

            # Interpolate from action defaults (initial) to terminal
            init_kp = kp_default.unsqueeze(0)
            init_dr = dr_default.unsqueeze(0)
            new_kp = init_kp + p * (target_kp - init_kp)
            new_dr = init_dr + p * (target_dr - init_dr)
        else:
            # No terminal specified — randomize around action defaults
            new_kp = kp_default.unsqueeze(0).repeat(n, 1)
            new_kp[:, :3] *= s_xyz
            new_kp[:, 3:] *= s_rpy
            new_dr = dr_default.unsqueeze(0).repeat(n, 1)
            new_dr[:, :3] *= s_dr_xyz
            new_dr[:, 3:] *= s_dr_rpy

        self._action_term._kp[env_ids] = new_kp
        self._action_term._kd[env_ids] = 2.0 * torch.sqrt(new_kp) * new_dr


class randomize_rel_cartesian_osc_gains_fixed(randomize_rel_cartesian_osc_gains):
    """Same as randomize_rel_cartesian_osc_gains but always applies scale_range (no curriculum)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.scale_progress = 1.0


class adr_sysid_curriculum(ManagerTermBase):
    """Automatic Domain Randomization curriculum for sysid event terms.

    Monitors the mean success rate from ``MultiResetManager``'s ``SuccessMonitor``
    and linearly ramps the ``scale_progress`` attribute of the target event terms
    from 0 (no friction/armature) to 1 (full sysid randomization).

    Updates are gated by ``update_every_n_steps`` (env steps via ``common_step_counter``)
    to ensure the update rate is independent of the number of environments.

    When success_rate > ``success_threshold_up``, ``scale_progress`` increases by ``delta``.
    When success_rate < ``success_threshold_down``, ``scale_progress`` decreases by ``delta``.

    If ``warmup_success_threshold`` is set, the bang-bang controller is suppressed
    until mean success rate reaches this threshold (latching: once warmed up, stays
    warmed up even if success later dips).
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._event_term_names: list[str] = cfg.params["event_term_names"]
        self._reset_event_name: str = cfg.params["reset_event_name"]
        self._initial_scale_progress: float = cfg.params.get("initial_scale_progress", 0.0)
        self._warmup_threshold: float | None = cfg.params.get("warmup_success_threshold")
        self._warmed_up: bool = self._warmup_threshold is None
        # Cache references to the event term instances (populated lazily)
        self._event_terms: list = []
        self._reset_term: object | None = None
        self._resolved = False
        # Step-gated update tracking
        self._last_update_step: int = -1
        self._last_state: dict[str, float] = {
            "scale_progress": self._initial_scale_progress,
            "mean_success_rate": 0.0,
        }

    def _resolve_terms(self):
        """Lazily resolve event term references (event manager may not be ready at __init__)."""
        if self._resolved:
            return
        self._resolved = True
        em = self._env.event_manager
        self._event_terms = []
        for name in self._event_term_names:
            term_cfg = em.get_term_cfg(name)
            self._event_terms.append(term_cfg.func)
        reset_cfg = em.get_term_cfg(self._reset_event_name)
        self._reset_term = reset_cfg.func
        if self._initial_scale_progress > 0.0:
            for term in self._event_terms:
                term.scale_progress = max(term.scale_progress, self._initial_scale_progress)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        event_term_names: list[str],
        reset_event_name: str,
        success_threshold_up: float = 0.7,
        success_threshold_down: float = 0.3,
        delta: float = 0.01,
        update_every_n_steps: int = 160,
        initial_scale_progress: float = 0.0,
        warmup_success_threshold: float | None = None,
    ) -> dict[str, float]:
        self._resolve_terms()

        # Only update once every N env steps (agnostic to num_envs)
        current_step = env.common_step_counter
        if (current_step - self._last_update_step) < update_every_n_steps:
            return self._last_state
        self._last_update_step = current_step

        # Get mean success rate across all tasks
        if not hasattr(self._reset_term, "success_monitor"):
            self._last_state = {"scale_progress": self._event_terms[0].scale_progress if self._event_terms else 0.0}
            return self._last_state

        success_rates = self._reset_term.success_monitor.get_success_rate()
        mean_success = success_rates.mean().item()

        # Warmup gate: hold scale_progress until success exceeds threshold
        if not self._warmed_up:
            if mean_success >= self._warmup_threshold:
                self._warmed_up = True
            else:
                self._last_state = {
                    "scale_progress": self._event_terms[0].scale_progress if self._event_terms else 0.0,
                    "mean_success_rate": mean_success,
                }
                return self._last_state

        # Update scale_progress based on thresholds
        current_progress = self._event_terms[0].scale_progress if self._event_terms else 0.0
        if mean_success > success_threshold_up:
            current_progress = min(1.0, current_progress + delta)
        elif mean_success < success_threshold_down:
            current_progress = max(0.0, current_progress - delta)

        # Apply to all target event terms
        for term in self._event_terms:
            term.scale_progress = current_progress

        self._last_state = {
            "scale_progress": current_progress,
            "mean_success_rate": mean_success,
        }
        return self._last_state


class action_scale_curriculum(ManagerTermBase):
    """Curriculum that gradually tightens action scales on the OSC action term.

    Linearly interpolates the per-axis ``_scale`` tensor from ``initial_scales``
    to ``target_scales`` as progress goes from 0 to 1.  This limits the maximum
    per-step EE motion without saturating the PD controller (unlike pose-error
    clipping), preserving gradient signal for RL.

    Uses the same success-rate monitoring as ``adr_sysid_curriculum``: progress
    increases when success_rate > ``success_threshold_up`` and decreases when
    < ``success_threshold_down``.
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._action_name: str = cfg.params["action_name"]
        self._reset_event_name: str = cfg.params["reset_event_name"]
        self._action_term = None
        self._reset_term = None
        self._resolved = False
        self._last_update_step: int = -1
        self._progress: float = cfg.params.get("initial_progress", 0.0)
        self._last_state: dict[str, float] = {"scale_progress": self._progress, "mean_success_rate": 0.0}

    def _resolve(self):
        if self._resolved:
            return
        self._resolved = True
        from .actions.task_space_actions import RelCartesianOSCAction

        action_term = self._env.action_manager._terms.get(self._action_name)
        if action_term is None or not isinstance(action_term, RelCartesianOSCAction):
            raise ValueError(f"Action term '{self._action_name}' is not a RelCartesianOSCAction.")
        self._action_term = action_term

        em = self._env.event_manager
        reset_cfg = em.get_term_cfg(self._reset_event_name)
        self._reset_term = reset_cfg.func

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        action_name: str,
        reset_event_name: str,
        target_scales: list[float],
        initial_scales: list[float],
        success_threshold_up: float = 0.7,
        success_threshold_down: float = 0.3,
        delta: float = 0.005,
        update_every_n_steps: int = 200,
        initial_progress: float = 0.0,
    ) -> dict[str, float]:
        self._resolve()

        current_step = env.common_step_counter
        if (current_step - self._last_update_step) < update_every_n_steps:
            return self._last_state
        self._last_update_step = current_step

        if not hasattr(self._reset_term, "success_monitor"):
            self._last_state = {"scale_progress": self._progress, "mean_success_rate": 0.0}
            return self._last_state

        success_rates = self._reset_term.success_monitor.get_success_rate()
        mean_success = success_rates.mean().item()

        if mean_success > success_threshold_up:
            self._progress = min(1.0, self._progress + delta)
        elif mean_success < success_threshold_down:
            self._progress = max(0.0, self._progress - delta)

        initial = torch.tensor(initial_scales, device=env.device, dtype=torch.float32)
        target = torch.tensor(target_scales, device=env.device, dtype=torch.float32)
        effective = initial + self._progress * (target - initial)

        self._action_term._scale = effective

        self._last_state = {
            "scale_progress": self._progress,
            "mean_success_rate": mean_success,
        }
        return self._last_state


class obs_noise_curriculum(ManagerTermBase):
    """Curriculum that gradually increases uniform noise on observation terms.

    Monitors success rate and linearly ramps the half-range on the specified
    observation terms' ``AdditiveUniformNoiseCfg`` from ``initial_half_range``
    to ``target_half_range`` as progress goes from 0 to 1.  At full progress
    the noise is U(-target_half_range, +target_half_range).
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._obs_group: str = cfg.params["obs_group"]
        self._obs_term_names: list[str] = cfg.params["obs_term_names"]
        self._reset_event_name: str = cfg.params["reset_event_name"]
        self._reset_term = None
        self._obs_term_cfgs: list = []
        self._resolved = False
        self._last_update_step: int = -1
        self._progress: float = 0.0
        self._last_state: dict[str, float] = {"scale_progress": 0.0, "mean_success_rate": 0.0}

    def _resolve(self):
        if self._resolved:
            return
        self._resolved = True
        om = self._env.observation_manager
        term_names = om._group_obs_term_names[self._obs_group]
        term_cfgs = om._group_obs_term_cfgs[self._obs_group]
        name_to_cfg = dict(zip(term_names, term_cfgs))
        for name in self._obs_term_names:
            if name not in name_to_cfg:
                raise ValueError(f"Obs term '{name}' not found in group '{self._obs_group}'. Available: {term_names}")
            cfg = name_to_cfg[name]
            if cfg.noise is None:
                raise ValueError(
                    f"Obs term '{name}' has no noise config. Set noise=AdditiveUniformNoiseCfg(n_min=0.0, n_max=0.0)."
                )
            self._obs_term_cfgs.append(cfg)

        em = self._env.event_manager
        reset_cfg = em.get_term_cfg(self._reset_event_name)
        self._reset_term = reset_cfg.func

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        obs_group: str,
        obs_term_names: list[str],
        reset_event_name: str,
        target_half_range: float,
        initial_half_range: float = 0.0,
        success_threshold_up: float = 0.7,
        success_threshold_down: float = 0.3,
        delta: float = 0.005,
        update_every_n_steps: int = 200,
    ) -> dict[str, float]:
        self._resolve()

        current_step = env.common_step_counter
        if (current_step - self._last_update_step) < update_every_n_steps:
            return self._last_state
        self._last_update_step = current_step

        if not hasattr(self._reset_term, "success_monitor"):
            self._last_state = {"scale_progress": self._progress, "mean_success_rate": 0.0}
            return self._last_state

        success_rates = self._reset_term.success_monitor.get_success_rate()
        mean_success = success_rates.mean().item()

        if mean_success > success_threshold_up:
            self._progress = min(1.0, self._progress + delta)
        elif mean_success < success_threshold_down:
            self._progress = max(0.0, self._progress - delta)

        effective_hr = initial_half_range + self._progress * (target_half_range - initial_half_range)
        for term_cfg in self._obs_term_cfgs:
            term_cfg.noise.n_min = -effective_hr
            term_cfg.noise.n_max = effective_hr

        self._last_state = {
            "scale_progress": self._progress,
            "mean_success_rate": mean_success,
        }
        return self._last_state


class randomize_visual_appearance_multiple_meshes(ManagerTermBase):
    """Randomize the visual appearance (texture or color) of multiple mesh bodies using Replicator API.

    This unified function can randomize either textures or solid colors on mesh bodies.
    Use ``texture_prob`` to control the probability of applying textures vs solid colors:
    - ``texture_prob=1.0``: Always use textures (default)
    - ``texture_prob=0.0``: Always use solid colors
    - ``0 < texture_prob < 1``: Randomly choose between texture and color each reset

    Texture paths can be provided via:
    1. ``texture_paths`` parameter (list of full paths)
    2. ``texture_config_path`` parameter (path to a YAML file)

    Colors can be specified as:
    1. A dict with ``r``, ``g``, ``b`` keys mapping to (low, high) ranges
    2. A list of RGB tuples to choose from

    Parameters:
    - ``texture_prob``: Probability of using texture vs color (default 1.0 = always texture)
    - ``colors``: Color specification for solid color mode
    - ``diffuse_tint_range``: RGB tint multiplier for texture mode, e.g. ((0.8, 0.8, 0.8), (1.0, 1.0, 1.0))

    .. note::
        Requires :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to be False.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term."""
        super().__init__(cfg, env)

        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        texture_paths = cfg.params.get("texture_paths")
        texture_config_path = cfg.params.get("texture_config_path")
        event_name = cfg.params.get("event_name")
        mesh_names: list[str] = cfg.params.get("mesh_names", [])

        # Core parameters
        self.texture_prob = cfg.params.get("texture_prob", 1.0)  # 1.0 = always texture, 0.0 = always color
        self.diffuse_tint_range = cfg.params.get("diffuse_tint_range")  # ((r,g,b), (r,g,b))
        self.colors = cfg.params.get("colors", {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)})
        self.color_event_name = f"{event_name}_color"

        # Material property ranges (DEXTRAH-aligned defaults)
        self._texture_scale_range = cfg.params.get("texture_scale_range", (0.7, 5.0))
        self._roughness_range = cfg.params.get("roughness_range", (0.0, 1.0))
        self._metallic_range = cfg.params.get("metallic_range", (0.0, 1.0))
        self._specular_range = cfg.params.get("specular_range", (0.0, 1.0))

        # Load texture paths from YAML config if provided
        if texture_config_path is not None:
            texture_paths = utils.load_asset_paths_from_config(
                texture_config_path, cache_subdir="textures", skip_validation=False
            )
            logging.info(f"[{event_name}] Loaded {len(texture_paths)} texture paths.")
        if self.texture_prob > 0 and (texture_paths is None or len(texture_paths) == 0):
            raise RuntimeError(
                f"[{event_name}] texture_prob={self.texture_prob} but no texture paths loaded. "
                f"Check texture_config_path={texture_config_path}"
            )
        if texture_paths:
            non_local = [p for p in texture_paths if not p.startswith("/")]
            if non_local:
                raise RuntimeError(
                    f"[{event_name}] {len(non_local)} texture paths are non-local (Nucleus) "
                    "and will silently fail if Nucleus is unreachable. "
                    f"First 3: {non_local[:3]}. "
                    "Use only local/cloud-cached textures."
                )
            missing = [p for p in texture_paths if not os.path.exists(p)]
            if missing:
                raise RuntimeError(
                    f"[{event_name}] {len(missing)}/{len(texture_paths)} texture files missing on disk. "
                    f"First 3: {missing[:3]}"
                )

        # check to make sure replicate_physics is set to False
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual appearance with scene replication enabled."
                " Please set 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]
        asset_prim_path = asset.cfg.prim_path

        # create the affected prim path pattern
        if len(mesh_names) == 0:
            pattern_with_visuals = f"{asset_prim_path}/.*/visuals"
            matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
            if matching_prims:
                prim_path_pattern = pattern_with_visuals
            else:
                prim_path_pattern = f"{asset_prim_path}/.*"
                carb.log_info(
                    f"Pattern '{pattern_with_visuals}' found no prims. Falling back to '{prim_path_pattern}'."
                )
        else:
            mesh_prim_paths = []
            for mesh_name in mesh_names:
                if not mesh_name.startswith("/"):
                    mesh_name = "/" + mesh_name
                mesh_prim_paths.append(f"{asset_prim_path}{mesh_name}")
            prim_path_pattern = "|".join(mesh_prim_paths)

        # Store texture paths and RNG
        self.texture_paths = texture_paths
        unique_seed = hash(event_name) % (2**31)
        self.texture_rng = rep.rng.ReplicatorRNG(seed=unique_seed)
        self.prim_path_pattern = prim_path_pattern

        # Get prims and create materials
        stage = sim_utils.SimulationContext.instance().stage
        prims_group = rep.functional.get.prims(path_pattern=prim_path_pattern, stage=stage)
        num_prims = len(prims_group)

        if num_prims == 0:
            raise RuntimeError(
                f"[randomize_visual_appearance_multiple_meshes] No prims found matching: {prim_path_pattern}. "
                "Check mesh_names and asset_cfg."
            )

        # Disable instanceable on prims
        for prim in prims_group:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        # Create OmniPBR materials and bind them to the prims
        self.material_prims = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
        )
        self._stage = stage
        self._texture_verified = False

        # Cache shader prims for direct USD access (avoids Replicator pipeline race conditions)
        from pxr import Sdf, UsdShade

        self._shader_prims = []
        for i, mat_prim in enumerate(self.material_prims):
            mat_path = str(mat_prim.GetPath()) if hasattr(mat_prim, "GetPath") else str(mat_prim)
            shader_prim = stage.GetPrimAtPath(Sdf.Path(f"{mat_path}/Shader"))
            if not shader_prim.IsValid():
                raise RuntimeError(f"[{event_name}] Shader not found at {mat_path}/Shader after material creation.")
            self._shader_prims.append(shader_prim)

            # Force direct USD material binding (Replicator bind_prims can silently fail)
            material = UsdShade.Material(mat_prim)
            target_prim = prims_group[i]
            UsdShade.MaterialBindingAPI.Apply(target_prim)
            UsdShade.MaterialBindingAPI(target_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

        # Ensure material property inputs exist on each shader
        _required_inputs = {
            "texture_scale": Sdf.ValueTypeNames.Float2,
            "reflection_roughness_constant": Sdf.ValueTypeNames.Float,
            "metallic_constant": Sdf.ValueTypeNames.Float,
            "specular_level": Sdf.ValueTypeNames.Float,
        }
        for shader_prim in self._shader_prims:
            shader = UsdShade.Shader(shader_prim)
            props = shader_prim.GetPropertyNames()
            for attr_name, attr_type in _required_inputs.items():
                if f"inputs:{attr_name}" not in props:
                    shader.CreateInput(attr_name, attr_type)

        # Parse color config for direct USD color generation
        if isinstance(self.colors, dict):
            self._color_low = np.array([self.colors[key][0] for key in ["r", "g", "b"]])
            self._color_high = np.array([self.colors[key][1] for key in ["r", "g", "b"]])
        else:
            self._color_list = list(self.colors)
            self._color_low = None
            self._color_high = None

        # Apply initial randomization so envs don't start with default appearance
        self(env, torch.arange(env.num_envs, device=env.device), **cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str] | None = None,
        texture_config_path: str | None = None,
        mesh_names: list[str] = [],
        texture_prob: float = 1.0,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]] | None = None,
        diffuse_tint_range: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        texture_scale_range: tuple[float, float] | None = None,
        roughness_range: tuple[float, float] | None = None,
        metallic_range: tuple[float, float] | None = None,
        specular_range: tuple[float, float] | None = None,
    ):
        if not self._shader_prims:
            return

        from pxr import Sdf

        rng = self.texture_rng.generator
        num_prims = len(self._shader_prims)

        # Per-prim texture vs color decision
        use_texture_mask = rng.random(size=num_prims) < self.texture_prob

        # Pre-generate random material properties (shared by both modes)
        rand_roughness = rng.uniform(self._roughness_range[0], self._roughness_range[1], size=num_prims)
        rand_metallic = rng.uniform(self._metallic_range[0], self._metallic_range[1], size=num_prims)
        rand_specular = rng.uniform(self._specular_range[0], self._specular_range[1], size=num_prims)

        # Pre-generate texture-mode data
        random_textures = None
        if self.texture_paths and use_texture_mask.any():
            random_textures = rng.choice(self.texture_paths, size=num_prims)
            for tex_path in random_textures:
                if tex_path.startswith("/") and not os.path.exists(tex_path):
                    raise RuntimeError(
                        f"[randomize_visual_appearance] Texture file not found: {tex_path}. "
                        "Local texture paths must exist on disk."
                    )

        # Pre-generate color-mode data
        random_colors = None
        if not use_texture_mask.all():
            if self._color_low is not None:
                random_colors = rng.uniform(self._color_low, self._color_high, size=(num_prims, 3))
            else:
                indices = rng.integers(0, len(self._color_list), size=num_prims)
                random_colors = np.array([self._color_list[i] for i in indices])

        n_tex = int(use_texture_mask.sum())
        n_col = num_prims - n_tex
        logging.debug(f"[{event_name}] {n_tex} TEXTURE / {n_col} COLOR -> {num_prims} prims")

        with Sdf.ChangeBlock():
            for i, shader_prim in enumerate(self._shader_prims):
                # Material properties (both modes)
                shader_prim.GetAttribute("inputs:reflection_roughness_constant").Set(float(rand_roughness[i]))
                shader_prim.GetAttribute("inputs:metallic_constant").Set(float(rand_metallic[i]))
                shader_prim.GetAttribute("inputs:specular_level").Set(float(rand_specular[i]))

                if use_texture_mask[i] and random_textures is not None:
                    shader_prim.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath(random_textures[i]))
                    s = float(rng.uniform(self._texture_scale_range[0], self._texture_scale_range[1]))
                    shader_prim.GetAttribute("inputs:texture_scale").Set(Gf.Vec2f(s, s))

                    if self.diffuse_tint_range is not None:
                        t = rng.uniform(self.diffuse_tint_range[0], self.diffuse_tint_range[1], size=3)
                        shader_prim.GetAttribute("inputs:diffuse_tint").Set(
                            Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))
                        )
                else:
                    shader_prim.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath(""))
                    if random_colors is not None:
                        shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(
                            Gf.Vec3f(float(random_colors[i][0]), float(random_colors[i][1]), float(random_colors[i][2]))
                        )

        if not self._texture_verified and random_textures is not None and use_texture_mask.any():
            first_tex_idx = int(np.argmax(use_texture_mask))
            self._verify_texture_applied(random_textures[first_tex_idx], event_name)
            self._texture_verified = True

    def _verify_texture_applied(self, expected_texture: str, event_name: str):
        """One-time check that textures are actually being applied by reading back from USD."""
        shader_prim = self._shader_prims[0]
        shader_path = str(shader_prim.GetPath())
        tex_attr = shader_prim.GetAttribute("inputs:diffuse_texture")
        if not tex_attr or not tex_attr.IsValid():
            raise RuntimeError(
                f"[{event_name}] Texture verification failed: 'inputs:diffuse_texture' attribute "
                f"not found on {shader_path}."
            )
        current_val = tex_attr.Get()
        logging.debug(
            f"[{event_name}] Texture verify: shader={shader_path}, value={current_val}, expected={expected_texture}"
        )
        if current_val is None or str(current_val) == "":
            raise RuntimeError(
                f"[{event_name}] Texture verification failed: diffuse_texture is empty after "
                f"USD Set. Expected: {expected_texture}."
            )


class implicit_to_explicit_swap(ManagerTermBase):
    """One-shot curriculum that swaps the arm actuator from ImplicitActuator to
    an explicit actuator (e.g. DelayedDCMotor) once the ADR sysid curriculum
    reaches ``scale_progress == 1.0``.

    After the swap, ``randomize_arm_from_sysid`` (which looks up
    ``robot.actuators[actuator_name]`` each call) will automatically pick up
    the new explicit actuator and start setting delay buffers.

    Set ``swap_at_init=True`` to trigger the swap on the first call regardless
    of ``scale_progress`` (useful when resuming from a checkpoint where the
    swap had already occurred).
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._swapped = False
        self._swap_at_init: bool = cfg.params.get("swap_at_init", False)
        self._robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._actuator_name: str = cfg.params["actuator_name"]
        self._explicit_arm_cfg = cfg.params["explicit_arm_cfg"]
        self._sysid_event_name: str = cfg.params["sysid_event_name"]
        self._sysid_term = None
        self._resolved = False

    def _resolve(self):
        if self._resolved:
            return
        self._resolved = True
        em = self._env.event_manager
        term_cfg = em.get_term_cfg(self._sysid_event_name)
        self._sysid_term = term_cfg.func

    def _do_swap(self, env: ManagerBasedEnv) -> dict[str, object]:
        old_actuator = self._robot.actuators[self._actuator_name]
        new_actuator = self._explicit_arm_cfg.class_type(
            cfg=self._explicit_arm_cfg,
            joint_names=old_actuator.joint_names,
            joint_ids=old_actuator.joint_indices,
            num_envs=self._robot.num_instances,
            device=self._robot.device,
        )
        self._robot.actuators[self._actuator_name] = new_actuator
        joint_ids = old_actuator.joint_indices
        self._robot.write_joint_stiffness_to_sim(0.0, joint_ids=joint_ids)
        self._robot.write_joint_damping_to_sim(0.0, joint_ids=joint_ids)
        self._robot.write_joint_effort_limit_to_sim(1.0e9, joint_ids=joint_ids)
        self._robot.write_joint_velocity_limit_to_sim(new_actuator.velocity_limit, joint_ids=joint_ids)
        self._robot._data.default_joint_stiffness[:, joint_ids] = new_actuator.stiffness
        self._robot._data.default_joint_damping[:, joint_ids] = new_actuator.damping

        self._swapped = True
        carb.log_info(
            f"[implicit_to_explicit_swap] Swapped '{self._actuator_name}' from "
            f"{type(old_actuator).__name__} to {type(new_actuator).__name__} "
            f"at step {env.common_step_counter}"
        )
        return {"actuator_swapped": True, "swap_step": env.common_step_counter}

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        asset_cfg: SceneEntityCfg,
        actuator_name: str,
        explicit_arm_cfg,
        sysid_event_name: str,
        swap_at_init: bool = False,
    ) -> dict[str, object]:
        if self._swapped:
            return {"actuator_swapped": True}

        if self._swap_at_init:
            return self._do_swap(env)

        self._resolve()
        if self._sysid_term.scale_progress < 1.0:
            return {"actuator_swapped": False, "scale_progress": self._sysid_term.scale_progress}

        return self._do_swap(env)
