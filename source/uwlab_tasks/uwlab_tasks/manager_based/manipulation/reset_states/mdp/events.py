# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for manipulation tasks."""

import numpy as np
import os
import scipy.stats as stats
import tempfile
import torch
import trimesh
import trimesh.transformations as tra

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
from isaaclab.utils.assets import retrieve_file_path
from pxr import UsdGeom

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils

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
                asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


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
        self.base_path: str = cfg.params.get("base_path")
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

        # Compute grasp dataset path using object hash
        self.grasp_dataset_path = self._compute_grasp_dataset_path()

        # Load and pre-compute grasp data for fast sampling
        self._load_and_precompute_grasps(env)

    def _compute_grasp_dataset_path(self) -> str:
        """Compute grasp dataset path using hash of the fixed asset (insertive object)."""
        usd_path = self.fixed_asset.cfg.spawn.usd_path
        object_hash = utils.compute_assembly_hash(usd_path)
        return f"{self.base_path}/{object_hash}.pt"

    def _load_and_precompute_grasps(self, env):
        """Load Torch (.pt) grasp data and convert to optimized tensors."""
        # Handle URL or local path
        local_path = retrieve_file_path(self.grasp_dataset_path)
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
        base_path: str,
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
        self.base_path: str = cfg.params.get("base_path")
        self.receptive_object_cfg: SceneEntityCfg = cfg.params.get("receptive_object_cfg")
        self.receptive_object: RigidObject = env.scene[self.receptive_object_cfg.name]
        self.insertive_object_cfg: SceneEntityCfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object: RigidObject = env.scene[self.insertive_object_cfg.name]

        # Pose range for sampling variations
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b", dict())
        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)

        # Compute partial assembly dataset path using object hash
        self.partial_assembly_dataset_path = self._compute_partial_assembly_dataset_path()

        # Load and pre-compute partial assembly data for fast sampling
        self._load_and_precompute_partial_assemblies(env)

    def _compute_partial_assembly_dataset_path(self) -> str:
        """Compute partial assembly dataset path using hash of insertive and receptive objects."""
        insertive_usd_path = self.insertive_object.cfg.spawn.usd_path
        receptive_usd_path = self.receptive_object.cfg.spawn.usd_path
        object_hash = utils.compute_assembly_hash(insertive_usd_path, receptive_usd_path)
        return f"{self.base_path}/{object_hash}.pt"

    def _load_and_precompute_partial_assemblies(self, env):
        """Load Torch (.pt) partial assembly data and convert to optimized tensors."""
        local_path = retrieve_file_path(self.partial_assembly_dataset_path)
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
        base_path: str,
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

        base_paths: list[str] = cfg.params.get("base_paths", [])
        probabilities: list[float] = cfg.params.get("probs", [])

        if not base_paths:
            raise ValueError("No base paths provided")
        if len(base_paths) != len(probabilities):
            raise ValueError("Number of base paths must match number of probabilities")

        # Compute dataset paths using object hash
        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        receptive_usd_path = env.scene["receptive_object"].cfg.spawn.usd_path
        reset_state_hash = utils.compute_assembly_hash(insertive_usd_path, receptive_usd_path)

        # Generate dataset paths using provided base paths
        dataset_files = []
        for base_path in base_paths:
            dataset_files.append(f"{base_path}/{reset_state_hash}.pt")

        # Load all datasets
        self.datasets = []
        num_states = []
        rank = int(os.getenv("RANK", "0"))
        download_dir = os.path.join(tempfile.gettempdir(), f"rank_{rank}")
        for dataset_file in dataset_files:
            # Handle both local files and URLs
            local_file_path = retrieve_file_path(dataset_file, download_dir=download_dir)

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
        base_paths: list[str],
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
            self._env.scene.reset_to(states_to_reset_from["initial_state"], env_ids=current_env_ids, is_relative=True)

        # Reset velocities
        robot: Articulation = self._env.scene["robot"]
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]), env_ids=env_ids)


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


def randomize_operational_space_controller_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    action_name: str,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
    operation: str = "scale",
    distribution: str = "log_uniform",
) -> None:
    """Randomize operational space controller motion stiffness and damping gains.

    This function randomizes the motion_stiffness_task and motion_damping_ratio_task parameters
    of an operational space controller. The first three terms (xyz) and last three terms (ypr)
    are randomized together to maintain consistency within translational and rotational components.

    Args:
        env: The environment instance.
        env_ids: The environment indices to randomize. If None, all environments are randomized.
        action_name: The name of the action term to randomize.
        stiffness_distribution_params: The distribution parameters for stiffness (min, max).
        damping_distribution_params: The distribution parameters for damping ratio (min, max).
        operation: The operation to perform on the gains. Currently supports "scale" and "add".
        distribution: The distribution to sample from. Currently supports "log_uniform".

    Raises:
        ValueError: If the action is not found or is not an operational space controller action.
        ValueError: If an unsupported distribution is specified.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Get the action term
    action_term = env.action_manager._terms.get(action_name)
    if action_term is None:
        raise ValueError(f"Action term '{action_name}' not found in action manager.")

    # Check if it's an operational space controller action
    if not hasattr(action_term, "_osc") or not hasattr(action_term._osc, "cfg"):
        raise ValueError(f"Action term '{action_name}' does not appear to be an operational space controller.")

    controller = action_term._osc

    # Check distribution type
    if operation != "scale":
        raise ValueError(f"Operation '{operation}' not supported. Only 'scale' is supported.")
    if distribution not in ["uniform", "log_uniform"]:
        raise ValueError(
            f"Distribution '{distribution}' not supported. Only 'uniform' and 'log_uniform' are supported."
        )

    # Sample random multipliers for stiffness (xyz and ypr separately)
    if distribution == "uniform":
        stiff_xyz_multiplier = (
            torch.rand(len(env_ids), device=env.device)
            * (stiffness_distribution_params[1] - stiffness_distribution_params[0])
            + stiffness_distribution_params[0]
        )

        stiff_rpy_multiplier = (
            torch.rand(len(env_ids), device=env.device)
            * (stiffness_distribution_params[1] - stiffness_distribution_params[0])
            + stiffness_distribution_params[0]
        )
    else:  # log_uniform
        log_min_stiff = torch.log(torch.tensor(stiffness_distribution_params[0], device=env.device))
        log_max_stiff = torch.log(torch.tensor(stiffness_distribution_params[1], device=env.device))

        stiff_xyz_multiplier = torch.exp(
            torch.rand(len(env_ids), device=env.device) * (log_max_stiff - log_min_stiff) + log_min_stiff
        )

        stiff_rpy_multiplier = torch.exp(
            torch.rand(len(env_ids), device=env.device) * (log_max_stiff - log_min_stiff) + log_min_stiff
        )

    # Sample random multipliers for damping (xyz and ypr separately)
    if distribution == "uniform":
        damp_xyz_multiplier = (
            torch.rand(len(env_ids), device=env.device)
            * (damping_distribution_params[1] - damping_distribution_params[0])
            + damping_distribution_params[0]
        )

        damp_rpy_multiplier = (
            torch.rand(len(env_ids), device=env.device)
            * (damping_distribution_params[1] - damping_distribution_params[0])
            + damping_distribution_params[0]
        )
    else:  # log_uniform
        log_min_damp = torch.log(torch.tensor(damping_distribution_params[0], device=env.device))
        log_max_damp = torch.log(torch.tensor(damping_distribution_params[1], device=env.device))

        damp_xyz_multiplier = torch.exp(
            torch.rand(len(env_ids), device=env.device) * (log_max_damp - log_min_damp) + log_min_damp
        )

        damp_rpy_multiplier = torch.exp(
            torch.rand(len(env_ids), device=env.device) * (log_max_damp - log_min_damp) + log_min_damp
        )

    # Apply randomization to motion stiffness gains
    # Original gains from config
    original_stiffness = torch.tensor(controller.cfg.motion_stiffness_task, device=env.device)

    # Create new stiffness values for each environment
    new_stiffness = torch.zeros((len(env_ids), 6), device=env.device)
    new_stiffness[:, 0:3] = original_stiffness[0:3] * stiff_xyz_multiplier.unsqueeze(-1)  # xyz
    new_stiffness[:, 3:6] = original_stiffness[3:6] * stiff_rpy_multiplier.unsqueeze(-1)  # rpy

    # Update the controller's motion stiffness gains
    controller._motion_p_gains_task[env_ids] = torch.diag_embed(new_stiffness)
    # Apply selection matrix to zero out non-controlled axes
    controller._motion_p_gains_task[env_ids] = (
        controller._selection_matrix_motion_task[env_ids] @ controller._motion_p_gains_task[env_ids]
    )

    # Apply randomization to motion damping gains
    # Original damping ratios from config
    original_damping = torch.tensor(controller.cfg.motion_damping_ratio_task, device=env.device)

    # Create new damping values for each environment
    new_damping_ratios = torch.zeros((len(env_ids), 6), device=env.device)
    new_damping_ratios[:, 0:3] = original_damping[0:3] * damp_xyz_multiplier.unsqueeze(-1)  # xyz
    new_damping_ratios[:, 3:6] = original_damping[3:6] * damp_rpy_multiplier.unsqueeze(-1)  # rpy

    # Update the controller's motion damping gains
    # Damping = 2 * sqrt(stiffness) * damping_ratio
    controller._motion_d_gains_task[env_ids] = torch.diag_embed(
        2 * torch.diagonal(controller._motion_p_gains_task[env_ids], dim1=-2, dim2=-1).sqrt() * new_damping_ratios
    )
