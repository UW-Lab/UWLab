# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import pybullet as p
from isaaclab.utils import math as math_utils


class BulletArticulationKinematicsData:
    def __init__(self, dof_dim: int, link_dim: int, device: str):
        self.dof_dim = dof_dim
        self.link_dim = link_dim
        self.device = device
        self.reset()

    def reset(self):
        self.link_names: list[str] = []
        self.dof_names: list[str] = []
        self.dof_types: list[str] = []
        self.dof_indices = torch.zeros((1, self.dof_dim), device=self.device)

        self.link_transforms = torch.zeros((1, self.link_dim, 7), device=self.device)
        self.link_velocities = torch.zeros((1, self.link_dim, 6), device=self.device)
        self.link_mass = torch.zeros((1, self.link_dim), device=self.device)
        self.link_inertia = torch.zeros((1, self.link_dim, 9), device=self.device)
        self.link_coms = torch.zeros((1, self.link_dim, 7), device=self.device)

        self.mass_matrix = torch.zeros((1, self.dof_dim, self.dof_dim), device=self.device)

        self.dof_positions = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_velocities = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_accelerations = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_torques = torch.zeros((1, self.dof_dim), device=self.device)

        self.dof_position_target = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_velocity_target = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_torque_target = torch.zeros((1, self.dof_dim), device=self.device)

        self.dof_stiffness = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_armatures = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_frictions = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_damping = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_limits = torch.zeros((1, self.dof_dim, 2), device=self.device)
        self.dof_max_forces = torch.zeros((1, self.dof_dim), device=self.device)
        self.dof_max_velocity = torch.zeros((1, self.dof_dim), device=self.device)

        self.jacobians = torch.zeros((1, self.link_dim, 6, self.dof_dim), device=self.device)


dof_types_dict: dict[int, str] = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
}


class BulletArticulationKinematics:
    def __init__(self, urdf_path, is_fixed_base, debug_visualize, dt, device):
        """
        Initialize PyBullet in DIRECT mode and load the URDF at urdf_path.
        """
        # Connect in DIRECT mode (no GUI)
        self.debug_visualize = debug_visualize
        if debug_visualize:
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        self._is_fixed_base = is_fixed_base
        # Load URDF, useFixedBase=True to keep it from falling due to gravity
        self.articulation = p.loadURDF(urdf_path, useFixedBase=is_fixed_base, physicsClientId=self.client_id)
        # Query how many joints (which also defines how many child links).
        self._num_joints = p.getNumJoints(self.articulation, physicsClientId=self.client_id)
        self._num_links = self._num_joints + 1

        # Identify which joints are actually movable (revolute or prismatic)
        self._dof_indicies = []
        for j in range(self._num_joints):
            info = p.getJointInfo(self.articulation, j, physicsClientId=self.client_id)
            joint_type = info[2]  # 0=REVOLUTE, 1=PRISMATIC, 4=FIXED, ...
            if joint_type not in [p.JOINT_FIXED]:
                self._dof_indicies.append(j)

        # Number of degrees of freedom (movable joints)
        self._num_dofs = len(self._dof_indicies)
        print("Total Links:", self._num_links, "Total joints:", self._num_joints, "Movable DoF:", self._num_dofs)

        # Initialize the state storage
        self.articulation_view_data = BulletArticulationKinematicsData(self._num_dofs, self._num_links, device)
        self.device = device

        self.populate_joint_state()
        self.populate_link_transforms()
        self.populate_link_velocities()
        self.populate_link_mass()
        self.populate_inertia()

    def close(self):
        """
        Signal the rendering thread to stop and wait for it to join.
        """
        self.pause = True

    @property
    def fixed_base(self):
        return self._is_fixed_base

    @property
    def num_links(self):
        return self._num_links

    @property
    def num_dof(self):
        return self._num_dofs

    @property
    def joint_names(self):
        return self.articulation_view_data.dof_names

    @property
    def link_names(self):
        return self.articulation_view_data.link_names

    def render(self):
        p.stepSimulation(physicsClientId=self.client_id)

    def get_root_link_transform(self, clone: bool = True) -> torch.Tensor:
        data = self.articulation_view_data.link_transforms[:, 0]
        return data.clone() if clone else data

    def get_root_link_velocity(self, clone: bool = True) -> torch.Tensor:
        data = self.articulation_view_data.link_velocities[:, 0]
        return data.clone() if clone else data

    def get_link_transforms(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.link_transforms[:, body_indices]
        return data.clone() if clone else data

    def get_link_velocities(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.link_velocities[:, body_indices]
        return data.clone() if clone else data

    def get_link_coms(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        """in body local frames"""
        data = self.articulation_view_data.link_coms[:, body_indices]
        return data.clone() if clone else data

    def get_link_masses(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.link_mass[:, body_indices]
        return data.clone() if clone else data

    def get_link_inertias(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.link_inertia[:, body_indices]
        return data.clone() if clone else data

    def get_dof_limits(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_limits[:, :, body_indices]
        return data.clone() if clone else data

    def get_dof_positions(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_positions[:, body_indices]
        return data.clone() if clone else data

    def get_dof_position_targets(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_position_target[:, body_indices]
        return data.clone() if clone else data

    def get_dof_velocities(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_velocities[:, body_indices]
        return data.clone() if clone else data

    def get_dof_velocity_targets(
        self,
        body_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_velocity_target[:, body_indices]
        return data.clone() if clone else data

    def get_dof_max_velocities(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_max_velocity[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_torques(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_torques[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_max_forces(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_max_forces[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_stiffnesses(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ):
        data = self.articulation_view_data.dof_stiffness[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_dampings(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ) -> torch.Tensor:
        data = self.articulation_view_data.dof_damping[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_frictions(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ):
        data = self.articulation_view_data.dof_frictions[:, joint_indices]
        return data.clone() if clone else data

    def get_dof_armatures(
        self,
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
        clone: bool = True,
    ):
        data = self.articulation_view_data.dof_armatures[:, joint_indices]
        return data.clone() if clone else data

    def get_mass_matrix(self) -> torch.Tensor:
        """
        Return the mass matrix for the current state.
        """
        # 1) Gather the current joint positions from PyBullet
        joint_positions = self.get_dof_positions()

        # 2) Compute the mass matrix
        mass_matrix = p.calculateMassMatrix(
            bodyUniqueId=self.articulation, objPositions=joint_positions[0].tolist(), physicsClientId=self.client_id
        )

        # 3) Convert the mass matrix to a torch tensor
        mass_matrix = torch.tensor(mass_matrix, device=self.device)

        return mass_matrix

    def get_jacobian(self) -> torch.Tensor:
        """
        Return the Jacobian for each link in shape (num_links, 6, num_joints).

        - num_links
        - 6 = [dPos/dq (3 rows), dRot/dq (3 rows)]
        - num_joints = total joints
        """
        # 1) Gather the current joint positions from PyBullet
        joint_positions = self.get_dof_positions()

        # 2) For each link, calculate the Jacobian
        jacobians = torch.zeros((1, self._num_links, 6, self._num_dofs), device=self.device)
        for link_idx in range(0, self._num_links):
            linJ, angJ = p.calculateJacobian(
                bodyUniqueId=self.articulation,
                linkIndex=link_idx - 1,
                localPosition=[0, 0, 0],
                objPositions=joint_positions[0].tolist(),
                objVelocities=torch.zeros_like(joint_positions)[0].tolist(),
                objAccelerations=torch.zeros_like(joint_positions)[0].tolist(),
                physicsClientId=self.client_id,
            )
            # linJ, angJ are (3, num_joints)
            linJ = torch.tensor(linJ, device=self.device)
            angJ = torch.tensor(angJ, device=self.device)
            jacobians[0, link_idx, :3, :] = linJ
            jacobians[0, link_idx, 3:, :] = angJ

        return jacobians

    def set_masses(
        self,
        masses: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        link_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        """
        Set the mass of each link in PyBullet at runtime using p.changeDynamics.
        NOTE: If you pass an invalid mass for the base (index -1), PyBullet might ignore it
              or throw an error.
        """
        # Transfer the data from the user-supplied tensor to our internal state
        # (in case you want to keep an internal copy).
        self.articulation_view_data.link_mass[0, link_indices] = masses.to(self.device)

        # Actually call p.changeDynamics to update the mass in Bullet
        if isinstance(link_indices, slice):
            effective_indices = range(self._num_links)[link_indices]
        elif isinstance(link_indices, torch.Tensor):
            effective_indices = link_indices.tolist()
        else:
            effective_indices = link_indices

        for i, link_idx in enumerate(effective_indices):
            mass_value = masses[i].item() if len(masses.shape) > 0 else masses.item()
            p.changeDynamics(
                bodyUniqueId=self.articulation, linkIndex=link_idx, mass=mass_value, physicsClientId=self.client_id
            )

    def set_root_velocities(
        self,
        root_velocities: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        pass

    def set_dof_limits(
        self,
        limits: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_limits[indices, joint_indices] = limits.to(self.device)

    def set_dof_positions(
        self,
        positions: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        with torch.inference_mode(mode=True):
            # Also store internally
            self.articulation_view_data.dof_positions[indices, joint_indices] = positions.to(self.device)

            if isinstance(joint_indices, slice):
                effective_indices = self._dof_indicies[joint_indices]
            elif isinstance(joint_indices, torch.Tensor):
                effective_indices = [self._dof_indicies[i] for i in joint_indices.tolist()]
            else:
                effective_indices = [self._dof_indicies[i] for i in joint_indices]
            # For each movable dof
            for idx, j_id in enumerate(effective_indices):
                p.resetJointState(
                    bodyUniqueId=self.articulation,
                    jointIndex=j_id,
                    targetValue=positions[0][idx].item(),
                    targetVelocity=0.0,
                    physicsClientId=self.client_id,
                )
            self.populate_link_transforms()

    def set_dof_position_targets(
        self,
        positions: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_position_target[indices, joint_indices] = positions.to(self.device)

    def set_dof_velocities(
        self,
        velocities: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_velocities[indices, joint_indices] = velocities.to(self.device)

        if isinstance(joint_indices, slice):
            effective_indices = self._dof_indicies[joint_indices]
        if isinstance(joint_indices, torch.Tensor):
            effective_indices = [self._dof_indicies[i] for i in joint_indices.tolist()]

        for idx, j_id in enumerate(effective_indices):
            # get current joint position so we don't overwrite it
            cur_state = p.getJointState(self.articulation, j_id, physicsClientId=self.client_id)
            cur_pos = cur_state[0]
            p.resetJointState(
                bodyUniqueId=self.articulation,
                jointIndex=j_id,
                targetValue=cur_pos,
                targetVelocity=velocities[0][idx].item(),
                physicsClientId=self.client_id,
            )
        self.populate_link_velocities()

    def set_dof_velocity_targets(
        self,
        velocities: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_velocity_target[indices, joint_indices] = velocities.to(self.device)

    def set_dof_torques(
        self,
        torques: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_torques[indices, joint_indices] = torques.to(self.device)

    def set_dof_states(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        efforts: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.set_dof_positions(positions, indices, joint_indices)
        self.set_dof_velocities(velocities, indices, joint_indices)
        self.set_dof_torques(efforts, indices, joint_indices)

    def set_dof_stiffnesses(
        self,
        stiffness: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_stiffness[indices, joint_indices] = stiffness.to(self.device)

    def set_dof_dampings(
        self,
        damping: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_damping[indices, joint_indices] = damping.to(self.device)

    def set_dof_armatures(
        self,
        armatures: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_armatures[indices, joint_indices] = armatures.to(self.device)

    def set_dof_frictions(
        self,
        frictions: torch.Tensor,
        indices: list[int] | torch.Tensor | slice = slice(None),
        joint_indices: list[int] | torch.Tensor | slice = slice(None),
    ):
        self.articulation_view_data.dof_frictions[indices, joint_indices] = frictions.to(self.device)

    def forward_kinematics(self, positions: torch.Tensor) -> torch.Tensor:
        self.current_joint_positions = self.articulation_view_data.dof_positions.clone()
        # 1) Set the joint positions in PyBullet
        self.set_dof_positions(positions)

        # 2) Get the link transforms
        link_transforms = self.get_link_transforms()

        # 3) Reset the joint positions to the original state
        self.set_dof_positions(self.current_joint_positions)

        return link_transforms

    def set_dof_targets(self, positions, velocities, torques):
        self.set_dof_position_targets(positions)
        self.set_dof_velocity_targets(velocities)
        self.set_dof_torques(torques)

    def populate_joint_state(self):
        dof_names = []
        dof_types = []
        link_names = []
        for j in range(self._num_dofs):
            info = p.getJointInfo(self.articulation, self._dof_indicies[j], physicsClientId=self.client_id)
            dof_index: int = info[0]
            dof_name: str = info[1].decode("utf-8")
            dof_type: int = info[2]
            damping: float = info[6]
            friction: float = info[7]
            lower_limit: float = info[8]
            upper_limit: float = info[9]
            max_forces: float = info[10]
            max_velocity: float = info[11]
            link_name = info[12].decode("utf-8")

            if dof_type is not p.JOINT_FIXED:
                self.articulation_view_data.dof_indices[:, j] = dof_index
                dof_names.append(dof_name)
                dof_types.append(dof_types_dict[dof_type])
                self.articulation_view_data.dof_damping[:, j] = damping
                self.articulation_view_data.dof_frictions[:, j] = friction
                self.articulation_view_data.dof_limits[:, j, 0] = lower_limit
                self.articulation_view_data.dof_limits[:, j, 1] = upper_limit
                self.articulation_view_data.dof_max_forces[:, j] = max_forces
                self.articulation_view_data.dof_max_velocity[:, j] = max_velocity

            link_names.append(link_name)

        self.articulation_view_data.dof_names = dof_names
        self.articulation_view_data.dof_types = dof_types

        base, _ = p.getBodyInfo(self.articulation, physicsClientId=self.client_id)
        self.articulation_view_data.link_names = [base.decode("utf-8")] + link_names

    def populate_link_transforms(self):
        link_world_pose = torch.zeros_like(self.articulation_view_data.link_transforms)
        # populate the base link
        base_pose = p.getBasePositionAndOrientation(self.articulation, physicsClientId=self.client_id)
        pos = base_pose[0]
        quat = base_pose[1]  # (x,y,z,w)
        self.articulation_view_data.link_transforms[:, 0, :3] = torch.tensor(pos, device=self.device)
        self.articulation_view_data.link_transforms[:, 0, 3:] = torch.tensor(quat, device=self.device)

        for link_idx in range(1, self._num_links):
            link_state = p.getLinkState(
                self.articulation,
                link_idx - 1,
                computeLinkVelocity=True,
                computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )
            link_world_position: list[float] = link_state[0]
            link_world_orientation: list[float] = link_state[1]  # (x,y,z,w)

            link_world_pose[:, link_idx, :3] = torch.tensor(link_world_position)
            link_world_pose[:, link_idx, 3:] = torch.tensor(link_world_orientation)

        self.articulation_view_data.link_transforms = link_world_pose.to(self.device)

    def populate_link_velocities(self):
        link_world_velocity = torch.zeros_like(self.articulation_view_data.link_velocities)

        lin_vel, ang_vel = p.getBaseVelocity(self.articulation, physicsClientId=self.client_id)
        link_world_velocity[:, 0, :3] = torch.tensor(lin_vel)
        link_world_velocity[:, 0, 3:] = torch.tensor(ang_vel)
        for link_idx in range(1, self._num_links):
            link_state = p.getLinkState(
                self.articulation,
                link_idx - 1,
                computeLinkVelocity=True,
                computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )
            world_link_linear_velocity: list[float] = link_state[6]
            world_link_angular_velocity: list[float] = link_state[7]
            link_world_velocity[:, link_idx, :3] = torch.tensor(world_link_linear_velocity)
            link_world_velocity[:, link_idx, 3:] = torch.tensor(world_link_angular_velocity)

        self.articulation_view_data.link_velocities = link_world_velocity.to(self.device)

    def populate_link_mass(self):
        dyn_info = p.getDynamicsInfo(self.articulation, -1, physicsClientId=self.client_id)
        self.articulation_view_data.link_mass[:, 0] = dyn_info[0]

        for link_idx in range(1, self._num_links):
            # 3) Get the mass of this link
            dyn_info = p.getDynamicsInfo(self.articulation, link_idx - 1, physicsClientId=self.client_id)
            self.articulation_view_data.link_mass[:, link_idx] = dyn_info[0]

    def populate_inertia(self):
        # Initialize output: (batch=1, num_links, 9)
        inertias = torch.zeros_like(self.articulation_view_data.link_inertia)
        coms = torch.zeros_like(self.articulation_view_data.link_coms)
        for link_idx in range(0, self._num_links):
            # Retrieve the dynamics info from PyBullet
            dyn_info = p.getDynamicsInfo(self.articulation, link_idx - 1, physicsClientId=self.client_id)

            local_inertia_diag = torch.tensor(dyn_info[2])  # shape [3]
            local_inertial_pos = torch.tensor(dyn_info[3])  # (x, y, z)
            local_inertia_quat = torch.tensor(dyn_info[4])  # shape [4], (x,y,z,w)

            # 1) Diagonal inertia in a 3×3 matrix
            Idiag = torch.diag(local_inertia_diag)

            # 2) Convert PyBullet quaternion (x,y,z,w) into a rotation matrix
            #    Adjust for any function that expects (w, x, y, z) vs. (x, y, z, w).
            #    For example:
            #      local_inertia_quat -> (x, y, z, w)
            #      convert_quat(...)   -> reorder to (w, x, y, z) if needed.
            R = math_utils.matrix_from_quat(
                math_utils.convert_quat(local_inertia_quat, to="wxyz")  # type: ignore
            )  # shape (3,3)

            # 3) Compute the full inertia in link frame: I = R * Idiag * R^T
            Ifull = R @ Idiag @ R.transpose(0, 1)  # shape (3,3)

            # 4) Flatten row-major into a length-9 vector
            inertias[0, link_idx, :] = Ifull.reshape(-1)
            coms[:, link_idx, :3] = local_inertial_pos
            coms[:, link_idx, 3:] = local_inertia_quat

        self.articulation_view_data.link_inertia[:] = inertias.to(self.device)
        self.articulation_view_data.link_coms[:] = coms.to(self.device)

    def __del__(self):
        p.disconnect()
