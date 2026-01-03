# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from abc import abstractmethod
from typing import TypedDict


class SharedDataSchema(TypedDict):
    is_running: bool
    close: bool
    link_names: list[str]
    dof_names: list[str]
    dof_types: list[str]
    pos: torch.Tensor
    vel: torch.Tensor
    torque: torch.Tensor
    pos_target: torch.Tensor
    vel_target: torch.Tensor
    eff_target: torch.Tensor
    link_transforms: torch.Tensor
    link_velocities: torch.Tensor
    link_mass: torch.Tensor
    link_inertia: torch.Tensor
    link_coms: torch.Tensor
    mass_matrix: torch.Tensor
    dof_stiffness: torch.Tensor
    dof_armatures: torch.Tensor
    dof_frictions: torch.Tensor
    dof_damping: torch.Tensor
    dof_limits: torch.Tensor
    dof_max_forces: torch.Tensor
    dof_max_velocity: torch.Tensor
    jacobians: torch.Tensor


class ArticulationView:
    """
    An abstract interface for querying and setting the state of an articulated mechanism.
    This interface aims to provide a unified set of methods for interacting with:
        - Physics engines (e.g., PhysX, MuJoCo, Bullet, etc.)
        - Real robot hardware
        - Simulated or software-only kinematics/dynamics backends

    The subclass implementing this interface must handle how these queries and commands
    are reflected in the underlying system.
    """

    def __init__(self):
        """
        Initializes the articulation view.

        Subclasses must implement:
            - Initialization and any required backend setup.
            - Necessary internal references for subsequent get/set methods.

        Raises:
            NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def play(self):
        """
        Play the articulation(s) in the view.
        """
        raise NotImplementedError

    @abstractmethod
    def pause(self):
        """
        Pause the articulation(s) in the view.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """
        Pause the articulation(s) in the view.
        """
        raise NotImplementedError

    @property
    def count(self) -> int:
        """
        Number of articulation instances being managed by this view.

        E.g., if your environment has N separate copies of the same robot,
        this property would return N.

        Returns:
            int: The total number of articulation instances.
        """
        raise NotImplementedError

    @property
    def fixed_base(self) -> bool:
        """
        Indicates whether the articulation(s) in this view has a fixed base.

        A fixed-base articulation does not move freely in space
        (e.g., a robotic arm bolted to a table),
        whereas a floating-base articulation can move freely.

        Returns:
            bool: True if the articulation is fixed-base, False otherwise.
        """
        raise NotImplementedError

    @property
    def dof_count(self) -> int:
        """
        Number of degrees of freedom (DOFs) for the articulation(s) in this view.

        Returns:
            int: Count of all DOFs for the managed articulation(s).
        """
        raise NotImplementedError

    @property
    def max_fixed_tendons(self) -> int:
        """
        Maximum number of 'fixed tendons' (sometimes known as cables or passively constrained joints).

        Returns:
            int: The number of fixed tendon connections in the articulation(s).
        """
        raise NotImplementedError

    @property
    def num_bodies(self) -> int:
        """
        Number of rigid bodies (links) in the articulation(s).

        Returns:
            int: Total number of rigid bodies in the articulation(s).
        """
        raise NotImplementedError

    @property
    def joint_names(self) -> list[str]:
        """
        Ordered list of joint names in the articulation(s).

        Returns:
            list[str]: Names of the joints in order of their DOF indices.
        """
        raise NotImplementedError

    @property
    def fixed_tendon_names(self) -> list[str]:
        """
        Ordered list of names for the fixed tendons (cables) in the articulation(s).

        Returns:
            list[str]: Names of the fixed tendons, if any.
        """
        raise NotImplementedError

    @property
    def body_names(self) -> list[str]:
        """
        Ordered list of body (link) names in the articulation(s).

        Returns:
            list[str]: Names of the rigid bodies in order of their indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_root_transforms(self) -> torch.Tensor:
        """
        Get the root poses (position + orientation) of each articulation instance.

        This typically refers to the base link or root transform
        of a floating/fixed-base articulation in world coordinates.

        Returns:
            torch.Tensor: A tensor of shape (count, 7),
                          each row containing [px, py, pz, qw, qx, qy, qz].
        """
        raise NotImplementedError

    @abstractmethod
    def get_root_velocities(self) -> torch.Tensor:
        """
        Get the linear and angular velocities of the root link(s) of each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, 6),
                          each row containing [vx, vy, vz, wx, wy, wz].
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_accelerations(self) -> torch.Tensor:
        """
        Get the link accelerations for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 6),
                          containing [ax, ay, az, alpha_x, alpha_y, alpha_z] for each link.
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_transforms(self) -> torch.Tensor:
        """
        Get the transforms (position + orientation) of each link for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 7),
                          for [px, py, pz, qx, qy, qz, qw] per link.
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_velocities(self) -> torch.Tensor:
        """
        Get the linear and angular velocities of each link in the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 6),
                          containing [vx, vy, vz, wx, wy, wz] per link.
        """
        raise NotImplementedError

    @abstractmethod
    def get_coms(self) -> torch.Tensor:
        """
        Get the center-of-mass (COM) positions of each link or the entire articulation.

        Depending on the implementation, this can mean:
          - Per-link COM (shape = (count, num_bodies, 3))
          - Single COM for the entire system (shape = (count, 3))

        Returns:
            torch.Tensor: COM positions in the world or local coordinate frame
                          depending on the backend's convention.
        """
        raise NotImplementedError

    @abstractmethod
    def get_masses(self) -> torch.Tensor:
        """
        Get the masses for each link in the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies),
                          containing the mass of each link.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inertias(self) -> torch.Tensor:
        """
        Get the inertial tensors (often expressed in link-local frames) for each link.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 3, 3),
                          containing the inertia matrix for each link.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_positions(self) -> torch.Tensor:
        """
        Get the joint positions for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count)
                          with joint angles or positions.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_velocities(self) -> torch.Tensor:
        """
        Get the joint velocities for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          with joint velocity values.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_max_velocities(self) -> torch.Tensor:
        """
        Get the maximum velocity limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing velocity limits per joint.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_max_forces(self) -> torch.Tensor:
        """
        Get the maximum force (torque) limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          with torque/force limits per joint.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_stiffnesses(self) -> torch.Tensor:
        """
        Get the joint stiffness values for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing the stiffness for each DOF.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_dampings(self) -> torch.Tensor:
        """
        Get the joint damping values for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing the damping for each DOF.
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_armatures(self) -> torch.Tensor:
        """
        Get the armature values for each joint in each articulation instance.

        The 'armature' is sometimes used to represent an inertia-like term
        used in certain simulation backends or real hardware compensations.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count).
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_friction_coefficients(self) -> torch.Tensor:
        """
        Get the friction coefficients for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count).
        """
        raise NotImplementedError

    @abstractmethod
    def get_dof_limits(self) -> torch.Tensor:
        """
        Get the joint position limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count, 2),
                          where the last dimension stores [lower_limit, upper_limit].
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_stiffnesses(self) -> torch.Tensor:
        """
        Get the stiffness values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the stiffness value for each tendon.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_dampings(self) -> torch.Tensor:
        """
        Get the damping values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the damping value for each tendon.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_limit_stiffnesses(self) -> torch.Tensor:
        """
        Get the limit stiffness values for each fixed tendon.

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the limit stiffness for each tendon.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_limits(self) -> torch.Tensor:
        """
        Get the limit range for each fixed tendon, which might represent
        min/max constraints on the tendon length or tension.

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons, 2),
                          containing [lower_limit, upper_limit] for each tendon.
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_rest_lengths(self) -> torch.Tensor:
        """
        Get the rest lengths for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons).
        """
        raise NotImplementedError

    @abstractmethod
    def get_fixed_tendon_offsets(self) -> torch.Tensor:
        """
        Get the offset values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons).
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_actuation_forces(self, forces: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the actuation forces (torques) for the specified articulation instances.

        Args:
            forces (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                   specifying the commanded forces/torques.
            indices (torch.Tensor): A tensor of indices specifying which
                                    articulation instances to apply these forces.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_position_targets(self, positions: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the position targets for the specified articulation instances,
        if the underlying controller or simulation uses position-based control.

        Args:
            positions (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      specifying desired joint positions.
            indices (torch.Tensor): Indices of articulation instances to apply these targets.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_positions(self, positions: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Hard-set the joint positions for the specified articulation instances.
        Usually used for resetting or overriding joint states directly.

        Args:
            positions (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      specifying the new positions.
            indices (torch.Tensor): Indices of articulation instances to set positions for.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_velocity_targets(self, velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the velocity targets for the specified articulation instances,
        if the underlying controller or simulation uses velocity-based control.

        Args:
            velocities (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                       specifying desired joint velocities.
            indices (torch.Tensor): Indices of articulation instances to apply these targets.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_velocities(self, velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Hard-set the joint velocities for the specified articulation instances.
        Usually used for resetting or overriding joint states directly.

        Args:
            velocities (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                       specifying new joint velocities.
            indices (torch.Tensor): Indices of articulation instances to set velocities for.
        """
        raise NotImplementedError

    @abstractmethod
    def set_root_transforms(self, root_poses_xyzw: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the root transforms (position + orientation) for each articulation instance.
        Orientation is expected in (x, y, z, w) format.

        Args:
            root_poses_xyzw (torch.Tensor): A tensor of shape (len(indices), 7),
                                            containing [px, py, pz, qx, qy, qz, qw].
            indices (torch.Tensor): Indices of articulation instances to set transforms for.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_stiffnesses(self, stiffness: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint stiffness values for the specified articulation instances.

        Args:
            stiffness (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      with new stiffness values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_dampings(self, damping: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint damping values for the specified articulation instances.

        Args:
            damping (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                    with new damping values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_armatures(self, armatures: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint armature values for the specified articulation instances.

        Args:
            armatures (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                      specifying new armature values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_friction_coefficients(self, friction_coefficients: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the friction coefficients for each joint of the specified articulation instances.

        Args:
            friction_coefficients (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                                  specifying new friction values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_max_velocities(self, max_velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the maximum allowed velocities for each joint in the specified articulation instances.

        Args:
            max_velocities (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                           specifying new velocity limits.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_max_forces(self, max_forces: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the maximum allowed forces (torques) for each joint in the specified articulation instances.

        Args:
            max_forces (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                       specifying new force/torque limits.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_dof_limits(self, limits: torch.Tensor, indices: torch.Tensor):
        """
        Set new position limits (lower/upper) for each joint in the specified articulation instances.

        Args:
            limits (torch.Tensor): A tensor of shape (len(indices), dof_count, 2),
                                   specifying [lower_limit, upper_limit] for each joint.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def set_fixed_tendon_properties(
        self,
        fixed_tendon_stiffness: torch.Tensor,
        fixed_tendon_damping: torch.Tensor,
        fixed_tendon_limit_stiffness: torch.Tensor,
        fixed_tendon_limit: torch.Tensor,
        fixed_tendon_rest_length: torch.Tensor,
        fixed_tendon_offset: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Set the properties of fixed tendons (cables) for the specified articulation instances.

        Args:
            fixed_tendon_stiffness (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons),
                                                   specifying the stiffness for each tendon.
            fixed_tendon_damping (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons),
                                                 specifying the damping for each tendon.
            fixed_tendon_limit_stiffness (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons),
                                                        specifying the limit stiffness for each tendon.
            fixed_tendon_limit (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons, 2),
                                               specifying [lower_limit, upper_limit] for each tendon.
            fixed_tendon_rest_length (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons),
                                                     specifying the rest length for each tendon.
            fixed_tendon_offset (torch.Tensor): A tensor of shape (len(indices), max_fixed_tendons),
                                                specifying the offset for each tendon.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_forces_and_torques_at_position(
        self,
        force_data: torch.Tensor,
        torque_data: torch.Tensor,
        position_data: torch.Tensor,
        indices: torch.Tensor,
        is_global: bool,
    ) -> None:
        """
        Apply forces and torques to bodies at specified positions (in local or global coordinates).

        Args:
            force_data (torch.Tensor): A tensor of shape (len(indices), nbodies, 3),
                                       specifying the force to be applied.
            torque_data (torch.Tensor): A tensor of shape (len(indices), nbodies, 3),
                                        specifying the torque to be applied.
            position_data (torch.Tensor): A tensor of shape (len(indices), nbodies, 3),
                                          specifying the point of application
                                          (in local or world coordinates).
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
            is_global (bool):
                If True, forces/torques are expressed in world/global coordinates.
                Otherwise, they are in local link coordinates.
        """
        raise NotImplementedError

    @abstractmethod
    def _initialize_default_data(self, data):
        """
        Initialize the default data for the articulation view.
        """
        raise NotImplementedError
