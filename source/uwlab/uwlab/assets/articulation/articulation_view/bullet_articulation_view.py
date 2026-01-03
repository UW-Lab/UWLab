# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import queue
import threading
import time
import torch
import torch.multiprocessing as mp
from typing import TYPE_CHECKING

from . import ArticulationView
from .utils.articulation_kinematics import BulletArticulationKinematics

if TYPE_CHECKING:
    from ..articulation_data import ArticulationData
    from ..articulation_drive import ArticulationDrive
    from . import BulletArticulationViewCfg, SharedDataSchema


def init_shared_data(count: int, ndof: int, nbodies: int) -> SharedDataSchema:
    data: SharedDataSchema = {
        "is_running": False,
        "close": False,
        "link_names": [],
        "dof_names": [],
        "dof_types": [],
        "pos": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "vel": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "torque": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "pos_target": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "vel_target": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "eff_target": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "link_transforms": torch.zeros(count, nbodies, 7, device="cpu").share_memory_(),
        "link_velocities": torch.zeros(count, nbodies, 6, device="cpu").share_memory_(),
        "link_mass": torch.zeros(count, nbodies, device="cpu").share_memory_(),
        "link_inertia": torch.zeros(count, nbodies, 9, device="cpu").share_memory_(),
        "link_coms": torch.zeros(count, nbodies, 7, device="cpu").share_memory_(),
        "mass_matrix": torch.zeros(count, ndof, ndof, device="cpu").share_memory_(),
        "dof_stiffness": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "dof_armatures": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "dof_frictions": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "dof_damping": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "dof_limits": torch.zeros(count, ndof, 2, device="cpu").share_memory_(),
        "dof_max_forces": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "dof_max_velocity": torch.zeros(count, ndof, device="cpu").share_memory_(),
        "jacobians": torch.zeros(count, nbodies, 6, ndof, device="cpu").share_memory_(),
    }
    return data


class BulletArticulationView(ArticulationView):
    """
    LeapHand Implementation of Articulation View
    """

    def __init__(
        self,
        device: str,
        cfg: BulletArticulationViewCfg,
    ):
        """
        Initializes the articulation view.
        """
        # setup hand
        self.device = device
        self._dt = cfg.dt
        self._urdf = cfg.urdf
        self._debug_visualize = cfg.debug_visualize
        self._isaac_joint_names = cfg.isaac_joint_names

        self._dummy_mode = cfg.dummy_mode

        if cfg.use_multiprocessing:
            self.manager = mp.Manager()
            self.init_event = mp.Event()
            self.cmd_queue = mp.Queue()
            self.ack_queue = mp.Queue()
            # Create a shared dictionary to communicate with the child process
            self.shared_data: SharedDataSchema = self.manager.dict()  # type: ignore
            self._proc = mp.Process(target=self._run, args=(self.init_event,))
        else:
            # Threading shares memory within the same process, so no need for mp.Manager
            self.manager = None
            self.init_event = threading.Event()
            self.cmd_queue = queue.Queue()
            self.ack_queue = queue.Queue()
            self.shared_data: SharedDataSchema = {}  # type: ignore
            self._proc = threading.Thread(target=self._run, daemon=True, args=(self.init_event,))

        # store data that will never be retrieved from kinematic articulation
        self.drive_cfg = cfg.drive_cfg

        # Spawn the child process
        self._proc.start()
        self.init_event.wait()

    def _run(self, init_event: threading.Event):
        # Loop until closed
        _kinematic = BulletArticulationKinematics(self._urdf, True, self._debug_visualize, dt=self._dt, device="cpu")
        self.shared_data.update(init_shared_data(self.count, _kinematic.num_dof, _kinematic.num_links))
        self._populate_shared_data(_kinematic)
        _drive: ArticulationDrive = None  # type: ignore
        if not self._dummy_mode:
            _drive = self.drive_cfg.class_type(cfg=self.drive_cfg)
            _drive_joint_names = _drive.ordered_joint_names
            self._isaac_to_real_idx = [
                self._isaac_joint_names.index(name) for name in _drive_joint_names if name in self._isaac_joint_names
            ]
            self._real_to_isaac_idx = [
                _drive_joint_names.index(name) for name in self._isaac_joint_names if name in _drive_joint_names
            ]

            self._bullet_to_real_idx = [
                self.shared_data["dof_names"].index(name)
                for name in _drive_joint_names
                if name in self.shared_data["dof_names"]
            ]
            self._real_to_bullet_idx = [
                _drive_joint_names.index(name) for name in self.shared_data["dof_names"] if name in _drive_joint_names
            ]

        self._isaac_to_bullet_idx = [
            self._isaac_joint_names.index(name)
            for name in self.shared_data["dof_names"]
            if name in self._isaac_joint_names
        ]
        self._bullet_to_isaac_idx = [
            self.shared_data["dof_names"].index(name)
            for name in self._isaac_joint_names
            if name in self.shared_data["dof_names"]
        ]

        init_event.set()

        next_poll_time = time.time() + self._dt
        while True:
            if self.shared_data["close"]:
                break

            self._process_blocking_commands(_kinematic, _drive)

            if self.shared_data["is_running"]:
                now = time.time()
                if now >= next_poll_time:
                    next_poll_time = now + self._dt
                    if self._dummy_mode:
                        self._read_write_dummy_states()
                        _kinematic.set_dof_states(
                            self.shared_data["pos"][:, self._isaac_to_bullet_idx],
                            self.shared_data["vel"][:, self._isaac_to_bullet_idx],
                            self.shared_data["torque"][:, self._isaac_to_bullet_idx],
                        )

                        self.shared_data["link_coms"][:] = _kinematic.get_link_coms()
                        self.shared_data["link_transforms"][:] = _kinematic.get_link_transforms()
                        self.shared_data["link_velocities"][:] = _kinematic.get_link_velocities()
                    else:
                        # 1) read pos/vel from hardware
                        pos, vel, eff = _drive.read_dof_states()
                        self.shared_data["pos"][:, self._real_to_isaac_idx] = pos[:, self._real_to_isaac_idx]
                        self.shared_data["vel"][:, self._real_to_isaac_idx] = vel[:, self._real_to_isaac_idx]
                        self.shared_data["torque"][:, self._real_to_isaac_idx] = eff[:, self._real_to_isaac_idx]

                        _kinematic.set_dof_states(
                            self.shared_data["pos"][:, self._isaac_to_bullet_idx],
                            self.shared_data["vel"][:, self._isaac_to_bullet_idx],
                            self.shared_data["torque"][:, self._isaac_to_bullet_idx],
                        )

                        self.shared_data["link_coms"][:] = _kinematic.get_link_coms()
                        self.shared_data["link_transforms"][:] = _kinematic.get_link_transforms()
                        self.shared_data["link_velocities"][:] = _kinematic.get_link_velocities()

                        # 2) write target to hardware and kinematic
                        _kinematic.set_dof_targets(
                            self.shared_data["pos_target"][:, self._isaac_to_bullet_idx],
                            self.shared_data["vel_target"][:, self._isaac_to_bullet_idx],
                            self.shared_data["eff_target"][:, self._isaac_to_bullet_idx],
                        )
                        _drive.write_dof_targets(
                            pos_target=self.shared_data["pos_target"][:, self._isaac_to_real_idx],
                            vel_target=self.shared_data["vel_target"][:, self._isaac_to_real_idx],
                            eff_target=self.shared_data["eff_target"][:, self._isaac_to_real_idx],
                        )
                        if self._debug_visualize:
                            _kinematic.render()
                    # Sleep till next poll time
                    time.sleep(max(next_poll_time - time.time(), 0))
                else:
                    # now is less than next poll time, sleep till next poll time
                    time.sleep(next_poll_time - now)
            else:
                # If not running, keep sleeping
                time.sleep(self._dt)

        # Done, disconnect hardware
        _drive.close()
        _kinematic.close()
        print("DynamixelWorker: Child process stopped")

    def _process_blocking_commands(self, _kinematic: BulletArticulationKinematics, _drive: ArticulationDrive):
        if not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            match cmd:
                case "set_dof_stiffnesses":
                    _kinematic.set_dof_stiffnesses(self.shared_data["dof_stiffness"][:, self._isaac_to_bullet_idx])
                    if not self._dummy_mode:
                        _drive.set_dof_stiffnesses(self.shared_data["dof_stiffness"][:, self._bullet_to_real_idx])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_stiffnesses"})
                case "set_dof_armatures":
                    _kinematic.set_dof_armatures(self.shared_data["dof_armatures"][:, self._isaac_to_bullet_idx])
                    if not self._dummy_mode:
                        _drive.set_dof_armatures(self.shared_data["dof_armatures"][:, self._bullet_to_real_idx])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_armatures"})
                case "set_dof_frictions":
                    _kinematic.set_dof_frictions(self.shared_data["dof_frictions"][:, self._isaac_to_bullet_idx])
                    if not self._dummy_mode:
                        _drive.set_dof_frictions(self.shared_data["dof_frictions"][:, self._bullet_to_real_idx])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_frictions"})
                case "set_dof_dampings":
                    _kinematic.set_dof_dampings(self.shared_data["dof_damping"][:, self._isaac_to_bullet_idx])
                    if not self._dummy_mode:
                        _drive.set_dof_dampings(self.shared_data["dof_damping"][:, self._bullet_to_real_idx])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_dampings"})
                case "set_dof_limits":
                    _kinematic.set_dof_limits(self.shared_data["dof_limits"][:, self._isaac_to_bullet_idx])
                    if not self._dummy_mode:
                        _drive.set_dof_limits(self.shared_data["dof_limits"][:, self._bullet_to_real_idx])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_limits"})

    def _populate_shared_data(self, _kinematic: BulletArticulationKinematics):
        self.shared_data["link_names"] = _kinematic.link_names
        self.shared_data["dof_names"] = _kinematic.joint_names

        self.shared_data["pos"][:] = _kinematic.get_dof_positions(clone=False)
        self.shared_data["vel"][:] = _kinematic.get_dof_velocities(clone=False)
        self.shared_data["torque"][:] = _kinematic.get_dof_torques(clone=False)
        self.shared_data["pos_target"][:] = _kinematic.get_dof_position_targets(clone=False)
        self.shared_data["vel_target"][:] = _kinematic.get_dof_velocity_targets(clone=False)
        # self.shared_data["eff_target"][:]

        self.shared_data["link_transforms"][:] = _kinematic.get_link_transforms(clone=False)
        self.shared_data["link_velocities"][:] = _kinematic.get_link_velocities(clone=False)
        self.shared_data["link_mass"][:] = _kinematic.get_link_masses(clone=False)
        self.shared_data["link_inertia"][:] = _kinematic.get_link_inertias(clone=False)
        self.shared_data["link_coms"][:] = _kinematic.get_link_coms(clone=False)
        self.shared_data["mass_matrix"][:] = _kinematic.get_mass_matrix()
        self.shared_data["dof_stiffness"][:] = _kinematic.get_dof_stiffnesses(clone=False)
        self.shared_data["dof_armatures"][:] = _kinematic.get_dof_armatures(clone=False)
        self.shared_data["dof_frictions"][:] = _kinematic.get_dof_frictions(clone=False)
        self.shared_data["dof_damping"][:] = _kinematic.get_dof_dampings(clone=False)
        self.shared_data["dof_limits"][:] = _kinematic.get_dof_limits(clone=False)
        self.shared_data["dof_max_forces"][:] = _kinematic.get_dof_max_forces(clone=False)
        self.shared_data["dof_max_velocity"][:] = _kinematic.get_dof_max_velocities(clone=False)
        self.shared_data["jacobians"][:] = _kinematic.get_jacobian()

    def _read_write_dummy_states(self):
        self.shared_data["pos"] = self.shared_data["pos_target"]
        self.shared_data["vel"] = self.shared_data["vel_target"]
        self.shared_data["torque"] = self.shared_data["eff_target"]

    # Public API
    # -------------------------------------------------------------------------
    # Rules:
    # Share the Data, not Object!
    # Below method should not be called in the child process, namely self._run,
    # Implementation of below methods should not directly access fields used for child process, e.g:
    # _kinematic
    # _drive

    def play(self):
        self.shared_data["is_running"] = True

    def pause(self):
        self.shared_data["is_running"] = False

    def close(self):
        self.shared_data["is_running"] = False
        self.shared_data["close"] = True
        if self._proc is not None:
            self._proc.join()
            self._proc = None

        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    @property
    def count(self) -> int:
        """
        Number of articulation instances being managed by this view.

        E.g., if your environment has N separate copies of the same robot,
        this property would return N.

        Returns:
            int: The total number of articulation instances.
        """
        return 1

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
        return True

    @property
    def dof_count(self) -> int:
        """
        Number of degrees of freedom (DOFs) for the articulation(s) in this view.

        Returns:
            int: Count of all DOFs for the managed articulation(s).
        """
        return self.shared_data["dof_names"].__len__()

    @property
    def max_fixed_tendons(self) -> int:
        """
        Maximum number of 'fixed tendons' (sometimes known as cables or passively constrained joints).

        Returns:
            int: The number of fixed tendon connections in the articulation(s).
        """
        return 0

    @property
    def num_bodies(self) -> int:
        """
        Number of rigid bodies (links) in the articulation(s).

        Returns:
            int: Total number of rigid bodies in the articulation(s).
        """
        return self.shared_data["link_names"].__len__()

    @property
    def joint_names(self) -> list[str]:
        """
        Ordered list of joint names in the articulation(s).

        Returns:
            list[str]: Names of the joints in order of their DOF indices.
        """
        return self.shared_data["dof_names"].copy()

    @property
    def fixed_tendon_names(self) -> list[str]:
        """
        Ordered list of names for the fixed tendons (cables) in the articulation(s).

        Returns:
            list[str]: Names of the fixed tendons, if any.
        """
        return []

    @property
    def body_names(self) -> list[str]:
        """
        Ordered list of body (link) names in the articulation(s).

        Returns:
            list[str]: Names of the rigid bodies in order of their indices.
        """
        return self.shared_data["link_names"].copy()

    def get_root_transforms(self) -> torch.Tensor:
        """
        Get the root poses (position + orientation) of each articulation instance.

        This typically refers to the base link or root transform
        of a floating/fixed-base articulation in world coordinates.

        Returns:
            torch.Tensor: A tensor of shape (count, 7),
                          each row containing [px, py, pz, qw, qx, qy, qz].
        """
        return self.shared_data["link_transforms"][:, 0, :].clone().to(self.device)

    def get_root_velocities(self) -> torch.Tensor:
        """
        Get the linear and angular velocities of the root link(s) of each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, 6),
                          each row containing [vx, vy, vz, wx, wy, wz].
        """
        return self.shared_data["link_velocities"][:, 0, :].clone().to(self.device)

    def get_link_accelerations(self) -> torch.Tensor:
        """
        Get the link accelerations for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 6),
                          containing [ax, ay, az, alpha_x, alpha_y, alpha_z] for each link.
        """
        print("getter function: get_link_accelerations, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.num_bodies, 6), dtype=torch.float32, device=self.device)

    def get_link_transforms(self) -> torch.Tensor:
        """
        Get the transforms (position + orientation) of each link for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 7),
                          for [px, py, pz, qw, qx, qy, qz] per link.
        """
        return self.shared_data["link_transforms"].clone().to(self.device)

    def get_link_velocities(self) -> torch.Tensor:
        """
        Get the linear and angular velocities of each link in the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 6),
                          containing [vx, vy, vz, wx, wy, wz] per link.
        """
        return self.shared_data["link_velocities"].clone().to(self.device)

    def get_coms(self) -> torch.Tensor:
        """
        Get the center-of-mass (COM) positions of each link or the entire articulation.

        Depending on the implementation, this can mean:
          - Per-link COM (shape = (count, num_bodies, 7))
          - Single COM for the entire system (shape = (count, 7))

        Returns:
            torch.Tensor: COM positions in the world or local coordinate frame
                          depending on the backend's convention.
        """
        return self.shared_data["link_coms"].clone().to(self.device)

    def get_masses(self) -> torch.Tensor:
        """
        Get the masses for each link in the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies),
                          containing the mass of each link.
        """
        link_mass = self.shared_data["link_mass"].clone().to(self.device)
        return link_mass

    def get_inertias(self) -> torch.Tensor:
        """
        Get the inertial tensors (often expressed in link-local frames) for each link.

        Returns:
            torch.Tensor: A tensor of shape (count, num_bodies, 9),
                          containing the inertia matrix for each link.
        """
        return self.shared_data["link_inertia"].to(self.device)

    def get_dof_positions(self) -> torch.Tensor:
        """
        Get the joint positions for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count)
                          with joint angles or positions.
        """
        return self.shared_data["pos"].clone().to(self.device)

    def get_dof_velocities(self) -> torch.Tensor:
        """
        Get the joint velocities for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          with joint velocity values.
        """
        return self.shared_data["vel"].clone().to(self.device)

    def get_dof_torques(self) -> torch.Tensor:
        """
        Get the joint torques for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          with joint velocity values.
        """
        return self.shared_data["torque"].clone().to(self.device)

    def get_dof_max_velocities(self) -> torch.Tensor:
        """
        Get the maximum velocity limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing velocity limits per joint.
        """
        return self.shared_data["dof_max_velocity"].clone().to(self.device)

    def get_dof_max_forces(self) -> torch.Tensor:
        """
        Get the maximum force (torque) limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          with torque/force limits per joint.
        """
        # max force expects cpu tensor
        return self.shared_data["dof_max_forces"].clone().to("cpu")

    def get_dof_stiffnesses(self) -> torch.Tensor:
        """
        Get the joint stiffness values for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing the stiffness for each DOF.
        """
        return self.shared_data["dof_stiffness"].clone().to(self.device)

    def get_dof_dampings(self) -> torch.Tensor:
        """
        Get the joint damping values for each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count),
                          containing the damping for each DOF.
        """
        return self.shared_data["dof_damping"].clone().to(self.device)

    def get_dof_armatures(self) -> torch.Tensor:
        """
        Get the armature values for each joint in each articulation instance.

        The 'armature' is sometimes used to represent an inertia-like term
        used in certain simulation backends or real hardware compensations.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count).
        """
        return self.shared_data["dof_armatures"].clone().to(self.device)

    def get_dof_friction_coefficients(self) -> torch.Tensor:
        """
        Get the friction coefficients for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count).
        """
        return self.shared_data["dof_frictions"].clone().to(self.device)

    def get_dof_limits(self) -> torch.Tensor:
        """
        Get the joint position limits for each joint in each articulation instance.

        Returns:
            torch.Tensor: A tensor of shape (count, dof_count, 2),
                          where the last dimension stores [lower_limit, upper_limit].
        """
        return self.shared_data["dof_limits"].clone().to(self.device)

    def get_fixed_tendon_stiffnesses(self) -> torch.Tensor:
        """
        Get the stiffness values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the stiffness value for each tendon.
        """
        print("getter function: get_fixed_tendon_stiffnesses, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def get_fixed_tendon_dampings(self) -> torch.Tensor:
        """
        Get the damping values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the damping value for each tendon.
        """
        print("getter function: get_fixed_tendon_dampings, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def get_fixed_tendon_limit_stiffnesses(self) -> torch.Tensor:
        """
        Get the limit stiffness values for each fixed tendon.

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons),
                          with the limit stiffness for each tendon.
        """
        print(
            "getter function: get_fixed_tendon_limit_stiffnesses, result are place holder values, do not rely on them"
        )
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def get_fixed_tendon_limits(self) -> torch.Tensor:
        """
        Get the limit range for each fixed tendon, which might represent
        min/max constraints on the tendon length or tension.

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons, 2),
                          containing [lower_limit, upper_limit] for each tendon.
        """
        print("getter function: get_fixed_tendon_limits, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def get_fixed_tendon_rest_lengths(self) -> torch.Tensor:
        """
        Get the rest lengths for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons).
        """
        print("getter function: get_fixed_tendon_rest_lengths, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def get_fixed_tendon_offsets(self) -> torch.Tensor:
        """
        Get the offset values for each fixed tendon across the articulation(s).

        Returns:
            torch.Tensor: A tensor of shape (count, max_fixed_tendons).
        """
        print("getter function: get_fixed_tendon_offsets, result are place holder values, do not rely on them")
        return torch.zeros((self.count, self.dof_count), dtype=torch.float32, device=self.device)

    def set_dof_actuation_forces(self, forces: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the actuation forces (torques) for the specified articulation instances.

        Args:
            forces (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                   specifying the commanded forces/torques.
            indices (torch.Tensor): A tensor of indices specifying which
                                    articulation instances to apply these forces.
        """
        if torch.any(forces != 0):
            print(
                "calling placeholder function: set_dof_actuation_forces with non-zero values, but function is not"
                " implemented"
            )

    def set_dof_position_targets(self, positions: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the position targets for the specified articulation instances,
        if the underlying controller or simulation uses position-based control.

        Args:
            positions (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      specifying desired joint positions.
            indices (torch.Tensor): Indices of articulation instances to apply these targets.
        """
        self.shared_data["pos_target"][:] = positions.cpu()

    def set_dof_positions(self, positions: torch.Tensor, indices: torch.Tensor, threshold: float = 1e-2) -> None:
        """
        Hard-set the joint positions for the specified articulation instances.
        Usually used for resetting or overriding joint states directly.

        Args:
            positions (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      specifying the new positions.
            indices (torch.Tensor): Indices of articulation instances to set positions for.
        """
        count = 0
        while torch.sum(self.get_dof_positions() - positions).abs() > threshold:
            self.set_dof_position_targets(positions, indices)
            count += 1
            if count > 50:
                break

    def set_dof_velocity_targets(self, velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the velocity targets for the specified articulation instances,
        if the underlying controller or simulation uses velocity-based control.

        Args:
            velocities (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                       specifying desired joint velocities.
            indices (torch.Tensor): Indices of articulation instances to apply these targets.
        """
        if torch.any(velocities != 0):
            print(
                "calling placeholder function: set_dof_velocity_targets with non-zero values, but function is not"
                " implemented"
            )
        # the interface is correct but driver doesn't support velocity control yet
        self.shared_data["vel_target"][:] = velocities.cpu()

    def set_dof_velocities(self, velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Hard-set the joint velocities for the specified articulation instances.
        Usually used for resetting or overriding joint states directly.

        Args:
            velocities (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                       specifying new joint velocities.
            indices (torch.Tensor): Indices of articulation instances to set velocities for.
        """
        if torch.any(velocities != 0):
            print(
                "calling placeholder function: set_dof_velocities with non-zero values, but function is not implemented"
            )
        pass

    def set_root_transforms(self, root_poses_xyzw: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the root transforms (position + orientation) for each articulation instance.
        Orientation is expected in (x, y, z, w) format.

        Args:
            root_poses_xyzw (torch.Tensor): A tensor of shape (len(indices), 7),
                                            containing [px, py, pz, qx, qy, qz, qw].
            indices (torch.Tensor): Indices of articulation instances to set transforms for.
        """
        if torch.any(root_poses_xyzw != 0):
            print(
                "calling placeholder function: set_root_transforms with non-zero values, but function is not"
                " implemented"
            )
        pass

    def set_root_velocities(self, root_velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the root velocities (linvel + angvel) for each articulation instance.

        Args:
            root_velocities (torch.Tensor): A tensor of shape (len(indices), 6),
                                            containing [x, y, z, rx, ry, rz].
            indices (torch.Tensor): Indices of articulation instances to set transforms for.
        """
        if torch.any(root_velocities != 0):
            print(
                "calling placeholder function: set_root_transforms with non-zero values, but function is not"
                " implemented"
            )
        pass

    def set_dof_stiffnesses(self, stiffness: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint stiffness values for the specified articulation instances.

        Args:
            stiffness (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                      with new stiffness values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        self.shared_data["dof_stiffness"][:] = stiffness.cpu()
        self.cmd_queue.put("set_dof_stiffnesses")
        ack = self.ack_queue.get()
        if ack != "OK":
            print(f"Warning: Child returned ack={ack}")

    def set_dof_dampings(self, damping: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint damping values for the specified articulation instances.

        Args:
            damping (torch.Tensor): A tensor of shape (len(indices), dof_count)
                                    with new damping values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        self.shared_data["dof_damping"][:] = damping.cpu()
        self.cmd_queue.put("set_dof_dampings")
        ack = self.ack_queue.get()
        if ack != "OK":
            print(f"Warning: Child returned ack={ack}")

    def set_dof_armatures(self, armatures: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the joint armature values for the specified articulation instances.

        Args:
            armatures (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                      specifying new armature values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        self.shared_data["dof_armatures"][:] = armatures.cpu()
        self.cmd_queue.put("set_dof_armatures")
        ack = self.ack_queue.get()
        if ack != "OK":
            print(f"Warning: Child returned ack={ack}")

    def set_dof_friction_coefficients(self, friction_coefficients: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the friction coefficients for each joint of the specified articulation instances.

        Args:
            friction_coefficients (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                                  specifying new friction values.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        self.shared_data["dof_frictions"][:] = friction_coefficients.cpu()
        self.cmd_queue.put("set_dof_frictions")
        ack = self.ack_queue.get()
        if ack != "OK":
            print(f"Warning: Child returned ack={ack}")

    def set_dof_max_velocities(self, max_velocities: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the maximum allowed velocities for each joint in the specified articulation instances.

        Args:
            max_velocities (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                           specifying new velocity limits.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        if torch.any(max_velocities != 0):
            print(
                "calling placeholder function: set_dof_max_velocities with non-zero values, but function is not"
                " implemented"
            )
        pass

    def set_dof_max_forces(self, max_forces: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Set the maximum allowed forces (torques) for each joint in the specified articulation instances.

        Args:
            max_forces (torch.Tensor): A tensor of shape (len(indices), dof_count),
                                       specifying new force/torque limits.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        if torch.any(max_forces != 0):
            print(
                "calling placeholder function: set_dof_max_forces with non-zero values, but function is not implemented"
            )
        pass

    def set_dof_limits(self, limits: torch.Tensor, indices: torch.Tensor):
        """
        Set new position limits (lower/upper) for each joint in the specified articulation instances.

        Args:
            limits (torch.Tensor): A tensor of shape (len(indices), dof_count, 2),
                                   specifying [lower_limit, upper_limit] for each joint.
            indices (torch.Tensor): Indices specifying which articulation instances to affect.
        """
        self.shared_data["dof_limits"][:] = limits.cpu()
        self.cmd_queue.put("set_dof_limits")
        ack = self.ack_queue.get()
        if ack != "OK":
            print(f"Warning: Child returned ack={ack}")

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
        raise NotImplementedError("Real does not support applying virtual forces.")

    def _initialize_default_data(self, data: ArticulationData):
        self.set_dof_position_targets(data.default_joint_pos, indices=torch.arange(self.count))
        self.set_dof_velocity_targets(data.default_joint_vel, indices=torch.arange(self.count))
