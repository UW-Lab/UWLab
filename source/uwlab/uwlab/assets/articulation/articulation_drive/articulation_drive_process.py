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
from abc import abstractmethod
from typing import TYPE_CHECKING

from .articulation_drive import ArticulationDrive

if TYPE_CHECKING:
    from . import ArticulationDriveCfg, ArticulationDriveData


# This class provides a sketch for dedicated process for handling hardware communication in an ArticulationDrive,
# rather than forcing every drive in a “view” to share a single process.

#     Why?
#         If you have multiple drives (e.g. an arm with two hands), each drive might otherwise block or slow down others when talking to hardware.
#         By creating a separate process per drive, each drive handles its own hardware I/O without impacting the rest of the system.

#     How it Works:
#         On creation, every drive class spawns either a multiprocessing Process or a thread (depending on configuration).
#         This separate worker loop regularly reads from and writes to the hardware (position, velocity, torque, etc.).
#         Commands from the main thread (such as setting stiffness or limits) go through a queue and get handled in the worker process.

#     Benefits:
#         No single bottleneck from multiple drives all sharing one process.
#         More responsive hardware communication for each drive.
#         Cleaner separation of logic—each drive manages its own thread/process.

# Before this Design
#     - ArticulationView1(Process1)                      - ArticulationView2(Process2)
#         - ArticulationDrive1                               - ArticulationDrive1
#         - ArticulationDrive2
#         - ArticulationDrive3

# as you may see, ArticulationView2 maybe fine, but ArticulationView1 may have a bottleneck due
# multiple hardware calls and some maybe blocking.

# After this Design:
#     - ArticulationView1(Process1)                      - ArticulationView2(Process4 thread1)
#         - ArticulationDrive1(Process2)                               - ArticulationDrive(Process4 thread2)
#         - ArticulationDrive2(Process3 thread1)
#         - ArticulationDrive3(Process3 thread2)


# Use this design when you suspect a single thread/process for all drives within a articulation View
# might become slow or blocking due to heavy hardware calls or large simulations.


class ArticulationDriveDedicatedProcess(ArticulationDrive):
    def __init__(self, cfg: ArticulationDriveCfg):
        self._dt = cfg.dt
        if cfg.use_multiprocessing:
            self.manager = mp.Manager()
            self.init_event = mp.Event()
            self.cmd_queue = mp.Queue()
            self.ack_queue = mp.Queue()
            # Create a shared dictionary to communicate with the child process
            self.shared_data: ArticulationDriveData = self.manager.dict()  # type: ignore
            self._proc = mp.Process(target=self._run, args=(self.init_event,))
        else:
            # Threading shares memory within the same process, so no need for mp.Manager
            self.manager = None
            self.init_event = threading.Event()
            self.cmd_queue = queue.Queue()
            self.ack_queue = queue.Queue()
            self.shared_data: ArticulationDriveData = {}  # type: ignore
            self._proc = threading.Thread(target=self._run, daemon=True, args=(self.init_event,))

    """
    Below method should be implemented by the child class
    """

    @property
    def ordered_joint_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def set_dof_stiffnesses(self, stiffnesses: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_armatures(self, armatures: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_frictions(self, frictions: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_dampings(self, dampings: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_limits(self, limits: torch.Tensor) -> None:
        raise NotImplementedError

    """Below methods should not be called by any other class
    """

    def _run(self, init_event: threading.Event):
        init_event.set()

        next_poll_time = time.time() + self._dt
        while True:
            if self.shared_data["close"]:
                break

            self._process_blocking_commands()

            if self.shared_data["is_running"]:
                now = time.time()
                if now >= next_poll_time:
                    next_poll_time = now + self._dt
                    # 1) read pos/vel from hardware
                    pos, vel, eff = self.read_dof_states()
                    self.shared_data["pos"][:] = pos
                    self.shared_data["vel"][:] = vel
                    self.shared_data["torque"][:] = eff

                    # 2) write target to hardware and kinematic
                    self.write_dof_targets(
                        pos_target=self.shared_data["pos_target"],
                        vel_target=self.shared_data["vel_target"],
                        eff_target=self.shared_data["eff_target"],
                    )
                    # Sleep till next poll time
                    time.sleep(max(next_poll_time - time.time(), 0))
                else:
                    # now is less than next poll time, sleep till next poll time
                    time.sleep(next_poll_time - now)
            else:
                # If not running, keep sleeping
                time.sleep(self._dt)

        # Done, disconnect hardware
        self.close()
        print("DynamixelWorker: Child process stopped")

    def _process_blocking_commands(self):
        if not self.cmd_queue.empty():
            cmd = self.cmd_queue.get()
            match cmd:
                case "set_dof_stiffnesses":
                    self.set_dof_stiffnesses(self.shared_data["dof_stiffness"])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_stiffnesses"})
                case "set_dof_armatures":
                    self.set_dof_armatures(self.shared_data["dof_armatures"])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_armatures"})
                case "set_dof_frictions":
                    self.set_dof_frictions(self.shared_data["dof_frictions"])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_frictions"})
                case "set_dof_dampings":
                    self.set_dof_dampings(self.shared_data["dof_damping"])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_dampings"})
                case "set_dof_limits":
                    self.set_dof_limits(self.shared_data["dof_limits"])
                    self.ack_queue.put({"status": "OK", "command": "set_dof_limits"})

    """
    Below methods are for users to call
    """
