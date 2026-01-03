# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from . import recorders

##
# State recorders.
##


# Stable state recorder.
@configclass
class StableStateRecorderCfg(RecorderTermCfg):
    """Configuration for the initial state recorder term."""

    class_type: type[RecorderTerm] = recorders.StableStateRecorder


@configclass
class StableStateRecorderManagerCfg(RecorderManagerBaseCfg):
    """Recorder configurations for recording actions and states."""

    record_pre_reset_states = StableStateRecorderCfg()


# Grasp relative pose recorder.
@configclass
class GraspRelativePoseRecorderCfg(RecorderTermCfg):
    """Configuration for the grasp relative pose recorder term."""

    class_type: type[RecorderTerm] = recorders.GraspRelativePoseRecorder
    robot_name: str = MISSING
    object_name: str = MISSING
    gripper_body_name: str = MISSING


@configclass
class GraspRelativePoseRecorderManagerCfg(RecorderManagerBaseCfg):
    """Configuration for the grasp relative pose recorder manager."""

    def __init__(self, *args, **kwargs):
        """Initialize with explicit parameters for cleaner interface.

        Args:
            robot_name: Name of the robot in the scene to track.
            object_name: Name of the object in the scene to track.
            gripper_body_name: Name of the gripper body for pose tracking.
            **kwargs: Additional arguments passed to parent class.
        """
        robot_name = kwargs.pop("robot_name", MISSING)
        object_name = kwargs.pop("object_name", MISSING)
        gripper_body_name = kwargs.pop("gripper_body_name", MISSING)
        super().__init__(args, **kwargs)
        self.record_grasp_relative_pose = GraspRelativePoseRecorderCfg(
            robot_name=robot_name,
            object_name=object_name,
            gripper_body_name=gripper_body_name,
        )
