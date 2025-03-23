# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for different controllers and motion-generators.

Controllers or motion generators are responsible for closed-loop tracking of a given command. The
controller can be a simple PID controller or a more complex controller such as impedance control
or inverse kinematics control. The controller is responsible for generating the desired joint-level
commands to be sent to the robot.
"""

from .differential_ik import MultiConstraintDifferentialIKController
from .differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
