# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for ASTEROID environments.

Re-exports everything from the OmniReset MDP and adds the pick-specific terms.
"""

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import *  # noqa: F401, F403

from .commands_cfg import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .recorders import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
