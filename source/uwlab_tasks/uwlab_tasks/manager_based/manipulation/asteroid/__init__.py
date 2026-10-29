# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ASTEROID environments.

Pick-only variants of the OmniReset UR5e + Robotiq 2F-85 tasks used by ASTEROID
(iterative in-context exploration + distillation, see ``scripts/ASTEROID``).

Everything here builds on :mod:`uwlab_tasks.manager_based.manipulation.omnireset`:
scene, MDP terms and environment configs are subclassed and only the pick-specific
deltas live in this package (no receptive object, pick-height success, coupled
domain randomization, tactile data-collection observations).
"""

import os

ASTEROID_DATASETS_DIR = os.environ.get("ASTEROID_DATASETS_DIR", "Datasets/CubePick")
"""Root of the locally recorded reset-state / grasp datasets used by the ASTEROID configs.

Layout mirrors the OmniReset asset hub, keyed by the insertive object only::

    <ASTEROID_DATASETS_DIR>/Resets/<Object>/resets_<ResetType>.pt
    <ASTEROID_DATASETS_DIR>/Grasps/<Object>/grasps.pt

Override with the ``ASTEROID_DATASETS_DIR`` environment variable.
"""
