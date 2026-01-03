# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .ext_cfg import (
    ContextInitializerCfg,
    SupplementaryLossesCfg,
    SupplementarySampleTermsCfg,
    SupplementaryTrainingCfg,
)
from .loss_ext import *
from .patches import patch_agent_with_supplementary_loss
from .sample_ext import *
