# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import hydra
import isaaclab_tasks  # noqa: F401
import pytest

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@pytest.mark.parametrize("override", ["env.scene.terrain=gap", "env.scene.terrain=all"])
def test_hydra_group_override(override: str):
    """Test the hydra configuration system for group overriding behavior with two choices."""

    @hydra_task_compose("UW-Position-Advance-Skills-Spot-v0", "rsl_rl_cfg_entry_point", [override])
    def main(env_cfg, agent_cfg):
        keys = list(env_cfg.scene.terrain.terrain_generator.sub_terrains.keys())
        if override.endswith("=gap"):
            assert len(keys) == 1
            assert keys[0] == "gap"
        else:
            # full set should contain multiple terrains including 'gap'
            assert len(keys) > 1
            assert "gap" in keys

    main()
    # clean up
    hydra.core.global_hydra.GlobalHydra.instance().clear()
