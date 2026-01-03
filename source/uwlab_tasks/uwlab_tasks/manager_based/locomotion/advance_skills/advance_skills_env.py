# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .advance_skills_base_env import AdvanceSkillsBaseEnvCfg
from .terrains import EXTREME_STAIR, GAP, IRREGULAR_PILLAR_OBSTACLE, PIT, SLOPE_INV, SQUARE_PILLAR_OBSTACLE


class GapEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"gap": GAP}


class PitEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"pit": PIT}


class ExtremeStairEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"pit": EXTREME_STAIR}


class SlopeInvEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"slope_inv": SLOPE_INV}


class SquarePillarObstacleEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"square_pillar_obstacle": SQUARE_PILLAR_OBSTACLE}


class IrregularPillarObstacleEnvConfig(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {"irregular_pillar_obstacle": IRREGULAR_PILLAR_OBSTACLE}


class AdvanceSkillsEnvCfg(AdvanceSkillsBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.sub_terrains = {
            "gap": GAP,
            "pit": PIT,
            "extreme_stair": EXTREME_STAIR,
            "slope_inv": SLOPE_INV,
            "square_pillar_obstacle": SQUARE_PILLAR_OBSTACLE,
            "irregular_pillar_obstacle": IRREGULAR_PILLAR_OBSTACLE,
        }
