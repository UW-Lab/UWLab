# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This sub-module provides methods to create different terrains using the ``trimesh`` library.

In contrast to the height-field representation, the trimesh representation does not
create arbitrarily small triangles. Instead, the terrain is represented as a single
tri-mesh primitive. Thus, this representation is more computationally and memory
efficient than the height-field representation, but it is not as flexible.
"""

from .basic_mesh_terrains_cfg import (
    MeshBoxTerrainCfg,
    MeshFloatingRingTerrainCfg,
    MeshGapTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPitTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRailsTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    MeshRepeatedCylindersTerrainCfg,
    MeshRepeatedPyramidsTerrainCfg,
    MeshStarTerrainCfg,
)
from .mesh_terrains_cfg import (
    CachedTerrainGenCfg,
    MeshBalanceBeamsTerrainCfg,
    MeshDiversityBoxTerrainCfg,
    MeshObjTerrainCfg,
    MeshPassageTerrainCfg,
    MeshSteppingBeamsTerrainCfg,
    MeshStonesEverywhereTerrainCfg,
    MeshStructuredTerrainCfg,
    TerrainGenCfg,
)
