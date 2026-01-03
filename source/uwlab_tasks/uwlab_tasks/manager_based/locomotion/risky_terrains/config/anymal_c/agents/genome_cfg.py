# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from uwlab.genes import GenomeCfg
from uwlab.genes.gene import gene_mdp as mdp
from uwlab.genes.gene.gene_cfg import FloatGeneCfg

GENOMIC_MUTATION_PROFILE = {
    # SCENE
    # "scene.height_scanner.pattern_cfg": {
    #     "resolution": FloatGeneCfg(
    #         phase=["mutate"], mutation_func=mdp.random_float, mutation_args=(0.03, 0.1)),
    #     "size": FloatTupleGeneCfg(
    #         element_length=2,
    #         tuple_type="descend",
    #         phase=["mutate"],
    #         mutation_func=mdp.random_float,
    #         mutation_args=(1.00, 2.50),
    #     )
    # },
    # REWARDS
    "rewards": {
        ".": {"weight": FloatGeneCfg(phase=["mutate"], mutation_func=mdp.add_fraction, mutation_args=(0.3,))},
    },
    # CURRICULUM
    # "curriculum.terrain_levels.params": {
    #     "promotion_fraction": FloatGeneCfg(
    #         phase=["mutate"], mutation_func=mdp.random_float, mutation_args=(0.6, 0.95)),
    #     "demotion_fraction": FloatGeneCfg(
    #         phase=["mutate"], mutation_func=mdp.random_float, mutation_args=(0.01, 0.4))
    # },
}


@configclass
class RiskyTerrainAnymalCGenomeCfg(GenomeCfg):
    genomic_mutation_profile: dict = GENOMIC_MUTATION_PROFILE

    genomic_constraint_profile: dict = {}

    seed: int = 32
