# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from uwlab.genes.gene import GeneOperatorBase, GeneOperatorBaseCfg, TupleGeneBaseCfg

if TYPE_CHECKING:
    from .genome_cfg import GenomeCfg


class Genome:
    env_cfg: ManagerBasedRLEnvCfg | None = None

    agent_cfg: RslRlOnPolicyRunnerCfg | None = None

    """
    The Genome class models a genetic structure with specific configuration environments (env_cfg and agent_cfg)
    and a genomic_regulatory_profile. It allows for genetic operations like mutation, breedingm and cloning of an
    genome. The class also supports the construction of a genetic dictionary where all gene are listed in a dictionary
    """

    def __init__(self, cfg: GenomeCfg):
        """Initializes an Genome instance by copying the provided configurations and profile and setting up an empty
        genetic dictionary.
            cfg (GenomeGeneticCfg): The regulatory profile for gene mutation where rule of mutation is specified.
        """
        self.genomic_regulatory_profile = cfg.genomic_mutation_profile
        self.genomic_group_processor = cfg.genomic_constraint_profile
        self.genetic_dictionary: dict[str, GeneOperatorBase] = {}
        self.genetic_groups: dict[str, list] = {}
        self.genetic_mutation_phases: dict[str, list[GeneOperatorBase]] = {"init": [], "mutate": [], "breed": []}
        self.tuple_genetic_groups: dict[str, list] = {"descend": [], "ascend": [], "equal": [], "symmetric": []}
        self.np_random = np.random.default_rng(cfg.seed)

    @property
    def my_genetic_manual(self) -> dict[str, GeneOperatorBase]:
        return self.genetic_dictionary

    def activate(self, env_cfg, agent_cfg):
        """Activates the genome by constructing the genetic modulation linkages. This involves recursively traversing the
        environment and regulatory profile, generating a dictionary of genes (genetic_dictionary) that access to all
        genetic functions.

        Args:
            env_cfg (ManagerBasedRLEnvCfg): The configuration of environment this genome lives in.
            agent_cfg (RslRlOnPolicyRunnerCfg): The configuration of agent that trains this genome.
        """
        # reset these fields are necessary because the genome may be re-activated,
        # which means some of genes may be deleted, and some new genes may be added, the new gene bridge needs
        # to be re-constructed
        self.env_cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.genetic_dictionary: dict[str, GeneOperatorBase] = {}
        self.genetic_groups: dict[str, list] = {}
        self.genetic_mutation_phases: dict[str, list[GeneOperatorBase]] = {"init": [], "mutate": [], "breed": []}
        self.tuple_genetic_groups: dict[str, list] = {"descend": [], "ascend": [], "equal": [], "symmetric": []}

        genetic_dictionary = {}
        cfgs, keys = self._recursively_construct_genetic_modulation_linkage(
            self.env_cfg, self.genomic_regulatory_profile, []
        )
        for cfg, args in zip(cfgs, keys):
            cfg.mutation_rate = self.np_random.random()
            retrival_func_list = []
            for i in range(len(args)):
                if "[]" in args[i][:2]:
                    retrival_func_list.append(lambda src_dict, key: dict.get(src_dict, key))
                    args[i] = args[i][2:]  # type: ignore
                elif "." in args[i][0]:
                    retrival_func_list.append(lambda src_class, key: enhanced_attrgetter(key)(src_class))
                    args[i] = args[i][1:]  # type: ignore
                else:
                    retrival_func_list.append(lambda src_class, key: enhanced_attrgetter(key)(src_class))

            if isinstance(cfg, TupleGeneBaseCfg):
                # process tuple type genes
                tuple_gene = []
                for i in range(cfg.element_length):  # type: ignore
                    cfg.element_idx = i
                    gene_operator = cfg.class_type((retrival_func_list, args), cfg, rng=self.np_random)
                    gene_identifier = ".".join(args) + f"_{i}"
                    genetic_dictionary[gene_identifier] = gene_operator
                    tuple_gene.append(gene_operator)

                    # Process Group
                    if cfg.group not in self.genetic_groups:
                        self.genetic_groups[cfg.group] = []
                    self.genetic_groups[cfg.group].append(gene_operator)
                    # Process Phase
                    for phase in cfg.phase:
                        self.genetic_mutation_phases[phase].append(gene_operator)
                self.tuple_genetic_groups[cfg.tuple_type].append(tuple_gene)
            else:
                # process single number type genes
                gene_operator = cfg.class_type((retrival_func_list, args), cfg, rng=self.np_random)
                genetic_dictionary[".".join(args)] = gene_operator
                # Process Group
                if cfg.group not in self.genetic_groups:
                    self.genetic_groups[cfg.group] = []
                self.genetic_groups[cfg.group].append(gene_operator)
                # Process Phase
                for phase in cfg.phase:
                    self.genetic_mutation_phases[phase].append(gene_operator)
        self.genetic_dictionary = genetic_dictionary

    def gene_initialize(self):
        for gene in self.genetic_mutation_phases["init"]:
            gene.mutate(self.env_cfg)
        # init functions may modify the env_cfg, some genes may be deleted,
        # and genes bridged earlier may be invalid, so we need to re-activate
        self.activate(self.env_cfg, self.agent_cfg)

        return self.env_cfg, self.agent_cfg

    def mutate(self):
        # Step1: mutation
        for gene in self.genetic_mutation_phases["mutate"]:
            gene.mutate(self.env_cfg)

        # Step2: Apply rules that make sure float are sanity checked

        # apply rules for those that are tuple
        for tuple_gene, tuples in self.tuple_genetic_groups.items():
            if tuple_gene == "equal":
                for tuple in tuples:
                    val = [gene.get(self.env_cfg) for gene in tuple]
                    val = np.array(val)
                    new_val = np.average(val).item()
                    for gene in tuple:
                        gene.set(self.env_cfg, new_val)
            elif tuple_gene == "ascend":
                for tuple in tuples:
                    val = [gene.get(self.env_cfg) for gene in tuple]
                    val = np.array(val)
                    new_val = np.sort(val)
                    for i, gene in enumerate(tuple):
                        gene.set(self.env_cfg, new_val[i].item())
            elif tuple_gene == "descend":
                for tuple in tuples:
                    val = [gene.get(self.env_cfg) for gene in tuple]
                    val = np.array(val)
                    new_val = np.sort(val)[::-1]
                    for i, gene in enumerate(tuple):
                        gene.set(self.env_cfg, new_val[i].item())
            elif tuple_gene == "symmetric":
                for tuple in tuples:
                    val = [gene.get(self.env_cfg) for gene in tuple]
                    val = np.array(val)
                    avg = np.abs(np.average(val))
                    new_val = np.array([-avg, avg])
                    for i, gene in enumerate(tuple):
                        gene.set(self.env_cfg, new_val[i].item())

        # apply extra customly designed rules
        for group_key, gene_list in self.genetic_groups.items():
            if group_key == "any":
                continue
            val_list = [gene.get(self.env_cfg) for gene in gene_list]
            func, arg = self.genomic_group_processor[group_key]
            new_val_list = func(val_list, *arg)
            for i, gene in enumerate(gene_list):
                gene.set(self.env_cfg, new_val_list[i])

    def breed(self, other_genome: Genome):
        for gene in self.genetic_mutation_phases["breed"]:
            val = gene.get(self.env_cfg)
            try:
                other_val = gene.get(other_genome.env_cfg)
            # if the gene is not in the other genome, skip
            # this error is accepted because the gene may not be in the other genome
            except TypeError:
                continue
            if val is None or other_val is None:
                continue
            gene.breed(self.env_cfg, other_genome.env_cfg)

    def clone(self):
        return Genome(self.env_cfg.copy(), self.agent_cfg.copy(), self.genomic_regulatory_profile.copy())  # type: ignore

    def _recursively_construct_genetic_modulation_linkage(
        self, roots, genomic_regulatory_profile, keys
    ) -> tuple[list[GeneOperatorBaseCfg], list[str]]:
        """A private method that recursively traverses the environment configuration (roots) and the regulatory profile
        (genomic_regulatory_profile) to construct the genetic modulation linkages. This method identifies all the relevant genes and
        attributes based on the genomic_regulatory_profile and returns the path_to_gene and gene_detail necessary for creating the
        genetic dictionary.

        Args:
            roots (dict|class): The current level of the environment configuration being traversed.
            genomic_regulatory_profile (dict): The current level of the regulatory profile that provides instructions on which attributes/keys to extract.
            keys (list): A list of keys (attribute paths) accumulated during recursion.

        Returns:
            gene_detail list[(scale, type)]: A list of guides for the identified genes mutation rules.
            path_to_gene list[str]: A list of keys indicating the attribute path to retrieve the data in env_cfg.
        """
        gene_detail = []
        path_to_gene = []
        profile = genomic_regulatory_profile
        if isinstance(profile, GeneOperatorBaseCfg):
            gene_detail.append(profile)
            path_to_gene.append(keys)
        elif isinstance(roots, dict):
            for k, v in profile.items():
                if k == ".":
                    for ext_key, val in roots.items():
                        for guide_key in profile["."].keys():
                            if enhanced_attrgetter(guide_key)(val) is not None:
                                sub_guid, sub_keys = self._recursively_construct_genetic_modulation_linkage(
                                    val, profile["."], keys + split_keys(ext_key)
                                )
                                gene_detail.extend(sub_guid)
                                path_to_gene.extend(sub_keys)
                                break
                else:
                    if k in roots:
                        attr = roots[k]
                        if attr is not None:
                            sub_guid, sub_keys = self._recursively_construct_genetic_modulation_linkage(
                                attr, profile[k], keys + split_keys(k)
                            )
                            gene_detail.extend(sub_guid)
                            path_to_gene.extend(sub_keys)

        elif hasattr(roots, "__dict__"):
            for k, v in profile.items():
                if k == ".":
                    for ext_key in dir(roots):
                        attr = getattr(roots, ext_key)
                        for guide_key in profile["."].keys():
                            if hasattr(attr, guide_key):
                                sub_guid, sub_keys = self._recursively_construct_genetic_modulation_linkage(
                                    attr, profile["."], keys + split_keys(ext_key)
                                )
                                gene_detail.extend(sub_guid)
                                path_to_gene.extend(sub_keys)
                                break
                else:
                    if enhanced_attrgetter(k)(roots) is not None:
                        attr = enhanced_attrgetter(k)(roots)
                        if attr is not None:
                            sub_guid, sub_keys = self._recursively_construct_genetic_modulation_linkage(
                                attr, profile[k], keys + split_keys(k)
                            )
                            gene_detail.extend(sub_guid)
                            path_to_gene.extend(sub_keys)

        else:
            gene_detail.append(profile)
            path_to_gene.append(keys)

        return gene_detail, path_to_gene


def enhanced_attrgetter(attr_string):
    def getter(obj):
        parts = re.split(r"(\.|\[|\])", attr_string)  # Split the string by '.', '[', and ']'
        current_obj = obj
        # inside_bracket = False

        for part in parts:
            if not part or part in ".[]":
                continue
            try:
                if isinstance(current_obj, dict):
                    current_obj = current_obj[part]  # Access dictionary key
                else:
                    current_obj = getattr(current_obj, part)  # Access attribute
            except (AttributeError, KeyError, TypeError):
                return None  # Return None if an attribute or key is not found
        return current_obj

    return getter


def split_keys(attr_string):
    parts = re.split(r"(\.|\[|\])", attr_string)  # Split the string by '.', '[', and ']'
    inside_bracket = False
    keys = []
    for part in parts:
        if not part or part in ".":
            continue
        if part == "[":
            inside_bracket = True  # Set flag when entering a bracket
            continue
        if part == "]":
            inside_bracket = False  # Set flag when exiting a bracket
            continue
        if inside_bracket:
            keys.append(f"[]{part}")
        else:
            keys.append(f".{part}")
    return keys
