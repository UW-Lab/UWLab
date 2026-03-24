# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import warp as wp
from isaaclab.assets import RigidObject
from isaaclab.sim.utils import get_first_matching_child_prim
from pxr import UsdPhysics

from . import utils
from .rigid_object_hasher import RigidObjectHasher

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .collision_analyzer_cfg import CollisionAnalyzerCfg

# from isaaclab.markers import VisualizationMarkers
# from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

# ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloudDebug")
# ray_cfg.markers["hit"].radius = 0.005
# visualizer = VisualizationMarkers(ray_cfg)


class CollisionAnalyzer:

    cfg: CollisionAnalyzerCfg

    def __init__(self, cfg: CollisionAnalyzerCfg, env: ManagerBasedRLEnv):
        self.cfg = cfg
        self.asset: RigidObject = env.scene[cfg.asset_cfg.name]
        self.obstacles: list[RigidObject] = [env.scene[cfg.name] for cfg in cfg.obstacle_cfgs]
        # we support passing in a Articulation(joint connected rigidbodies) as asset, but that requires we collect all
        # body names user intended to generate collision checks
        body_names = (
            self.asset.body_names
            if cfg.asset_cfg.body_names is None
            else [self.asset.body_names[i] for i in cfg.asset_cfg.body_ids]
        )
        if isinstance(body_names, str):
            body_names = [body_names]

        self.body_ids = []
        self.local_pts = []
        for i, body_name in enumerate(body_names):
            # start = time.perf_counter()
            prim = get_first_matching_child_prim(
                self.asset.cfg.prim_path.replace(".*", "0", 1),  # we use the 0th env prim as template
                predicate=lambda p: p.GetName() == body_name and p.HasAPI(UsdPhysics.RigidBodyAPI),
            )
            local_pts = utils.sample_object_point_cloud(
                num_envs=env.num_envs,
                num_points=cfg.num_points,
                prim_path_pattern=str(prim.GetPath()).replace("env_0", "env_.*", 1),
                device=env.device,
            )
            if local_pts is not None:
                self.local_pts.append(local_pts.view(env.num_envs, 1, cfg.num_points, 3))
                self.body_ids.append(self.asset.body_names.index(body_name))
            # pc_time = time.perf_counter() - start
        self.local_pts = torch.cat(self.local_pts, dim=1)
        self.body_ids = torch.tensor(self.body_ids, dtype=torch.int, device=env.device)

        self.num_coll_per_obstacle_per_env = torch.empty(
            (len(self.obstacles), env.num_envs), dtype=torch.int32, device=env.device
        )
        self.obstacle_root_scales = torch.empty(
            (len(self.obstacles), env.num_envs, 3), dtype=torch.float, device=env.device
        )
        env_handles = [[[] for _ in range(env.num_envs)] for _ in range(len(self.obstacles))]
        obstacle_relative_transforms = [[[] for _ in range(env.num_envs)] for _ in range(len(self.obstacles))]

        for i, obstacle in enumerate(self.obstacles):
            # start = time.perf_counter()
            obs_h = RigidObjectHasher(env.num_envs, prim_path_pattern=obstacle.cfg.prim_path, device=env.device)
            for prim, p_hash, rel_tf, eid in zip(
                obs_h.collider_prims,
                obs_h.collider_prim_hashes,
                obs_h.collider_prim_relative_transforms,
                obs_h.collider_prim_env_ids,
            ):
                # convert each USD prim → Warp mesh...
                obstacle_relative_transforms[i][eid].append(rel_tf)
                if p_hash.item() in obs_h.get_warp_mesh_store():
                    env_handles[i][eid].append(obs_h.get_warp_mesh_store()[p_hash.item()].id)
                else:
                    wp_mesh = utils.prim_to_warp_mesh(prim, device=env.device, relative_to_world=False)
                    obs_h.get_warp_mesh_store()[p_hash.item()] = wp_mesh
                    env_handles[i][eid].append(wp_mesh.id)

            self.num_coll_per_obstacle_per_env[i] = torch.bincount(obs_h.collider_prim_env_ids)
            self.obstacle_root_scales[i] = obs_h.root_prim_scales
            # pc_time = time.perf_counter() - start
            # print(f"Sampled {len(obs_h.collider_prims)} wp meshes at '{obstacle.cfg.prim_path}' in {pc_time:.3f}s")
        self.max_prims = torch.max(self.num_coll_per_obstacle_per_env).item()
        handle_list = []
        rel_transform = []
        for i in range(len(env_handles)):
            handle_list.extend([torch.tensor(lst, dtype=torch.int64, device=env.device) for lst in env_handles[i]])
            rel_transform.extend([torch.cat((tf), dim=0) for tf in obstacle_relative_transforms[i]])
        self.handles_tensor = pad_sequence(handle_list, batch_first=True, padding_value=0).view(
            len(self.obstacles), env.num_envs, -1
        )
        self.collider_rel_transform = pad_sequence(rel_transform, batch_first=True, padding_value=0).view(
            len(self.obstacles), env.num_envs, -1, 10
        )

    def __call__(self, env: ManagerBasedRLEnv, env_ids: torch.Tensor):
        pos_w = (
            self.asset.data.body_link_pos_w[env_ids][:, self.body_ids]
            .unsqueeze(2)
            .expand(-1, -1, self.cfg.num_points, 3)
        )
        quat_w = (
            self.asset.data.body_link_quat_w[env_ids][:, self.body_ids]
            .unsqueeze(2)
            .expand(-1, -1, self.cfg.num_points, 4)
        )
        cloud = math_utils.quat_apply(quat_w, self.local_pts[env_ids]) + pos_w  # omit scale 1

        obstacles_pos_w = torch.cat(
            [
                obstacle.data.root_pos_w[env_ids].view(-1, 1, 1, 3).expand(-1, -1, self.cfg.num_points, 3)
                for obstacle in self.obstacles
            ],
            dim=0,
        )
        obstacles_quat_w = torch.cat(
            [
                obstacle.data.root_quat_w[env_ids].view(-1, 1, 1, 4).expand(-1, cloud.shape[1], self.cfg.num_points, 4)
                for obstacle in self.obstacles
            ],
            dim=0,
        )
        obstacles_scale_w = (
            self.obstacle_root_scales[:, env_ids].view(-1, 1, 1, 3).expand(-1, -1, self.cfg.num_points, 3)
        )
        could_root = (
            math_utils.quat_apply_inverse(
                obstacles_quat_w, (cloud.repeat(len(self.obstacles), 1, 1, 1)) - obstacles_pos_w
            )
            / obstacles_scale_w
        )

        total_points = len(self.body_ids) * self.cfg.num_points * len(self.obstacles)
        queries = wp.from_torch(could_root.reshape(-1, 3), dtype=wp.vec3)
        handles = wp.from_torch(
            self.handles_tensor[:, env_ids].view(-1), dtype=wp.uint64
        )  # (num_obstacles * len(env_ids) * max_prims,)
        counts = wp.from_torch(self.num_coll_per_obstacle_per_env[:, env_ids].view(-1), dtype=wp.int32)
        coll_rel_pos = wp.from_torch(self.collider_rel_transform[:, env_ids, :, :3].view(-1, 3), dtype=wp.vec3)
        coll_rel_quat = wp.from_torch(self.collider_rel_transform[:, env_ids, :, 3:7].view(-1, 4), dtype=wp.quat)
        coll_rel_scale = wp.from_torch(self.collider_rel_transform[:, env_ids, :, 7:10].view(-1, 3), dtype=wp.vec3)
        sign_w = wp.zeros((len(env_ids) * total_points,), dtype=float, device=env.device)
        wp.launch(
            utils.get_signed_distance,
            dim=len(env_ids) * total_points,
            inputs=[
                queries,
                handles,
                counts,
                coll_rel_pos,
                coll_rel_quat,
                coll_rel_scale,
                float(self.cfg.max_dist),
                self.cfg.min_dist != 0.0,
                len(env_ids),
                len(self.body_ids) * self.cfg.num_points,
                self.max_prims,
            ],
            outputs=[sign_w],
            device=env.device,
        )
        signs = (
            wp.to_torch(sign_w)
            .view(len(self.obstacles), len(env_ids), len(self.body_ids), self.cfg.num_points)
            .amin(dim=0)
        )
        # collision_points = cloud[(signs < 0.0)]
        # for i in range(500):
        #     env.sim.render()
        #     visualizer.visualize(collision_points.view(-1, 3))

        coll_free_mask = signs.amin(dim=(1, 2)) >= self.cfg.min_dist
        return coll_free_mask
