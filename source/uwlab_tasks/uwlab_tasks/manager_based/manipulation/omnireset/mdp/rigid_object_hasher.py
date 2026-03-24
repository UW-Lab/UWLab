# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import warp as wp
from isaaclab.sim import get_all_matching_child_prims
from pxr import Gf, Usd, UsdGeom, UsdPhysics

HASH_STORE = {"warp_mesh_store": {}, "__stage_id__": None}


class RigidObjectHasher:
    """Compute per-root and per-collider 64-bit hashes of transform+geometry."""

    def __init__(self, num_envs, prim_path_pattern, device="cpu"):
        self.prim_path_pattern = prim_path_pattern
        self.device = device
        # Invalidate cache if USD stage changed between runs (minimal, self-contained).
        stage_id = stage_utils.get_current_stage_id()
        prev_stage_id = HASH_STORE.get("__stage_id__")
        if prev_stage_id is not None and stage_id is not None and prev_stage_id != stage_id:
            HASH_STORE.clear()
            HASH_STORE["warp_mesh_store"] = {}
            HASH_STORE["__stage_id__"] = stage_id
        elif prev_stage_id is None and stage_id is not None:
            HASH_STORE["__stage_id__"] = stage_id

        if prim_path_pattern in HASH_STORE:
            return

        HASH_STORE[prim_path_pattern] = {
            "num_roots": 0,
            "collider_prims": [],
            "collider_prim_hashes": [],
            "collider_prim_env_ids": [],
            "collider_prim_relative_transforms": [],
            "root_prim_hashes": [],
            "root_prim_scales": [],
        }
        stor = HASH_STORE[prim_path_pattern]
        xform_cache = UsdGeom.XformCache()
        prim_paths = [prim_path_pattern.replace(".*", f"{i}", 1) for i in range(num_envs)]

        num_roots = len(prim_paths)
        collider_prim_env_ids = []
        collider_prims: list[Usd.Prim] = []
        collider_prim_relative_transforms = []
        collider_prim_hashes = []
        root_prim_hashes = []
        root_prim_scales = []
        for i in range(num_roots):
            # 1: Get all child prims that are colliders, count them, and store their belonging env id
            coll_prims = get_all_matching_child_prims(
                prim_paths[i],
                predicate=lambda p: p.GetTypeName() in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")
                and p.HasAPI(UsdPhysics.CollisionAPI),
                traverse_instance_prims=True,
            )
            if len(coll_prims) == 0:
                return
            collider_prims.extend(coll_prims)
            collider_prim_env_ids.extend([i] * len(coll_prims))

            # 2: Get relative transforms of all collider prims
            root_xf = xform_cache.GetLocalToWorldTransform(prim_utils.get_prim_at_path(prim_paths[i]))
            root_tf = Gf.Transform(root_xf)
            rel_tfs = []
            root_prim_scales.append(torch.tensor(root_tf.GetScale()))
            for prim in coll_prims:
                child_xf = xform_cache.GetLocalToWorldTransform(prim)
                rel_mat_tf = Gf.Transform(child_xf * root_xf.GetInverse())
                rel_quat = rel_mat_tf.GetRotation().GetQuat()
                rel_t = torch.tensor(rel_mat_tf.GetTranslation())
                rel_q = torch.tensor([rel_quat.GetReal(), *rel_quat.GetImaginary()])
                rel_s = torch.tensor(rel_mat_tf.GetScale())
                rel_tfs.append(torch.cat([rel_t, rel_q, rel_s]))
            rel_tfs = torch.cat(rel_tfs)
            collider_prim_relative_transforms.append(rel_tfs)

            # 3: Store the collider prims hash
            root_hash = hashlib.sha256()
            for prim, prim_rel_tf in zip(coll_prims, rel_tfs.numpy()):
                h = hashlib.sha256()
                h.update(
                    np.round(prim_rel_tf * 50).astype(np.int64)
                )  # round so small, +-2cm tol, difference won't cause issue
                prim_type = prim.GetTypeName()
                h.update(prim_type.encode("utf-8"))
                if prim_type == "Mesh":
                    verts = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float32)
                    h.update(verts.tobytes())
                else:
                    if prim_type == "Cube":
                        s = UsdGeom.Cube(prim).GetSizeAttr().Get()
                        h.update(np.float32(s).tobytes())
                    elif prim_type == "Sphere":
                        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                        h.update(np.float32(r).tobytes())
                    elif prim_type == "Cylinder":
                        c = UsdGeom.Cylinder(prim)
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                    elif prim_type == "Capsule":
                        c = UsdGeom.Capsule(prim)
                        h.update(c.GetAxisAttr().Get().encode("utf-8"))
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                    elif prim_type == "Cone":
                        c = UsdGeom.Cone(prim)
                        h.update(np.float32(c.GetRadiusAttr().Get()).tobytes())
                        h.update(np.float32(c.GetHeightAttr().Get()).tobytes())
                collider_hash = h.digest()
                root_hash.update(collider_hash)
                collider_prim_hashes.append(int.from_bytes(collider_hash[:8], "little", signed=True))
            small = int.from_bytes(root_hash.digest()[:8], "little", signed=True)
            root_prim_hashes.append(small)

        stor["num_roots"] = num_roots
        stor["collider_prims"] = collider_prims
        stor["collider_prim_hashes"] = torch.tensor(collider_prim_hashes, dtype=torch.int64, device="cpu")
        stor["collider_prim_env_ids"] = torch.tensor(collider_prim_env_ids, dtype=torch.int64, device="cpu")
        stor["collider_prim_relative_transforms"] = torch.cat(collider_prim_relative_transforms).view(-1, 10).to("cpu")
        stor["root_prim_hashes"] = torch.tensor(root_prim_hashes, dtype=torch.int64, device="cpu")
        stor["root_prim_scales"] = torch.stack(root_prim_scales).to("cpu")

    @property
    def num_root(self) -> int:
        return self.get_val("num_roots")

    @property
    def root_prim_hashes(self) -> torch.Tensor:
        return self.get_val("root_prim_hashes").to(self.device)

    @property
    def root_prim_scales(self) -> torch.Tensor:
        """Get the root prim transforms."""
        return self.get_val("root_prim_scales").to(self.device)

    @property
    def collider_prim_relative_transforms(self) -> torch.Tensor:
        return self.get_val("collider_prim_relative_transforms").to(self.device)

    @property
    def collider_prim_hashes(self) -> torch.Tensor:
        return self.get_val("collider_prim_hashes").to(self.device)

    @property
    def collider_prims(self) -> list[Usd.Prim]:
        return self.get_val("collider_prims")

    @property
    def collider_prim_env_ids(self) -> torch.Tensor:
        return self.get_val("collider_prim_env_ids").to(self.device)

    def get_val(self, key: str):
        """Get the hash store for the hasher."""
        return HASH_STORE.get(self.prim_path_pattern, {}).get(key)

    def set_val(self, key: str, val: any):
        if isinstance(val, torch.Tensor):
            val = val.to("cpu")
        HASH_STORE[self.prim_path_pattern][key] = val

    def get_warp_mesh_store(self) -> dict[int, wp.Mesh]:
        """Get the warp mesh store for the hasher."""
        return HASH_STORE["warp_mesh_store"]

    def get_hash_store(self) -> dict[int, any]:
        """Get the entire hash store"""
        return HASH_STORE
