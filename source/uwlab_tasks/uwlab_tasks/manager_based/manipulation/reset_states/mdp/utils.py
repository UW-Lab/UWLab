# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import hashlib
import io
import logging
import numpy as np
import os
import random
import tempfile
import torch
import trimesh
import yaml
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from urllib.parse import urlparse

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
import omni
import warp as wp
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.warp import convert_to_warp_mesh
from pxr import UsdGeom
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from .rigid_object_hasher import RigidObjectHasher

# ---- module-scope caches ----
_PRIM_SAMPLE_CACHE: dict[tuple[str, int], np.ndarray] = {}  # (prim_hash, num_points) -> (N,3) in root frame
_FINAL_SAMPLE_CACHE: dict[str, np.ndarray] = {}  # env_hash -> (num_points,3) in root frame


def clear_pointcloud_caches():
    _PRIM_SAMPLE_CACHE.clear()
    _FINAL_SAMPLE_CACHE.clear()


@lru_cache(maxsize=None)
def _load_mesh_tensors(prim):
    tm = prim_to_trimesh(prim)
    verts = torch.from_numpy(tm.vertices.astype("float32"))
    faces = torch.from_numpy(tm.faces.astype("int64"))
    return verts, faces


def sample_object_point_cloud(
    num_envs: int,
    num_points: int,
    prim_path_pattern: str,
    device: str = "cuda",  # assume GPU
    rigid_object_hasher: RigidObjectHasher | None = None,
    seed: int = 42,
) -> torch.Tensor | None:
    """Generating point cloud given the path regex expression. This methood samples point cloud on ALL colliders
    falls under the prim path pattern. It is robust even if there are different numbers of colliders under the same
    regex expression. e.g. envs_0/object has 2 colliders, while envs_1/object has 4 colliders. This method will ensure
    each object has exactly num_points pointcloud regardless of number of colliders. If detected 0 collider, this method
    will return None, indicating no pointcloud can be sampled.

    To save memory and time, this method utilize RigidObjectHasher to make sure collider that hash to the same key will
    only be sampled once. It worths noting there are two kinds of hash:

    collider hash, and root hash. As name suggest, collider hash describes the uniqueness of collider from the view of root,
    collider hash is generated at atomic level and can not be representing aggregated. The root hash describes the
    uniqueness of aggregate of root, and can be hash that represent aggregate of multiple components that composes root.

    Be mindful that root's transform: translation, quaternion, scale, do no account for root's hash

    Args:
        num_envs (int): _description_
        num_points (int): _description_
        prim_path_pattern (str): _description_
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        torch.Tensor | None: _description_
    """
    hasher = (
        rigid_object_hasher
        if rigid_object_hasher is not None
        else RigidObjectHasher(num_envs, prim_path_pattern, device=device)
    )

    if hasher.num_root == 0:
        return None

    replicated_env = torch.all(hasher.root_prim_hashes == hasher.root_prim_hashes[0])
    if replicated_env:
        # Pick env 0’s colliders
        mask_env0 = hasher.collider_prim_env_ids == 0
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p, m in zip(hasher.collider_prims, mask_env0) if m])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms[mask_env0]
    else:
        # Build all envs's colliders
        verts_list, faces_list = zip(*[_load_mesh_tensors(p) for p in hasher.collider_prims])
        meshes = Meshes(verts=[v.to(device) for v in verts_list], faces=[f.to(device) for f in faces_list])
        rel_tf = hasher.collider_prim_relative_transforms
    with temporary_seed(seed):
        # Uniform‐surface sample then scale to root
        samp = sample_points_from_meshes(meshes, num_points * 2)
        local, _ = sample_farthest_points(samp, K=num_points)
        t_rel, q_rel, s_rel = rel_tf[:, :3].unsqueeze(1), rel_tf[:, 3:7].unsqueeze(1), rel_tf[:, 7:].unsqueeze(1)
        # here is apply_forward not apply_inverse, because when mesh loaded, it is unscaled. But inorder to view it from
        # root, you need to apply forward transformation of root->child, which is exactly tqs_root_child.
        root = math_utils.quat_apply(q_rel.expand(-1, num_points, -1), local * s_rel) + t_rel

        # Merge Colliders
        if replicated_env:
            buf = root.reshape(1, -1, 3)
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged.view(1, num_points, 3).expand(num_envs, -1, -1) * hasher.root_prim_scales.unsqueeze(1)
        else:
            # 4) Scatter each collider into a padded per‐root buffer
            env_ids = hasher.collider_prim_env_ids.to(device)  # (M,)
            counts = torch.bincount(env_ids, minlength=hasher.num_root)  # (num_root,)
            max_c = int(counts.max().item())
            buf = torch.zeros((hasher.num_root, max_c * num_points, 3), device=device, dtype=root.dtype)
            # track how many placed in each root
            placed = torch.zeros_like(counts)
            for i in range(len(hasher.collider_prims)):
                r = int(env_ids[i].item())
                start = placed[r].item() * num_points
                buf[r, start : start + num_points] = root[i]
                placed[r] += 1
            # 5) One batch‐FPS to merge per‐root
            merged, _ = sample_farthest_points(buf, K=num_points)
            result = merged * hasher.root_prim_scales.unsqueeze(1)

    return result


def _triangulate_faces(prim) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    faces = []
    it = iter(indices)
    for cnt in counts:
        poly = [next(it) for _ in range(cnt)]
        for k in range(1, cnt - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
    return np.asarray(faces, dtype=np.int64)


def create_primitive_mesh(prim) -> trimesh.Trimesh:
    prim_type = prim.GetTypeName()
    if prim_type == "Cube":
        size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        return trimesh.creation.box(extents=(size, size, size))
    elif prim_type == "Sphere":
        r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
        return trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif prim_type == "Cylinder":
        c = UsdGeom.Cylinder(prim)
        return trimesh.creation.cylinder(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Capsule":
        c = UsdGeom.Capsule(prim)
        return trimesh.creation.capsule(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    elif prim_type == "Cone":  # Cone
        c = UsdGeom.Cone(prim)
        return trimesh.creation.cone(radius=c.GetRadiusAttr().Get(), height=c.GetHeightAttr().Get())
    else:
        raise KeyError(f"{prim_type} is not a valid primitive mesh type")


def prim_to_trimesh(prim, relative_to_world=False) -> trimesh.Trimesh:
    if prim.GetTypeName() == "Mesh":
        mesh = UsdGeom.Mesh(prim)
        verts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        faces = _triangulate_faces(prim)
        mesh_tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    else:
        mesh_tm = create_primitive_mesh(prim)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # shape (4,4)
        mesh_tm.apply_transform(tf)

    return mesh_tm


def fps(points: torch.Tensor, n_samples: int, memory_threashold=2 * 1024**3) -> torch.Tensor:  # 2 GiB
    device = points.device
    N = points.shape[0]
    elem_size = points.element_size()
    bytes_needed = N * N * elem_size
    if bytes_needed <= memory_threashold:
        dist_mat = torch.cdist(points, points)
        sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
        min_dists = torch.full((N,), float("inf"), device=device)
        farthest = torch.randint(0, N, (1,), device=device)
        for j in range(n_samples):
            sampled_idx[j] = farthest
            min_dists = torch.minimum(min_dists, dist_mat[farthest].view(-1))
            farthest = torch.argmax(min_dists)
        return sampled_idx
    logging.warning(f"FPS fallback to iterative (needed {bytes_needed} > {memory_threashold})")
    sampled_idx = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float("inf"), device=device)
    farthest = torch.randint(0, N, (1,), device=device)
    for j in range(n_samples):
        sampled_idx[j] = farthest
        dist = torch.norm(points - points[farthest], dim=1)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return sampled_idx


def prim_to_warp_mesh(prim, device, relative_to_world=False) -> wp.Mesh:
    if prim.GetTypeName() == "Mesh":
        mesh_prim = UsdGeom.Mesh(prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    else:
        mesh = create_primitive_mesh(prim)
        points = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.int32)

    if relative_to_world:
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T  # (4,4)
        points = (points @ tf[:3, :3].T) + tf[:3, 3]

    wp_mesh = convert_to_warp_mesh(points, indices, device=device)
    return wp_mesh


@wp.kernel
def get_signed_distance(
    queries: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * n_points, 3]
    mesh_handles: wp.array(dtype=wp.uint64),  # [n_obstacles * E_bad * max_prims]
    prim_counts: wp.array(dtype=wp.int32),  # [n_obstacles * E_bad]
    coll_rel_pos: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * max_prims, 3]
    coll_rel_quat: wp.array(dtype=wp.quat),  # [n_obstacles * E_bad * max_prims, 4]
    coll_rel_scale: wp.array(dtype=wp.vec3),  # [n_obstacles * E_bad * max_prims, 3]
    max_dist: float,
    check_dist: bool,
    num_envs: int,
    num_points: int,
    max_prims: int,
    signs: wp.array(dtype=float),  # [E_bad * n_points]
):
    tid = wp.tid()
    per_obstacle_stride = num_envs * num_points
    obstacle_idx = tid // per_obstacle_stride
    rem = tid - obstacle_idx * per_obstacle_stride
    env_id = rem // num_points  # this env_id is index of arange(0, len(env_id)), its sequence, not selective indexing
    q = queries[tid]
    # accumulator for the lowest‐sign (start large)
    best_signed_dist = max_dist
    obstacle_env_base = obstacle_idx * num_envs * max_prims + env_id * max_prims
    prim_id = obstacle_idx * num_envs + env_id

    for p in range(prim_counts[prim_id]):
        index = obstacle_env_base + p
        mid = mesh_handles[index]
        if mid != 0:
            q1 = q - coll_rel_pos[index]
            q2 = wp.quat_rotate_inv(coll_rel_quat[index], q1)
            crs = coll_rel_scale[index]
            q3 = wp.vec3(q2.x / crs.x, q2.y / crs.y, q2.z / crs.z)
            mp = wp.mesh_query_point(mid, q3, max_dist)
            if mp.result:
                if check_dist:
                    closest = wp.mesh_eval_position(mid, mp.face, mp.u, mp.v)
                    local_dist = q3 - closest
                    unscaled_local_dist = wp.vec3(local_dist.x * crs.x, local_dist.y * crs.y, local_dist.z * crs.z)
                    delta_root = wp.quat_rotate(coll_rel_quat[index], unscaled_local_dist)
                    dist = wp.length(delta_root)
                    signed_dist = dist * mp.sign
                else:
                    signed_dist = mp.sign
                if signed_dist < best_signed_dist:
                    best_signed_dist = signed_dist
    signs[tid] = best_signed_dist


@contextmanager
def temporary_seed(seed: int, restore_numpy: bool = True, restore_python: bool = True):
    # snapshot states
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state() if restore_numpy else None
    py_state = random.getstate() if restore_python else None

    try:
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            torch_utils.set_seed(seed)
        yield
    finally:
        # restore everything
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if np_state is not None:
            np.random.set_state(np_state)
        if py_state is not None:
            random.setstate(py_state)


def read_metadata_from_usd_directory(usd_path: str) -> dict:
    """Read metadata from metadata.yaml in the same directory as the USD file."""
    # Get the directory containing the USD file
    usd_dir = os.path.dirname(usd_path)

    # Look for metadata.yaml in the same directory
    metadata_path = os.path.join(usd_dir, "metadata.yaml")
    rank = int(os.getenv("RANK", "0"))
    download_dir = os.path.join(tempfile.gettempdir(), f"rank_{rank}")
    with open(retrieve_file_path(metadata_path, download_dir=download_dir)) as f:
        metadata_file = yaml.safe_load(f)

    return metadata_file


def compute_assembly_hash(*usd_paths: str) -> str:
    """Compute a hash for an assembly based on the USD file paths.

    Args:
        *usd_paths: Variable number of USD file paths

    Returns:
        A hash string that uniquely identifies the combination of objects
    """
    # Extract path suffixes and sort to ensure consistent hash regardless of input order
    sorted_paths = sorted(urlparse(path).path for path in usd_paths)
    combined = "|".join(sorted_paths)

    full_hash = hashlib.md5(combined.encode()).hexdigest()
    return full_hash
