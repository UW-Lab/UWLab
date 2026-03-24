# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import functools
import io
import logging
import numpy as np
import os
import random
import shutil
import tempfile
import torch
import trimesh
import yaml
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import PurePosixPath
from urllib.parse import urlparse

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
import omni
import warp as wp
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.warp import convert_to_warp_mesh
from pxr import UsdGeom
from pytorch3d.ops import sample_farthest_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

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


def get_temp_dir(rank: int | None = None) -> str:
    """Get a user/job-specific temporary directory under /tmp/uwlab/.

    Creates a directory structure that avoids conflicts between users and jobs:
    /tmp/uwlab/{uid}/{job_id}/{rank}/

    Args:
        rank: Process rank (defaults to RANK env var or 0)

    Returns:
        Path to the temporary directory (created if it doesn't exist)
    """
    if rank is None:
        rank = int(os.getenv("RANK", "0"))

    uid = os.getuid()
    job_id = os.getenv("SLURM_JOB_ID") or os.getenv("PBS_JOBID") or "local"

    download_dir = os.path.join("/tmp", "uwlab", str(uid), str(job_id), f"rank_{rank}")
    os.makedirs(download_dir, mode=0o700, exist_ok=True)

    return download_dir


def safe_retrieve_file_path(url: str, download_dir: str | None = None) -> str:
    """Resolve a file path, downloading from the cloud if necessary.

    For HTTPS URLs and local paths the unified :func:`resolve_cloud_path`
    handles download + persistent caching.  Nucleus (``omniverse://``)
    paths still fall back to Isaac Lab's :func:`retrieve_file_path`.
    """
    from uwlab_assets import resolve_cloud_path

    if url.startswith(("http://", "https://")) or os.path.isfile(url):
        return resolve_cloud_path(url)

    # Nucleus / omni.client fallback
    if download_dir is None:
        download_dir = get_temp_dir()
    os.makedirs(download_dir, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=download_dir, prefix=".dl_")
    try:
        downloaded = retrieve_file_path(url, download_dir=tmp_dir)
        abs_tmp = os.path.abspath(tmp_dir)
        if not os.path.abspath(downloaded).startswith(abs_tmp + os.sep):
            return downloaded
        target = os.path.join(download_dir, os.path.basename(downloaded))
        os.rename(downloaded, target)
        return target
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@functools.cache
def read_metadata_from_usd_directory(usd_path: str) -> dict:
    """Read metadata from metadata.yaml in the same directory as the USD file.

    Results are memoised per *usd_path* so each asset's metadata is
    downloaded and parsed at most once per process.
    """
    usd_dir = os.path.dirname(usd_path)
    metadata_path = os.path.join(usd_dir, "metadata.yaml")
    local_path = safe_retrieve_file_path(metadata_path, download_dir=get_temp_dir())
    with open(local_path) as f:
        metadata_file = yaml.safe_load(f)

    return metadata_file


def object_name_from_usd(usd_path: str) -> str:
    """Extract the canonical object name from a USD asset path.

    Uses the parent directory name, which is unique across the asset tree.
    Works identically for local paths and S3 URLs.

    Example: ``'.../Props/Custom/Peg/peg.usd'`` -> ``'Peg'``
    """
    return PurePosixPath(urlparse(usd_path).path).parent.name


def compute_pair_dir(*usd_paths: str) -> str:
    """Derive a human-readable directory name from one or more USD asset paths.

    Names are sorted alphabetically and joined with ``'__'``.

    Examples:
        Single object:  ``('...Peg/peg.usd',)`` -> ``'Peg'``
        Object pair:    ``('...Peg/peg.usd', '...PegHole/peg_hole.usd')`` -> ``'Peg__PegHole'``
    """
    return "__".join(sorted(object_name_from_usd(p) for p in usd_paths))


def load_asset_paths_from_config(
    config_path: str,
    cache_subdir: str = "",
    skip_validation: bool = True,
) -> list[str]:
    """Load asset paths from YAML config file.

    Args:
        config_path: Path to the YAML config file.
        cache_subdir: Subdirectory name for local caching of cloud assets (e.g., "hdris", "textures").
        skip_validation: If True, skip expensive omni.client.stat() validation for Nucleus paths.

    Returns:
        List of asset paths ready to use.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    def collect_paths(obj):
        paths = []
        if isinstance(obj, dict):
            for value in obj.values():
                paths.extend(collect_paths(value))
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    paths.append(item)
                else:
                    paths.extend(collect_paths(item))
        return paths

    local_and_nucleus_paths = []
    cloud_paths = []
    for section, paths_obj in config.items():
        section_paths = collect_paths(paths_obj)
        for p in section_paths:
            if section == "isaac_nucleus":
                local_and_nucleus_paths.append(f"{ISAAC_NUCLEUS_DIR}/{p}")
            elif section == "local":
                local_and_nucleus_paths.append(p)
            elif section == "cloud":
                if p.startswith("http://") or p.startswith("https://"):
                    cloud_paths.append(p)
                else:
                    cloud_paths.append(f"{UWLAB_CLOUD_ASSETS_DIR}/{p}")
            else:
                local_and_nucleus_paths.append(f"{NVIDIA_NUCLEUS_DIR}/{p}")

    # Download cloud assets to local cache
    cached_cloud_paths = []
    if cloud_paths:
        cached_cloud_paths = _download_cloud_assets(cloud_paths, cache_subdir)

    # Build final path list
    valid_paths = list(cached_cloud_paths)
    if skip_validation:
        valid_paths.extend(local_and_nucleus_paths)
    else:
        skipped_nucleus = []
        for path in local_and_nucleus_paths:
            if path.startswith("/"):
                if os.path.exists(path):
                    valid_paths.append(path)
            else:
                skipped_nucleus.append(path)
        if skipped_nucleus:
            logging.warning(
                f"[load_asset_paths_from_config] Skipped {len(skipped_nucleus)} Nucleus paths "
                f"(non-local, unreliable). Using {len(valid_paths)} local paths only."
            )

    # Validate that we actually have usable paths
    if not valid_paths:
        raise RuntimeError(
            f"[load_asset_paths_from_config] No valid asset paths loaded from {config_path}.\n"
            f"  Nucleus paths found: {len(local_and_nucleus_paths)}\n"
            f"  Cloud paths found: {len(cloud_paths)}\n"
            f"  Cloud paths cached locally: {len(cached_cloud_paths)}\n"
            "  Check that your Nucleus server is running or cloud assets are downloadable."
        )

    # Validate local paths are accessible (local and cached cloud only; Nucleus paths
    # are validated lazily by the renderer since omni.client.stat is expensive)
    inaccessible = []
    for p in valid_paths:
        if p.startswith("/") and not os.path.exists(p):
            inaccessible.append(p)
    if inaccessible:
        logging.warning(
            f"[load_asset_paths_from_config] {len(inaccessible)}/{len(valid_paths)} local paths are inaccessible. "
            f"First 3: {inaccessible[:3]}"
        )
        valid_paths = [p for p in valid_paths if p not in inaccessible]

    total_local = sum(1 for p in valid_paths if p.startswith("/"))
    total_nucleus = sum(1 for p in valid_paths if p.startswith("omniverse://"))
    logging.info(
        f"[load_asset_paths_from_config] Loaded {len(valid_paths)} paths from {config_path} "
        f"({total_local} local, {total_nucleus} nucleus)"
    )

    return valid_paths


def _download_cloud_assets(cloud_urls: list[str], cache_subdir: str = "", num_workers: int = 8) -> list[str]:
    """Download cloud URLs to local cache, return local paths.

    Delegates to :func:`resolve_cloud_path` for each URL so all cloud
    assets share the same persistent cache and atomic-download logic.
    Downloads are parallelized with *num_workers* threads and a live
    progress line with elapsed time is printed.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from uwlab_assets import resolve_cloud_path

    n = len(cloud_urls)
    to_download = [u for u in cloud_urls if not os.path.isfile(_cached_local_path(u))]
    needs_download = len(to_download)

    if needs_download == 0:
        return [resolve_cloud_path(url) for url in cloud_urls]

    tag = cache_subdir or "cloud"
    print(f"[INFO] Downloading {needs_download}/{n} {tag} assets ({num_workers} workers) ...")
    t0 = time.monotonic()

    downloaded = 0
    futures = {}
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for url in to_download:
            futures[pool.submit(resolve_cloud_path, url)] = url
        for future in as_completed(futures):
            future.result()
            downloaded += 1
            elapsed = time.monotonic() - t0
            rate = downloaded / elapsed
            eta = (needs_download - downloaded) / rate if rate > 0 else 0
            print(
                f"\r  [{downloaded}/{needs_download}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining",
                end="",
                flush=True,
            )

    elapsed = time.monotonic() - t0
    print(f"\n[INFO] Finished downloading {needs_download} {tag} assets in {elapsed:.0f}s.")

    return [resolve_cloud_path(url) for url in cloud_urls]


def _cached_local_path(url: str) -> str:
    """Return the expected local cache path for a cloud URL without downloading."""
    from uwlab_assets import _extract_relative_path

    rel = _extract_relative_path(url)
    return os.path.join(os.path.expanduser("~"), ".cache", "uwlab", "assets", rel)


# ---- OSC / script helpers (unscaled action = Cartesian delta) ----
def target_pose_to_action(
    ee_pos_b: torch.Tensor,
    ee_quat_b: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> torch.Tensor:
    """Compute arm action (6-DOF delta) so RelCartesianOSC tracks target pose.

    For Sysid env (unscaled action): action = delta_pose. Used by sysid/plot
    scripts that step the env with waypoint targets.
    """
    delta_pos = target_pos - ee_pos_b
    quat_err = math_utils.quat_mul(target_quat, math_utils.quat_inv(ee_quat_b))
    axis_angle = math_utils.axis_angle_from_quat(quat_err)
    return torch.cat([delta_pos, axis_angle], dim=-1)


def settle_robot(
    robot, sim, default_joint_pos, default_joint_vel, arm_joint_ids, sim_dt, headless=True, settle_steps=10
):
    """Hard-reset settle: write desired state repeatedly, then final write without stepping."""
    for _ in range(settle_steps):
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        robot.write_data_to_sim()
        sim.step(render=not headless)
        robot.update(sim_dt)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.write_data_to_sim()
    robot.update(sim_dt)
