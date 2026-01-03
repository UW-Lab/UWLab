# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import warp as wp  # Warp (https://github.com/NVIDIA/warp)
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from . import patch_sampling_cfg as patch_cfg


def uniform_sample_multiple_ranges(
    ranges: list[tuple[float, float]],
    sample_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample `sample_size` values from the provided list of (min, max) ranges.
    Each sampled value is drawn from one interval chosen uniformly at random.
    """
    if not ranges:
        raise ValueError("`ranges` cannot be empty")

    # Number of intervals
    num_intervals = len(ranges)

    # Randomly select which interval each sample should come from
    interval_indices = torch.randint(low=0, high=num_intervals, size=(sample_size,), device=device)

    # Prepare the output buffer
    samples = torch.empty(sample_size, device=device, dtype=torch.float32)

    # For each interval, sample the required number of values
    for i in range(num_intervals):
        mask = interval_indices == i
        count_i = mask.sum()
        if count_i > 0:
            low, high = ranges[i]
            samples[mask] = torch.empty(count_i, device=device).uniform_(low, high)

    return samples


def find_piecewise_range_flat_patches(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.PieceWiseRangeFlatPatchSamplingCfg,
) -> torch.Tensor:
    """
    Finds flat patches of a given radius in the input mesh, but now supports
    multiple intervals for x, y, and z to give you more control over where to sample.

    Args:
        wp_mesh: The Warp mesh to find patches in.
        num_patches: The desired number of patches to find.
        patch_radius: The radii used to form patches (float or list of floats).
        origin: The origin defining the center of the search space (in mesh frame).
        x_range: A list of (min, max) intervals for X sampling (in mesh frame).
        y_range: A list of (min, max) intervals for Y sampling (in mesh frame).
        z_range: A list of (min, max) intervals for Z filtering (in mesh frame).
        max_height_diff: The maximum allowed distance between lowest and highest
                         points on a patch to consider it valid.
        max_iterations: The maximum number of rejection-sampling iterations.

    Returns:
        A (num_patches, 3) torch.Tensor containing the valid flat patch centers,
        in the mesh frame, offset so that the origin is subtracted at the end.

    Raises:
        RuntimeError: If the function cannot find valid patches within max_iterations.
    """
    device = wp.device_to_torch(wp_mesh.device)

    # Handle patch_radius input
    if isinstance(cfg.patch_radius, float):
        patch_radius = [cfg.patch_radius]

    # Convert origin to a torch tensor (on the correct device)
    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).float().to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.float().to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    # --- 1) Clip each interval to the bounding box of the mesh and shift by origin
    # Convert mesh points to numpy for min/max
    mesh_pts = wp_mesh.points.numpy()
    mesh_xmin, mesh_xmax = mesh_pts[:, 0].min(), mesh_pts[:, 0].max()
    mesh_ymin, mesh_ymax = mesh_pts[:, 1].min(), mesh_pts[:, 1].max()

    x_range = [cfg.x_range] if isinstance(cfg.x_range, tuple) else cfg.x_range
    y_range = [cfg.y_range] if isinstance(cfg.y_range, tuple) else cfg.y_range
    z_range = [cfg.z_range] if isinstance(cfg.z_range, tuple) else cfg.z_range

    # For x-ranges
    x_range_clipped = []
    for low, high in x_range:
        new_low = max(low + origin[0].item(), mesh_xmin)
        new_high = min(high + origin[0].item(), mesh_xmax)
        if new_low < new_high:
            x_range_clipped.append((new_low, new_high))

    # For y-ranges
    y_range_clipped = []
    for low, high in y_range:
        new_low = max(low + origin[1].item(), mesh_ymin)
        new_high = min(high + origin[1].item(), mesh_ymax)
        if new_low < new_high:
            y_range_clipped.append((new_low, new_high))

    # For z-ranges, we won't clip by mesh bounding box (optional),
    # but we shift them by the origin's Z:
    z_range_shifted = []
    for low, high in z_range:
        new_low = low + origin[2].item()
        new_high = high + origin[2].item()
        z_range_shifted.append((new_low, new_high))

    if not x_range_clipped:
        raise ValueError("No valid x-ranges remain after clipping to bounding box.")
    if not y_range_clipped:
        raise ValueError("No valid y-ranges remain after clipping to bounding box.")
    if not z_range_shifted:
        raise ValueError("z_range cannot be empty.")

    # --- 2) Create a ring of points around (0, 0) in the XY plane to query patch validity
    angle = torch.linspace(0, 2 * np.pi, 10, device=device)
    query_x = []
    query_y = []
    for radius in patch_radius:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)  # (num_radii*10, 1)
    query_y = torch.cat(query_y).unsqueeze(1)  # (num_radii*10, 1)
    # shape: (num_radii*10, 3)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    # Buffers to keep track of invalid patches
    points_ids = torch.arange(cfg.num_patches, device=device)
    flat_patches = torch.zeros(cfg.num_patches, 3, device=device)

    # --- 3) Rejection sampling
    iter_count = 0
    while len(points_ids) > 0 and iter_count < cfg.max_iterations:
        # (A) Sample X and Y from the multiple intervals
        pos_x = uniform_sample_multiple_ranges(x_range_clipped, len(points_ids), device)
        pos_y = uniform_sample_multiple_ranges(y_range_clipped, len(points_ids), device)

        # Store the new (x, y)
        flat_patches[points_ids, 0] = pos_x
        flat_patches[points_ids, 1] = pos_y

        # (B) Raycast from above (z=100, say) straight down
        # Build the 3D query points for each patch
        # shape after unsqueeze: (n_ids, 1, 3) + (query_points) => (n_ids, num_radii*10, 3)
        points = flat_patches[points_ids].unsqueeze(1) + query_points
        # start from 'far above' in Z
        points[..., 2] = 100.0

        # direction is straight down
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        # Flatten for raycasting
        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        # Reshape back to (n_ids, num_radii*10, 3)
        heights = ray_hits.view(points.shape)[..., 2]

        # We'll set the patch center's final Z as the last set of ring hits
        # so that e.g. flat_patches[:, 2] is the Z of the center ring point
        flat_patches[points_ids, 2] = heights[..., -1]

        # (C) Check validity:
        #  1) The patch ring must lie entirely within at least one z-range interval
        #     We'll check each ring point's Z to see if it's within ANY of the z_range.
        #     If the ring fails in all intervals, it's invalid.
        z_ok_mask = torch.zeros(len(points_ids), dtype=torch.bool, device=device)
        for zlow, zhigh in z_range_shifted:
            in_this_range = (heights >= zlow) & (heights <= zhigh)
            # We only say "ok" if *all* ring points are within the range
            # for that interval:
            fully_in_this_interval = in_this_range.all(dim=1)  # shape: (len(points_ids))
            z_ok_mask |= fully_in_this_interval

        #  2) Height difference check
        #     For all ring points, difference between min and max must be <= max_height_diff
        height_diff = heights.max(dim=1)[0] - heights.min(dim=1)[0]

        # Final "not valid" condition
        not_valid = (~z_ok_mask) | (height_diff > cfg.max_height_diff)

        # Filter out the invalid patch IDs
        points_ids = points_ids[not_valid]

        iter_count += 1

    # If we still have leftover invalid patches, raise an error
    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches within the maximum number of iterations!\n"
            f"  Iterations: {iter_count}\n"
            f"  Still invalid patches: {len(points_ids)}\n"
            "  Consider adjusting your ranges or max_height_diff."
        )

    # Return the flat patches, subtracting the origin to keep consistency
    # with the original function's behavior of returning in "mesh frame minus origin".
    return flat_patches - origin


def find_flat_patches(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.FlatPatchSamplingCfg,
) -> torch.Tensor:
    """Finds flat patches of given radius in the input mesh.

    The function finds flat patches of given radius based on the search space defined by the input ranges.
    The search space is characterized by origin in the mesh frame, and the x, y, and z ranges. The x and y
    ranges are used to sample points in the 2D region around the origin, and the z range is used to filter
    patches based on the height of the points.

    The function performs rejection sampling to find the patches based on the following steps:

    1. Sample patch locations in the 2D region around the origin.
    2. Define a ring of points around each patch location to query the height of the points using ray-casting.
    3. Reject patches that are outside the z range or have a height difference that is too large.
    4. Keep sampling until all patches are valid.

    Args:
        wp_mesh: The warp mesh to find patches in.
        num_patches: The desired number of patches to find.
        patch_radius: The radii used to form patches. If a list is provided, multiple patch sizes are checked.
            This is useful to deal with holes or other artifacts in the mesh.
        origin: The origin defining the center of the search space. This is specified in the mesh frame.
        x_range: The range of X coordinates to sample from.
        y_range: The range of Y coordinates to sample from.
        z_range: The range of valid Z coordinates used for filtering patches.
        max_height_diff: The maximum allowable distance between the lowest and highest points
            on a patch to consider it as valid. If the difference is greater than this value,
            the patch is rejected.

    Returns:
        A tensor of shape (num_patches, 3) containing the flat patches. The patches are defined in the mesh frame.

    Raises:
        RuntimeError: If the function fails to find valid patches. This can happen if the input parameters
            are not suitable for finding valid patches and maximum number of iterations is reached.
    """
    # set device to warp mesh device
    device = wp.device_to_torch(wp_mesh.device)

    # resolve inputs to consistent type
    # -- patch radii
    patch_radius = [cfg.patch_radius] if isinstance(cfg.patch_radius, float) else cfg.patch_radius

    # -- origin
    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).to(torch.float).to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    # create ranges for the x and y coordinates around the origin.
    # The provided ranges are bounded by the mesh's bounding box.
    x_range = (
        max(cfg.x_range[0] + origin[0].item(), wp_mesh.points.numpy()[:, 0].min()),
        min(cfg.x_range[1] + origin[0].item(), wp_mesh.points.numpy()[:, 0].max()),
    )
    y_range = (
        max(cfg.y_range[0] + origin[1].item(), wp_mesh.points.numpy()[:, 1].min()),
        min(cfg.y_range[1] + origin[1].item(), wp_mesh.points.numpy()[:, 1].max()),
    )
    z_range = (
        cfg.z_range[0] + origin[2].item(),
        cfg.z_range[1] + origin[2].item(),
    )

    # create a circle of points around (0, 0) to query validity of the patches
    # the ring of points is uniformly distributed around the circle
    angle = torch.linspace(0, 2 * np.pi, 10, device=device)
    query_x = []
    query_y = []
    for radius in patch_radius:
        query_x.append(radius * torch.cos(angle))
        query_y.append(radius * torch.sin(angle))
    query_x = torch.cat(query_x).unsqueeze(1)  # dim: (num_radii * 10, 1)
    query_y = torch.cat(query_y).unsqueeze(1)  # dim: (num_radii * 10, 1)
    # dim: (num_radii * 10, 3)
    query_points = torch.cat([query_x, query_y, torch.zeros_like(query_x)], dim=-1)

    # create buffers
    # -- a buffer to store indices of points that are not valid
    points_ids = torch.arange(cfg.num_patches, device=device)
    # -- a buffer to store the flat patches locations
    flat_patches = torch.zeros(cfg.num_patches, 3, device=device)

    # sample points and raycast to find the height.
    # 1. Reject points that are outside the z_range or have a height difference that is too large.
    # 2. Keep sampling until all points are valid.
    iter_count = 0
    while len(points_ids) > 0 and iter_count < 10000:
        # sample points in the 2D region around the origin
        pos_x = torch.empty(len(points_ids), device=device).uniform_(*x_range)
        pos_y = torch.empty(len(points_ids), device=device).uniform_(*y_range)
        flat_patches[points_ids, :2] = torch.stack([pos_x, pos_y], dim=-1)

        # define the query points to check validity of the patch
        # dim: (num_patches, num_radii * 10, 3)
        points = flat_patches[points_ids].unsqueeze(1) + query_points
        points[..., 2] = 100.0
        # ray-cast direction is downwards
        dirs = torch.zeros_like(points)
        dirs[..., 2] = -1.0

        # ray-cast to find the height of the patches
        ray_hits = raycast_mesh(points.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]
        heights = ray_hits.view(points.shape)[..., 2]
        # set the height of the patches
        # note: for invalid patches, they would be overwritten in the next iteration
        #   so it's safe to set the height to the last value
        flat_patches[points_ids, 2] = heights[..., -1]

        # check validity
        # -- height is within the z range
        not_valid = torch.any(torch.logical_or(heights < z_range[0], heights > z_range[1]), dim=1)
        # -- height difference is within the max height difference
        not_valid = torch.logical_or(not_valid, (heights.max(dim=1)[0] - heights.min(dim=1)[0]) > cfg.max_height_diff)

        # remove invalid patches indices
        points_ids = points_ids[not_valid]
        # increment count
        iter_count += 1

    # check all patches are valid
    if len(points_ids) > 0:
        raise RuntimeError(
            "Failed to find valid patches! Please check the input parameters."
            f"\n\tMaximum number of iterations reached: {iter_count}"
            f"\n\tNumber of invalid patches: {len(points_ids)}"
            f"\n\tMaximum height difference: {cfg.max_height_diff}"
        )

    # return the flat patches (in the mesh frame)
    return flat_patches - origin


def find_flat_patches_by_radius(
    wp_mesh: wp.Mesh,
    origin: np.ndarray | torch.Tensor | tuple[float, float, float],
    cfg: patch_cfg.FlatPatchSamplingByRadiusCfg,
) -> torch.Tensor:
    """Finds flat patches of given radius in the input mesh by sampling patch
    centers in a circular region around `origin`.

    Instead of taking x_range, y_range, this function takes radius_range (min, max)
    and uniformly samples:
       - radius in [radius_range[0], radius_range[1]]
       - angle in [0, 2*pi]
    Then, (x, y) = radius * cos(angle), radius * sin(angle), around `origin`.

    The function uses rejection sampling to ensure patches are valid according to:
      1. The patch ring is fully within the z_range.
      2. The ring’s height difference is no greater than max_height_diff.

    Args:
        wp_mesh: The Warp mesh to find patches in.
        origin: The origin defining the center of the circular region for patch sampling.
                Specified in the mesh frame.
        cfg: A configuration object with the following attributes:
            - num_patches (int): Number of patches to find.
            - patch_radius (float | list[float]): Single or multiple radii for the validation ring.
            - radius_range (tuple[float, float]): The min/max radius used for sampling patch centers.
            - z_range (tuple[float, float]): The min/max Z used for validating patches.
            - max_height_diff (float): The maximum allowed height difference across patch ring.
            - max_iterations (int): The maximum number of iterations for rejection sampling.

    Returns:
        A torch.Tensor of shape (num_patches, 3) containing the patch centers in the mesh frame,
        offset so that `origin` is subtracted at the end.

    Raises:
        RuntimeError: If the function fails to find valid patches within `max_iterations`.
    """
    device = wp.device_to_torch(wp_mesh.device)

    # -- handle patch_radius input
    if isinstance(cfg.patch_radius, float):
        patch_radius = [cfg.patch_radius]
    else:
        patch_radius = cfg.patch_radius

    # -- resolve the origin to a torch tensor (on the correct device)
    if isinstance(origin, np.ndarray):
        origin = torch.from_numpy(origin).float().to(device)
    elif isinstance(origin, torch.Tensor):
        origin = origin.float().to(device)
    else:
        origin = torch.tensor(origin, dtype=torch.float, device=device)

    # -- expand z_range by origin
    z_range_shifted = (cfg.z_range[0] + origin[2].item(), cfg.z_range[1] + origin[2].item())

    # -- create ring (circle) of points around (0, 0) to test patch "flatness"
    # We'll sample 10 angles around the circle for each candidate patch radius
    angle = torch.linspace(0, 2 * np.pi, 10, device=device)  # shape: (10,)
    ring_x = []
    ring_y = []
    for radius in patch_radius:
        ring_x.append(radius * torch.cos(angle))  # shape: (10,)
        ring_y.append(radius * torch.sin(angle))  # shape: (10,)

    ring_x = torch.cat(ring_x).unsqueeze(1)  # shape: (num_radii * 10, 1)
    ring_y = torch.cat(ring_y).unsqueeze(1)  # shape: (num_radii * 10, 1)
    # final ring of shape: (num_radii * 10, 3)
    ring_points = torch.cat([ring_x, ring_y, torch.zeros_like(ring_x)], dim=-1)

    # -- Prepare arrays for sampling
    # We'll keep track of which patch indices are "still invalid" and require more sampling
    patch_ids = torch.arange(cfg.num_patches, device=device)
    flat_patches = torch.zeros((cfg.num_patches, 3), device=device)

    # -- Rejection sampling
    iteration = 0
    while len(patch_ids) > 0 and iteration < cfg.max_iterations:
        # (1) Sample radius in [r_min, r_max]
        r_min, r_max = cfg.radius_range
        cur_radius = torch.empty(len(patch_ids), device=device).uniform_(r_min, r_max)
        # (2) Sample angle in [0, 2*pi]
        cur_angle = torch.empty(len(patch_ids), device=device).uniform_(0, 2 * np.pi)

        # Convert polar -> cartesian
        pos_x = cur_radius * torch.cos(cur_angle)
        pos_y = cur_radius * torch.sin(cur_angle)

        # Store new (x, y) in our patch buffer
        flat_patches[patch_ids, 0] = pos_x + origin[0]
        flat_patches[patch_ids, 1] = pos_y + origin[1]

        # Raycast ring points from "far above" in Z
        # shape for ring offset: (len(patch_ids), num_radii * 10, 3)
        ring_in_world = flat_patches[patch_ids].unsqueeze(1) + ring_points
        ring_in_world[..., 2] = 100.0  # set starting Z for the ray
        # directions: straight down
        dirs = torch.zeros_like(ring_in_world)
        dirs[..., 2] = -1.0

        # Flatten for raycasting
        ray_hits = raycast_mesh(ring_in_world.view(-1, 3), dirs.view(-1, 3), wp_mesh)[0]  # shape: (N, 3)

        # Reshape back to (len(patch_ids), num_radii * 10, 3)
        ring_hits_3d = ray_hits.view(ring_in_world.shape)
        # We'll define the patch center's final Z from the last ring point's Z
        flat_patches[patch_ids, 2] = ring_hits_3d[..., -1, 2]

        # (3) Check validity:
        #  -- all ring points must lie fully in the z_range
        heights = ring_hits_3d[..., 2]  # shape: (len(patch_ids), num_radii*10)
        out_of_range = (heights < z_range_shifted[0]) | (heights > z_range_shifted[1])
        #  -- height difference must be within max_height_diff
        height_diff = heights.max(dim=1)[0] - heights.min(dim=1)[0]
        # "not valid" if out of range or if height diff is too large
        not_valid = out_of_range.any(dim=1) | (height_diff > cfg.max_height_diff)

        # Keep only the patches that are invalid for another iteration
        patch_ids = patch_ids[not_valid]
        iteration += 1

    if len(patch_ids) > 0:
        # We ran out of iterations but still have invalid patches
        raise RuntimeError(
            f"Failed to find valid patches within {cfg.max_iterations} iterations.\n"
            f"Still invalid patches: {len(patch_ids)}.\n"
            "Consider relaxing your constraints or increasing max_iterations."
        )

    # Return patch centers in the "mesh frame minus origin"
    # (i.e., subtract origin so that final returned coords are consistent
    #  with the original code's behavior)
    return flat_patches - origin
