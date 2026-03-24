# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compute the bottom offset of a USD asset.

The bottom offset is the distance from the asset's origin to the lowest
point of any mesh in the USD file. This value is used to spawn objects
flush on the table surface.

Usage:
    python scripts_v2/tools/compute_bottom_offset.py /path/to/asset.usd
"""

import argparse
import numpy as np

from pxr import Usd, UsdGeom


def get_world_points(mesh: UsdGeom.Mesh, time=Usd.TimeCode.Default()) -> np.ndarray:
    """Get mesh vertices transformed to world space."""
    points = np.array(mesh.GetPointsAttr().Get(time), dtype=np.float64)
    if len(points) == 0:
        return points
    xform = UsdGeom.Xformable(mesh.GetPrim())
    world_transform = xform.ComputeLocalToWorldTransform(time)
    mat = np.array(world_transform, dtype=np.float64).T
    ones = np.ones((len(points), 1), dtype=np.float64)
    homogeneous = np.hstack([points, ones])
    return (homogeneous @ mat.T)[:, :3]


def compute_bottom_offset(usd_path: str, verbose: bool = False) -> float:
    """Compute distance from origin to lowest mesh vertex in the USD."""
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise FileNotFoundError(f"Cannot open USD: {usd_path}")

    all_points = []
    mesh_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            pts = get_world_points(mesh)
            if len(pts) > 0:
                all_points.append(pts)
                mesh_count += 1

    if not all_points:
        raise ValueError(f"No meshes found in {usd_path}")

    pts = np.vstack(all_points)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    min_z = bbox_min[2]

    if verbose:
        print(f"Meshes found: {mesh_count}")
        print(f"Total vertices: {len(pts)}")
        print(f"Bounding box min: ({bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f})")
        print(f"Bounding box max: ({bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f})")
        size = bbox_max - bbox_min
        print(f"Size: ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})")

    return abs(min_z)


def main():
    parser = argparse.ArgumentParser(description="Compute bottom offset of a USD asset.")
    parser.add_argument("usd_path", type=str, help="Path to the USD file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print bounding box details.")
    args = parser.parse_args()

    offset = compute_bottom_offset(args.usd_path, verbose=args.verbose)
    print(f"bottom_offset: {offset:.6f}")


if __name__ == "__main__":
    main()
