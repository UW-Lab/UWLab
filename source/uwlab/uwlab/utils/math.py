# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional


def create_axis_remap_function(forward: str = "x", left: str = "y", up: str = "z", device: str = "cpu"):
    """Creates a function to remap and reorient the axes of input tensors.

    This function generates a new function that remaps the axes of input tensors
    according to the specified forward, left, and up axes. The resulting
    function can be used to reorder and reorient both positional and rotational data.

    Args:
        forward: The axis that should be mapped to the primary axis (e.g., "x" or "-x").
        left: The axis that should be mapped to the secondary axis (e.g., "y" or "-y").
        up: The axis that should be mapped to the tertiary axis (e.g., "z" or "-z").
        device: The device on which the resulting tensors will be processed (e.g., "cpu" or "cuda").

    Returns:
        A function that takes two tensors as inputs:
        - positions (torch.Tensor): A positional tensor of shape (N, 3).
        - rotations (torch.Tensor): A rotational tensor of shape (N, 3).

        The output function returns a tuple containing:
        - new_positions (torch.Tensor): The remapped positional tensor of shape (N, 3).
        - new_rotations (torch.Tensor): The remapped rotational tensor of shape (N, 3).

    Example:
        .. code-block:: python

            remap_fn = create_axis_remap_function(forward="z", left="-x", up="y")
            new_positions, new_rotations = remap_fn(positions, rotations)
    """
    # Define the mapping from axis labels to indices
    axis_to_index = {"x": 0, "y": 1, "z": 2}

    # Create a mapping for the new axis order
    new_axis_order = [forward, left, up]
    indices = []
    signs = []

    for axis in new_axis_order:
        sign = 1
        if axis.startswith("-"):
            sign = -1
            axis = axis[1:]

        index = axis_to_index[axis]
        indices.append(index)
        signs.append(sign)

    signs = torch.tensor(signs, device=device)

    def remap_positions_and_rotations(positions: torch.Tensor | None, rotations: torch.Tensor | None) -> tuple:
        """Remaps the positions and rotations tensors according to the specified axis order.

        Args:
            positions: Input positional tensor of shape (N, 3).
            rotations: Input rotational tensor of shape (N, 3).

        Returns:
            A tuple containing the remapped positional and rotational tensors, both of shape (N, 3).
            if respective input is not None
        """
        # Apply sign first, then reorder the axes
        new_positions = (positions * signs)[:, indices] if positions is not None else None
        new_rotations = (rotations * signs)[:, indices] if rotations is not None else None
        return new_positions, new_rotations

    return remap_positions_and_rotations
