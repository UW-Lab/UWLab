# Copyright (c) 2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path

from pxr import Sdf


def write_usd_entry_layer(generated_path: str, output_path: str) -> str:
    """Expose structured importer output without relocating its assets.

    Args:
        generated_path: Root USD layer produced by the importer.
        output_path: Requested USD entry-layer filename.

    Returns:
        Absolute path to the requested entry layer.
    """
    generated_path = os.path.abspath(generated_path)
    output_path = os.path.abspath(output_path)
    source = Sdf.Layer.FindOrOpen(generated_path)
    if source is None:
        raise ValueError(f"Unable to open generated USD layer: {generated_path}")
    if generated_path == output_path:
        return output_path
    if os.path.exists(output_path) and os.path.samefile(generated_path, output_path):
        raise ValueError("Requested output aliases the generated layer; choose a distinct filename.")
    entry = Sdf.Layer.CreateAnonymous()
    for key in source.pseudoRoot.ListInfoKeys():
        if key not in {"subLayers", "subLayerOffsets"}:
            entry.pseudoRoot.SetInfo(key, source.pseudoRoot.GetInfo(key))
    entry.subLayerPaths = [Path(os.path.relpath(generated_path, os.path.dirname(output_path))).as_posix()]
    if not entry.Export(output_path):
        raise RuntimeError(f"Unable to write USD entry layer: {output_path}")
    return output_path
