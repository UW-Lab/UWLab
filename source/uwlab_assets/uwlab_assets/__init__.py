# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import os
import toml

# Conveniences to other module directories via relative paths
UWLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
UWLAB_ASSETS_DATA_DIR = os.path.join(UWLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
UWLAB_ASSETS_METADATA = toml.load(os.path.join(UWLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""
UWLAB_CLOUD_ASSETS_DIR = "https://uwlab-assets.s3.us-west-004.backblazeb2.com"
# Configure the module-level variables
__version__ = UWLAB_ASSETS_METADATA["package"]["version"]
