# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'uwlab_tasks' python package."""

import os
import platform
import sys
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = []

is_linux_x86_64 = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"

wheel_by_py = {
    "cp311": (
        "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
        "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp311-cp311-linux_x86_64.whl"
    ),
    "cp310": (
        "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
        "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp310-cp310-linux_x86_64.whl"
    ),
}

if is_linux_x86_64 and py in wheel_by_py:
    INSTALL_REQUIRES.append(f"pytorch3d @ {wheel_by_py[py]}")

# Installation operation
setup(
    name="uwlab_tasks",
    author="UW and Isaac Lab Project Developers",
    maintainer="UW and Isaac Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=["uwlab_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
