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
INSTALL_REQUIRES = [
    # OmniReset dependencies
    "rtree",
    "zarr==2.18.3",
    "numcodecs==0.13.1",
    "cmaes",
]

# pytorch3d is only needed by the collision analyzer, which runs in the
# dataset-generation tasks (reset states, grasp sampling, partial assemblies).
# omnireset/mdp/utils.py imports it lazily so the RL tasks stay usable without it,
# and it is exposed as the ``collision`` extra (``uwlab.sh -i`` installs it).
#
# There is no sdist fallback on purpose -- building pytorch3d from source needs a
# matching CUDA toolkit and takes a long time -- so a wheel is only added when a
# prebuilt one exists for this python/CUDA combination. The build is tied to the
# torch it was compiled against, so keep this table in sync with the torch pin in
# uwlab.sh (`ensure_cuda_torch`) and check a new pin actually loads.
#
#   cp310/cp311 -> 0.7.8 + pt2.7.0 + cu128   (Isaac Lab 2.3.2 stack)
#   cp312       -> 0.7.9 + pt2.10.0 + cu128  (Isaac Lab 3.0 stack)
#
# cp312 needs 0.7.9: 0.7.8's newest cp312 build is pt2.8.0. No pt2.11 build is
# published for either release, so cp312 takes the newest (pt2.10.0); it loads
# against the torch 2.11.0 that Isaac Sim 6.0.1 pins (CUDA ops verified).
is_linux_x86_64 = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
py = f"cp{sys.version_info.major}{sys.version_info.minor}"

wheel_by_py = {
    "cp312": (
        "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.9/"
        "pytorch3d-0.7.9%2Bpt2.10.0cu128-cp312-cp312-linux_x86_64.whl"
    ),
    "cp311": (
        "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
        "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp311-cp311-linux_x86_64.whl"
    ),
    "cp310": (
        "https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8/"
        "pytorch3d-0.7.8%2Bpt2.7.0cu128-cp310-cp310-linux_x86_64.whl"
    ),
}

EXTRAS_REQUIRE = {
    "collision": [f"pytorch3d @ {wheel_by_py[py]}"] if is_linux_x86_64 and py in wheel_by_py else [],
}

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
    extras_require=EXTRAS_REQUIRE,
    packages=["uwlab_tasks"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
