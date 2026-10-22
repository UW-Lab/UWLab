# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'uwlab_rl' python package."""

import itertools
import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    # rsl_rl's WandbSummaryWriter still passes ``wandb.Settings(start_method="thread")``;
    # wandb removed that field and its Settings model rejects unknown fields, so newer
    # releases fail at the first log call. 0.19.x is the last series that accepts it.
    "wandb>=0.19.6,<0.20",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

# Extra dependencies for RL agents
# Pinned to a commit, not a branch: an unpinned git dependency makes a rebuild
# silently install a different API than the one this code was written against.
# Must be a commit on UW-Lab/rsl_rl with the rsl-rl >= 5.0 API that Isaac Lab 3.0's
# isaaclab_rl requires, including HeteroscedasticGaussianDistribution (rsl-rl 5.3).
# Bump together with the Isaac Lab commit pinned in uwlab.sh.
# Released UW-Lab/rsl_rl integration after UW-Lab/rsl_rl#6 merged.
RSL_RL_REPO = "https://github.com/UW-Lab/rsl_rl.git"
RSL_RL_COMMIT = "2c3bf18001a5e2a78527e9ea368b7ea31700a2c5"  # uw-v5.4.1 (UW-Lab/rsl_rl#6)
EXTRAS_REQUIRE = {
    "rsl-rl": [
        # Update this pin alongside compatible UWLab changes.
        f"rsl-rl-lib @ git+{RSL_RL_REPO}@{RSL_RL_COMMIT}",
    ],
}

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

# Installation operation
setup(
    name="uwlab_rl",
    author="UW Lab Project Developers",
    maintainer="UW Lab Project Developers",
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.12,<3.13",
    install_requires=INSTALL_REQUIRES,
    dependency_links=PYTORCH_INDEX_URL,
    extras_require=EXTRAS_REQUIRE,
    packages=["uwlab_rl"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 6.1.0",
    ],
    zip_safe=False,
)
