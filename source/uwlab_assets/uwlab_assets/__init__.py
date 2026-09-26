# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing asset and sensor configurations."""

import logging
import os
import toml
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Conveniences to other module directories via relative paths
UWLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""
UWLAB_ASSETS_DATA_DIR = os.path.join(UWLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""
UWLAB_ASSETS_METADATA = toml.load(os.path.join(UWLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

UWLAB_CLOUD_ASSETS_REPO = "https://huggingface.co/datasets/UW-Lab/uwlab-assets"
"""HuggingFace dataset repository holding the cloud assets."""

UWLAB_CLOUD_ASSETS_REVISION = "83860532010b2737aa80d6e8621235ee554186f0"  # branch isaaclab3-ea
"""Pinned commit of :data:`UWLAB_CLOUD_ASSETS_REPO`.

Pinned rather than a branch name so that asset changes on HuggingFace are opt-in: bump this
constant deliberately when new assets or datasets are published. Isaac Lab 3.0 EA assets and
converted state checkpoints live on ``isaaclab3-ea``. The ``isaaclab3`` branch retains beta
checkpoints with the legacy observation order; ``main`` keeps the Isaac Lab 2.x files.
"""

UWLAB_CLOUD_ASSETS_DIR = f"{UWLAB_CLOUD_ASSETS_REPO}/resolve/{UWLAB_CLOUD_ASSETS_REVISION}"


def _extract_revision_and_relative_path(url: str) -> tuple[str, str]:
    """Split a HuggingFace resolve URL into ``(revision, repo-relative path)``.

    Example:
        ``https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/isaaclab3/Props/Custom/Peg/peg.usd``
        -> ``("isaaclab3", "Props/Custom/Peg/peg.usd")``
    """
    parsed = urlparse(url)
    parts = parsed.path.strip("/").split("/")
    try:
        idx = parts.index("resolve")
        return parts[idx + 1], "/".join(parts[idx + 2 :])
    except (ValueError, IndexError):
        return "", parsed.path.strip("/")


def _extract_relative_path(url: str) -> str:
    """Strip the HuggingFace resolve-URL prefix, returning the repo-relative path."""
    return _extract_revision_and_relative_path(url)[1]


def _urlretrieve_quiet(url: str, dest: str) -> None:
    """Download *url* to *dest* silently."""
    req = urllib.request.urlopen(url)
    chunk_size = 1 << 16  # 64 KiB
    with open(dest, "wb") as f:
        while True:
            chunk = req.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    req.close()


def resolve_cloud_path(path: str) -> str:
    """Resolve a cloud asset path to a local file, downloading if needed.

    * Local paths (including already-cached files) are returned immediately.
    * HTTPS URLs are downloaded once to ``~/.cache/uwlab/assets/<revision>/<relative>``
      and the local cached path is returned on subsequent calls. The revision is part
      of the key so that bumping :data:`UWLAB_CLOUD_ASSETS_REVISION` never serves a
      file cached from another revision.
    * Downloads are atomic (write to a temp file, then ``os.rename``).
    """
    if not path.startswith(("http://", "https://")):
        return path

    revision, rel = _extract_revision_and_relative_path(path)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "uwlab", "assets", revision)
    local = os.path.join(cache_dir, rel)

    if os.path.isfile(local):
        return local

    os.makedirs(os.path.dirname(local), exist_ok=True)
    tmp = f"{local}.tmp.{os.getpid()}"
    try:
        logger.info(f"Downloading {rel} ...")
        _urlretrieve_quiet(path, tmp)
        os.rename(tmp, local)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    return local


# Configure the module-level variables
__version__ = UWLAB_ASSETS_METADATA["package"]["version"]
