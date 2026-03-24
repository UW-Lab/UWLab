# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
Zarr Dataset File Handler
This module provides a Zarr-based dataset file handler that works with
all manager-based environments in Isaac Lab, compatible with diffusion policy format.
DESIGN OVERVIEW:
==================
The ZarrDatasetFileHandler is designed to automatically extract and record episode data
from Isaac Lab environments to Zarr format using the ReplayBuffer structure from diffusion policy.
It uses a configuration-based approach to determine which observations and actions to record.
The Zarr format expects the dataset to contain:
- data/ (group containing all episode data)
  - actions (array with shape [T, action_dim])
  - obs/ (group containing observations)
    - observation_key_1 (array with shape [T, ...])
    - observation_key_2 (array with shape [T, ...])
    - ... (each observation stored as separate key)
  - rewards (array with shape [T])
  - dones (array with shape [T])
- meta/ (group containing metadata)
  - episode_ends (array with episode end indices)
KEY FEATURES:
============
1. CONFIGURATION-DRIVEN:
   - Uses environment observation and action managers automatically
   - Supports both regular observations and state observations
2. AUTOMATIC FEATURE EXTRACTION:
   - Analyzes environment's observation and action managers automatically
   - Handles nested observation structures with group-based access
   - Automatically detects and processes video/image features
   - Supports different action term types
3. FLEXIBLE OBSERVATION HANDLING:
   - All observations: saved as "obs/{key}" (separate keys)
   - Support for observations from different groups (policy, critic, etc.)
   - Automatic tensor shape analysis and feature specification
4. UNIVERSAL COMPATIBILITY:
   - Works with any manager-based environment
   - No hardcoded assumptions about observation or action structure
   - Adapts to different environment types automatically
5. EFFICIENT STORAGE:
   - Uses Zarr compression for efficient storage
   - Supports chunking for large datasets
   - Compatible with diffusion policy ReplayBuffer
6. IMAGE SUPPORT:
   - Automatically detects image observations (RGB format)
   - Saves image observations as zarr arrays with optimized chunking
   - Uses blosc compression for fast access during training
USAGE PATTERNS:
==============
1. Basic Usage (Zero Configuration):
   ```python
   # Automatically records all available observations
   handler = ZarrDatasetFileHandler()
   handler.create("dataset.zarr")
   ```
2. Image Support:
   ```python
   # Automatically detects and processes image data
   # Handles [B, H, W, C] format and stores efficiently as zarr arrays
   ```
USAGE:
=====
The handler automatically detects and records all available observations from the environment.
No configuration is required, but you can optionally provide configuration for customization:

```python
# Basic usage - automatically records all observations
handler = ZarrDatasetFileHandler()
handler.create("dataset.zarr")
```

This handler provides a streamlined way to record Isaac Lab environments to Zarr datasets
with zero configuration required, compatible with diffusion policy format.
"""

import numpy as np
import shutil
import torch
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numcodecs
import zarr
from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase
from isaaclab.utils.datasets.episode_data import EpisodeData


class ZarrDatasetFileHandler(DatasetFileHandlerBase):
    """Zarr dataset file handler for storing episode data.

    Automatically records episode data to Zarr format compatible with diffusion policy ReplayBuffer.
    Optimized for large-scale datasets with efficient chunking and compression.
    Saves image observations as zarr arrays with memory-optimized chunking.

    Args:
        chunk_size: Chunk size for temporal dimension (default: 1000 for memory efficiency)
        image_chunk_size: Chunk size for image arrays (default: 100 for memory efficiency)
        image_keys: List of observation keys that should be treated as images (default: None, auto-detect)

    Example Usage:
        ```python
        handler = ZarrDatasetFileHandler()
        handler.create("dataset.zarr", env_name="my_env")
        ```
    """

    def __init__(self, chunk_size: int = 5000, image_chunk_size: int = 50, image_keys: list[str] | None = None):
        """Initialize the Zarr dataset file handler.

        Args:
            chunk_size: Chunk size for temporal dimension (non-image data)
            image_chunk_size: Chunk size for image arrays (memory-optimized)
            image_keys: List of observation keys that should be treated as images
        """
        self._dataset = None
        self._dataset_path = None
        self._env_name = None
        self._episode_count = 0
        self._compressor = None
        self._chunk_size = chunk_size
        self._image_chunk_size = image_chunk_size
        self._image_keys = image_keys

        # Set up compression for all data
        self._compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)

    def create(self, file_path: str, env_name: str | None = None, overwrite: bool = True):
        """Create a new dataset file.

        Args:
            file_path: Path to the dataset file (must end with .zarr)
            env_name: Optional name for the environment (used in metadata)
            overwrite: If True, silently remove an existing dataset at this path.
                If False, raise ValueError when the path already exists.
        """
        if not file_path.endswith(".zarr"):
            raise ValueError("Dataset file path must end with .zarr")

        self._dataset_path = Path(file_path)

        if self._dataset_path.exists():
            if not overwrite:
                raise ValueError(
                    f"Dataset already exists at {self._dataset_path}. Pass overwrite=True or remove it manually."
                )
            print(f"Removing existing dataset at {self._dataset_path}")
            shutil.rmtree(self._dataset_path)

        # Initialize environment name
        self._env_name = env_name or "isaac_lab_env"

        # Use default task description
        self._task_description = "Custom task"

        # Create Zarr dataset structure
        try:
            # Create root group
            self._dataset = zarr.group(str(self._dataset_path))

            # Create data group (will be used later)
            self._dataset.create_group("data")

            # Create meta group
            meta_group = self._dataset.create_group("meta")

            # Initialize episode_ends array
            meta_group.zeros("episode_ends", shape=(0,), dtype=np.int64, compressor=None)

            # Add environment name to metadata
            self._dataset.attrs["env_name"] = self._env_name
            self._dataset.attrs["task_description"] = self._task_description

        except Exception as e:
            raise RuntimeError(f"Failed to create Zarr dataset: {e}")

        self._episode_count = 0

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        raise NotImplementedError("Open not implemented for Zarr handler")

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return self._env_name

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        if self._dataset is None:
            return []
        return [f"episode_{i:06d}" for i in range(self._episode_count)]

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
        """
        if self._dataset is None or episode.is_empty():
            return

        # Convert Isaac Lab episode data to Zarr format and save
        self._convert_and_save_episode(episode)

        # Only increment episode count if using default indexing
        if demo_id is None:
            self._episode_count += 1

    def _convert_and_save_episode(self, episode: EpisodeData):
        """Convert Isaac Lab episode data to Zarr format and save it."""
        episode_dict = episode.data

        if "actions" not in episode_dict or "obs" not in episode_dict:
            raise ValueError("Episode must contain actions and observations")

        num_frames = episode_dict["actions"].shape[0]

        # Process all observations together
        obs_dict = episode_dict["obs"]
        processed_obs = self._process_observations_for_episode(obs_dict)

        episode_data = {
            "actions": episode_dict["actions"].cpu().numpy(),
            "obs": processed_obs,
            "rewards": episode_dict.get("rewards", torch.zeros(num_frames)).cpu().numpy(),
            "dones": episode_dict.get("dones", torch.cat([torch.zeros(num_frames - 1), torch.ones(1)])).cpu().numpy(),
        }

        # Save episode data to Zarr
        self._save_episode_to_zarr(episode_data)

    def _process_observations_for_episode(self, obs_dict: dict[str, Any]) -> dict[str, np.ndarray]:
        """Process observations for an entire episode."""
        episode_obs = {}
        for obs_key, value in obs_dict.items():
            try:
                episode_obs[obs_key] = value.cpu().numpy()
            except Exception as e:
                print(f"Error processing observation '{obs_key}': {e}")
        return episode_obs

    def _save_episode_to_zarr(self, episode_data: dict[str, Any]):
        """Save episode data to Zarr format."""
        if self._dataset is None:
            raise RuntimeError("Dataset not initialized")

        data_group = self._dataset["data"]
        meta_group = self._dataset["meta"]
        episode_ends = meta_group["episode_ends"]

        # Get current episode end and calculate new end
        current_end = int(episode_ends[-1]) if len(episode_ends) > 0 else 0
        episode_length = len(episode_data["actions"])
        new_end = current_end + episode_length

        # Extend arrays and add episode data
        for key, value in episode_data.items():
            if key == "obs":
                for obs_key, obs_value in value.items():
                    self._extend_or_create_array(data_group, f"obs/{obs_key}", obs_value, episode_length)
            else:
                self._extend_or_create_array(data_group, key, value, episode_length)

        # Update episode ends
        episode_ends.resize(len(episode_ends) + 1)
        episode_ends[-1] = int(new_end)

    def _extend_or_create_array(self, group, key: str, data: np.ndarray, episode_length: int):
        """Extend existing array or create new one with episode data."""
        if key in group:
            # Extend existing array
            arr = group[key]
            arr.resize(arr.shape[0] + episode_length, *arr.shape[1:])
            arr[-episode_length:] = data
        else:
            # Create new array with optimized chunking
            if self._is_image_array(data):
                # Use image-optimized chunking and compression
                chunks = (self._image_chunk_size,) + data.shape[1:]
            else:
                # Use standard chunking for non-image data
                chunks = (self._chunk_size,) + data.shape[1:]

            group.create_dataset(key, data=data, chunks=chunks, dtype=data.dtype, compressor=self._compressor)

    def _is_image_array(self, data: np.ndarray) -> bool:
        """Check if array is an image array (4D with shape [T, H, W, C])."""
        return data.ndim == 4 and data.shape[-1] in [1, 3, 4]

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        """Load episode data from the file."""
        raise NotImplementedError("Load episode not implemented for Zarr handler")

    def flush(self):
        """Flush any pending data to disk."""
        # Zarr handles flushing automatically
        pass

    def close(self):
        """Close the dataset file handler."""
        # Clear references
        self._dataset = None

    def add_env_args(self, env_args: dict):
        pass
