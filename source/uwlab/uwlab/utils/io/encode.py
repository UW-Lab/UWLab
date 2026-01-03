# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for file I/O with yaml."""

import io
import logging
import pickle
import yaml
from typing import Any

from isaaclab.utils import class_to_dict

logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def encode_pickle(data: Any) -> bytes:
    """Saves data into a pickle file safely.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    try:
        # Serialize the dictionary to a bytes buffer using pickle
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)  # Reset buffer pointer to the beginning

        serialized_data = buffer.read()
        logger.info("Data successfully serialized to pickle format.")
        return serialized_data

    except pickle.PicklingError as pe:
        logger.error(f"PicklingError: Failed to serialize data: {pe}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during pickle serialization: {e}")
        raise


def encode_dict_to_yaml(data: dict | object, sort_keys: bool = False) -> bytes:
    """Serializes data into a YAML-formatted byte string.

    Args:
        data (dict | object): The data to serialize. Can be a dictionary or any serializable object.
        sort_keys (bool, optional): Whether to sort the keys in the output YAML. Defaults to False.

    Returns:
        bytes: The serialized YAML byte string.

    Raises:
        TypeError: If the data provided is not serializable to a dictionary.
        yaml.YAMLError: If an error occurs during YAML serialization.
        Exception: For any other exceptions that occur during serialization.
    """
    try:
        # Convert data into a dictionary if it's not already one
        if not isinstance(data, dict):
            data = class_to_dict(data)  # Assumes class_to_dict is defined elsewhere
            logger.debug("Converted object to dictionary for YAML serialization.")

        # Serialize the dictionary to a YAML-formatted string
        yaml_str = yaml.dump(data, sort_keys=sort_keys)
        logger.info("Data successfully serialized to YAML format.")

        # Encode the YAML string to bytes (UTF-8)
        yaml_bytes = yaml_str.encode("utf-8")
        logger.debug("YAML string encoded to bytes successfully.")

        return yaml_bytes

    except TypeError as type_error:
        logger.error(f"TypeError: Data provided is not serializable to a dictionary: {type_error}")
        raise

    except yaml.YAMLError as yaml_error:
        logger.error(f"YAMLError: Failed to serialize data to YAML: {yaml_error}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during YAML serialization: {e}")
        raise
