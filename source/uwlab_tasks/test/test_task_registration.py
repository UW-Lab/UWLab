# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys


def test_task_registration_without_simulator():
    """Register discoverable task IDs without importing their MDP implementations."""
    program = """
import ast
import importlib.abc
import sys
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from isaaclab.app import AppLauncher

class RejectTaskMdp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('uwlab_tasks.') and '.mdp' in fullname:
            raise AssertionError(f'Task registration eagerly imported {fullname}')

sys.meta_path.insert(0, RejectTaskMdp())
with patch.object(AppLauncher, '__init__', side_effect=AssertionError('Simulator launch forbidden')):
    import uwlab_tasks

expected = set()
root = Path(uwlab_tasks.__file__).parent
for path in root.rglob('__init__.py'):
    parents = [parent for parent in path.parents if parent != root and root in parent.parents]
    if not all((parent / '__init__.py').is_file() for parent in parents):
        continue
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == 'gym.register':
            for keyword in node.keywords:
                if keyword.arg == 'id' and isinstance(keyword.value, ast.Constant):
                    expected.add(keyword.value.value)
assert expected
assert expected <= set(gym.registry), sorted(expected - set(gym.registry))
assert 'OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0' in expected
assert not any(name.startswith('uwlab_tasks.') and '.mdp' in name for name in sys.modules)
print(f'Registered {len(expected)} declared task IDs without loading their MDPs')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
