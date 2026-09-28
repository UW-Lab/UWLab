# Copyright (c) 2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Sdf, Usd, UsdGeom

from scripts.tools.usd_output import write_usd_entry_layer

ROOT = Path(__file__).resolve().parents[3]


class ConverterCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return vars(self)


ConverterCfg.JointDriveCfg = ConverterCfg
ConverterCfg.PDGainsCfg = ConverterCfg


def make_imported_asset(directory):
    directory.mkdir(parents=True)
    part = Usd.Stage.CreateNew(str(directory / "part.usda"))
    prim = UsdGeom.Cube.Define(part, "/Robot/Part")
    prim.CreateSizeAttr(2.5)
    part.GetRootLayer().Save()
    generated = Usd.Stage.CreateNew(str(directory / "robot.usda"))
    robot = generated.DefinePrim("/Robot", "Xform")
    generated.SetDefaultPrim(robot)
    UsdGeom.SetStageUpAxis(generated, "Y")
    UsdGeom.SetStageMetersPerUnit(generated, 0.01)
    generated.GetRootLayer().subLayerPaths = ["part.usda"]
    generated.GetRootLayer().Save()
    return directory / "robot.usda"


@pytest.mark.parametrize("kind", ["urdf", "mjcf"])
@pytest.mark.parametrize("extension", ["usd", "usda", "usdc"])
def test_converter_main_honors_requested_filename(tmp_path, kind, extension):
    source = tmp_path / f"input.{kind}"
    source.touch()
    output = tmp_path / "output" / f"custom name.{extension}"
    previews = []

    def convert(cfg):
        path = make_imported_asset(Path(cfg.usd_dir) / "robot")
        return SimpleNamespace(usd_path=str(path))

    args = SimpleNamespace(
        input=str(source),
        output=str(output),
        fix_base=False,
        merge_joints=False,
        joint_stiffness=100.0,
        joint_damping=1.0,
        joint_target_type="position",
        merge_mesh=False,
        collision_from_visuals=False,
        collision_type="convexHull",
        self_collision=False,
        import_physics_scene=False,
    )
    namespace = {
        "args_cli": args,
        "os": os,
        "check_file_path": os.path.isfile,
        "print_dict": lambda *a, **kw: None,
        "PhysicsCfg": object,
        "launch_simulation": lambda **kw: contextlib.nullcontext(None),
        "preview": lambda path, cfg: previews.append(path),
        "write_usd_entry_layer": write_usd_entry_layer,
        "UrdfConverterCfg": ConverterCfg,
        "MjcfConverterCfg": ConverterCfg,
        "UrdfConverter": convert,
        "MjcfConverter": convert,
    }
    tree = ast.parse((ROOT / f"scripts/tools/convert_{kind}.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
    namespace["main"]()
    assert output.is_file()
    assert previews == [str(output)]
    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/Robot")
    assert UsdGeom.GetStageUpAxis(stage) == "Y"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.01
    assert UsdGeom.Cube(stage.GetPrimAtPath("/Robot/Part")).GetSizeAttr().Get() == 2.5
    assert stage.GetRootLayer().subLayerPaths == ["robot/robot.usda"]


def test_same_output_layer_is_not_rewritten(tmp_path):
    generated = make_imported_asset(tmp_path / "robot")
    before = generated.read_bytes()
    assert write_usd_entry_layer(str(generated), str(generated)) == str(generated)
    assert generated.read_bytes() == before


@pytest.mark.parametrize("symlink", [False, True])
def test_output_alias_cannot_overwrite_generated_layer(tmp_path, symlink):
    generated = make_imported_asset(tmp_path / "robot")
    output = tmp_path / "alias.usda"
    if symlink:
        output.symlink_to(generated)
    else:
        os.link(generated, output)
    before = generated.read_bytes()
    with pytest.raises(ValueError, match="aliases"):
        write_usd_entry_layer(str(generated), str(output))
    assert generated.read_bytes() == before
