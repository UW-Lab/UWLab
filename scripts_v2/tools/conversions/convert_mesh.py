# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.utils.dict import print_dict

from uwlab_assets.props.workbench.workbench_conversion_cfg import WORKBENCH_CONVERSION_CFG

from uwlab.sim.converters.mesh_converter import MeshConverter


def main():
    for mesh_converter_cfg in WORKBENCH_CONVERSION_CFG:
        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input Mesh file: {mesh_converter_cfg.asset_path}")
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)  # type: ignore
        print("-" * 80)
        print("-" * 80)

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        # print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
