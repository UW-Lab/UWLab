# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import os

import isaacsim.core.utils.prims as prim_utils
import omni
import omni.kit.commands
from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from isaaclab.sim.schemas import schemas
from isaaclab.sim.utils import clone, export_prim_to_file, get_all_matching_child_prims, safe_set_attribute_on_usd_prim
from isaacsim.coreutils.extensions import enable_extension
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

from .mesh_converter_cfg import MeshConverterCfg


def apply_material_binding(stage: Usd.Stage, prim_path: str, material_path: str) -> None:
    """Applies a material to a given prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return

    # Ensure that the MaterialBindingAPI is applied
    if not prim.HasAPI(UsdShade.MaterialBindingAPI):
        UsdShade.MaterialBindingAPI.Apply(prim)

    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    # Unbind any previous materials before binding a new one
    material_binding_api.UnbindAllBindings()
    material_binding_api.Bind(
        material=material,
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose=UsdShade.Tokens.allPurpose,
    )


def apply_collision_properties(stage: Usd.Stage, prim_path: str, cfg: MeshConverterCfg) -> None:
    """Applies collision properties to a given prim."""
    collision_prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    if not collision_prim:
        return

    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)  # type: ignore
    mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
    # -- Collider properties such as offset, scale, etc.
    if cfg.collision_props:
        schemas.define_collision_properties(prim_path=collision_prim.GetPath(), cfg=cfg.collision_props, stage=stage)
    visibility_attr = collision_prim.GetAttribute("visibility")
    if visibility_attr:
        visibility_attr.Set("invisible", time=Usd.TimeCode.Default())


def remove_all_other_prims(stage: Usd.Stage, keep_prefix_paths: list[str]) -> None:
    """Removes all prims except /World and /World/<keep_prefix_paths>.

    Args:
        stage: The USD stage.
        keep_prefix_paths: A list of path prefixes to keep. All prims under these prefixes
                           (and under /World) will not be removed.
    """
    prims_to_remove = []
    root_path = stage.GetDefaultPrim().GetPath()
    for prim in stage.Traverse():
        ppath = prim.GetPath()

        if ppath == root_path:
            continue
        # Check if the prim matches any of the keep prefixes
        if not any(ppath.HasPrefix(Sdf.Path(prefix)) for prefix in keep_prefix_paths):
            prims_to_remove.append(ppath)

    for ppath in prims_to_remove:
        stage.RemovePrim(ppath)


@clone
def spawn_preview_surface(prim_path: str, cfg) -> Usd.Prim:
    """Create a preview surface prim and override the settings with the given config.

    A preview surface is a physically-based surface that handles simple shaders while supporting
    both *specular* and *metallic* workflows. All color inputs are in linear color space (RGB).
    For more information, see the `documentation <https://openusd.org/release/spec_usdpreviewsurface.html>`__.

    The function calls the USD command `CreatePreviewSurfaceMaterialPrim`_ to create the prim.

    .. _CreatePreviewSurfaceMaterialPrim: https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd.commands/omni.usd.commands.CreatePreviewSurfaceMaterialPrimCommand.html

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # spawn material if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        omni.kit.commands.execute("CreatePreviewSurfaceMaterialPrim", mtl_path=prim_path, select_new_prim=False)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")
    # obtain prim
    prim = prim_utils.get_prim_at_path(f"{prim_path}/Shader")
    # apply properties
    cfg = cfg.to_dict()
    del cfg["func"]
    for attr_name, attr_value in cfg.items():
        safe_set_attribute_on_usd_prim(prim, f"inputs:{attr_name}", attr_value, camel_case=True)
    # return prim
    return prim


class MeshConverter(AssetConverterBase):
    """Converter for a mesh file in OBJ / STL / FBX format to a USD file.

    This class wraps around the `omni.kit.asset_converter`_ extension to provide a lazy implementation
    for mesh to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    To make the asset instanceable, we must follow a certain structure dictated by how USD scene-graph
    instancing and physics work. The rigid body component must be added to each instance and not the
    referenced asset (i.e. the prototype prim itself). This is because the rigid body component defines
    properties that are specific to each instance and cannot be shared under the referenced asset. For
    more information, please check the `documentation <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#instancing-rigid-bodies>`_.

    Due to the above, we follow the following structure:

    * ``{prim_path}`` - The root prim that is an Xform with the rigid body and mass APIs if configured.
    * ``{prim_path}/geometry`` - The prim that contains the mesh and optionally the materials if configured.
      If instancing is enabled, this prim will be an instanceable reference to the prototype prim.

    .. _omni.kit.asset_converter: https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html

    .. caution::
        When converting STL files, Z-up convention is assumed, even though this is not the default for many CAD
        export programs. Asset orientation convention can either be modified directly in the CAD program's export
        process or an offset can be added within the config in Isaac Lab.

    """

    cfg: MeshConverterCfg
    """The configuration instance for mesh to USD conversion."""

    def __init__(self, cfg: MeshConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for mesh to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MeshConverterCfg):
        """Generate USD from OBJ, STL or FBX.

        It stores the asset in the following format:

        /file_name (default prim)
          |- /mesh_file_name <- Made instanceable if requested
            |- /Looks
            |- /mesh

        Args:
            cfg: The configuration for conversion of mesh to USD.

        Raises:
            RuntimeError: If the conversion using the Omniverse asset converter fails.
        """
        # resolve mesh name and format
        mesh_file_basename, mesh_file_format = os.path.basename(cfg.asset_path).split(".")
        mesh_file_format = mesh_file_format.lower()

        # Convert USD
        asyncio.get_event_loop().run_until_complete(
            self._convert_mesh_to_usd(
                in_file=cfg.asset_path, out_file=self.usd_path, prim_path=f"/{mesh_file_basename}"
            )
        )
        # Open converted USD stage
        # note: This opens a new stage and does not use the stage created earlier by the user
        # create a new stage
        stage: Usd.Stage = Usd.Stage.Open(self.usd_path)  # type: ignore
        # add USD to stage cache
        stage_id = UsdUtils.StageCache.Get().Insert(stage)  # type: ignore

        stage.DefinePrim(f"/{mesh_file_basename}/visuals", "Xform")
        stage.DefinePrim(f"/{mesh_file_basename}/collisions", "Xform")
        # Get the default prim (which is the root prim) -- "/{mesh_file_basename}"
        xform_prim = stage.GetDefaultPrim()
        root_path = f"/{mesh_file_basename}"
        root_prim = stage.GetPrimAtPath(f"/{mesh_file_basename}")
        visual_prim = stage.GetPrimAtPath(f"{root_prim.GetPath()}/visuals")
        collisions_prim = stage.GetPrimAtPath(f"{root_prim.GetPath()}/collisions")

        # Collect meshes
        meshes = []
        for child in stage.GetPseudoRoot().GetChildren():
            found_meshes = get_all_matching_child_prims(
                child.GetPath(), lambda prim: prim.GetTypeName() == "Mesh", stage=stage
            )
            meshes.extend(found_meshes)
        if not meshes:
            print(f"No meshes found in {self.usd_path}")

        # Process each mesh with visual material, physic material and collision properties
        for count, mesh_prim in enumerate(meshes):
            # Create visual copy
            visual_prim_path = f"{root_path}/visuals/mesh_{count:03d}"
            visual_material_path = f"{root_path}/visuals/Looks"

            Sdf.CopySpec(
                stage.GetEditTarget().GetLayer(),
                str(mesh_prim.GetPath()),
                stage.GetEditTarget().GetLayer(),
                visual_prim_path,
            )
            if cfg.visual_material_props:
                visual_cfg = cfg.visual_material_props
                visual_cfg.func(prim_path=visual_material_path, cfg=visual_cfg, stage=stage)
                apply_material_binding(stage, visual_prim_path, visual_material_path)
            else:
                Sdf.CopySpec(
                    stage.GetEditTarget().GetLayer(),
                    f"{root_path}/temp/Looks",
                    stage.GetEditTarget().GetLayer(),
                    visual_material_path,
                )
                apply_material_binding(stage, visual_prim_path, visual_material_path + "/DefaultMaterial")

            # Create collision copy
            collisions_prim_path = f"{root_path}/collisions/mesh_{count:03d}"
            Sdf.CopySpec(
                stage.GetEditTarget().GetLayer(),
                str(mesh_prim.GetPath()),
                stage.GetEditTarget().GetLayer(),
                collisions_prim_path,
            )

            # Apply collision properties
            if cfg.collision_props:
                apply_collision_properties(stage, collisions_prim_path, cfg)

            if cfg.physics_material_props is not None:
                physics_material_path = f"{root_path}/collisions/PhysicsMaterial"
                cfg.physics_material_props.func(
                    prim_path=physics_material_path, cfg=cfg.physics_material_props, stage=stage
                )
            apply_material_binding(stage, collisions_prim_path, physics_material_path)

        # Remove extraneous prims
        remove_all_other_prims(stage, [visual_prim.GetPath().pathString, collisions_prim.GetPath().pathString])
        # Delete the old Xform and make the new Xform the default prim
        stage.SetDefaultPrim(xform_prim)
        # Handle instanceable
        # Create a new Xform prim that will be the prototype prim
        instanceable_mesh_path = os.path.join(".", "Props", f"{mesh_file_basename}.usd")
        if cfg.make_instanceable:
            # Export Xform to a file so we can reference it from all instances
            export_prim_to_file(
                path=os.path.join(self.usd_dir, instanceable_mesh_path),
                source_prim_path=root_path,
                stage=stage,
            )
            # Delete the original prim that will now be a reference
            visual_prim_path = visual_prim.GetPath().pathString
            collision_prim_path = collisions_prim.GetPath().pathString
            omni.kit.commands.execute("DeletePrims", paths=[visual_prim_path, collision_prim_path], stage=stage)
            # Update references to exported Xform and make it instanceable
            stage.DefinePrim(visual_prim_path, typeName="Xform")
            stage.DefinePrim(collision_prim_path, typeName="Xform")
            visual_undef_prim = stage.GetPrimAtPath(visual_prim_path)
            collision_undef_prim = stage.GetPrimAtPath(collision_prim_path)
            visual_undef_prim_ref: Usd.References = visual_undef_prim.GetReferences()
            collision_undef_prim_ref: Usd.References = collision_undef_prim.GetReferences()
            visual_undef_prim_ref.AddReference(assetPath=instanceable_mesh_path, primPath=visual_prim_path)  # type: ignore
            collision_undef_prim_ref.AddReference(assetPath=instanceable_mesh_path, primPath=collision_prim_path)  # type: ignore
            visual_undef_prim.SetInstanceable(True)
            collision_undef_prim.SetInstanceable(True)

        # Apply mass and rigid body properties after everything else
        # Properties are applied to the top level prim to avoid the case where all instances of this
        #   asset unintentionally share the same rigid body properties
        # apply mass properties
        if cfg.mass_props is not None:
            schemas.define_mass_properties(prim_path=xform_prim.GetPath(), cfg=cfg.mass_props, stage=stage)
        # apply rigid body properties
        if cfg.rigid_props is not None:
            schemas.define_rigid_body_properties(prim_path=xform_prim.GetPath(), cfg=cfg.rigid_props, stage=stage)

        # Save changes to USD stage
        stage.Save()
        if stage_id is not None:
            UsdUtils.StageCache.Get().Erase(stage_id)  # type: ignore

    """
    Helper methods.
    """

    @staticmethod
    async def _convert_mesh_to_usd(
        in_file: str, out_file: str, prim_path: str = "/World", load_materials: bool = True
    ) -> bool:
        """Convert mesh from supported file types to USD.

        This function uses the Omniverse Asset Converter extension to convert a mesh file to USD.
        It is an asynchronous function and should be called using `asyncio.get_event_loop().run_until_complete()`.

        The converted asset is stored in the USD format in the specified output file.
        The USD file has Y-up axis and is scaled to meters.

        The asset hierarchy is arranged as follows:

        .. code-block:: none
            prim_path (default prim)
                |- /geometry/Looks
                |- /geometry/mesh

        Args:
            in_file: The file to convert.
            out_file: The path to store the output file.
            prim_path: The prim path of the mesh.
            load_materials: Set to True to enable attaching materials defined in the input file
                to the generated USD mesh. Defaults to True.

        Returns:
            True if the conversion succeeds.
        """
        enable_extension("omni.kit.asset_converter")
        enable_extension("omni.usd.metrics.assembler")

        import omni.kit.asset_converter
        import omni.usd
        from omni.metrics.assembler.core import get_metrics_assembler_interface

        # Create converter context
        converter_context = omni.kit.asset_converter.AssetConverterContext()
        # Set up converter settings
        # Don't import/export materials
        converter_context.ignore_materials = not load_materials
        converter_context.ignore_animations = True
        converter_context.ignore_camera = True
        converter_context.ignore_light = True
        # Merge all meshes into one
        converter_context.merge_all_meshes = True
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        # This does not work right now :(, so we need to scale the mesh manually
        converter_context.use_meter_as_world_unit = True
        converter_context.baking_scales = True
        # Uses double precision for all transform ops.
        converter_context.use_double_precision_to_usd_transform_op = True

        # Create converter task
        instance = omni.kit.asset_converter.get_instance()
        out_file_non_metric = out_file.replace(".usd", "_non_metric.usd")
        task = instance.create_converter_task(in_file, out_file_non_metric, None, converter_context)  # type: ignore
        # Start conversion task and wait for it to finish
        success = True
        while True:
            success = await task.wait_until_finished()
            if not success:
                await asyncio.sleep(0.1)
            else:
                break

        temp_stage = Usd.Stage.CreateInMemory()  # type: ignore
        UsdGeom.SetStageUpAxis(temp_stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(temp_stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(temp_stage, 1.0)  # type: ignore

        base_prim = temp_stage.DefinePrim(prim_path, "Xform")
        prim = temp_stage.DefinePrim(f"{prim_path}/temp", "Xform")
        prim.GetReferences().AddReference(out_file_non_metric)
        cache = UsdUtils.StageCache.Get()
        cache.Insert(temp_stage)  # type: ignore
        stage_id = cache.GetId(temp_stage).ToLongInt()  # type: ignore
        get_metrics_assembler_interface().resolve_stage(stage_id)
        temp_stage.SetDefaultPrim(base_prim)
        temp_stage.Export(out_file)
        return success
