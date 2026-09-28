Create a New Task
=================

This guide walks through adding a custom object pair (insertive + receptive) so you can train OmniReset policies on your own tasks.

----

Step 1: Prepare Meshes in Blender
----------------------------------

All mesh-level properties must be set in Blender before export. Isaac Sim does not modify mesh geometry on import, so scale and orientation must be baked into the vertices here.

In a single Blender session:

1. Rescale objects to real-world dimensions (meters). ``Ctrl+A`` > Rotation & Scale to bake transforms.
2. Reorient so Z-axis points up when the object is resting on a table: ``Tab`` > Edit Mode, ``A`` to select all, rotate as needed (e.g. ``R X 90``).
3. Set origin: right-click > Set Origin > Origin to Center of Mass (Volume).
4. Place both objects in assembled pose, record the relative transform for ``assembled_pose`` to be used in Step 4.
5. Export each object as ``.usdz``.

.. raw:: html

   <div style="text-align: center; margin: 20px 0;">
     <video width="640" controls>
       <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/omnireset-new-task-blender-tutorial.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>

----

Step 2: Set Up USD in Isaac Sim
--------------------------------

Import the exported mesh into Isaac Sim and restructure it into the standard asset format:

1. Import the ``.usdz`` into a new USD stage.
2. Separate the mesh into a visual mesh (with materials) and a collision mesh (invisible, with SDF physics collider).
3. Save as ``.usd``.

.. tip::

   If objects appear in the wrong place in Step 6, try adding the Rigid Body component to the root prim, setting the SDF collider only on the collision mesh, and removing physics from the visual mesh.

.. raw:: html

   <div style="text-align: center; margin: 20px 0;">
     <video width="640" controls>
       <source src="https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main/Media/OmniReset/omnireset-new-task-isaacsim-tutorial.mp4" type="video/mp4">
       Your browser does not support the video tag.
     </video>
   </div>

----

Step 3: Compute Bottom Offset
-------------------------------

The bottom offset is the distance from the object's origin to its lowest point, used for spawning objects flush on the table. Run on each ``.usd`` from Step 2:

.. code:: bash

   python scripts_v2/tools/compute_bottom_offset.py /path/to/object.usd

Example output::

   bottom_offset: 0.056658

Record these values for Step 4.

----

Step 4: Create Metadata
-------------------------

Create a ``metadata.yaml`` file **in the same folder** as each ``.usd``:

.. code:: text

   My_Insertive_Object/
     my_insertive_object.usd
     metadata.yaml

   My_Receptive_Object/
     my_receptive_object.usd
     metadata.yaml

The metadata has the following fields:

- ``assembled_offset``: Transform from the insertive object to this object in the assembled pose. Always identity for the insertive object; for the receptive object, use the relative transform recorded in Step 1.
- ``bottom_offset``: Transform from origin to the bottom of the object. The Z value is the **negative** of the script output from Step 3.
- ``success_thresholds`` (receptive only): How tightly the policy must align parts. Use ``position: 0.0025, orientation: 0.025`` for tight-fit tasks (e.g. screw insertion). For looser tasks (e.g. cube stacking), try ``position: 0.005, orientation: 0.05``. May need to tune depending on the task.

**Insertive object** example:

.. code:: yaml

   assembled_offset:
     pos: [0.0, 0.0, 0.0]
     quat: [1.0, 0.0, 0.0, 0.0]
   bottom_offset:
     pos: [0.0, 0.0, -0.056658]
     quat: [1.0, 0.0, 0.0, 0.0]

**Receptive object** example:

.. code:: yaml

   assembled_offset:
     pos: [0.012, 0.0, 0.035]
     quat: [1.0, 0.0, 0.0, 0.0]
   bottom_offset:
     pos: [0.0, 0.0, -0.010169]
     quat: [1.0, 0.0, 0.0, 0.0]
   success_thresholds:
     position: 0.0025
     orientation: 0.025

----

Step 5: Register Object Variants
----------------------------------

Add your objects to the ``variants`` dictionary in **4 config files**:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Config File
     - Purpose
   * - ``partial_assemblies_cfg.py``
     - Partial assembly collection
   * - ``grasp_sampling_cfg.py``
     - Grasp pose sampling
   * - ``reset_states_cfg.py``
     - Reset state generation
   * - ``rl_state_cfg.py``
     - RL training & evaluation


All files live under:

.. code:: text

   source/uwlab_tasks/.../omnireset/config/ur5e_robotiq_2f85/

Add to ``variants["scene.insertive_object"]``:

.. code:: python

   "my_insertive_object": make_insertive_object(
       "/absolute/path/to/my_insertive_object.usd"
   ),

Add to ``variants["scene.receptive_object"]``:

.. code:: python

   "my_receptive_object": make_receptive_object(
       "/absolute/path/to/my_receptive_object.usd"
   ),

.. tip::

   Use local absolute paths during development. Switch to ``UWLAB_CLOUD_ASSETS_DIR`` when sharing.

----

Step 6: Verify Setup
----------------------

**Check assembled pose offset** by running partial assemblies:

.. code:: bash

   python scripts_v2/tools/record_partial_assemblies.py \
       --task OmniReset-PartialAssemblies-v0 \
       --num_envs 10 --num_trajectories 10 --headless \
       env.scene.insertive_object=my_insertive_object env.scene.receptive_object=my_receptive_object

If objects are misaligned or upside down, revisit Step 1.

**Check bottom offset** by generating a small set of reset states and visualizing:

.. code:: bash

   python scripts_v2/tools/record_reset_states.py \
       --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
       --num_envs 4 --num_reset_states 8 --headless \
       env.scene.insertive_object=my_insertive_object env.scene.receptive_object=my_receptive_object

.. code:: bash

   python scripts_v2/tools/visualize_reset_states.py \
       --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
       --num_envs 4 --dataset_dir ./Datasets/OmniReset \
       env.scene.insertive_object=my_insertive_object env.scene.receptive_object=my_receptive_object

Confirm the receptive object sits flush on the table. If it's floating or clipping, adjust the ``bottom_offset`` in Step 4.

Once everything looks correct, proceed to :doc:`rl_training` to generate full reset states and train.

----

Known Limitations
------------------

**Grasp sampling.** The grasp sampler does not always produce valid grasps for adversarial or unusually shaped objects. You may need to tune sampling parameters in the grasp sampling config. Additionally, the collision checker may be inaccurate when operating at mm-level precision, so ``min_dist`` may need task-specific tuning at the moment.

We are actively working on removing the grasp sampling requirement from OmniReset entirely, which will make the pipeline more general and also eliminate the need for mm-level collision checking.
