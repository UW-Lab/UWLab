Installing UW Lab
===================

UW Lab builds on top of IsaacLab and IsaacSim. Please follow the below instructions to install UW Lab.


.. note::

   If you use Conda, we recommend using `Miniconda <https://docs.anaconda.com/miniconda/miniconda-other-installer-links/>`_.

Install Isaac Lab and Isaac Sim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Please follow one of three ways to install Isaac Lab and Isaac Sim:
   For best experience with vscode development, we recommend using Binary Installation.
   For easy and quick installation, we recommend using pip installation.
   For advanced users, we recommend using IsaacSim and IsaacLab pip installation.

   `IsaacLab with IsaacSim pip installation <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html>`_


   `IsaacLab with IsaacSim binary Installation <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html>`_


   `IsaacSim and IsaacLab pip installation <https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/isaaclab_pip_installation.html>`_


Please also go through **Verifying the Isaac Lab Installation:** section in above links,
If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.isaacsim.omniverse.nvidia.com//latest/isaac_sim_forums.html>`__.


Install UW Lab
~~~~~~~~~~~~~~~~

-  Make sure that your virtual environment is activated (if applicable)

-  Install UW Lab by cloning the repository and running the installation script

   .. code-block:: bash

      git clone https://github.com/UW-Lab/UWLab.git

-  Pip Install UW Lab in edible mode

   .. code-block:: bash

      cd UWLab
      ./uwlab.sh -i


Verify UW Lab Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Try running the following command to verify that UW Lab is installed correctly:

.. code:: bash

   python scripts/reinforcement_learning/rsl_rl/train.py --task UW-Position-Pit-Spot-v0 --num_envs 1024


Congratulations! You have successfully installed UW Lab.
