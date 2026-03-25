Cloning UW Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/uw-lab/UWLab/fork>`_ of the UW Lab repository to contribute
   to the project but this is not mandatory to use the framework. If you
   make a fork, please replace ``isaac-sim`` with your username
   in the following instructions.

Clone the UW Lab repository into your project's workspace:

.. tab-set::

   .. tab-item:: SSH

      .. code:: bash

         git clone git@github.com:uw-lab/UWLab.git

   .. tab-item:: HTTPS

      .. code:: bash

         git clone https://github.com/uw-lab/UWLab.git


We provide a helper executable `uwlab.sh <https://github.com/uw-lab/UWLab/blob/main/uwlab.sh>`_
and `uwlab.bat <https://github.com/uw-lab/UWLab/blob/main/uwlab.bat>`_ for Linux and Windows
respectively that provides utilities to manage extensions.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: text

         ./uwlab.sh --help

         usage: uwlab.sh [-h] [-i] [-f] [-p] [-s] [-t] [-o] [-v] [-d] [-n] [-c] -- Utility to manage UW Lab.

         optional arguments:
            -h, --help           Display the help content.
            -i, --install [LIB]  Install the extensions inside UW Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
            -f, --format         Run pre-commit to format the code and check lints.
            -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
            -s, --sim            Run the simulator executable (isaac-sim.sh) provided by Isaac Sim.
            -t, --test           Run all python pytest tests.
            -o, --docker         Run the docker container helper script (docker/container.sh).
            -v, --vscode         Generate the VSCode settings file from template.
            -d, --docs           Build the documentation from source using sphinx.
            -n, --new            Create a new external project or internal task from template.
            -c, --conda [NAME]   Create the conda environment for UW Lab. Default name is 'env_uwlab'.
            -u, --uv [NAME]      Create the uv environment for UW Lab. Default name is 'env_uwlab'.

   .. tab-item:: :icon:`fa-brands fa-windows` Windows
      :sync: windows

      .. code:: text

         uwlab.bat --help

         usage: uwlab.bat [-h] [-i] [-f] [-p] [-s] [-v] [-d] [-n] [-c] -- Utility to manage UW Lab.

         optional arguments:
            -h, --help           Display the help content.
            -i, --install [LIB]  Install the extensions inside UW Lab and learning frameworks (rl_games, rsl_rl, sb3, skrl) as extra dependencies. Default is 'all'.
            -f, --format         Run pre-commit to format the code and check lints.
            -p, --python         Run the python executable provided by Isaac Sim or virtual environment (if active).
            -s, --sim            Run the simulator executable (isaac-sim.bat) provided by Isaac Sim.
            -t, --test           Run all python pytest tests.
            -v, --vscode         Generate the VSCode settings file from template.
            -d, --docs           Build the documentation from source using sphinx.
            -n, --new            Create a new external project or internal task from template.
            -c, --conda [NAME]   Create the conda environment for UW Lab. Default name is 'env_uwlab'.
            -u, --uv [NAME]      Create the uv environment for UW Lab. Default name is 'env_uwlab'.
