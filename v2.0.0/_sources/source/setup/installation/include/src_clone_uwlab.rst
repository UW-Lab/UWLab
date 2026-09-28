Cloning UW Lab
~~~~~~~~~~~~~~~~~

.. note::

   We recommend making a `fork <https://github.com/uw-lab/UWLab/fork>`_ to contribute,
   but this is not required to use the framework. When using your fork, replace
   ``uw-lab`` with your GitHub username in the clone commands.

Clone UW Lab into your project's workspace:

.. tab-set::

   .. tab-item:: SSH

      .. code-block:: bash

         git clone git@github.com:uw-lab/UWLab.git

   .. tab-item:: HTTPS

      .. code-block:: bash

         git clone https://github.com/uw-lab/UWLab.git

We provide the Linux helper `uwlab.sh <https://github.com/uw-lab/UWLab/blob/main/uwlab.sh>`_
to manage installation, Python execution, tests, and documentation. From the checkout,
print the supported commands with:

.. code-block:: bash

   ./uwlab.sh --help

.. warning::

   This repository does not ship ``uwlab.bat``. Do not assume that Isaac Lab's
   Windows batch commands have UWLab equivalents; the helper instructions here
   use the Linux shell entry point.
