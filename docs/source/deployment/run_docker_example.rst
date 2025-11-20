Running an example with Docker
==============================

From the root of the UW Lab repository,  the ``docker`` directory contains all the Docker relevant files. These include the three files
(**Dockerfile**, **docker-compose.yaml**, **.env**) which are used by Docker, and an additional script that we use to interface with them,
**container.py**.

In this tutorial, we will learn how to use the UW Lab Docker container for development. For a detailed description of the Docker setup,
including installation and obtaining access to an Isaac Sim image, please reference the :ref:`deployment-docker`. For a description
of Docker in general, please refer to `their official documentation <https://docs.docker.com/get-started/overview/>`_.


Building the Container
~~~~~~~~~~~~~~~~~~~~~~

To build the UW Lab container from the root of the UW Lab repository, we will run the following:


.. code-block:: console

   python docker/container.py start


The terminal will first pull the base IsaacSim image, build the UW Lab image's additional layers on top of it, and run the UW Lab container.
This should take several minutes for the first build but will be shorter in subsequent runs as Docker's caching prevents repeated work.
If we run the command ``docker container ls`` on the terminal, the output will list the containers that are running on the system. If
everything has been set up correctly, a container with the ``NAME`` **uw-lab-base** should appear, similar to below:


.. code-block:: console

   CONTAINER ID   IMAGE               COMMAND   CREATED           STATUS         PORTS     NAMES
   483d1d5e2def   uw-lab-base      "bash"    30 seconds ago   Up 30 seconds             uw-lab-base


Once the container is up and running, we can enter it from our terminal.

.. code-block:: console

   python docker/container.py enter


On entering the UW Lab container, we are in the terminal as the superuser, ``root``. This environment contains a copy of the
UW Lab repository, but also has access to the directories and libraries of Isaac Sim. We can run experiments from this environment
using a few convenient aliases that have been put into the ``root`` **.bashrc**. For instance, we have made the **uwlab.sh** script
usable from anywhere by typing its alias ``uwlab``.

Additionally in the container, we have `bind mounted`_ the ``UWLab/source`` directory from the
host machine. This means that if we modify files under this directory from an editor on the host machine, the changes are
reflected immediately within the container without requiring us to rebuild the Docker image.

We will now run a sample script from within the container to demonstrate how to extract artifacts
from the UW Lab Docker container.

Executing the Script
~~~~~~~~~~~~~~~~~~~~

We will execute the script to produce a log, adding a ``--headless`` flag to our execution to prevent a GUI:

.. code-block:: bash

  uwlab -p scripts/tutorials/00_sim/log_time.py --headless


Now ``log.txt`` will have been produced at ``/workspace/uwlab/logs/docker_tutorial``. If we exit the container
by typing ``exit``, we will return to ``UWLab/docker`` in our host terminal environment. We can then enter
the following command to retrieve our logs from the Docker container and put them on our host machine:

.. code-block:: console

  ./container.py copy


We will see a terminal readout reporting the artifacts we have retrieved from the container. If we navigate to
``/uwlab/docker/artifacts/logs/docker_tutorial``, we will see a copy of the ``log.txt`` file which was produced
by the script above.

Each of the directories under ``artifacts`` corresponds to Docker `volumes`_ mapped to directories
within the container and the ``container.py copy`` command copies them from those `volumes`_ to these directories.

We could return to the UW Lab Docker terminal environment by running ``container.py enter`` again,
but we have retrieved our logs and wish to go inspect them. We can stop the UW Lab Docker container with the following command:

.. code-block:: console

  ./container.py stop


This will bring down the Docker UW Lab container. The image will persist and remain available for further use, as will
the contents of any `volumes`_. If we wish to free up the disk space taken by the image, (~20.1GB), and do not mind repeating
the build process when we next run ``./container.py start``, we may enter the following command to delete the **uw-lab-base** image:

.. code-block:: console

  docker image rm uw-lab-base

A subsequent run of ``docker image ls`` will show that the image tagged **uw-lab-base** is now gone. We can repeat the process for the
underlying NVIDIA container if we wish to free up more space. If a more powerful method of freeing resources from Docker is desired,
please consult the documentation for the `docker prune`_ commands.


.. _volumes: https://docs.docker.com/storage/volumes/
.. _bind mounted: https://docs.docker.com/storage/bind-mounts/
.. _docker prune: https://docs.docker.com/config/pruning/
