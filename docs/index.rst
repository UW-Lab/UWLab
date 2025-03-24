Overview
========

.. figure:: source/_static/cover.png
   :width: 100%
   :alt: Cover

Preamble
==================

**UW Lab** build upon the solid foundation laid by ``Isaac Lab`` / ``NVIDIA Isaac Sim``,
expanding its framework to embrace a broader spectrum of algorithms, robots, and environments. While adhering to the
principles of modularity, agility, openness, and battery-included as Isaac Lab.

In the short term, our mission is to unify and facilitate the research efforts of our colleagues within a single,
cohesive framework. Looking ahead, UW Lab envisions a future where AI, robotics, and the boundaries between
reality and digital world seamlessly converge, offering profound insights into the interaction and development of
intelligent systems.

We recognize that this is a long and evolving journey, which is why we place immense value on the journey of
development, prioritizing principled, flexible, and extensible structures over mere results. At UW Lab, we are
committed to crafting a value where the process is as significant as the outcome, fostering innovation that resonates
deeply with our vision.


License
=======

The UW Lab framework is open-sourced under the BSD-3-Clause license.
Please refer to :ref:`license` for more details.


Acknowledgement
===============

If you found UW Lab useful, we appreciate if you cite it in academic publications:

.. code:: bibtex

   @software{zhang2025uwlab,
      author       = {Zhang, Zhengyu and Yu, Feng and Castro, Mateo and Yin, Patrick and Peng, Quanquan and Scalise, Rosario},
      title        = {{UWLab}: A Simulation Platform for Robot Learning Environment},
      year         = {2025},
      url          = {https://github.com/UW-Lab/UWLab}
   }


UW Lab development initiated from the `Orbit <https://isaac-orbit.github.io/>`_ framework.
please cite orbit in academic publications as well in honor of the original authors.:

.. code:: bibtex

   @article{mittal2023orbit,
      author={Mittal, Mayank and Yu, Calvin and Yu, Qinxi and Liu, Jingzhou and Rudin, Nikita and Hoeller, David and Yuan, Jia Lin and Singh, Ritvik and Guo, Yunrong and Mazhar, Hammad and Mandlekar, Ajay and Babich, Buck and State, Gavriel and Hutter, Marco and Garg, Animesh},
      journal={IEEE Robotics and Automation Letters},
      title={Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments},
      year={2023},
      volume={8},
      number={6},
      pages={3740-3747},
      doi={10.1109/LRA.2023.3270034}
   }


Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   source/setup/installation/local_installation

.. toctree::
   :maxdepth: 1
   :caption: Publications
   :titlesonly:

   source/publications/pg1

.. toctree::
   :maxdepth: 3
   :caption: Overview
   :titlesonly:

   source/overview/isaac_environments
   source/overview/uw_environments

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/license

.. toctree::
    :hidden:
    :caption: Project Links

    UW Lab <https://github.com/UW-Lab/UWLab>
    Isaac Lab <https://github.com/isaac-sim/IsaacLab>
    NVIDIA Isaac Sim <https://docs.omniverse.nvidia.com/isaacsim/latest/index.html>
    NVIDIA PhysX <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/index.html>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _NVIDIA Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/latest/index.html
