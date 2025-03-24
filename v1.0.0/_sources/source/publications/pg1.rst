Parental Guidance(PG1)
======================

Links
-----
- **Paper on OpenReview:** `Parental Guidance: Efficient Lifelong Learning through Evolutionary Distillation <https://openreview.net/forum?id=mFaPH8JZLC>`_
- **GitHub Repository:** `UW Lab GitHub <https://github.com/UW-Lab/UWLab>`_

Authors
-------
**Zhengyu Zhang** †, **Quanquan Peng** ‡, **Rosario Scalise** †, **Byron Boots** †

† Paul G Allen School, University of Washington
‡ Shanghai Jiao Tong University

.. image:: ../../source/_static/publications/pg1/pg1.png
   :alt: Research Illustration
   :align: center

Abstract
--------
Developing robotic agents that can perform well in diverse environments while showing a variety of behaviors is
a key challenge in AI and robotics. Traditional reinforcement learning (RL) methods often create agents that specialize
in narrow tasks, limiting their adaptability and diversity. To overcome this, we propose a preliminary,
evolution-inspired framework that includes a reproduction module, similar to natural species reproduction,
balancing diversity and specialization. By integrating RL, imitation learning (IL), and a coevolutionary agent-terrain
curriculum, our system evolves agents continuously through complex tasks. This approach promotes adaptability,
inheritance of useful traits, and continual learning. Agents not only refine inherited skills but also surpass
their predecessors. Our initial experiments show that this method improves exploration efficiency and supports
open-ended learning, offering a scalable solution where sparse reward coupled with diverse terrain environments
induces a multi-task setting.


BibTex
----------
.. code:: bibtex

   @inproceedings{
      zhang2024blending,
      title={Blending Reinforcement Learning and Imitation Learning for Evolutionary Continual Learning},
      author={Zhengyu Zhang and Quanquan Peng and Rosario Scalise and Byron Boots},
      booktitle={[CoRL 2024] Morphology-Aware Policy and Design Learning Workshop (MAPoDeL)},
      year={2024},
      url={https://openreview.net/forum?id=d2VTtWOCMm}
   }
