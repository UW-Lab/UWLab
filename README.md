![Isaac Lab](docs/source/_static/uwlab.png)

---

# UW Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

**UW Lab**, open source projects built upon the robust foundation established by Isaac Lab, organized by UW(University of Washington) Robotic Students, aims to creates accelerated, unified, organized research in the field of robotics simulation in Isaac Ecosystem. This repo is designed and structured to reuse toolkit of ongoing IsaacLab development, track IsaacLab version, and extend the components from novelty and engineers from UW.

## Key Features

In addition to what IsaacLab provides, UW Lab brings:

- **Environments**: Cleaned Implementation of reputable environments in Manager-Based format
- **Sim to Real**: Providing robots and configuration that has been tested in UW Robotic Lab amd deliver the Simulation Setup that can directly transfer to reals


## Getting Started

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Contributing to UW Lab

Please refer to Isaac Lab
[contribution guideline](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html).


## Troubleshooting

Please for bug and troubleshooting [submit an issue](https://github.com/UW-Lab/UWLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/UW-Lab/UWLab/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/UW-Lab/UWLab/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## License

UW Lab is released under [BSD-3 License](LICENSE). The Isaac Lab framework is released under [BSD-3 License](LICENSE).

## Acknowledgement

If you found UW Lab useful, we appreciate if you cite it in academic publications:
```
@software{zhang2025uwlab,
  author       = {Zhang, Zhengyu and Yu, Feng and Castro, Mateo and Yin, Patrick and Peng, Quanquan and Scalise, Rosario},
  title        = {{UWLab}: A Simulation Platform for Robot Learning Environment},
  year         = {2025},
  url          = {https://github.com/UW-Lab/UWLab}
}
```
UW Lab development initialed from the Isaac Lab, and closely track the development of Isaac Lab. As gratitude we also appreciate if you cite Isaac Lab in academic publications:
```
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
```
