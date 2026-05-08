#!/usr/bin/env bash
# Wrapper to run a UWLab command inside isaac-sim container with the
# uwlab-sriyash conda env (lti clone with editable installs pointing at
# UWLab/source/* + UWLab/diffusion_policy + UWLab/.uwlab_rsl_rl).
set -eo pipefail
cd /mnt/storage/lti/UWLab
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate uwlab-sriyash
set +u
source /isaac-sim/setup_conda_env.sh
set -u
_py_ver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_conda_sp="$CONDA_PREFIX/lib/python${_py_ver}/site-packages"
# Conda env site-packages first (then Isaac Sim's pip_prebundle paths it sets).
export PYTHONPATH="${_conda_sp}${PYTHONPATH:+:$PYTHONPATH}"
unset _py_ver _conda_sp
exec "$@"
