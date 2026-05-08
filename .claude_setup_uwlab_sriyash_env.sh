#!/usr/bin/env bash
# Clone the lti conda env into uwlab-sriyash and swap the editable installs
# from UWLab-ICL paths to UWLab paths.
set -e
LOG=/mnt/storage/lti/UWLab/.claude_logs/setup_uwlab_sriyash.log
exec > >(tee -a "$LOG") 2>&1
echo "==== START: $(date) ===="

source /home/ubuntu/miniforge3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "uwlab-sriyash"; then
    echo "[setup] uwlab-sriyash env already exists; skipping clone."
else
    echo "[setup] cloning lti -> uwlab-sriyash (this takes 3-5 min)"
    conda create --name uwlab-sriyash --clone lti -y
fi

conda activate uwlab-sriyash

echo "[setup] uninstalling UWLab-ICL editable packages"
pip uninstall -y \
    diffusion_policy \
    rsl-rl-lib \
    uwlab \
    uwlab_tasks \
    uwlab_assets \
    uwlab_rl

echo "[setup] reinstalling editable packages from UWLab paths"
pip install --no-deps -e /mnt/storage/lti/UWLab/source/uwlab
pip install --no-deps -e /mnt/storage/lti/UWLab/source/uwlab_assets
pip install --no-deps -e /mnt/storage/lti/UWLab/source/uwlab_rl
pip install --no-deps -e /mnt/storage/lti/UWLab/source/uwlab_tasks
pip install --no-deps -e /mnt/storage/lti/UWLab/.uwlab_rsl_rl
pip install --no-deps -e /mnt/storage/lti/UWLab/diffusion_policy

echo "[setup] verifying editable mappings"
pip show -f diffusion_policy 2>/dev/null | grep -E "Name|Location|Editable" | head -5
pip show -f rsl-rl-lib 2>/dev/null | grep -E "Name|Location|Editable" | head -5
pip show -f uwlab_tasks 2>/dev/null | grep -E "Name|Location|Editable" | head -5

echo "==== DONE: $(date) ===="
