#!/bin/bash
#SBATCH -J g4-build-sbx
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
D=/datahub/harirai/$USER/gemma4
W=$D/build-tmp/$SLURM_JOB_ID; mkdir -p $W/tmp $W/cache
export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache PROOT_TMP_DIR=$W/tmp
# 1) sandbox image (mirror of Dockerfile.public)
apptainer build --force $D/images/sbx.sif $HOME/gemma4/code/sbx.def
apptainer exec $D/images/sbx.sif sh -c 'python3 --version; git --version; pytest --version; ls /wheels | wc -l'
# 2) host-side harness venv (Python 3.12, as on Kaggle)
export UV_CACHE_DIR=$D/uv-cache UV_PYTHON_INSTALL_DIR=$D/uv-python
V=$D/harness-venv
[ -x $V/bin/python ] || ~/.local/bin/uv venv -p 3.12 $V
~/.local/bin/uv pip install -p $V/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
~/.local/bin/uv pip install -p $V/bin/python $D/data/harness_wheels/*.whl
$V/bin/python -c "import swegemma, adk_submission, adk_eval_core, google.adk, litellm; print('host harness ok')"
rm -rf $W
