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
apptainer build --force $D/images/sbx2.sif $HOME/gemma4/code/sbx2.def
apptainer exec $D/images/sbx2.sif sh -c 'python3 --version; git --version; pytest --version; ls /wheels | wc -l'
rm -rf $W
