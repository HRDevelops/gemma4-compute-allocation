#!/bin/bash
#SBATCH -J g4-build-harness
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
OUT=/datahub/harirai/$USER/gemma4/images; mkdir -p $OUT
W=/datahub/harirai/$USER/gemma4/build-tmp/$SLURM_JOB_ID; mkdir -p $W/tmp $W/cache
export APPTAINER_TMPDIR=$W/tmp APPTAINER_CACHEDIR=$W/cache PROOT_TMP_DIR=$W/tmp
apptainer --version; t0=$(date +%s)
apptainer build --fakeroot --force $OUT/harness.sif $HOME/gemma4/code/harness.def || apptainer build --force $OUT/harness.sif $HOME/gemma4/code/harness.def
echo "built in $(( $(date +%s)-t0 ))s"; ls -la $OUT/harness.sif
apptainer exec $OUT/harness.sif sh -c 'git --version; python3 --version; python3 -c "import swegemma;print(swegemma.__version__ if hasattr(swegemma,\"__version__\") else \"swegemma ok\")"'
rm -rf $W
