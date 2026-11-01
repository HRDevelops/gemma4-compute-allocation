#!/bin/bash
#SBATCH -J g4-build-vllm
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH --qos=issai-zen4
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
VER=${VER:-v0.19.1}
OUT=$HOME/gemma4/images/vllm-${VER}.sif
LOCAL=$HOME/gemma4/cache/build-$SLURM_JOB_ID
mkdir -p $LOCAL/cache $LOCAL/tmp
df -h $HOME
export APPTAINER_CACHEDIR=$LOCAL/cache APPTAINER_TMPDIR=$LOCAL/tmp
apptainer build --force $LOCAL/vllm.sif docker://vllm/vllm-openai:${VER}
mv $LOCAL/vllm.sif $OUT
rm -rf $LOCAL
apptainer exec $OUT python3 -c "import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__)"
ls -lh $OUT
