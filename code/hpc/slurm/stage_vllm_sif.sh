#!/bin/bash
# Build the vLLM image on this node-local scratch if it is not there yet.
# Home quota is too small for the ~10 GB image, so it lives in /mnt/scratch per node.
# Usage: source-able (prints SIF path) or sbatch-able.
#SBATCH -J g4-stage-vllm
#SBATCH -p H100
#SBATCH -A issai
#SBATCH --qos=issai-h100
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
VER=${VER:-v0.19.1}
ROOT=/mnt/scratch/$USER/gemma4
SIF=$ROOT/images/vllm-${VER}   # sandbox directory (mksquashfs unavailable under proot)
mkdir -p $ROOT/images $ROOT/appt-cache $ROOT/appt-tmp
exec 9>$ROOT/images/.lock; flock 9
# scratch is purged nightly by file age: verify real contents, not just the marker
if [ ! -f "$SIF/.built" ] || [ ! -x "$SIF/usr/local/bin/vllm" ]; then rm -rf "$SIF" "$SIF.partial";
  echo "[$(hostname)] building $SIF"; start=$(date +%s)
  APPTAINER_CACHEDIR=$ROOT/appt-cache APPTAINER_TMPDIR=$ROOT/appt-tmp \
    apptainer build --force --sandbox $SIF.partial docker://vllm/vllm-openai:${VER}
  mv $SIF.partial $SIF; find $SIF -xdev -exec touch -h {} + 2>/dev/null || true; touch $SIF/.built
  rm -rf $ROOT/appt-tmp/*
  echo "[$(hostname)] built in $(( $(date +%s) - start ))s"
fi
du -sh $SIF
apptainer exec $SIF python3 -c "import vllm,torch;print(\"vllm\",vllm.__version__,\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"
echo "SIF=$SIF"
