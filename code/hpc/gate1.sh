#!/bin/bash
#SBATCH -J g4-gate1
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH -c 32
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
# usage: sbatch gate1.sh <gold|empty> [concurrency] [task ids...]
set -uo pipefail
MODE=$1; CONC=${2:-12}; shift 2 || shift $#
D=/datahub/harirai/$USER/gemma4
W=/mnt/scratch/$USER/gemma4-gate1-$SLURM_JOB_ID          # node-local, exec, fast
mkdir -p $W/tmp; export TMPDIR=$W/tmp   # private wheel-tar cache per job
cp $D/images/${SBX:-sbx2}.sif $W/sbx.sif
mkdir -p $W/usr_local_seed && apptainer -s exec --bind $W $W/sbx.sif cp -a /usr/local/. $W/usr_local_seed/ && test -x $W/usr_local_seed/bin/python3 || { echo "seed copy failed"; exit 1; }
OUT=$D/runs/gate1-$MODE-$SLURM_JOB_ID; mkdir -p $OUT
echo "[$(hostname)] mode=$MODE conc=$CONC out=$OUT"
t0=$(date +%s)
$D/harness-venv/bin/python $HOME/gemma4/code/gate1.py --mode $MODE --data $D/data --out $OUT --concurrency $CONC \
  --sif $W/sbx.sif --work $W --seed $W/usr_local_seed --wheels $D/data/${WHEELS:-wheels_plus} ${@:+--task-ids "$@"}
echo "elapsed $(( $(date +%s)-t0 ))s"
rm -rf $W
