#!/bin/bash
#SBATCH -J g4-probe -p ZEN4 -A issai -c 8 --mem=16G -t 00:30:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
D=/datahub/harirai/$USER/gemma4; W=/mnt/scratch/$USER/g4-probe-$SLURM_JOB_ID; mkdir -p $W/tmp
export TMPDIR=$W/tmp
cp $D/images/sbx2.sif $W/sbx.sif; mkdir -p $W/seed && apptainer -s exec --bind $W $W/sbx.sif cp -a /usr/local/. $W/seed/
$D/harness-venv/bin/python $HOME/gemma4/code/probe_env.py $W/sbx.sif $W $W/seed $D/data $D/data/wheels_plus "$@"
rm -rf $W
