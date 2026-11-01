#!/bin/bash
#SBATCH -J g4-dl-models
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH --qos=issai-zen4
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
set -euo pipefail
cd $HOME/gemma4
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in gemma-4-31B-it-qat-w4a16-ct gemma-4-12B-it-qat-w4a16-ct; do
  venv/bin/hf download google/$m --local-dir models/$m
  du -sh models/$m
done
