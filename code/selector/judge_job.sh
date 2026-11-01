#!/bin/bash
#SBATCH -J g4-judge
#SBATCH -p H100
#SBATCH -A issai
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:2
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 4:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
# usage: sbatch judge_job.sh <run_dir> ; reads <run_dir>/features.jsonl and <run_dir>/seed*/patches, writes judge_scores.jsonl
set -uo pipefail
RUN=$1
D=/datahub/harirai/$USER/gemma4; C=$HOME/gemma4/code
MODEL=$HOME/gemma4/models/gemma-4-31B-it-qat-w4a16-ct
W=/mnt/scratch/$USER/g4-judge-$SLURM_JOB_ID; mkdir -p $W/tmp; export TMPDIR=$W/tmp
NGPU=$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr , '\n' | wc -l)
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
bash $C/slurm/stage_vllm_sif.sh >/dev/null || { echo "vllm stage failed"; exit 1; }
IMG=/mnt/scratch/$USER/gemma4/images/vllm-v0.19.1
$D/harness-venv/bin/python - > $W/vllm_cmd.txt <<PY
from adk_submission import VllmConfig, VllmServer
c = VllmConfig(model="$MODEL", port=$PORT, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
               max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
               enable_lora=False, scheduler_reserve_full_isl=False, tensor_parallel_size=$NGPU)
cmd = VllmServer(c).build_cmd(); cmd[0] = "python3"
print("\n".join(cmd + ["--served-model-name", "gemma"]))
PY
S=/mnt/scratch/$USER/gemma4; mkdir -p $S/vllm-cache $S/triton-cache $S/inductor-cache
export VLLM_CACHE_ROOT=$S/vllm-cache TRITON_CACHE_DIR=$S/triton-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
mapfile -t VCMD < $W/vllm_cmd.txt
apptainer exec --nv --bind /mnt/scratch $IMG "${VCMD[@]}" > $HOME/gemma4/logs/vllm-judge-$SLURM_JOB_ID.log 2>&1 &
VPID=$!; t0=$(date +%s)
until curl -sf localhost:$PORT/v1/models >/dev/null; do
  kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -30 $HOME/gemma4/logs/vllm-judge-$SLURM_JOB_ID.log; exit 1; }
  [ $(( $(date +%s)-t0 )) -gt 1500 ] && { echo "vLLM start timeout"; exit 1; }
  sleep 10; done
echo "vLLM ready after $(( $(date +%s)-t0 ))s"
ts=$(date +%s)
$D/harness-venv/bin/python $C/selector/judge.py --features $RUN/features.jsonl --tasks $D/data/tasks.jsonl \
  --raw $RUN --port $PORT --out $RUN/judge_scores.jsonl --n 3 --conc 48
echo "judge took $(( $(date +%s)-ts ))s"
grep -E "generation throughput" $HOME/gemma4/logs/vllm-judge-$SLURM_JOB_ID.log | tail -1
kill $VPID; wait $VPID 2>/dev/null; rm -rf $W
