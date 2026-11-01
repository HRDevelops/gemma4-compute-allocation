#!/bin/bash
#SBATCH -J g4-agent
#SBATCH -p H100
#SBATCH -A issai
#SBATCH --qos=issai-h100
#SBATCH --gres=gpu:h100:2
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
# usage: sbatch agent_job.sh <run_name> <submission_dir> <concurrency> <seed> [seed ...] [-- task ids]
set -uo pipefail
RUN=$1; SUB=$2; CONC=$3; shift 3
SEEDS=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do SEEDS+=($1); shift; done; [ "${1:-}" = "--" ] && shift
IDS=("$@")
D=/datahub/harirai/$USER/gemma4; C=$HOME/gemma4/code
MODEL=$HOME/gemma4/models/gemma-4-31B-it-qat-w4a16-ct
W=/mnt/scratch/$USER/g4-agent-$SLURM_JOB_ID; mkdir -p $W/tmp; export TMPDIR=$W/tmp
NGPU=$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr , '\n' | wc -l)
PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
echo "[$(hostname)] run=$RUN seeds=${SEEDS[*]} conc=$CONC gpus=$CUDA_VISIBLE_DEVICES port=$PORT"

# images: vLLM (node-local, rebuilt if purged) + sandbox
bash $C/slurm/stage_vllm_sif.sh >/dev/null || { echo "vllm stage failed"; exit 1; }
IMG=/mnt/scratch/$USER/gemma4/images/vllm-v0.19.1
cp $D/images/sbx2.sif $W/sbx.sif
mkdir -p $W/usr_local_seed && apptainer -s exec --bind $W $W/sbx.sif cp -a /usr/local/. $W/usr_local_seed/ \
  && test -x $W/usr_local_seed/bin/python3 || { echo "seed copy failed"; exit 1; }

# vLLM command exactly as adk_submission builds it on Kaggle (only TP differs: 2xH100 vs 4xL4)
$D/harness-venv/bin/python - > $W/vllm_cmd.txt <<PY
from adk_submission import VllmConfig, VllmServer
c = VllmConfig(model="$MODEL", port=$PORT, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4",
               max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
               enable_lora=False, scheduler_reserve_full_isl=False, tensor_parallel_size=$NGPU)
cmd = VllmServer(c).build_cmd(); cmd[0] = "python3"
print("\n".join(cmd))
PY
echo "vLLM cmd: $(tr '\n' ' ' < $W/vllm_cmd.txt)"
S=/mnt/scratch/$USER/gemma4; mkdir -p $S/vllm-cache $S/triton-cache $S/inductor-cache
export VLLM_CACHE_ROOT=$S/vllm-cache TRITON_CACHE_DIR=$S/triton-cache TORCHINDUCTOR_CACHE_DIR=$S/inductor-cache
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
mapfile -t VCMD < $W/vllm_cmd.txt
apptainer exec --nv --bind /mnt/scratch $IMG "${VCMD[@]}" > $HOME/gemma4/logs/vllm-agent-$SLURM_JOB_ID.log 2>&1 &
VPID=$!
t0=$(date +%s)
until curl -sf localhost:$PORT/v1/models >/dev/null; do
  kill -0 $VPID 2>/dev/null || { echo "vLLM died"; tail -30 $HOME/gemma4/logs/vllm-agent-$SLURM_JOB_ID.log; exit 1; }
  [ $(( $(date +%s)-t0 )) -gt 1500 ] && { echo "vLLM start timeout"; exit 1; }
  sleep 10; done
echo "vLLM ready after $(( $(date +%s)-t0 ))s"; grep -E "KV cache size|Maximum concurrency" $HOME/gemma4/logs/vllm-agent-$SLURM_JOB_ID.log | tail -2

for SEED in "${SEEDS[@]}"; do
  OUT=$D/runs/$RUN/seed$SEED
  [ -f $OUT/DONE ] && { echo "seed $SEED already done"; continue; }
  ts=$(date +%s)
  $D/harness-venv/bin/python $C/run_agent.py --submission $SUB --seed $SEED --data $D/data --out $OUT \
    --model-path $MODEL --port $PORT --concurrency $CONC --tasks-csv $C/gate1_soundness.csv \
    --sif $W/sbx.sif --work $W --seed-usr-local $W/usr_local_seed --wheels $D/data/wheels_plus \
    ${IDS[@]:+--task-ids "${IDS[@]}"} 2>&1 | grep -E "^SEED=|Traceback|Error:" | head -20
  echo "seed $SEED took $(( $(date +%s)-ts ))s"; [ ${#IDS[@]} -eq 0 ] && touch $OUT/DONE
done
kill $VPID; wait $VPID 2>/dev/null; rm -rf $W
