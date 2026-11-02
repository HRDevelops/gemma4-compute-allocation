#!/bin/bash
# Re-download given runs from the HPC and recompute every analysis and figure.
# usage: bash update_all.sh <run_name> [run_name ...]
set -e
G=/d/kaggle_competition/gemma4; R=$G/results; C=$G/code/selector
for r in "$@"; do
  ssh nu-hpc "cd /datahub/harirai/\$USER/gemma4/runs/$r && tar czf ../${r}_sel.tgz seed*/patches seed*/traces seed*/task_results.jsonl" 2>&1 | grep -v -i "openssh\|upgraded\|quantum" || true
  mkdir -p $R/$r/raw && scp -q nu-hpc:/datahub/harirai/husnain.rasool/gemma4/runs/${r}_sel.tgz $R/$r/raw/
  (cd $R/$r/raw && tar xzf ${r}_sel.tgz && for d in seed*; do cp $d/task_results.jsonl $R/$r/${d}_task_results.jsonl; done)
  (cd $C && python features.py $R/$r/raw $R/$r $R/$r/features.jsonl)
done
cd $C; R=$(cygpath -m $R)
python frontier.py short=$R/baseline/features.jsonl,$R/baseline_holdout/features.jsonl medium=$R/baseline_medium/features.jsonl long=$R/baseline_long/features.jsonl thorough=$R/long_thorough/features.jsonl --csv $R/frontier.csv > $R/frontier.txt
python noise.py short=$R/short10_features.jsonl medium=$R/baseline_medium/features.jsonl long=$R/baseline_long/features.jsonl thorough=$R/long_thorough/features.jsonl > $R/noise.txt
python paired_tests.py $R > $R/paired_tests.txt
python make_figs.py $R $G/paper/figs
python usage.py short=$R/baseline/features.jsonl,$R/baseline_holdout/features.jsonl medium=$R/baseline_medium/features.jsonl long=$R/baseline_long/features.jsonl thorough=$R/long_thorough/features.jsonl > $R/usage_by_regime.txt
cat $R/usage_by_regime.txt $R/frontier.txt $R/noise.txt $R/paired_tests.txt
