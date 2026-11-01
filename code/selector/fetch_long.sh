#!/bin/bash
# Wait for long-budget jobs, download results, build features, run comparison.
F="grep -v -i openssh\|upgraded\|quantum"
ssh nu-hpc 'while squeue -h -j 35357,35358,35359 | grep -q .; do sleep 120; done; sacct -X -n -P -j 35357,35358,35359 -o JobName,State,NodeList,Elapsed; grep -hE "SEED=|took|Traceback|died|failed" ~/gemma4/logs/g4-long-*-3535[789].out; cd /datahub/harirai/$USER/gemma4/runs/baseline_long && tar czf ../baseline_long_sel.tgz seed*/patches seed*/traces seed*/task_results.jsonl' 2>&1 | eval $F
L=/d/kaggle_competition/gemma4/results/baseline_long; mkdir -p $L/raw
scp -q nu-hpc:/datahub/harirai/husnain.rasool/gemma4/runs/baseline_long_sel.tgz $L/raw/ && cd $L/raw && tar xzf baseline_long_sel.tgz
for d in seed*; do cp $d/task_results.jsonl $L/${d}_task_results.jsonl; done
cd /d/kaggle_competition/gemma4/code/selector
python features.py $L/raw $L $L/features.jsonl
python compare.py ../../results/baseline/features.jsonl $L/features.jsonl | tee $L/compare.txt
