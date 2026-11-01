#!/bin/bash
# Submit Gate 1 per repo; each repo's own released wheel is left out of injection.
set -euo pipefail
D=/datahub/harirai/$USER/gemma4/data
for short in fastapi rich requests httpx; do
  W=$D/wheels_eval_$short; rm -rf $W; mkdir -p $W
  for f in $D/wheels_plus/*.whl; do
    b=$(basename $f); dist=$(echo "${b%%-*}" | tr 'A-Z_.' 'a-z--')
    [ "$dist" = "$short" ] || ln -s $f $W/$b
  done
  echo "$short: $(ls $W | wc -l) wheels"
  IDS=$(python3 -c "import json;print(' '.join(t['instance_id'] for t in map(json.loads,open('$D/tasks.jsonl')) if t['repo'].split('/')[1]=='$short'))")
  for mode in gold empty; do
    WHEELS=wheels_eval_$short sbatch --parsable -J g4-gate1-$mode-$short --export=ALL,WHEELS=wheels_eval_$short ~/gemma4/code/gate1.sh $mode 24 $IDS
  done
done
