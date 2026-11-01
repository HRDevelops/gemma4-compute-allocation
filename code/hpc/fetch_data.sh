#!/bin/bash
#SBATCH -J g4-fetch-data
#SBATCH -p ZEN4
#SBATCH -A issai
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /shared/home/husnain.rasool/gemma4/logs/%x-%j.out
# Downloads competition files from pre-signed GCS URLs (urls.tsv: name<TAB>size<TAB>url).
set -uo pipefail
D=/datahub/harirai/$USER/gemma4/data
L=$HOME/gemma4/code/urls.tsv
mkdir -p $D
fetch() {  # name size url
  local n=$1 s=$2 u=$3 f=$D/$1
  if [ -f "$f" ] && [ "$s" != 0 ] && [ "$(stat -c %s "$f")" = "$s" ]; then return 0; fi
  mkdir -p "$(dirname "$f")"
  curl -sfL --retry 5 --retry-delay 5 -o "$f.part" "$u" && mv "$f.part" "$f" || echo "FAIL $n"
}
export -f fetch; export D
t0=$(date +%s)
awk -F'\t' '{print $1"\t"$2"\t"$3}' $L | tr '\t' '\n' | xargs -d '\n' -n 3 -P 12 bash -c 'fetch "$0" "$1" "$2"'
echo "done in $(( $(date +%s)-t0 ))s"
# verify sizes against the listing
awk -F'\t' -v D=$D '$2>0{f=D"/"$1; cmd="stat -c %s \""f"\" 2>/dev/null"; cmd|getline s; close(cmd); if(s!=$2){print "SIZE_MISMATCH",$1,s,$2; bad++}; s=""} END{print "mismatches:",bad+0}' $L
du -sh $D/*
