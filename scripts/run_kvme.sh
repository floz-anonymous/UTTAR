#!/bin/bash
# UTTAR on KVME, end to end. Usage: KVME=/path/to/KVME_ROOT UCLIP=/path/to/checkpoints.pt bash scripts/run_kvme.sh
set -euo pipefail
: "${KVME:?set KVME to the KVME dataset folder}"
: "${UCLIP:?set UCLIP to the released Ultrasound-CLIP checkpoints.pt}"
cd "$(dirname "$0")/.."
OUT=${OUT:-runs}
PY=${PY:-python}

$PY uttar/cache_features.py --kvme "$KVME" --ckpt "$UCLIP" --out $OUT/cache
$PY uttar/train.py --kvme "$KVME" --patch-cache $OUT/cache --out $OUT/uttar
for S in val test; do
  $PY uttar/infer.py --kvme "$KVME" --patch-cache $OUT/cache --run $OUT/uttar --split $S
  $PY uttar/render_qa.py --kvme "$KVME" --pred $OUT/uttar/predictions_$S.jsonl
  $PY uttar/baselines.py --kvme "$KVME" --ckpt "$UCLIP" --split $S --out $OUT/baselines
  $PY uttar/evaluate.py --kvme "$KVME" --pred $OUT/uttar/predictions_$S.jsonl \
      --baseline $OUT/baselines/predictions_$S.jsonl --out $OUT/uttar/eval_$S.json
done
