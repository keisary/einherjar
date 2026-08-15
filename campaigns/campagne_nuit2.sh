#!/bin/bash
# Nuit 2 (relance post-crash) : finir BTCUSD (5m seul, 15m/1h/4h deja termines au run nuit)
# puis XAUUSD (forex, 5 TF : 15m 1h 4h 5m 1d).
# Fichiers : log dans outputs/, tout le reste dans le repo (rien a la racine).
set -eo pipefail
cd /d/midas_v2/einherjar
LOG="outputs/campaign_nuit2.log"
: > "$LOG"
export PYTHONPATH=src

run_tf() {  # $1=asset $2=class $3=tf
  local asset="$1" cls="$2" tf="$3"
  echo "===== TF $tf ($asset) =====" >> "$LOG"
  for MODE in compare select refine admit; do
    echo "-- $MODE $tf ..." >> "$LOG"
    timeout 3600 python -m einherjar.research.discovery "$MODE" \
      --data-timeframe "$tf" --data-asset "$asset" --data-class "$cls" \
      --generators TypedGPGenerator --n-eval 1200 --taste-samples 400 --log-level INFO 2>&1 | tee -a "$LOG"
    rc=$?
    echo "[$tf] $MODE rc=$rc" >> "$LOG"
    if [ $rc -ne 0 ]; then echo "[$tf] ABANDON ($MODE rc=$rc)" >> "$LOG"; exit 1; fi
  done
}

# 1) BTCUSD : 5m restant.
run_tf BTCUSD crypto 5m

# 2) XAUUSD : 5 TF (forex).
for TF in 15m 1h 4h 5m 1d; do
  run_tf XAUUSD forex "$TF"
done

echo "===== FIN $(date '+%F %T') =====" >> "$LOG"