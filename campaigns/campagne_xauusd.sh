#!/bin/bash
# Campagne reelle XAUUSD (forex) — 5 TF : 15m 1h 4h 5m 1d.
# Contexte : fixes DSR/annualisation, CI iid, quotas demarrage, admission val complete (finalistes).
# Log : outputs/campaign_xauusd.log ; unite d'evolution : tasting (convergence ou n_eval 1200).
set -eo pipefail
cd /d/midas_v2/einherjar
LOG="outputs/campaign_xauusd.log"
: > "$LOG"
echo "== campagne XAUUSD 5 TF : $(date '+%F %T') ==" >> "$LOG"
for TF in 15m 1h 4h 5m 1d; do
  echo "===== TF $TF (XAUUSD) =====" >> "$LOG"
  for MODE in compare select refine admit; do
    echo "-- $MODE $TF ..." >> "$LOG"
    timeout 7200 python -m einherjar.research.discovery "$MODE" --data-timeframe "$TF" --data-asset XAUUSD --data-class forex --generators TypedGPGenerator --n-eval 1200 --taste-samples 400 --log-level INFO 2>&1 | tee -a "$LOG"
    rc=$?
    echo "[$TF] $MODE rc=$rc" >> "$LOG"
    if [ $rc -ne 0 ]; then
      echo "[$TF] ABANDON ($MODE rc=$rc)" >> "$LOG"
      exit 1
    fi
  done
  echo "[$TF] TERMINE" >> "$LOG"
done
echo "===== FIN $(date '+%F %T') =====" >> "$LOG"