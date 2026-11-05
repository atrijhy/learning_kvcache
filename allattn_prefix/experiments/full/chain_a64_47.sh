#!/usr/bin/env bash
# Capacity control: AllAttn-A64 (all 32 layers 64 tokens, 10.5M params vs old Hybrid G512-A64 78.9M) — EnvScaler overfit
# at the author lr 1e-5, after the G512A64 TACO run on 4-7.
O=/root/local_runs/ovf
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin${PYTHONPATH:+:$PYTHONPATH}
export PREP=/root/kvmodels/Qwen3.5-4B-AllAttn-A64 TAG=a64 TRAIN_SCRIPT=/root/opsd_plugin/internal/train_domain.bash \
  EXPORT_PY=/root/opsd_plugin/allattn/export_allattn_prefix.py TC_SLOTS=/root/opsd_plugin/internal/teacher_ceiling_slots.sh
until grep -q "CHAIN G512A64 47 DONE" $O/chain_g512a64_47.log 2>/dev/null; do sleep 60; done
bash $O/ovf_lr3.sh 4,5,6,7 1 envscaler 1e-5 > $O/chain_lr_env_a64.log 2>&1
echo "[$(date +%T)] CHAIN A64 47 DONE"
