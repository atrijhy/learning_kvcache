#!/usr/bin/env bash
# 4-7: after the G512A64 TACO overfit: formal TACO, then the AllAttn-A64 capacity control (EnvScaler overfit, lr 1e-5).
O=/root/local_runs/ovf
until grep -q "CHAIN G512A64 47 DONE" $O/chain_g512a64_47.log 2>/dev/null; do sleep 60; done
bash $O/full_allattn.sh 4,5,6,7 taco 8182 > $O/chain_full_g512a64_taco.log 2>&1
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin${PYTHONPATH:+:$PYTHONPATH}
export PREP=/root/kvmodels/Qwen3.5-4B-AllAttn-A64 TAG=a64 TRAIN_SCRIPT=/root/opsd_plugin/internal/train_domain.bash \
  EXPORT_PY=/root/opsd_plugin/allattn/export_allattn_prefix.py TC_SLOTS=/root/opsd_plugin/internal/teacher_ceiling_slots.sh
bash $O/ovf_lr3.sh 4,5,6,7 1 envscaler 1e-5 > $O/chain_lr_env_a64.log 2>&1
echo "[$(date +%T)] CHAIN FULL 47 DONE"
