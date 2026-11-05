#!/usr/bin/env bash
# 4-7 (replaces chain_full_47.sh, 2026-11-04 23:15): wait for the running formal TACO; if it died (vLLM sample_tokens
# hang), resume from its last checkpoint with new rollout seeds; then the AllAttn-A64 capacity control.
O=/root/local_runs/ovf
while pgrep -f "full_allattn.sh 4,5,6,7 taco" >/dev/null; do sleep 60; done
if ! grep -q "FULL G512A64 DONE taco" $O/chain_full_g512a64_taco.log; then
  for seed in 4321 777; do
    cp $O/train_full_g512a64_taco_lr1e-5.log $O/train_full_g512a64_taco_lr1e-5_crash_$(date +%H%M).log
    HYDRA_EXTRA="actor_rollout_ref.rollout.seed=$seed" bash $O/full_allattn2.sh 4,5,6,7 taco 8182 >> $O/chain_full_g512a64_taco.log 2>&1 && break
  done
fi
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin${PYTHONPATH:+:$PYTHONPATH}
export PREP=/root/kvmodels/Qwen3.5-4B-AllAttn-A64 TAG=a64 TRAIN_SCRIPT=/root/opsd_plugin/internal/train_domain.bash \
  EXPORT_PY=/root/opsd_plugin/allattn/export_allattn_prefix.py TC_SLOTS=/root/opsd_plugin/internal/teacher_ceiling_slots.sh
bash $O/ovf_lr3.sh 4,5,6,7 1 envscaler 1e-5 > $O/chain_lr_env_a64.log 2>&1
echo "[$(date +%T)] CHAIN FULL 47 DONE"
