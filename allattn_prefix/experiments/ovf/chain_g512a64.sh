#!/usr/bin/env bash
# MAIN fixed structure, consistent with the author's sizes (GDN 512 / attention 64): Qwen3.5-4B-AllAttn-G512A64 — the
# 24 GDN layers' 512-token prefix becomes a residual softmax-attention branch (no decay); the 8 attention layers keep
# the author's A64 branch unchanged. Overfit tests at the author lr 1e-5.
#  03: EnvScaler then AWM (now);  47: TACO after the running A512 TACO run.   Usage: chain_g512a64.sh {03|47}
O=/root/local_runs/ovf
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin${PYTHONPATH:+:$PYTHONPATH}
export PREP=/root/kvmodels/Qwen3.5-4B-AllAttn-G512A64 TAG=g512a64 TRAIN_SCRIPT=/root/opsd_plugin/internal/train_domain.bash \
  EXPORT_PY=/root/opsd_plugin/allattn/export_allattn_prefix.py TC_SLOTS=/root/opsd_plugin/internal/teacher_ceiling_slots.sh
if [[ $1 == 03 ]]; then
  bash $O/ovf_lr3.sh 0,1,2,3 0 envscaler 1e-5 > $O/chain_lr_env_g512a64.log 2>&1
  bash $O/ovf_lr3.sh 0,1,2,3 0 awm 1e-5 > $O/chain_lr_awm_g512a64.log 2>&1
else
  until grep -q "LR OVF DONE taco 1e-5" $O/chain_lr_taco1e-5_allattn.log; do sleep 60; done
  bash $O/ovf_lr3.sh 4,5,6,7 1 taco 1e-5 > $O/chain_lr_taco_g512a64.log 2>&1
fi
echo "[$(date +%T)] CHAIN G512A64 $1 DONE"
