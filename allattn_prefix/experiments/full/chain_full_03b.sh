#!/usr/bin/env bash
# 0-3 (relaunch 2026-11-04 23:15): EnvScaler formal run died at step 39 on the vLLM sample_tokens hang -> resume from
# global_step_20 with a new rollout seed (retry once more with another seed if it dies again), then formal AWM.
O=/root/local_runs/ovf
for seed in 4321 777; do
  HYDRA_EXTRA="actor_rollout_ref.rollout.seed=$seed" bash $O/full_allattn2.sh 0,1,2,3 envscaler 8181 >> $O/chain_full_g512a64_envscaler.log 2>&1 && break
  cp $O/train_full_g512a64_envscaler_lr1e-5.log $O/train_full_g512a64_envscaler_lr1e-5_crash_seed$seed.log
done
bash $O/full_allattn2.sh 0,1,2,3 awm 8181 > $O/chain_full_g512a64_awm.log 2>&1
echo "[$(date +%T)] CHAIN FULL 03 DONE"
