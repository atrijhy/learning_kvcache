#!/usr/bin/env bash
# 0-3: after the G512A64 overfit chain (EnvScaler, AWM): formal EnvScaler, then formal AWM.
O=/root/local_runs/ovf
until grep -q "CHAIN G512A64 03 DONE" $O/chain_g512a64_03.log 2>/dev/null; do sleep 60; done
bash $O/full_allattn.sh 0,1,2,3 envscaler 8181 > $O/chain_full_g512a64_envscaler.log 2>&1
bash $O/full_allattn.sh 0,1,2,3 awm 8181 > $O/chain_full_g512a64_awm.log 2>&1
echo "[$(date +%T)] CHAIN FULL 03 DONE"
