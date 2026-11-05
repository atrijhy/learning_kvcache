#!/usr/bin/env bash
# 0-3 (2026-11-05 02:40, replaces chain_full_03b after the EnvScaler formal run): the EnvScaler prefix scored BELOW base
# (BFCL 65.46 vs 67.93, HE 82.9 vs 90.2). Sanity check first: the untrained AllAttn-G512A64 (V=0, must equal base)
# through the same greedy suite / vLLM path; then the formal AWM run.
O=/root/local_runs/ovf
while pgrep -f "greedy_suite.sh /root/kvmodels/prefix_full_g512a64_envscaler" >/dev/null && ! grep -q "generation done" $O/eval_full_g512a64_envscaler_lr1e-5.log; do sleep 30; done
while pgrep -f "vllm serve /root/kvmodels/prefix_full_g512a64_envscaler" >/dev/null; do sleep 10; done
echo "[$(date +%T)] sanity eval: untrained G512A64"
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin:/efs/kadocjiang/learning_kvcache_upstream/verl:/efs/kadocjiang/learning_kvcache_upstream
bash /root/local_runs/greedy/greedy_suite.sh /root/kvmodels/Qwen3.5-4B-AllAttn-G512A64 0,1,2,3 8183 > $O/eval_init_g512a64.log 2>&1
echo "[$(date +%T)] sanity eval generation done"
bash $O/full_allattn2.sh 0,1,2,3 awm 8181 > $O/chain_full_g512a64_awm.log 2>&1
echo "[$(date +%T)] CHAIN FULL 03 DONE"
