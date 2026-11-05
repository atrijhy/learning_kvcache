#!/usr/bin/env bash
# Unbiased base for the absorption test: the picked tasks were chosen BECAUSE the base student scored low on them in the
# screen (n=4, seed 1234), so that number is biased low (regression to the mean). Re-measure the base student on the same
# tasks with a fresh seed (777), n=4.  Usage: base_reeval.sh <gpus> <slot> <dom...>
gpus=$1 slot=$2; shift 2
O=/root/local_runs/ovf TC=/root/local_runs/teacher_ceiling
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
OV="actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.seed=777"
for dom in "$@"; do
  [ -f $O/data/${dom}_ab/picked.json ] || continue
  mkdir -p $O/out_base777
  STUDENT_MODEL=/root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64 EXTRA_OVERRIDES="$OV" DATA_DIR=$O/data OUT_DIR=$O/out_base777 JOBS=${dom}_ab QTAG=base777_$dom SLOT=$slot SLOT_GPUS=$gpus \
    bash $TC/teacher_ceiling_slots.sh > $O/out_base777/pipe_$dom.log 2>&1
  echo "[$(date +%T)] base777 $dom done"
done
