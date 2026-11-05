#!/usr/bin/env bash
# Teacher ceiling: does the base model (identity zero-init prefix) solve the training tasks better when it sees
# the privileged advice exactly as the OPSD teacher does ("\n\n"+ground_truth appended to the last user turn)?
# DATA_DIR / OUT_DIR / SLOT / SLOT_GPUS / EXTRA_OVERRIDES: reuse for other validation-only passes (e.g. the TACO pool pass).
# Validation-only runs of the training stack (same agent loops, tool envs and rewards), A = plain, B = with advice.
set -uo pipefail
root=/efs/kadocjiang/learning_kvcache_upstream
data=${DATA_DIR:-/root/local_runs/teacher_ceiling/data} out=${OUT_DIR:-/root/local_runs/teacher_ceiling/out}
queue=$out/queue_${QTAG:-main}.txt; mkdir -p "$out"; printf '%s\n' ${JOBS:-awm_A awm_B envscaler_A envscaler_B taco_A taco_B} > "$queue"
log() { echo "[$(date +%F' '%T)] $*"; }
next_job() { ( flock 9; line=$(head -1 "$queue"); [[ -n "$line" ]] && sed -i 1d "$queue"; echo "$line" ) 9>"$queue.lock"; }
cleanup_slot() {
  local slot=$1 sig p pids
  for sig in TERM KILL; do
    pids=(); for p in /proc/[0-9]*; do grep -qz "^VERL_ZMQ_NAMESPACE=tc_slot${slot}_" "$p/environ" 2>/dev/null && pids+=("${p#/proc/}"); done
    (( ${#pids[@]} )) && kill -s "$sig" "${pids[@]}" 2>/dev/null; sleep 15
  done
}
run_one() {  # <slot> <gpus> <job>
  local slot=$1 gpus=$2 job=$3 domain=${3%_*}
  log "slot$slot ($gpus): $job"
  PATH=/root/kv_venv/bin:$PATH PYTHON_BIN=/root/kv_venv/bin/python \
    RAY_TMPDIR=/tmp/kvtc$slot VERL_ZMQ_NAMESPACE=tc_slot${slot}_$job RAY_ADDRESS=local \
    RUN_MODE=full TRAINER_MODE=sync RESUME_MODE=disable TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS= \
    ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=$gpus \
    TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 \
    MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152 \
    PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152 \
    ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 \
    GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 DATALOADER_WORKERS=4 \
    VAL_BEFORE_TRAIN=True TEST_FREQ=20 SAVE_FREQ=1000 USE_WANDB=0 \
    ROLLOUT_ENFORCE_EAGER=False TEACHER_ENFORCE_EAGER=False \
    ACTOR_LR=1e-5 MODEL=${STUDENT_MODEL:-/root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64} TEACHER_MODEL=/root/kvmodels/Qwen3.5-4B \
    DATA_ROOT=$data/$job EXPERIMENT_ROOT=$out CHECKPOINT_DIR=/tmp/kvtc_ckpt_$job \
    ROLLOUT_DATA_DIR=$out/rollouts_$job RUN_NAME=teacher_ceiling_$job \
    bash "${TRAIN_SCRIPT:-$root/prefix_tuning/virtual_prefix/internal/train_domain.bash}" "$domain" \
    trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
    +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.object_store_memory=8589934592 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
    trainer.val_only=True trainer.validation_data_dir=$out/val_$job hydra.run.dir=$out/hydra_$job ${EXTRA_OVERRIDES:-} \
    > "$out/$job.log" 2>&1 < /dev/null
  log "slot$slot: $job rc=$?"; cleanup_slot "$slot"
}
slot_loop() { local job; while job=$(next_job) && [[ -n "$job" ]]; do run_one "$1" "$2" "$job"; done; }
for p in 8899 8900 8901; do curl -sf -m 5 localhost:$p/health >/dev/null || log "WARNING: tool service $p not healthy"; done
slot_loop ${SLOT:-0} ${SLOT_GPUS:-0,1,2,3} &
wait; log "TEACHER CEILING DONE"
