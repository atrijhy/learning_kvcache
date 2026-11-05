#!/usr/bin/env bash
# OPSD overfit at a higher prefix lr (the 5e-5 runs absorbed ~0 on their own training tasks while prefix SFT at 2e-3
# absorbed +24 pp on the same TACO tasks: rule out the 40x lr gap before blaming the OPSD signal). Same picked tasks,
# same 20 epochs / 40 steps, same absorption eval (student prompt, no hint, n=4, seed 1234) at steps 40/20/10.
# Usage: ovf_lr.sh <gpus> <slot> <dom> <lr> [wait_file wait_marker]   env TAG=<suffix> HYDRA_EXTRA="<overrides>" for loss variants
gpus=$1 slot=$2 dom=$3 lr=$4 wf=$5 wm=$6
O=/root/local_runs/ovf TC=/root/local_runs/teacher_ceiling root=/efs/kadocjiang/learning_kvcache_upstream
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
log() { echo "[$(date +%T)] $*"; }
OV="actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.seed=1234"
[[ -n $wf ]] && until grep -q "$wm" $wf 2>/dev/null; do sleep 60; done
name=ovf_opsd_${dom}_lr$lr${TAG:+_$TAG}
extra=""; [[ $dom == taco ]] && extra="MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152 PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152"
log "$dom OPSD overfit lr $lr train"
env PATH=/root/kv_venv/bin:$PATH PYTHON_BIN=/root/kv_venv/bin/python RAY_TMPDIR=/tmp/r$(echo $name | md5sum | cut -c1-8) VERL_ZMQ_NAMESPACE=opsd_$name RAY_ADDRESS=local \
  RUN_MODE=full TRAINER_MODE=sync RESUME_MODE=auto TOTAL_EPOCHS=20 TOTAL_TRAINING_STEPS= \
  ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=$gpus \
  TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 $extra \
  ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 DATALOADER_WORKERS=4 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=1000 SAVE_FREQ=10 USE_WANDB=0 ROLLOUT_ENFORCE_EAGER=False TEACHER_ENFORCE_EAGER=False \
  ACTOR_LR=$lr MODEL=${PREP:-/root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64} TEACHER_MODEL=/root/kvmodels/Qwen3.5-4B \
  DATA_ROOT=$O/data/${dom}_ab EXPERIMENT_ROOT=$O CHECKPOINT_DIR=$O/ckpt/$name ROLLOUT_DATA_DIR=$O/rollouts/$name RUN_NAME=$name \
  bash "${TRAIN_SCRIPT:-$root/prefix_tuning/virtual_prefix/internal/train_domain.bash}" $dom \
  trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
  +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=8589934592 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
  trainer.max_actor_ckpt_to_keep=null hydra.run.dir=$O/hydra/$name ${HYDRA_EXTRA:-} > $O/train_$name.log 2>&1 < /dev/null
rc=$?; log "$dom lr $lr train rc=$rc ($(grep -oE 'Training Progress: +[0-9]+%[^|]*\|[^|]*\| *[0-9]+/[0-9]+' $O/train_$name.log | tail -1))"
[[ $rc == 0 ]] || { log "LR OVF FAILED $dom $lr"; exit 1; }
for s in 40 20 10; do
  [[ -d $O/ckpt/$name/global_step_$s ]] || continue
  m=/root/kvmodels/prefix_${name}_step$s
  [[ -e $m/config.json ]] || CUDA_VISIBLE_DEVICES= PYTHONPATH=$root/verl:$root $root/.venv/bin/python ${EXPORT_PY:-$root/workspace/prefix_training/export_hybrid_prefix.py} \
    --checkpoint $O/ckpt/$name/global_step_$s/actor --prepared ${PREP:-/root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64} --output $m | tail -1
  mkdir -p $O/out_ab_lr${lr}${TAG:+_$TAG}_step$s
  STUDENT_MODEL=$m EXTRA_OVERRIDES="$OV" DATA_DIR=$O/data OUT_DIR=$O/out_ab_lr${lr}${TAG:+_$TAG}_step$s JOBS=${dom}_ab QTAG=ab_${name}_$s SLOT=$slot SLOT_GPUS=$gpus \
    bash ${TC_SLOTS:-$TC/teacher_ceiling_slots.sh} > $O/out_ab_lr${lr}${TAG:+_$TAG}_step$s/pipe_$dom.log 2>&1
  log "$dom lr $lr absorb step$s done"
done
log "LR OVF DONE $dom $lr ${TAG:-}"
