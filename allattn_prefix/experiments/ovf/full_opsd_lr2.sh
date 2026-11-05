#!/usr/bin/env bash
# (copy of full_opsd_lr.sh with per-domain eval) Author-recipe OPSD on the FULL author data of one domain (opsd_mixed_1625_v1, 1625 tasks, 1 epoch = 68 steps,
# Hybrid G512/A64 zero-init, top-64 forward KL) but at prefix lr $lr — the overfit test showed lr 2e-3 doubles absorption
# on EnvScaler (39.6 -> 60.9 vs 50.0 at 5e-5). Then BFCL (author suite) on the final prefix, to compare with base and the
# earlier lr-5e-5 author-recipe prefix (BFCL v3 68.77). Usage: full_opsd_lr.sh <gpus> <dom> <lr>
gpus=$1 dom=$2 lr=$3 port=${4:-8179}
O=/root/local_runs/ovf root=/efs/kadocjiang/learning_kvcache_upstream U=$root/traj_data/opsd_mixed_1625_v1
log() { echo "[$(date +%T)] $*"; }
name=full_opsd_${dom}_lr$lr
extra=""; [[ $dom == taco ]] && extra="MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152 PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152"
bench=bfcl; [[ $dom == taco ]] && bench="humaneval+_s mbpp+_s lcb_s"
log "$dom full OPSD lr $lr"
env PATH=/root/kv_venv/bin:$PATH PYTHON_BIN=/root/kv_venv/bin/python RAY_TMPDIR=/tmp/ray_$name VERL_ZMQ_NAMESPACE=opsd_$name RAY_ADDRESS=local \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_TIMEOUT:-1800} RUN_MODE=full TRAINER_MODE=sync RESUME_MODE=auto TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS= \
  ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=$gpus \
  TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 $extra \
  ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 DATALOADER_WORKERS=4 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=1000 SAVE_FREQ=20 USE_WANDB=0 ROLLOUT_ENFORCE_EAGER=False TEACHER_ENFORCE_EAGER=False \
  ACTOR_LR=$lr MODEL=/root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64 TEACHER_MODEL=/root/kvmodels/Qwen3.5-4B \
  DATA_ROOT=$U EXPERIMENT_ROOT=$O CHECKPOINT_DIR=$O/ckpt/$name ROLLOUT_DATA_DIR=$O/rollouts/$name RUN_NAME=$name \
  bash "$root/prefix_tuning/virtual_prefix/internal/train_domain.bash" $dom \
  trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
  +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=8589934592 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
  trainer.max_actor_ckpt_to_keep=null hydra.run.dir=$O/hydra/$name ${HYDRA_EXTRA:-} > $O/train_$name.log 2>&1 < /dev/null
rc=$?; log "$dom full lr $lr train rc=$rc ($(grep -oE 'Training Progress: +[0-9]+%[^|]*\|[^|]*\| *[0-9]+/[0-9]+' $O/train_$name.log | tail -1))"
[[ $rc == 0 ]] || exit 1
s=$(cat $O/ckpt/$name/latest_checkpointed_iteration.txt); m=/root/kvmodels/prefix_${name}_step$s
CUDA_VISIBLE_DEVICES= PYTHONPATH=$root/verl:$root $root/.venv/bin/python $root/workspace/prefix_training/export_hybrid_prefix.py \
  --checkpoint $O/ckpt/$name/global_step_$s/actor --prepared /root/kvmodels/Qwen3.5-4B-Hybrid-G512-A64 --output $m | tail -1
OUT_ROOT=/root/eval_local/author_suite_full_opsd EAGER=0 PATH=/root/.local/bin:$PATH BENCHMARKS="$bench" \
  bash $root/workspace/prefix_training/run_author_suite_v3_ngpu.sh $m $gpus $port > $O/eval_$name.log 2>&1
log "FULL OPSD DONE $dom $lr"
