#!/usr/bin/env bash
# Formal author-config OPSD on the MAIN fixed structure (AllAttn-G512A64): full author data opsd_mixed_1625_v1 of one
# domain, 1 epoch (67/66 steps), lr 1e-5, batch 24, top-64 fwd KL, sync V0 path, 3 actor + 1 teacher GPU, domain
# thinking/turn/tool-length defaults from train_domain.bash (= reproduce_20260923.sh). Differences from the author run:
# no in-training val (VAL_BEFORE_TRAIN/TEST_FREQ only evaluate, they do not change training) and CUDA graphs on
# (ROLLOUT/TEACHER_ENFORCE_EAGER=False, same as all ovf runs). Then export + greedy author-table eval (BFCL v3 all,
# HumanEval+/MBPP+, LCB v6) for every domain prefix.   Usage: full_allattn.sh <gpus> <dom> <port>
gpus=$1 dom=$2 port=$3
O=/root/local_runs/ovf root=/efs/kadocjiang/learning_kvcache_upstream U=$root/traj_data/opsd_mixed_1625_v1
PREP=/root/kvmodels/Qwen3.5-4B-AllAttn-G512A64
export OPSD_PLUGIN=1 PYTHONPATH=/root/opsd_plugin:$root/verl:$root VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900
log() { echo "[$(date +%T)] $*"; }
name=full_g512a64_${dom}_lr1e-5
extra=""; [[ $dom == taco ]] && extra="MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152 PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152"
log "$dom FULL OPSD G512A64 lr 1e-5 train"
env PATH=/root/kv_venv/bin:$PATH PYTHON_BIN=/root/kv_venv/bin/python RAY_TMPDIR=/tmp/r$(echo $name | md5sum | cut -c1-8) VERL_ZMQ_NAMESPACE=opsd_$name RAY_ADDRESS=local \
  RUN_MODE=full TRAINER_MODE=sync RESUME_MODE=auto TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS= \
  ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=$gpus \
  TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 $extra \
  ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 DATALOADER_WORKERS=4 \
  VAL_BEFORE_TRAIN=False TEST_FREQ=1000 SAVE_FREQ=20 USE_WANDB=0 ROLLOUT_ENFORCE_EAGER=False TEACHER_ENFORCE_EAGER=False \
  ACTOR_LR=1e-5 MODEL=$PREP TEACHER_MODEL=/root/kvmodels/Qwen3.5-4B \
  DATA_ROOT=$U EXPERIMENT_ROOT=$O CHECKPOINT_DIR=$O/ckpt/$name ROLLOUT_DATA_DIR=$O/rollouts/$name RUN_NAME=$name \
  bash /root/opsd_plugin/internal/train_domain.bash $dom \
  trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
  +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory=8589934592 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
  actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
  trainer.max_actor_ckpt_to_keep=null hydra.run.dir=$O/hydra/$name > $O/train_$name.log 2>&1 < /dev/null
rc=$?; log "$dom full G512A64 train rc=$rc ($(grep -oE 'Training Progress: +[0-9]+%[^|]*\|[^|]*\| *[0-9]+/[0-9]+' $O/train_$name.log | tail -1))"
[[ $rc == 0 ]] || { log "FULL G512A64 FAILED $dom"; exit 1; }
s=$(cat $O/ckpt/$name/latest_checkpointed_iteration.txt); m=/root/kvmodels/prefix_${name}_step$s
CUDA_VISIBLE_DEVICES= $root/.venv/bin/python /root/opsd_plugin/allattn/export_allattn_prefix.py \
  --checkpoint $O/ckpt/$name/global_step_$s/actor --prepared $PREP --output $m | tail -1
bash /root/local_runs/greedy/greedy_suite.sh $m $gpus $port > $O/eval_$name.log 2>&1
log "FULL G512A64 DONE $dom (eval generation done; LCB judging continues in background)"
