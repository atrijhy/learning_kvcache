#!/usr/bin/env bash
# 一个域的正式 OPSD（原配方：opsd_mixed 全量、1 epoch=67/66 步、lr 1e-5、batch 24、top-64 forward KL、
# V0 同步路径、3 actor + 1 teacher GPU）→ 导出 prefix → greedy 评测。vLLM 卡死（sample_tokens hang）时自动
# 换 rollout seed 从最近 checkpoint 续跑（每 20 步存一次）。
#   train_full.sh <4个GPU> <awm|envscaler|taco> <评测端口> <初始化目录> <tag>
#   例：train_full.sh 0,1,2,3 envscaler 8181 $MODELS/Qwen3.5-4B-AllAttn-G512A64-vs01 vs01
# 产物：$WORK/train_full_<tag>_<dom>.log，$WORK/ckpt/...，$MODELS/prefix_full_<tag>_<dom>_step<N>，$EVAL_OUT/<同名>/
# 额外 hydra 覆盖：HYDRA_EXTRA="..."（追加在最后，不要再写 rollout.seed）。
# SEEDS（默认 "default 4321 777"）：依次尝试的 rollout seed，第一个是正常训练用的，后面是卡死后续跑用的。
#   做"换 seed 重复实验"时用 SEEDS="11 4321 777" 并换一个 tag。
source "$(dirname "$0")/env.sh"
gpus=$1 dom=$2 port=$3 PREP=$4 tag=$5
[[ -f $PREP/config.json && -n $tag ]] || { echo "用法见脚本头"; exit 2; }
name=full_${tag}_${dom}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WORK/chain_$name.log"; }
extra=(); [[ $dom == taco ]] && extra=(MAX_PROMPT_LENGTH=28672 MAX_RESPONSE_LENGTH=16384 MAX_MODEL_LEN=49152
  PPO_MAX_TOKEN_LEN_PER_GPU=49152 MAX_NUM_BATCHED_TOKENS=49152)
export OPSD_PLUGIN=1 PYTHONPATH=$PLUGIN:$REPO_ROOT/verl:$REPO_ROOT VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900 REPO_ROOT
train() {  # $1 = rollout seed（空 = 默认）
  local seedarg=(); [[ -n $1 ]] && seedarg=(actor_rollout_ref.rollout.seed=$1)
  env PATH=$VENV_BIN:$PATH PYTHON_BIN=$KV_PY RAY_TMPDIR=/tmp/r$(echo $name | md5sum | cut -c1-8) \
    VERL_ZMQ_NAMESPACE=opsd_$name RAY_ADDRESS=local \
    RUN_MODE=full TRAINER_MODE=sync RESUME_MODE=auto TOTAL_EPOCHS=1 TOTAL_TRAINING_STEPS= \
    ACTOR_GPUS=3 TEACHER_GPUS=1 CUDA_VISIBLE_DEVICES=$gpus \
    TRAIN_BATCH_SIZE=24 PPO_MINI_BATCH_SIZE=24 DISTILLATION_TOPK=64 DISTILLATION_TOPK_CHUNK_SIZE=512 "${extra[@]}" \
    ROLLOUT_TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 GPU_MEMORY_UTILIZATION=0.65 NUM_WORKERS=24 FILTER_WORKERS=4 \
    DATALOADER_WORKERS=4 VAL_BEFORE_TRAIN=False TEST_FREQ=1000 SAVE_FREQ=20 USE_WANDB=0 \
    ROLLOUT_ENFORCE_EAGER=False TEACHER_ENFORCE_EAGER=False \
    ACTOR_LR=1e-5 MODEL=$PREP TEACHER_MODEL=$BASE_MODEL \
    DATA_ROOT=$DATA EXPERIMENT_ROOT=$WORK CHECKPOINT_DIR=$WORK/ckpt/$name ROLLOUT_DATA_DIR=$WORK/rollouts/$name RUN_NAME=$name \
    bash "$PLUGIN/internal/train_domain.bash" $dom \
    trainer.use_v1=False +ray_kwargs.ray_init.address=local ray_kwargs.ray_init.num_cpus=28 \
    +ray_kwargs.ray_init.num_gpus=4 +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.object_store_memory=8589934592 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True \
    trainer.max_actor_ckpt_to_keep=null hydra.run.dir=$WORK/hydra/$name "${seedarg[@]}" ${HYDRA_EXTRA:-} \
    >> "$WORK/train_$name.log" 2>&1 < /dev/null
}
ok=0
read -ra SEEDLIST <<< "${SEEDS:-default 4321 777}"
for seed in "${SEEDLIST[@]}"; do
  [[ $seed == default ]] && seed=""
  log "$dom 训练开始 (tag=$tag, init=$PREP, seed=${seed:-default})"
  train "$seed" && { ok=1; break; }
  log "$dom 训练中断，rc=$?；换 seed 从最近 checkpoint 续跑"
  pkill -9 -f "hydra.run.dir=$WORK/hydra/$name" 2>/dev/null; sleep 60   # 清掉残留的 driver，ray worker 会随之退出
done
[[ $ok == 1 ]] || { log "$dom 训练失败（3 个 seed 都没跑完），看 $WORK/train_$name.log"; exit 1; }
s=$(cat "$WORK/ckpt/$name/latest_checkpointed_iteration.txt"); m=$MODELS/prefix_${name}_step$s
log "$dom 训练完成 step $s，导出到 $m"
CUDA_VISIBLE_DEVICES= "$KV_PY" "$PLUGIN/allattn/export_allattn_prefix.py" \
  --checkpoint "$WORK/ckpt/$name/global_step_$s/actor" --prepared "$PREP" --output "$m" | tail -1
bash "$(dirname "$0")/greedy_suite.sh" "$m" "$gpus" "$port" >> "$WORK/chain_$name.log" 2>&1
log "$dom 全部完成（LCB 判分在后台继续，约 20-30 分钟）"
