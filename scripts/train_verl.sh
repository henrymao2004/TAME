#!/usr/bin/env bash
# Train TAME, GRPO or DAPO with verl v0.7.1.
#   METHOD=tame DATASET=virl bash scripts/train_verl.sh
#   DRY_RUN=1 METHOD=tame DATASET=virl bash scripts/train_verl.sh   # print the command only
set -euo pipefail

METHOD=${METHOD:-tame}
DATASET=${DATASET:-virl}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
GPUS=${GPUS:-8}
DATA=${DATA:-data/$DATASET}
VERL_DATA=${VERL_DATA:-data/${DATASET}_verl}
OUT=${OUT:-runs/${DATASET}_${METHOD}_verl}
SAE=${SAE:-sae/$DATASET}
FEATURES=${FEATURES:-analysis/${DATASET}_step200/features.json}
STEPS=${STEPS:-200}
BATCH=${BATCH:-64}
ROLLOUTS=${ROLLOUTS:-8}
LORA_RANK=${LORA_RANK:-64}
TOKENS_PER_GPU=${TOKENS_PER_GPU:-16384}

TAME_ARGS=(+tame.method="$METHOD" +tame.dataset="$DATASET")
if [ "$METHOD" = tame ]; then
  TAME_ARGS+=(+tame.sae="$SAE" +tame.features="$FEATURES")
fi

CMD=(python -m tame.verl_adapter.main
  algorithm.adv_estimator=grpo
  data.train_files="$VERL_DATA/train.parquet"
  data.val_files="$VERL_DATA/val.parquet"
  data.train_batch_size="$BATCH"
  data.max_prompt_length=4096
  data.max_response_length=1024
  data.filter_overlong_prompts=True
  data.truncation=error
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.lora_rank="$LORA_RANK"
  actor_rollout_ref.model.lora_alpha=$((LORA_RANK * 2))
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.actor.optim.lr=1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size="$BATCH"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKENS_PER_GPU"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n="$ROLLOUTS"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  reward.custom_reward_function.path="$PWD/tame/verl_adapter/reward.py"
  reward.custom_reward_function.name=compute_score
  trainer.use_legacy_worker_impl=enable
  trainer.n_gpus_per_node="$GPUS"
  trainer.nnodes=1
  trainer.total_training_steps="$STEPS"
  trainer.save_freq=10
  trainer.test_freq=-1
  trainer.default_local_dir="$OUT"
  trainer.project_name=tame
  trainer.experiment_name="${DATASET}_${METHOD}"
  trainer.logger='["console"]'
  "${TAME_ARGS[@]}"
  "$@")

if [ "${DRY_RUN:-0}" = 1 ]; then
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

[ -f "$VERL_DATA/train.parquet" ] || python -m tame.verl_adapter.data --src "$DATA" --out "$VERL_DATA" --dataset "$DATASET"
exec "${CMD[@]}"
