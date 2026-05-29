#!/usr/bin/env bash

ray stop

set -x
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
IFS=',' read -r -a CUDA_DEVICE_IDS <<< "${CUDA_VISIBLE_DEVICES}"
VISIBLE_GPU_COUNT="${#CUDA_DEVICE_IDS[@]}"

ray start --head --num-gpus="${VISIBLE_GPU_COUNT}" --port=6393
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../tasks/proofwriter/train.parquet \
    data.val_files=../tasks/proofwriter/test.parquet \
    custom_reward_function.path=../tasks/proofwriter/reward_function.py \
    custom_reward_function.name=reward_function_verl \
    data.train_batch_size=200 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-3B-Instruct \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=30 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='catastrophic-forgetting-grpo' \
    trainer.experiment_name="Llama-3.2-3B-Instruct_grpo_proofwriter_s42" \
    trainer.n_gpus_per_node="${VISIBLE_GPU_COUNT}" \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=500 \
    trainer.total_epochs=500 \
    data.seed=42 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.ref.fsdp_config.seed=42 \
    "$@" 2>&1 | tee grpo_llama_proofwriter_3b_s42.log
