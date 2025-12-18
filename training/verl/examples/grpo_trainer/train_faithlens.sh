# #!/usr/bin/env bash

# source /mnt/public/share/users/sishuzheng-share/verl_rl/bin/activate




export OPENAI_API_KEY="ssz"
export OPENAI_BASE_URL="http://0.0.0.0:8000/v1"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6


wandb login ####### your wandb count


# Set flash-rl
# export SGLANG_PATCH=1
# export FLASHRL_LOGGING_LEVEL=DEBUG

# Set ray env and activate venv
export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export RANK=$MLP_ROLE_INDEX
export WORLD_SIZE=$MLP_WORKER_NUM

ray stop
if [ ${RANK:-0} -eq 0 ]; then
    if [ ${WORLD_SIZE:-1} -gt 1 ]; then
      ray start --head --port=$MASTER_PORT --node-ip-address=$MASTER_ADDR --dashboard-host=0.0.0.0
    else
      ray start --head --dashboard-host=0.0.0.0
    fi
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=path_to_rl_data/rl_data_train.parquet \
    data.val_files=ath_to_rl_data/rl_data_train.parquet \
    data.train_batch_size=112 \
    data.max_prompt_length=32768 \
    data.max_response_length=8196 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=path_to_sft_model/checkpoint-1119 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.n=7 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=your_path_to_our_repo/verl/verl/utils/reward_score/ours_reward_with_reason.py \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='halluverifer' \
    trainer.experiment_name='qwen3_8b_acc_format_reward' \
    trainer.n_gpus_per_node=7 \
    trainer.default_local_dir=output_dir_to_trained_model \
    trainer.nnodes=1 \
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]' \
    trainer.val_before_train=False \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@
else
    ray start --address=$MASTER_ADDR:$MASTER_PORT
    sleep 1d
fi