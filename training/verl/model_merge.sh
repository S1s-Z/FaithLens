# path to the trained model
MODEL_DIR=


python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $MODEL_DIR/your_model_name/global_step_100/actor \
    --target_dir $MODEL_DIR/your_model_name/global_step_100_hf
