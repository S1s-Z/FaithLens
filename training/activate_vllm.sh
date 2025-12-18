
export VLLM_LOGGING_LEVEL=ERROR

CUDA_VISIBLE_DEVICES=7 \
vllm serve local_path_to_llama/Llama-3.1-8B-Instruct \
  --served-model-name llama3-8b-instruct \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.7 \
  --api-key ssz \
  --dtype auto