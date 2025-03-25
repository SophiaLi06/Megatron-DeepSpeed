pip install -r requirements/pytorch\:24.07/requirements.txt
pip install deepspeed
export PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
export CHECKPOINT_PATH="new_checkpoints"
export TENSORBOARD_LOGS_PATH="workspace/logs"
export VOCAB_FILE="dataset/gpt/gpt2-vocab.json"
export MERGE_FILE="dataset/gpt/gpt2-merges.txt"
export DATA_PATH="dataset/gpt/my-gpt2_text_document"
export NCCL_SOCKET_IFNAME=enp37s0f0np0
export GLOO_SOCKET_IFNAME=enp37s0f0np0
export MASTER_PORT=5000