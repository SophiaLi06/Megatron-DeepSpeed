#!/bin/bash

RECREATE_CHECKPOINTS=0
# add an "-d" option to delete the previous checkpoint
USE_DEEPSPEED=0
# add an "-z" option to enable zero
while getopts "dz" opt; do
  case ${opt} in
    d )
      RECREATE_CHECKPOINTS=1
      ;;
    z )
      USE_DEEPSPEED=1
      ;;
    \? )
      echo "Usage: cmd [-d]"
      ;;
  esac
done

# clear the options
shift $((OPTIND -1))

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=10.0.1.16
MASTER_PORT=$7
NNODES=4
NODE_RANK=$6
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=$1 #<Specify path>
TENSORBOARD_LOGS_PATH=$2 #<Specify path>
VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$5 #<Specify path and file prefix>_text_document

if [ $RECREATE_CHECKPOINTS -eq 1 ]; then
    echo "Deleting previous checkpoint"
    rm -rf $CHECKPOINT_PATH
    mkdir $CHECKPOINT_PATH
fi

DS_CONFIG="./examples/ds_configs/gpt_ds_config.json"
GLOBAL_BATCH=64
MICRO_BATCH=2
ZERO_STAGE=2

PP_STAGE=4

# cat <<EOT > $DS_CONFIG
# {
#   "train_batch_size" : $GLOBAL_BATCH,
#   "train_micro_batch_size_per_gpu": $MICRO_BATCH,
#   "steps_per_print": 100,

#   "zero_optimization": {
#     "stage": $ZERO_STAGE,
#     "contiguous_gradients": true,
#     "overlap_comm": true,
#     "reduce_scatter": true,
#     "reduce_bucket_size": 1e8,
#     "allgather_bucket_size": 1e8,
#     "stage3_max_live_parameters": 1e9,
#     "stage3_max_reuse_distance": 1e9,
#     "stage3_prefetch_bucket_size": 1e7,
#     "stage3_param_persistence_threshold": 1e5,
#     "sub_group_size": 1e9
#   },

#   "fp16": {
#     "enabled": true,
#     "initial_scale_power": 12
#   },

#   "wall_clock_breakdown" : true
# }
# EOT

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 100,

  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": true,
    "reduce_scatter": true,
    "use_multi_rank_bucket_allreduce": false,
    "reduce_bucket_size": 5e7
  },

  "fp16": {
    "enabled": true,
    "initial_scale_power": 12
  },

  "wall_clock_breakdown" : true
}
EOT

ds_args=""
if [ $USE_DEEPSPEED -eq 1 ]; then
    echo "DeepSpeed is enabled"
    ds_args=" --deepspeed ${ds_args}"
    ds_args=" --no-pipeline-parallel ${ds_args}" 
    ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
    ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
    ds_args=" --deepspeed-activation-checkpointing ${ds_args}"
    # ds_args=" --no-persist-layer-norm ${ds_args}"
else
    echo "DeepSpeed is not enabled"
    ds_args=" --pipeline-model-parallel-size $PP_STAGE ${ds_args}"
fi


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.00015 \
    --train-iters 200 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-impl mmap \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $ds_args \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
