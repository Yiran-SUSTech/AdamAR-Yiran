# !/bin/bash
set -x

export nnodes=1
export nproc_per_node=8
export node_rank=0
export master_addr="localhost"
export master_port=29500

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_c2i_copy_GPU.py "$@"
