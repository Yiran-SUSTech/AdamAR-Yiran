#!/bin/bash
set -x

export nnodes=1
export nproc_per_node=8
export node_rank=0
export master_addr="localhost"
export master_port=29500

export MACA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_SAME_RAND_A100=1
export MCPYTORCH_DISABLE_PRINT=1

export MCCL_MAX_NCHANNELS=32
export MCCL_MIN_NCHANNELS=4
export MCCL_ALGO=TREE,RING
export MCCL_CROSS_NIC=0

export FORCE_ACTIVE_WAIT=2
export MCCL_FAST_WRITE_BACK=1
export MCCL_EARLY_WRITE_BACK=15

torchrun \
--nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
--master_addr=$master_addr --master_port=$master_port \
autoregressive/train/train_c2i.py "$@"
