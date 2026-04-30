#!/bin/bash

CONFIG=${1}
WORK_DIR=${2}

CODE_DIR=${3}

cd ${CODE_DIR}

echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"


# export PATH=/kaiwu_vepfs_volc/kaiwu_vepfs_volc/fuzuoyi/anaconda3/bin:$PATH
# source /kaiwu_vepfs_volc/kaiwu_vepfs_volc/perception/xiyunlong/my_envs/scripts/cosmos_py310
# source /kaiwu_vepfs_volc/kaiwu_vepfs_volc/fuzuoyi/act_scripts/cosmos_py310

# conda activate /kaiwu_vepfs_volc/kaiwu_vepfs_volc/perception/xiyunlong/my_envs/envs/giga

which python
which pip

export PYTHONPATH=${CODE_DIR}:$PYTHONPATH
export COSMOS_PREDICT2_ARGS='--checkpoints /mnt/afs/liyu1/git_models'
export QWORLD_HF_CHECKPOINTS_ROOT=/mnt/afs/liyu1/git_models
export QWORLD_FILE_CLIENT_NAME=tos

export IMAGINAIRE_OUTPUT_ROOT=$WORK_DIR


# export MACA_HOME=/opt/maca-3.3.0
# export PATH=$MACA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$MACA_HOME/lib64:$LD_LIBRARY_PATH
# export TORCH_COMPILE_DISABLE=1

echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"

TORCH_EXTENSIONS_DIR=/mnt/afs/liyu1/rely/tmp/tmp_deepspeed

python -c "import os;print('os.environ:',os.environ)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 2. 解析CUDA_VISIBLE_DEVICES，计算可见GPU数量
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" = "all" ]; then
    # 未设置/设置为all时，取全局GPU数量
    nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    # 解析逗号分隔的GPU列表，统计数量
    nproc_per_node=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi

echo "nproc_per_node: $nproc_per_node"

# COUNT=$(python -c "import torch; print(torch.cuda.device_count())")


# *********************************************************
export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_TRACE_BUFFER_SIZE=4096
# export NCCL_TIMEOUT=1800



# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_ASYNC_ERROR_HANDLING=0
# export NCCL_BLOCKING_WAIT=1
# export CUDA_LAUNCH_BLOCKING=1

# export TORCH_NCCL_BLOCKING_WAIT=1
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
# *******************************************************

export NCCL_IB_DISABLE=0
export NCCL_CROSS_NIC=0       # 固定每个网卡的连接通道
export NCCL_ALGO=TREE,RING

# export NCCL_ALGO=TREE
# export NCCL_ALGO=RING
# export OMP_NUM_THREADS=8

# export NCCL_P2P_DISABLE=0
# export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4 

# export NCCL_IB_SPLIT_PLANS=1
# export NCCL_IB_GID_INDEX=3 

# export NCCL_MAX_NCHANNELS=32
# export NCCL_MIN_NCHANNELS=4

# export NCCL_NET_GDR_LEVEL=2

# *******************************************************

# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_BLOCKING_WAIT=1

# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_IB_TIMEOUT=0
# export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR}
export PYTHONPATH=${CODE_DIR}:$PYTHONPATH


# 如果环境变量没设置，就用默认值
export MASTER_ADDR=${MLP_WORKER_0_HOST:-"127.0.0.1"}
export MASTER_PORT=${MLP_WORKER_0_PORT:-"29500"}
export WORLD_SIZE=${MLP_WORKER_NUM:-1}
export RANK=${MLP_ROLE_INDEX:-0}

export NUM_PROCESSES=$(($WORLD_SIZE * $nproc_per_node))

echo "ENV || NUM_PROCESSES: ${NUM_PROCESSES}"
echo "ENV || MASTER_ADDR = $MASTER_ADDR"
echo "ENV || MASTER_PORT = $MASTER_PORT"
echo "ENV || WORLD_SIZE  = $WORLD_SIZE"
echo "ENV || RANK        = $RANK"



# cp /kaiwu_vepfs_volc/kaiwu_vepfs_volc/fuzuoyi/petreloss.conf /mnt/lstore/


# torchrun --nnodes=$WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT --nproc-per-node=${nproc_per_node} \
#     ${CODE_DIR}/tools/train.py --config=${CONFIG} \
#     --work_dir="${WORK_DIR}"   2>&1 | tee ${WORK_DIR}/train_${RANK}.log


/opt/conda/bin/accelerate launch \
    --config_file /mnt/afs/liyu1/0312/qworld_2-1_training_copy/qworld_2-1_training_copy/_my_configs/deepspeed_yamls/accelerate_config_ht_zero1.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $WORLD_SIZE \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
     ${CODE_DIR}/tools/train_use_deepspeed.py --config=${CONFIG} --work_dir=${WORK_DIR} 2>&1 | tee ${WORK_DIR}/train_${RANK}.log






