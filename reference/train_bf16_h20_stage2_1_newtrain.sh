#!/bin/bash
# ============================================================
# BF16 H20 Stage 2-2 newtrain_new_config 训练脚本
#
# 特点：
# - 8 GPU 并行训练（H20）
# - BF16 精度训练
# - 完全对齐 yangsen 的配置（来自 chenmo/code/qworld）
# - 完整 11 个视频数据集 + 图片数据集
# - 最新预训练权重 step-43100
# - data_use_cache = True (启用缓存)
# ============================================================

WORK_POSTFIX=${1}

export CUDA_LAUNCH_BLOCKING=1

# 配置路径 - 使用独立目录
CURR_FILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG=${CURR_FILE_DIR}/task_model_config_fp8_h20_8gpu_stage2_1_kk_new.py
WORK_DIR=/mnt/afs/liyu1/workdir/qworld/2-1_single/0330
CODE_DIR=/mnt/afs/liyu1/0312/qworld_2-1_training_copy/qworld_2-1_training_copy

seed=42

echo "============================================================"
echo "BF16 H20 Stage 2-1 newtrain_new_config 训练"
echo "============================================================"
echo "Config: task_model_config_fp8_h20_8gpu_stage2_1_kk_new.py"
echo "GPU: 0-7 (8 卡 H20)"
echo "特点: 完整数据集 + 最新权重 step-43100 + data_use_cache=True"
echo "CURR_FILE_DIR: ${CURR_FILE_DIR}"
echo "WORK_DIR: ${WORK_DIR}"
echo "CODE_DIR: ${CODE_DIR}"
echo "============================================================"

if [ -d ${WORK_DIR} ]; then
    echo "WORK_DIR: ${WORK_DIR} exists."
else
    echo "mkdir WORK_DIR: ${WORK_DIR} ..."
    mkdir -p ${WORK_DIR}
fi

bash ${CURR_FILE_DIR}/train_core.sh ${CONFIG} ${WORK_DIR} ${CODE_DIR}
