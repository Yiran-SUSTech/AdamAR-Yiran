# !/bin/bash
set -x

torchrun \
--nnodes=1 --nproc_per_node=8 --node_rank=0 \
--master_port=12345 \
/mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/autoregressive/sample/sample_c2i_ddp_test.py \
--vq-ckpt ./pretrained_models/vq_ds16_c2i.pt \
"$@"
