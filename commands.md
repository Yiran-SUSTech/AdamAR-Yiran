# Pre-extract discrete codes of training images
bash /mnt/afs/intern/fangwenhan/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/intern/fangwenhan/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384 

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/intern/fangwenhan/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop_105 --ten-crop --crop-range 1.05 --image-size 384


bash /mnt/afs/intern/fangwenhan/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/intern/fangwenhan/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 256

# training with augmented data and load a pretrained chkpt, train it with learning rate scheduler on GPU
# train and generate tokens in each pass in parallel
bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop \
    --image-size 384 --num-data 20000\
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/c2i_B_384.pt --from_llamagen \
    --global-batch-size 256 --lr 0.00015 --epochs 100 --warmup_percent 0.01 --is-lr-scheduler \
    --ckpt-every 25 --log-every 100 \
    --no-compile --is-wandb-log --wandb_offline

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop \
    --image-size 256 \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/c2i_B_256.pt\
    --global-batch-size 256 --lr 0.00015 --epochs 100 --warmup_percent 0.01 --is-lr-scheduler \
    --ckpt-every 25 --log-every 100 \
    --no-compile --is-wandb-log --wandb_offline

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop \
    --num-data 256 \
    --image-size 256 --adam-block-size 1 \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B \
    --global-batch-size 256 --lr 0.00015 --epochs 1 --warmup_percent 0.01 --is-lr-scheduler \
    --ckpt-every 25 --log-every 1 \
    --no-compile --is-wandb-log --wandb_offline

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop \
    --num-data 256 \
    --image-size 256 --adam-block-size 8 --subpass-len 4 \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B \
    --global-batch-size 256 --lr 0.0001 --epochs 1 \
    --ckpt-every 25 --log-every 1 \
    --no-compile --is-wandb-log --wandb_offline

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop \
    --image-size 256 --adam-block-size 8 --subpass-len 4 \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B \
    --global-batch-size 256 --lr 0.0001 --epochs 100 \
    --ckpt-every 20 --log-every 100 \
    --no-compile --is-wandb-log --wandb_offline



bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop \
    --image-size 256 --adam-block-size 1 \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/010-GPT-B/checkpoints/0150120.pt \
    --global-batch-size 256 --lr 0.0001 --epochs 100 \
    --ckpt-every 25 --log-every 100 \
    --no-compile --is-wandb-log --wandb_offline
    

# training with augmented data and load a pretrained chkpt, train it with learning rate scheduler on GPU
# train and generate tokens in each pass in serial
bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-GPU/cloud_save \
    --code-path /mnt/afs/intern/fangwenhan/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop \
    --image-size 384 --num-data 20000\
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-B --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/c2i_B_384.pt --from_llamagen \
    --global-batch-size 256 --lr 0.00025 --epochs 10 --warmup_percent 0.01 --is-lr-scheduler \
    --ckpt-every 5 --log-every 10 \
    --no-compile --is-serial --is-wandb-log --wandb_offline



# Sampling
bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/useful_ckpt/total_ImageNet_gbs256_35e_parallel_augment_lrs.pt --gpt-model GPT-B \
    --image-size 384 --image-size-eval 256 --cfg-scale 1.0 \
    --sample-dir samples --num-fid-samples 100



bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/120e_parallel_fromSC_Imagenet_256/checkpoints/120epoch_256_para.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 \
    --sample-dir samples --num-fid-samples 10000

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/100e_parallel_fromSC_Imagenet_256_ABS8/checkpoints/0500400.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8\
    --sample-dir samples --num-fid-samples 50000

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i_test.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/120e_parallel_fromSC_Imagenet_256/checkpoints/120epoch_256_para.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 \
    --sample-dir samples --num-fid-samples 256

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/200e_parallel_fromLG_Imagenet_256_ABS8/checkpoints/1000800.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 --subpass-len 1\
    --sample-dir samples --num-fid-samples 50000

bash /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/intern/fangwenhan/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/010-GPT-B/checkpoints/0150120.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 1 \
    --sample-dir samples --num-fid-samples 50000
    
# Evaluation
python3 evaluations/c2i/evaluator.py \
    /mnt/afs/intern/fangwenhan/zyr/LlamaGenOri/evaluations/VIRTUAL_imagenet256_labeled.npz \
    /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/samples/GPT-B-100e_para_ABS8_unattach-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0--ABS-8.npz


python3 evaluations/c2i/evaluator.py \
    /mnt/afs/intern/fangwenhan/zyr/LlamaGenOri/evaluations/VIRTUAL_imagenet256_labeled.npz \
    /mnt/afs/intern/fangwenhan/zyr/AdamAR-GPU/results/20e_parallel_fromSC_Imagenet_256_ABS1_close_maxidx/checkpoints/samples/GPT-B-0100080-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0--ABS-1.npz
