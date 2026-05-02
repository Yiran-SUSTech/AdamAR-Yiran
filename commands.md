# Pre-extract discrete codes of training images
bash /mnt/afs/zhengmingkai/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/zhengmingkai/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384 

bash /mnt/afs/zhengmingkai/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/zhengmingkai/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_c2i_flip_ten_crop_105 --ten-crop --crop-range 1.05 --image-size 384


bash /mnt/afs/zhengmingkai/zyr/AdamAR/scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt --data-path /mnt/afs/zhengmingkai/zyr/imagenet/datasets/imagenet2/train --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode/imagenet_code_256_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 256

# training with augmented data and load a pretrained chkpt, train it with learning rate scheduler on GPU
# train and generate tokens in each pass in parallel

bash /mnt/afs/zhengmingkai/zyr/AdamAR-Yiran/scripts/autoregressive/train_c2i.sh \
    --cloud-save-path /mnt/datasets/AdamAR-Yiran/cloud_save \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
    --image-size 256 --adam-block-size 8 --pre_token_choose close_min \
    --freqs_cis_reorder_shceme output_reorder --interlacing_type adam --use_pass_aware_adaLN --use_class_aware_adaLN \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-Bcond --gpt-type c2i \
    --global-batch-size 2048 --min-lr 1e-5 --lr 0.0001 --max-lr 1e-4 --is-lr-scheduler --warmup_percent 0.25 --const_percent 0 --cosine_percent 0.75 --epochs 400 \
    --ckpt-every 100 --log-every 100 \
    --no-compile --is-wandb-log --wandb_offline


bash /mnt/afs/zhengmingkai/zyr/AdamAR-Yiran/scripts/autoregressive/train_c2i_metax.sh \
    --cloud-save-path /mnt/datasets/AdamAR-Yiran/cloud_save \
    --code-path /mnt/afs/zhengmingkai/zyr/ExtractedCode2/imagenet_code_256_c2i_flip_ten_crop \
    --image-size 256 --adam-block-size 8 --pre_token_choose close_min \
    --freqs_cis_reorder_shceme output_reorder --interlacing_type adam --use_pass_aware_adaLN --use_class_aware_adaLN \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-model GPT-Bcond --gpt-type c2i \
    --global-batch-size 256 --min-lr 1e-5 --lr 0.0001 --max-lr 1e-4 --is-lr-scheduler --warmup_percent 0.25 --const_percent 0 --cosine_percent 0.75 --epochs 4 \
    --ckpt-every 100 --log-every 200 --num-workers 8 --prefetch_factor 8 \
    --no-compile \
    --mixed-precision bf16 --gradient-accumulation-steps 4
    <!-- --is-wandb-log --wandb_offline -->

# Sampling
bash /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i_test.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/results/100e_parallel_fromSC_Imagenet_256_ABS8_closemin/checkpoints/100e_ABS8_closemin.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 --pre_token_choose close_min \
    --sample-dir samples --num-fid-samples 50000

bash /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/results/100e_serial_fromSC_Imagenet_256_ABS8/checkpoints/100e_serial_ABS8_spl1.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 --subpass-len 1 --pre_token_choose close_min \
    --sample-dir samples --num-fid-samples 50000

bash /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/results/100e_fromSC_Imagenet_256_ABS1_close_max/checkpoints/100e_ABS1_close_max.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 1 --pre_token_choose close_min \
    --sample-dir samples --num-fid-samples 50000


bash /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/results/300e_parallel_fromSC_Imagenet_256_ABS8_spl4_closeleftup_outreorder/checkpoints/100e_spl4_closeleftup_outreorder.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 --subpass-len 4 --pre_token_choose close_left_up \
    --sample-dir samples --num-fid-samples 50000

    
bash /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/scripts/autoregressive/sample_c2i.sh \
    --vq-ckpt /mnt/afs/zhengmingkai/zyr/pretrained_models/vq_ds16_c2i.pt \
    --gpt-ckpt /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/results/100e_parallel_fromSC_Imagenet_256_ABS8_lr10const_closeleftup_outreorder/checkpoints/100e_lr10const_closeleftup_outreorder.pt --gpt-model GPT-B \
    --image-size 256 --image-size-eval 256 --cfg-scale 2.0 --adam-block-size 8 --pre_token_choose close_left_up \
    --sample-dir samples --num-fid-samples 50000


# Evaluation
python3 evaluations/c2i/evaluator.py \
    /mnt/afs/zhengmingkai/zyr/LlamaGenOri/evaluations/VIRTUAL_imagenet256_labeled.npz \
    /mnt/afs/zhengmingkai/zyr/AdamAR-GPU/samples/GPT-B-100e_lr10const_closeleftup_outreorder-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0--ABS-8.npz
