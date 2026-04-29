# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import broadcast
from torch.optim.lr_scheduler import LambdaLR

from glob import glob
from copy import deepcopy
import os
import sys
import time
import inspect
import argparse
import math
from thop import profile, clever_format

import wandb
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.join(current_script_dir, '../../') # adjust the path based on your project structure
project_root_dir = os.path.abspath(project_root_dir)

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models
from autoregressive.models.utils.visulization import *
from tokenizer.tokenizer_image.vq_model import VQ_models

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def create_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def log_image(args, latent_size, vq_model, gpt_model, logger, device, epoch, rank):
    logger.info("start log image after each epoch")

    gpt_model.eval()
    class_labels = [0]
    c_indices = torch.tensor(class_labels, device=device)
    qzshape = [len(class_labels), args.codebook_embed_dim, latent_size, latent_size]
    logger.info("data preparation finish")

    index_sample = gpt_model.module.generate(c_indices, latent_size ** 2, cfg_scales=(args.cfg_scale, args.cfg_scale),
                    temperature=args.temperature, 
                    top_k=args.top_k,
                    top_p=args.top_p)
    logger.info("gpt model generation finish")
    samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
    logger.info("vq model decoding finish")
    # log to wandb
    image_np = samples.detach().cpu().numpy()
    image_np = image_np[0]
    image_np = (image_np + 1.0) / 2.0
    image_np = np.transpose(image_np, (1, 2, 0))
    image_np = (image_np * 255).astype(np.uint8)
    if rank == 0:
        wandb.log({"sample_tensor_image": wandb.Image(image_np, caption=f"Sample image after epoch {epoch}")})
    logger.info("finish sampling")


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def print_model_summary(model, logger, seq_len, con_len, device):
    """
    打印模型每一层的名称和参数量。
    """
    total_params = 0
    trainable_params = 0
    
    # 打印标题
    logger.info("=" * 60)
    logger.info("Model Parameter Summary:")
    logger.info("{:<50} {:>10} {:>10} {:>20}".format("Layer Name", "Params (K)", "Trainable", "Dtype"))
    logger.info("-" * 60)

    # 使用 named_parameters() 遍历所有参数
    for name, parameter in model.named_parameters():
        # 获取当前参数的元素数量
        param_count = parameter.numel()
        total_params += param_count
        
        # 检查参数是否可训练
        is_trainable = parameter.requires_grad
        if is_trainable:
            trainable_params += param_count
        
        # 打印当前层的信息
        logger.info("{:<50} {:>10.2f} {:>10} {:>20}".format(
            name,
            param_count / 1e3, # 转换为千 (K)
            "Yes" if is_trainable else "No",
            str(parameter.dtype)
        ))

    L_idx = seq_len 
    # 按照 forward_train(self, idx, cond_idx, input_pos, targets, valid) 的顺序
    idx_dummy = torch.ones((1, L_idx), dtype=torch.long, device=device) 
    cond_idx_dummy = torch.ones((1,), dtype=torch.long, device=device)

    # 重点：thop 要求输入的元组中**只包含张量**，因此 None 必须移除
    # 假设你的 forward 函数为：raw_model(idx, cond_idx)
    # 如果模型 forward(idx, cond_idx, input_pos, targets, valid) 
    # 在推理模式下可以接受 None，你需要检查 thop 是否会因为 None 报错。
    # 如果报错，请尝试定义一个只接受 (idx, cond_idx) 的 wrapper。

    # 假设你的模型只使用前两个输入 (idx, cond_idx)
    # ⚠️ 请确保这是你的模型 forward 函数所需的**全部且按顺序**的输入
    inputs = (idx_dummy, cond_idx_dummy) 

    # --- 2. 计算 MACs 和参数 ---
    macs, params = profile(model, inputs=inputs)

    # 改善输出格式
    macs, params = clever_format([macs, params], "%.3f")

    # 打印总结
    logger.info("-" * 60)
    logger.info("Total Model Parameters: {:.2f} M".format(total_params / 1e6))
    logger.info("Total Trainable Parameters: {:.2f} M".format(trainable_params / 1e6))
    logger.info("=" * 60)

    logger.info(f"Model FLOPs: {macs}") # 结果通常会以 GFlops 的形式显示
    logger.info(f"Model Params: {params}")

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    
    # make directories in rank 0
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)

        # create logger - logger will be created in rank 0
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.is_wandb_log:
            # init wandb
            wandb.require("core")
            wandb.init(project="Train-GPT-AdamAR-GPU", 
                    name=f"gbs-{args.global_batch_size}-wandb_step-{args.log_every}-image_size-{args.image_size}",
                    mode="offline" if args.wandb_offline else "online",)
    else:
        logger = create_logger(None)  # Create a logger that does not write to file

    logger.info(f"Args: {args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup data:
    dl_st = time.time()
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True
        # seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        sampler=sampler,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
        pin_memory=False
    )
    torch.manual_seed(seed)

    dl_et = time.time()
    logger.info(f"dataset load time: {dl_et - dl_st} sec.")

    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    total_data_num = len(dataset)
    # bs_per_rank = int(total_data_num // args.global_batch_size)
    steps_per_epoch = len(loader) # the number of batches loaded into the rank, also the number of steps per epoch
    total_training_steps = steps_per_epoch * args.epochs

    # Set default values for learning rate parameters
    if args.max_lr is None:
        args.max_lr = args.lr
    if args.min_lr is None:
        args.min_lr = args.lr * 0.1
    if args.cosine_percent is None:
        args.cosine_percent = 1.0 - args.warmup_percent - args.const_percent

    # Validate that percentages sum to 1.0
    total_percent = args.warmup_percent + args.const_percent + args.cosine_percent
    if abs(total_percent - 1.0) > 1e-6:
        logger.warning(f"Warning: warmup_percent ({args.warmup_percent}) + const_percent ({args.const_percent}) + cosine_percent ({args.cosine_percent}) = {total_percent}, which does not equal 1.0")

    warmup_steps = int(total_training_steps * args.warmup_percent)
    constant_steps = int(total_training_steps * args.const_percent)
    cosine_steps = int(total_training_steps * args.cosine_percent)
    cosine_start_step = warmup_steps + constant_steps

    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
    logger.info(f"Learning rate schedule: max_lr={args.max_lr}, min_lr={args.min_lr}")
    logger.info(f"warmup steps: {warmup_steps} ({args.warmup_percent*100:.1f}%), constant steps: {constant_steps} ({args.const_percent*100:.1f}%), cosine steps: {cosine_steps} ({args.cosine_percent*100:.1f}%), total training steps: {total_training_steps}")
    
    def lr_lambda(current_step: int):
        """
        Warmup -> Constant -> Cosine Annealing 调度策略。
        返回学习率相对于 optimizer 初始 lr 的乘数。

        - Warmup 阶段: 从 0 线性增长到 max_lr
        - Constant 阶段: 保持 max_lr
        - Cosine Annealing 阶段: 从 max_lr 按余弦曲线下降到 min_lr
        """
        if current_step < warmup_steps:
            # Warmup: 从 0 线性增长到 max_lr
            # 返回值范围: [0, max_lr/lr]
            return (args.max_lr / args.lr) * (float(current_step) / float(max(1, warmup_steps)))

        elif current_step < cosine_start_step:
            # Constant: 保持 max_lr
            return args.max_lr / args.lr

        elif current_step < total_training_steps:
            # Cosine Annealing: 从 max_lr 下降到 min_lr
            if cosine_steps <= 0:
                return args.min_lr / args.lr

            progress = float(current_step - cosine_start_step) / float(cosine_steps)
            progress = min(1.0, progress)

            # Cosine annealing formula: 从 1.0 下降到 0.0
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

            # 将 cosine_factor 映射到 [min_lr, max_lr] 范围
            current_lr = args.min_lr + (args.max_lr - args.min_lr) * cosine_factor

            return current_lr / args.lr
        else:
            # 超出训练步数，返回最小学习率
            return args.min_lr / args.lr

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        subpass_len=args.subpass_len,
        subpass_num=args.subpass_num,
        logger=logger,
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        adam_block_size=args.adam_block_size,
        pre_token_choose=args.pre_token_choose,
        freqs_cis_reorder_shceme=args.freqs_cis_reorder_shceme,
        interlacing_type=args.interlacing_type,
        target_aware_emb=args.target_aware_emb,
        is_adaLN=args.is_adaLN,
        num_inputreorder_modules=args.num_inputreorder_modules,
        use_pass_aware_adaLN=args.use_pass_aware_adaLN,
        use_class_aware_adaLN=args.use_class_aware_adaLN,
        use_kq_norm=args.use_kq_norm,
        class_pass_emb_dim=args.class_pass_emb_dim,
    ).to(device)

    # visualize passes and attention mask
    if rank == 0:
        visualize_passes(
            img_width = int(model.block_size ** 0.5),
            img_height = int(model.block_size ** 0.5),
            token_map=model.auto_regr_struct.token_map,
            decoded_masked_coords=model.auto_regr_struct.decoded_masked_coords,
            experiment_dir=experiment_dir
        )
        visualize_input_seq(
            img_width = int(model.block_size ** 0.5),
            img_height = int(model.block_size ** 0.5),
            token_map=model.auto_regr_struct.token_map,
            token_map_tensors=model.auto_regr_struct.token_map_tensors,
            decoded_masked_coords=model.auto_regr_struct.decoded_masked_coords,
            experiment_dir=experiment_dir
        )
        visualize_target_seq(
            img_width = int(model.block_size ** 0.5),
            img_height = int(model.block_size ** 0.5),
            token_map=model.auto_regr_struct.token_map,
            token_map_tensors=model.auto_regr_struct.token_map_tensors,
            decoded_masked_coords=model.auto_regr_struct.decoded_masked_coords,
            experiment_dir=experiment_dir
        )
        visualize_attention_mask(
            attention_mask=model.auto_regr_struct.training_attention_mask,
            experiment_dir=experiment_dir
        )
        visualize_sequences(
            img_width = int(model.block_size ** 0.5),
            img_height = int(model.block_size ** 0.5),
            token_map=model.auto_regr_struct.token_map,
            token_map_tensors=model.auto_regr_struct.token_map_tensors,
            decoded_masked_coords=model.auto_regr_struct.decoded_masked_coords,
            experiment_dir=experiment_dir
        )
    
    # --- 获取原始模型实例 ---
    if hasattr(model, 'module'):
        # model is DDP wrapped
        raw_model = model.module
    else:
        raw_model = model

    if hasattr(raw_model, '_orig_mod'):
        # model is torch.compile wrapped
        raw_model = raw_model._orig_mod

    # --- 打印模型参数总结 ---
    print_model_summary(model, logger, latent_size ** 2, args.cls_token_num, device)


    if args.is_wandb_log:
        # if we want to log images to wandb, we need to setup VQ model
        vq_model = VQ_models[args.vq_model](
            codebook_size=args.vocab_size,
            codebook_embed_dim=args.codebook_embed_dim)
        vq_model.to(device)
        vq_model.eval()
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        vq_model.load_state_dict(checkpoint["model"])
        del checkpoint
        logger.info(f"image tokenizer is loaded")
    else:
        vq_model = None

    # training args
    logger.info(f"{args}")    
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = create_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Prepare models for training:
    train_steps = 0
    start_epoch = 0
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=False)
        if args.ema:
            ema.load_state_dict(checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"])
        if not args.from_llamagen: # if the checkpoint is not from LlamaGen, load the state of optimizer
            optimizer.load_state_dict(checkpoint["optimizer"])
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights

    scheduler = None
    if args.is_lr_scheduler:
        # Create a learning rate scheduler
        if args.gpt_ckpt is not None:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=train_steps - 1)
        else:
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0

    model = DDP(model.to(device), device_ids=[args.gpu])
    

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    logger.info(f"Training for {args.epochs} epochs...")

    # to see how many batches have been loaded to this core
    logger.info(f"Rank {rank} has loaded {len(loader)} batches")
    for x, y in loader:
        logger.info(f"Rank {rank} has each batch in shape: {x.shape}")
        break 

    
    log_steps = 0
    wandb_steps = 0
    running_loss = 0
    
    start_time = time.time()

    def train_loop_fn(loader, epoch):
        # tracker = xm.RateTracker()
        nonlocal scaler, log_steps, train_steps
        nonlocal running_loss, wandb_steps, logger, start_time
        model.train()
        for step, (data, target) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            z_indices = data.reshape(data.shape[0], -1)
            c_indices = target.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]
            with torch.cuda.amp.autocast(dtype=ptdtype): # automatic mixed precision
                _, loss = model(cond_idx=c_indices, idx=z_indices)
                
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer) # gpu
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)  # zero out gradients
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)


            current_lr = optimizer.param_groups[0]['lr'] # get current learning rate that is used to update parameters

            if args.is_lr_scheduler:
                scheduler.step() # update learning rate

            # Log loss values:
            current_loss = loss.item()
            running_loss += current_loss
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                # steps_per_sec = log_steps / (end_time - start_time)
                avg_step_time = (end_time - start_time) / log_steps
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {current_loss:.4f}, "+
                            f"Avg Train Loss: {avg_loss:.4f}, "+
                            f"Avg Step Time = {avg_step_time:.2f} sec, "+
                            f"Current learning rate: {current_lr:.6f}")
                
                if args.is_wandb_log and rank == 0: # log to wandb
                    wandb.log({"Iteration": train_steps, 
                                "Avg_train_loss": avg_loss, 
                                "Learning rate": current_lr}, step=train_steps)
                dist.barrier()
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()
            
    # log an image before the training
    # log_image(args, latent_size, vq_model, model, logger, device, 0, rank=rank)

    dist.barrier()
    for epoch in range(start_epoch+1, args.epochs+1):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        train_loop_fn(loader, epoch)
        logger.info(f'Finish Training for {epoch} epoch...')

        # Save checkpoint:
        if epoch % args.ckpt_every == 0 and epoch > 0:
            if rank == 0:
                try:
                    logger.info(f'\n**************************'+
                            f'rank {rank} is saving the checkpoint'+
                            f'\n**************************')
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
                except Exception as e:
                    logger.error(f"saving checkpoint failed: {str(e)}", exc_info=True)
                    pass
            dist.barrier()
        
        # Sample one image after each epoch and log into wandb
        if args.is_wandb_log:
            logger.info("Logging image to wandb...")
            # log an image after each epoch
            # log_image(args, latent_size, vq_model, model, logger, device, epoch, rank=rank)

        dist.barrier()

    logger.info("Done!")
    logger.info("Resources cleaned up")
    
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if rank == 0 and args.is_wandb_log:
        # notify the end of training to wandb
        wandb.alert(
                    title="Training is Done",
                    text=f"The training is done, please check the results"
                )
        
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--adam-block-size", type=int, choices=[1,2,4,8,16], default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10) # log every log_every steps
    parser.add_argument("--ckpt-every", type=int, default=5000) # save checkpoint every ckpt_every epochs
    parser.add_argument("--is-wandb-log", action='store_true', default=False)
    parser.add_argument("--wandb_offline", action='store_true', default=False)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--num-datapoints", type=int, default=None, help="number of data points to train on")
    parser.add_argument("--num-data", type=int, default=None, help="number of data points to train on") # sample only first num_data files
    parser.add_argument("--from_llamagen", action='store_true', default=False)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--profiler_port", type=int, default=9012, help="the port of investigation") 
    parser.add_argument("--profile", action='store_true', default=True)
    parser.add_argument("--num-workers", type=int, default=24)  #############################################################
    parser.add_argument("--warmup_percent", type=float, default=0.01, help="the ratio of warm-up steps in total number of steps")
    parser.add_argument("--const_percent", type=float, default=0.7, help="the ratio of steps with constant lr in total number of steps")
    parser.add_argument("--cosine_percent", type=float, default=None, help="the ratio of cosine annealing steps in total number of steps, if None, it will be calculated as 1.0 - warmup_percent - const_percent")
    parser.add_argument("--max-lr", type=float, default=None, help="maximum learning rate to reach during warmup, if None, it will be set to lr")
    parser.add_argument("--min-lr", type=float, default=None, help="minimum learning rate at the end of cosine annealing, if None, it will be set to 0.1 * lr")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--is-lr-scheduler", action='store_true', default=False)
    parser.add_argument("--subpass-len", type=int, default=None, help="the length of each subpass, None means no subpass")
    parser.add_argument("--subpass-num", type=int, default=None, help="the number of subpasses within each pass, None means no subpass")
    parser.add_argument("--pre_token_choose", type=str, choices=['close_min', 'close_max', 'knn', 'transformer_choose',
                                                                 'close_unattach_min', 'close_unattach_max', 
                                                                 'close_left_up', 'close_center', 'ex_corner'], default="close_min")
    parser.add_argument("--freqs_cis_reorder_shceme", type=str, choices=['output_reorder', 'input_reorder', 'None'], default='None')
    parser.add_argument("--interlacing_type", type=str, choices=['adam', 'spin_adam', 'corner_adam'], default="adam", help="interlacing type, options: adam, spin_adam")
    parser.add_argument("--target_aware_emb", action='store_true', default=False)
    parser.add_argument("--is_adaLN", action='store_true', default=False)
    parser.add_argument("--num_inputreorder_modules", type=int, default=None, help="the number of input reorder modules for transformer_choose")
    parser.add_argument("--use_pass_aware_adaLN", action='store_true', default=False)
    parser.add_argument("--use_class_aware_adaLN", action='store_true', default=False)
    parser.add_argument("--use_kq_norm", action='store_true', default=False, help="whether to use qk norm in attention")
    parser.add_argument("--class_pass_emb_dim", type=int, default=None, help="the dimension of class and pass embedding when using pass-aware or class-aware AdaLN")

    args = parser.parse_args()
    main(args)
