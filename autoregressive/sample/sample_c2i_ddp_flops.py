# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os
import sys
from PIL import Image
import numpy as np
import math
import argparse

import time

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.join(current_script_dir, '../../') # adjust the path based on your project structure
project_root_dir = os.path.abspath(project_root_dir)

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

    
from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from utils.logger import create_logger
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def print_model_summary(model, logger, args):
    """
    打印模型参数表，并统计 generate() 方法生成一张图的精确 FLOPs (含 SDPA 矩阵乘法)。
    """
    import torch.nn.functional as F
    from functools import wraps
    
    device = next(model.parameters()).device
    latent_size = args.image_size // args.downsample_size
    num_tokens = latent_size ** 2

    # 1. 确定分析对象 (解开 torch.compile)
    model_to_analyze = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    # 打印参数分布表
    logger.info("\n" + parameter_count_table(model_to_analyze))

    # 2. 准备统计容器
    stats = {"linear_flops": 0, "attention_flops": 0}

    # 3. 定义 Linear 层 Hook
    def count_linear_flops(module, input, output):
        # 2 * I * O * N
        stats["linear_flops"] += 2 * module.in_features * output.numel()

    # 4. 定义 Attention (SDPA) 拦截器
    original_sdpa = F.scaled_dot_product_attention
    @wraps(original_sdpa)
    def tracked_sdpa(query, key, value, *args, **kwargs):
        # Q: (B, H, L, d), K: (B, H, S, d)
        # FLOPs = 4 * B * H * L * S * d
        b, h, l, d = query.shape
        s = key.shape[2]
        stats["attention_flops"] += 4 * b * h * l * s * d
        return original_sdpa(query, key, value, *args, **kwargs)

    # 5. 挂载统计工具
    hooks = []
    for m in model_to_analyze.modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(count_linear_flops))
    
    # 临时替换全局 F.scaled_dot_product_attention
    F.scaled_dot_product_attention = tracked_sdpa

    # 6. 执行模拟生成 (Batch Size = 1)
    logger.info(f"Running deterministic FLOPs analysis for generate()...")
    dummy_c = torch.randint(0, args.num_classes, (1,), device=device)
    
    try:
        with torch.no_grad():
            model_to_analyze.generate(
                dummy_c, 
                num_tokens, 
                cfg_scales=(args.cfg_scale, args.cfg_scale),
                temperature=args.temperature, 
                top_k=args.top_k,
                top_p=args.top_p
            )
    finally:
        # 确保无论是否报错都还原环境
        for h in hooks:
            h.remove()
        F.scaled_dot_product_attention = original_sdpa

    # 7. 格式化并输出
    total_flops = stats["linear_flops"] + stats["attention_flops"]
    
    def format_flops(n):
        if n > 1e12: return f"{n / 1e12:.3f} TFLOPs"
        if n > 1e9: return f"{n / 1e9:.3f} GFLOPs"
        return f"{n / 1e6:.3f} MFLOPs"

    logger.info("=" * 60)
    logger.info(f"{'Method Analyzed:':<30} gpt_model.generate()")
    logger.info(f"{'Batch Size:':<30} 1")
    logger.info(f"{'Latent Size:':<30} {latent_size}x{latent_size} ({num_tokens} tokens)")
    logger.info(f"{'Linear FLOPs:':<30} {format_flops(stats['linear_flops'])}")
    logger.info(f"{'Attention FLOPs:':<30} {format_flops(stats['attention_flops'])}")
    logger.info(f"{'Total Precision FLOPs:':<30} {format_flops(total_flops)}")
    logger.info(f"{'Params (M):':<30} {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    logger.info("=" * 60)
    
    # 重要：重置 KV Cache，防止干扰后续主循环
    # 这里的参数需要和你在 GPT 模型初始化时一致
    if hasattr(model_to_analyze, 'setup_caches'):
        model_to_analyze.setup_caches(
            max_batch_size=args.per_proc_batch_size * (2 if args.cfg_scale > 1.0 else 1),
            max_seq_length=num_tokens + args.cls_token_num,
            dtype=next(model.parameters()).dtype
        )

def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}--ABS-{args.adam_block_size}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")

    # create logger
    if rank == 0:
        logger = create_logger(sample_folder_dir)
        logger.info(f"logger directory created at {sample_folder_dir}")
    else:
        logger = create_logger(None)  # Create a logger that does not write to file

    logger.info(f"Args: {args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        subpass_len=args.subpass_len,
        subpass_num=args.subpass_num,
        logger=logger,
        vocab_size=args.codebook_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        adam_block_size=args.adam_block_size,
        pre_token_choose=args.pre_token_choose,
        freqs_cis_reorder_shceme=args.freqs_cis_reorder_shceme,
        interlacing_type=args.interlacing_type,
        target_aware_emb=args.target_aware_emb,
        is_adaLN=args.is_adaLN,
        num_inputreorder_modules=args.num_inputreorder_modules,
        use_pass_aware_adaLN=args.use_pass_aware_adaLN,
        use_class_aware_adaLN=args.use_class_aware_adaLN,
    ).to(device=device, dtype=precision)
    
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=False)
    gpt_model.load_state_dict(checkpoint["model"], strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    dist.barrier()
    
    # --- 获取原始模型实例 ---
    if hasattr(gpt_model, 'module'):
        # model is DDP wrapped
        raw_model = gpt_model.module
    else:
        raw_model = gpt_model

    if hasattr(raw_model, '_orig_mod'):
        # model is torch.compile wrapped
        raw_model = raw_model._orig_mod

    # --- 打印模型参数总结 ---
    # print_model_summary(raw_model, logger, args)
    if rank == 0:
        import torch.nn.functional as F
        from functools import wraps

        logger.info("\n正在统计 generate 方法的绝对精确 FLOPs (Linear + Attention)...")
        
        # 1. 计数变量
        stats = {"linear_flops": 0, "attention_flops": 0}
        
        # 2. Linear 层 Hook
        def count_linear_flops(module, input, output):
            # 2 * I * O * N
            stats["linear_flops"] += 2 * module.in_features * output.numel()

        # 3. Attention 拦截逻辑
        # SDPA 计算包含两个主要的矩阵乘法：
        # 1. Q @ K.T -> (B, H, L, S)  FLOPs = 2 * B * H * L * S * d_head
        # 2. Score @ V -> (B, H, L, d_head) FLOPs = 2 * B * H * L * S * d_head
        original_sdpa = F.scaled_dot_product_attention

        @wraps(original_sdpa)
        def tracked_sdpa(query, key, value, *args, **kwargs):
            # query shape: (B, H, L, d_head)
            # key shape:   (B, H, S, d_head)
            # 两次矩阵乘法的总 FLOPs = 4 * B * H * L * S * d_head
            b, h, l, d = query.shape
            s = key.shape[2]
            stats["attention_flops"] += 4 * b * h * l * s * d
            return original_sdpa(query, key, value, *args, **kwargs)

        # 4. 挂载 Hook 和拦截器
        hooks = []
        for module in gpt_model.modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(count_linear_flops))
        
        # 替换全局函数 (只在统计期间)
        F.scaled_dot_product_attention = tracked_sdpa

        # 5. 执行模拟推理
        dummy_cond = torch.randint(0, args.num_classes, (1,), device=device)
        max_tokens = latent_size ** 2
        
        with torch.no_grad():
            _ = gpt_model.generate(
                dummy_cond, max_tokens, 
                cfg_scales=(args.cfg_scale, args.cfg_scale),
                temperature=args.temperature, 
                top_k=args.top_k,
                top_p=args.top_p
            )

        # 6. 还原状态与清理
        for h in hooks:
            h.remove()
        F.scaled_dot_product_attention = original_sdpa

        # 7. 输出结果
        total_flops = stats["linear_flops"] + stats["attention_flops"]
        logger.info("="*50)
        logger.info(f"LlamaGen AR 推理绝对精确统计 (单张图):")
        logger.info(f"Linear 层 FLOPs:    {stats['linear_flops'] / 1e9:.2f} G")
        logger.info(f"Attention 层 FLOPs: {stats['attention_flops'] / 1e9:.2f} G")
        logger.info(f"总计 FLOPs:         {total_flops / 1e9:.4f} G")
        logger.info(f"Attention 占比:     {(stats['attention_flops'] / total_flops)*100:.2f}%")
        logger.info("="*50 + "\n")

        # 8. 重置 KV Cache
        gpt_model.setup_caches(
            max_batch_size=args.per_proc_batch_size * (2 if args.cfg_scale > 1.0 else 1),
            max_seq_length=latent_size**2 + args.cls_token_num,
            dtype=gpt_model.tok_embeddings.weight.dtype
        )

    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    start_t = time.time()

    ARtime = 0 ################################
    # c_idx = 0
    # c_indicess = [[541,  97, 191, 172, 553, 604, 103, 733],
    #               [907, 545, 202, 965, 295, 347, 802, 475],
    #               [856, 843, 973, 187, 369, 871, 718, 636],
    #               [986, 367, 638, 255, 994, 659, 778, 680],
    #               [914, 197,  57, 864, 691, 297, 607, 723],
    #               [102, 199, 936, 967, 276, 380,  12, 278],
    #               [740, 668, 940, 911, 601, 368, 996, 797],
    #               [804, 343, 371, 733, 686,  52, 459, 585],
    #               [970, 277, 511, 828, 732, 808, 927, 608],
    #               [493, 596, 938, 722, 394,  88, 783, 845]]
    has_measured_flops = False  # 标志位，确保只测一次
    for _ in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)
        # c_indices = torch.tensor(c_indicess[c_idx], device=device)
        # c_idx = (c_idx + 1)
        # print(c_indices)
        # class_idx = c_indices.item()
        qzshape = [len(c_indices), args.codebook_embed_dim, latent_size, latent_size]
###########################################################################################
        if rank == 0 and not has_measured_flops:
            from deepspeed.profiling.flops_profiler import FlopsProfiler
            
            model_to_analyze = gpt_model._orig_mod if hasattr(gpt_model, '_orig_mod') else gpt_model

            # --- 关键补丁：解决 AttributeError ---
            # 遍历模型，给所有 KVCache 模块注入初始值为 0 的 __flops__ 属性
            for m in model_to_analyze.modules():
                if 'KVCache' in m.__class__.__name__: # 匹配你的 KVCache 类名
                    if not hasattr(m, '__flops__'):
                        m.__flops__ = 0 
            # ------------------------------------

            prof = FlopsProfiler(model_to_analyze)
            single_c = c_indices[:1] 
            
            prof.start_profile()
            
            with torch.no_grad():
                model_to_analyze.generate(
                    single_c, 
                    latent_size ** 2, 
                    cfg_scales=(args.cfg_scale, args.cfg_scale),
                    temperature=args.temperature, 
                    top_k=args.top_k,
                    top_p=args.top_p
                )
            
            # 这一步现在不会报错了，因为它能找到 __flops__ = 0
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            
            logger.info("-" * 50)
            logger.info(f"✅ DeepSpeed 统计成功!")
            logger.info(f"单图总计算量: {flops / 1e9:.2f} GFLOPs")
            logger.info("-" * 50)

            prof.end_profile()
            
            # 重置缓存
            if hasattr(model_to_analyze, 'setup_caches'):
                model_to_analyze.setup_caches(
                    max_batch_size=args.per_proc_batch_size * (2 if args.cfg_scale > 1.0 else 1),
                    max_seq_length=latent_size**2 + args.cls_token_num,
                    dtype=next(model_to_analyze.parameters()).dtype
                )
            has_measured_flops = True
###########################################################################################
        # Sample from the model:
        dist.barrier()
        s_ARtime = time.time() ################################
        # index_sample, decoding_mask, rows_cols, tmp_inv = gpt_model.generate(c_indices, latent_size ** 2, cfg_scales=(args.cfg_scale, args.cfg_scale),
        #                 temperature=args.temperature, 
        #                 top_k=args.top_k,
        #                 top_p=args.top_p)
        index_sample = gpt_model.generate(c_indices, latent_size ** 2, cfg_scales=(args.cfg_scale, args.cfg_scale),
                        temperature=args.temperature, 
                        top_k=args.top_k,
                        top_p=args.top_p)
        e_ARtime = time.time() ################################
        ARtime += (e_ARtime - s_ARtime) ################################
        
        ####################################################################################################
        # qzshape = [len(c_indices)*10, args.codebook_embed_dim, latent_size, latent_size]
        # print(f"index_sample.shape: {index_sample.shape}")
        ####################################################################################################
        samples = vq_model.decode_code(index_sample, qzshape) # output value is between [-1, 1]
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        # total += global_batch_size * 10 # for check pass-wise image generation
        total += global_batch_size
    end_time = time.time()

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        logger.info("Done.")
        logger.info(f"Total time for sampling {total} images: {end_time - start_t:.2f} seconds")
        logger.info(f"Total time for AR: {ARtime:.2f} seconds, batch size in each GPU: {args.per_proc_batch_size}, global batch size: {global_batch_size}, \
              iterations: {iterations}, avg time for each batch: {ARtime/iterations:.4f} seconds")
    dist.barrier()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--compile", action='store_true', default=True)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0, help="classifier-free guidance scale")
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--adam-block-size", type=int, choices=[1,2,4,8,16], default=8)
    parser.add_argument("--subpass-len", type=int, default=None, help="the length of each subpass, None means no subpass")
    parser.add_argument("--subpass-num", type=int, default=None, help="the number of subpasses within each pass, None means no subpass")
    parser.add_argument("--pre_token_choose", type=str, choices=['close_min', 'close_max', 'knn', 'transformer_choose',
                                                                 'close_unattach_min', 'close_unattach_max', 
                                                                 'close_left_up', 'close_center', 'ex_corner'], default="close_min")
    parser.add_argument("--freqs_cis_reorder_shceme", type=str, choices=['output_reorder', 'input_reorder', 'None'], default='None')
    parser.add_argument("--interlacing_type", type=str, choices=['adam', 'spin_adam', 'corner_adam'], default="adam", help="interlacing type, options: adam, spin_adam")
    parser.add_argument("--target_aware_emb", action='store_true')
    parser.add_argument("--is_adaLN", action='store_true', default=False)
    parser.add_argument("--num_inputreorder_modules", type=int, default=None, help="the number of input reorder modules for transformer_choose")
    parser.add_argument("--use_pass_aware_adaLN", action='store_true', default=False)
    parser.add_argument("--use_class_aware_adaLN", action='store_true', default=False)

    args = parser.parse_args()
    main(args)