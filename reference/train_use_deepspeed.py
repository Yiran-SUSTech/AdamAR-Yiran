from datetime import datetime
import gc
import time
import argparse
import torch
import sys
import os
from copy import deepcopy
import pickle
import copy
import re
from typing import Any, Dict, Union
import torch.distributed as dist
from qworld.commons.file_utils import QWorldFileClient
import mmengine
# **********************************************
# inject fileclient - start
mmengine.fileio.file_client.FileClient._backends['petrel'] = QWorldFileClient
mmengine.fileio.file_client.FileClient._backends['tos'] = QWorldFileClient
mmengine.fileio.file_client.FileClient._prefix_to_backends['s3'] = QWorldFileClient
mmengine.fileio.file_client.FileClient._prefix_to_backends['tos'] = QWorldFileClient
mmengine.fileio.io.prefix_to_backends['tos'] = QWorldFileClient
# inject fileclient - end
# **********************************************


from mmengine.config import Config
from mmengine.dist import init_dist, get_dist_info
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, DistributedSampler

from qworld.pipelines.diffsynth._impl.trainers.utils import ModelLogger
from qworld.commons.profile_utils import timer_recorder
from qworld.datasets import build_dataset
from qworld.pipelines import build_model_pipeline
from qworld.commons.trainers.samplers.visual_bucketing_sampler_v1 import DistributedBucketIndexSampler
from qworld.commons.trainers.samplers.visual_bucketing_sampler_v2 import DistributedBucketIndexSampler_V2

def trace_handler_f(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", max_name_column_width=10000, max_src_column_width=10000, row_limit=-1))
    prof.export_chrome_trace("./test_trace_" + str(torch.distributed.get_rank()) + ".json")

def collate_do_nothing_but_extract(batchs):
    print('collate_do_nothing_but_extract: {}'.format(batchs))
    return batchs[0]

def collate_for_mix(batchs):
    assert len(batchs) == 1
    batchs = batchs[0]

    collate_fn = torch.utils.data.default_collate
    # collate_fn = torch.utils.data.default_convert
    return collate_fn(batchs)

def collate_extract_before_multi_sample_collate_fn(data):
    assert len(data) == 1
    res = collate_multi_sample_collate_fn(data[0])
    # print(res)
    return res

def collate_multi_sample_collate_fn(data):
    res = {}
    keys = list(data[0].keys())
    for key in keys:
        res[key] = []
    for item in data:
        for key in keys:
            res[key].append(item.get(key, None))
    return res


def warmup_lr_fn(step, warmup_steps=10000, min_alpha=0.01):
    # in accelerator, `scheduler steps` equals to `number of samples`
    # 10000 means warmup for 10000 samples, not iterations.
    if step < warmup_steps:
        # linear
        alpha = (1 - min_alpha) * step / warmup_steps + min_alpha
    else:
        alpha = 1
    return alpha


def parse_args():
    parser = argparse.ArgumentParser(description='TRAIN_MODEL_LOOP')
    parser.add_argument('--config', default='', help='train config file path')
    parser.add_argument('--work_dir', default='', help='the dir to save logs and models')

    # parser.add_argument(
    #     '--deterministic',
    #     default='False',
    #     help='whether to set deterministic options for CUDNN backend.')

    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # args.deterministic = str2bool(args.deterministic)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def train_main():

    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


    from datetime import timedelta
    # timeout = timedelta(seconds=3600)
    timeout = 1800

    init_dist(launcher='pytorch', timeout=timeout)
    rank, world_size = get_dist_info()
    print(f'rank : {rank} / world_size: {world_size}')
    
    # check gpus
    if rank == 0:
        print('check gpu communicate & sync ...')
    x = torch.randn([1],dtype=torch.float32).to(torch.cuda.current_device())
    dist.broadcast(x, src=0)
    if rank == 0:
        print('check gpu communicate & sync done')


    from transformers import set_seed
    set_seed(42)

    args = parse_args()


    cfg_path = args.config
    work_dir = args.work_dir

    if cfg_path == '':
        ValueError('config path is empty')
        exit()
    
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = work_dir


    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        

    save_steps = cfg.trainer_cfg.save_steps
    num_epochs = cfg.trainer_cfg.num_epochs
    gradient_accumulation_steps = cfg.trainer_cfg.gradient_accumulation_steps
    find_unused_parameters = cfg.trainer_cfg.find_unused_parameters

    accelerator = Accelerator(
        project_dir=work_dir,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )

    # print distribution info
    accelerator.print(f"{AcceleratorState()}")
    torch.cuda.set_device(accelerator.local_process_index)
    if accelerator.local_process_index == 0:
        # print(args)
        pass

    dataset = build_dataset(cfg.data.train)
    print('len of dataset: {}'.format(len(dataset)))

    data_loader_cfg = cfg.data_loader.train
    data_loader_d = dict(
        DataLoader =DataLoader,
    )
    sampler_d = dict(
        DistributedSampler= DistributedSampler,
        DistributedBucketIndexSampler=DistributedBucketIndexSampler,
        DistributedBucketIndexSampler_V2=DistributedBucketIndexSampler_V2,
    )

    data_sampler_cfg=data_loader_cfg.pop('sampler','')
    if data_sampler_cfg:
        data_sampler_type = data_sampler_cfg.pop('type')
        data_sampler = sampler_d[data_sampler_type](
            dataset=dataset,
            **data_sampler_cfg
        )
    
    else:
        data_sampler = None
    print('data_sampler_cfg:\n{}'.format(data_sampler_cfg))
    
    data_loader_type = data_loader_cfg.pop('type')

    collate_fn_type = data_loader_cfg.pop('collate_fn', None)
    if collate_fn_type:
        collate_fn_d = dict(
            do_nothing_but_extract = collate_do_nothing_but_extract,
            collate_for_mix_dataset = collate_for_mix,
            multi_sample_collate_fn=collate_multi_sample_collate_fn,
            extract_before_multi_sample_collate_fn=collate_extract_before_multi_sample_collate_fn,
        )
        collate_fn = collate_fn_d[collate_fn_type]
    else:
        # using default
        collate_fn = None

    data_loader_train = data_loader_d[data_loader_type](
        dataset= dataset,
        sampler=data_sampler,
        collate_fn=collate_fn,
        **data_loader_cfg,
    )
    print('data_loader_cfg:\n{}'.format(data_loader_cfg))

    auto_load_last_dit = cfg.trainer_cfg.get('auto_load_last_dit', True)
    load_iter = 0
    if auto_load_last_dit:
        sub_dirs = os.listdir(work_dir)
        availables = []
        for sub_dir in sub_dirs:
            steps = [int(x) for x in re.findall(r'step-(\d+).safetensors', sub_dir)]
            if len(steps) == 1:
                availables.append(steps[0])

        if len(availables) > 0:
            load_iter = max(availables)
            auto_load_from = f'{work_dir}/step-{load_iter}.safetensors' 
            print('using as pretrain: {}'.format(auto_load_from))

            # accelerator.state.num_steps = load_iter
            # model_logger.num_steps = load_iter
            cfg.pipeline.pretrained_dit = auto_load_from
        else:
            print('auto load None')

    pipeline = build_model_pipeline(cfg.pipeline)

    model = pipeline.model

    print('model:\n{}'.format(model))


    # **************************************************
    # train loop

    optimizer_d = dict(
        AdamW = torch.optim.AdamW,
    )

    scheduler_d = dict(
        ConstantLR= torch.optim.lr_scheduler.ConstantLR,
        LambdaLR = torch.optim.lr_scheduler.LambdaLR,
    )

    lambda_lr_fn_d =dict(
        warmup_lr_fn= warmup_lr_fn,
    )


    model_logger = ModelLogger(
        output_path=work_dir,
        **cfg.trainer_cfg.model_logger,
    )

    set_seed(42)

    optimizer_type = cfg.trainer_cfg.optimizer.pop('type','AdamW')

    optimizer = optimizer_d[optimizer_type](model.trainable_modules(), **cfg.trainer_cfg.optimizer)

    scheduler_type = cfg.trainer_cfg.scheduler.pop('type','ConstantLR')

    if scheduler_type == 'LambdaLR':
        lambda_fn_type = cfg.trainer_cfg.scheduler.pop('lambda_fn')
        lambda_fn = lambda step: lambda_lr_fn_d[lambda_fn_type](step, **cfg.trainer_cfg.scheduler)
        scheduler =  scheduler_d[scheduler_type](optimizer=optimizer, lr_lambda=lambda_fn)
    else:
        scheduler =  scheduler_d[scheduler_type](optimizer=optimizer, **cfg.trainer_cfg.scheduler)

    dataloader = data_loader_train
    print('len of dataloader-1: {}'.format(len(dataloader)))

    overide_train_micro_batch_size_per_gpu = cfg.trainer_cfg.get('overide_train_micro_batch_size_per_gpu', 0)
    if overide_train_micro_batch_size_per_gpu > 0:
        try:
            accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = overide_train_micro_batch_size_per_gpu
        except:
            pass

    dist.barrier()

    if cfg.trainer_cfg.get('prepare_dataloader', False):
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    dist.barrier()
    print('len of dataloader-2: {}'.format(len(dataloader)))

    resume = cfg.trainer_cfg.get('resume', '')
    auto_resume = cfg.trainer_cfg.get('auto_resume', True)
    auto_load_last_dit = cfg.trainer_cfg.get('auto_load_last_dit', True)
    skip_first_batches_in_resume = cfg.trainer_cfg.get('skip_first_batches_in_resume', True)

    start_epoch = 0
    skipped_dataloader = None

    if resume:
        pass
    elif auto_resume:
        sub_dirs = os.listdir(work_dir)
        availables = []
        for sub_dir in sub_dirs:
            steps = [int(x) for x in re.findall(r'step-(\d+)', sub_dir)]
            if len(steps) == 1:
                availables.append(steps[0])

        if len(availables) > 0:
            resume_iter = max(availables)
            resume = f'{work_dir}/step-{resume_iter}' 
            print('using as resume: {}'.format(resume))
        else:
            print('resume None')
    elif load_iter > 0:
        # auto load
        start_epoch = load_iter // len(dataloader)
        accelerator.state.num_steps = load_iter
        model_logger.num_steps = load_iter

    if resume is not None and resume != '':
        print(f'resuming from {resume}')
        accelerator.load_state(resume)
        resume_step = int(resume.split('-')[-1])
        # comment this because it is very very slow on huge dataset
        if skip_first_batches_in_resume:
            print('resume : skip_first_batches ...')
            skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=resume_step % len(dataloader))
        else:
            print('resume : donot replace dataloader for skip_first_batches ...')
        
        start_epoch = resume_step // len(dataloader)
        model_logger.num_steps = resume_step
        
        print(f'resume done')


    dist.barrier()

    print("start training...")

    # ========== PyTorch Profiler 配置 (BF16 原版) ==========
    # 设置为 True 启用性能分析（只分析前几个 step）
    ENABLE_PROFILER = False  # 性能测试完成，已关闭
    # 多 iteration 统计配置：
    # - wait=99: 跳过前 99 步（充分预热）
    # - warmup=1: 预热 1 步（第 100 步）
    # - active=20: 记录 20 步（第 101-120 步，充足样本做精确统计）
    PROFILER_WAIT_STEPS = 99    # 等待步数（跳过前 99 步）
    PROFILER_WARMUP_STEPS = 1   # 预热步数
    PROFILER_ACTIVE_STEPS = 20  # 记录步数（20 个 iter，文件约 2GB）
    PROFILER_OUTPUT_DIR = os.path.join(work_dir, 'profiler_output_bf16')
    
    pytorch_profiler = None
    if ENABLE_PROFILER and rank == 0:
        from torch.profiler import profile, ProfilerActivity, schedule
        os.makedirs(PROFILER_OUTPUT_DIR, exist_ok=True)
        
        # 定义按执行顺序的组件列表 (Forward + Backward)
        EXECUTION_ORDER_FWD = [
            # === 前置处理 ===
            ("TimeEmbedding", "1. Time Embedding", "timestep → t"),
            ("TimeProjection", "2. Time Projection", "t → t_mod (6×dim)"),
            ("TextEmbedding", "3. Text Embedding", "context 投影"),
            ("PatchEmbedding", "4. Patch Embedding", "Conv3d 提取 patch"),
            # === DiTBlock 内部 ===
            ("DiTBlock.Modulation", "5. Modulation", "计算 scale/shift/gate"),
            # Self-Attention
            ("SelfAttn.QKV_Proj", "6. Self-Attn Q/K/V 投影", "3 个 Linear"),
            ("SelfAttn.RoPE", "7. Self-Attn RoPE", "旋转位置编码"),
            ("SelfAttn.FlashAttn", "8. Self-Attn FlashAttn", "Q@K^T + Score@V"),
            ("SelfAttn.O_Proj", "9. Self-Attn O 投影", "1 个 Linear"),
            # GatedDeltaNet
            ("DiTBlock.GatedDeltaNet", "   [或] GatedDeltaNet", "Linear Attention"),
            # Cross-Attention
            ("CrossAttn.Q_Proj", "10. Cross-Attn Q 投影", "1 个 Linear"),
            ("CrossAttn.KV_Proj", "11. Cross-Attn K/V 投影", "2 个 Linear"),
            ("CrossAttn.FlashAttn", "12. Cross-Attn Attention", "Q@K^T@V"),
            ("CrossAttn.O_Proj", "13. Cross-Attn O 投影", "1 个 Linear"),
            # FFN
            ("FFN.Forward", "14. FFN Forward", "Up+Down 投影"),
            # === 后处理 ===
            ("Head", "15. Head", "最终输出投影"),
        ]
        
        # Backward 相关的 kernel 名称 (PyTorch autograd 自动生成)
        BACKWARD_KERNELS = [
            ("FlashAttnFunc", "FlashAttn Backward", "Attention 反向"),
            ("flash_bwd", "FlashAttn Backward (kernel)", "Attention 反向 kernel"),
            ("AddmmBackward", "Linear Backward (addmm)", "矩阵乘法反向"),
            ("MmBackward", "Linear Backward (mm)", "矩阵乘法反向"),
            ("SiluBackward", "SiLU Backward", "激活函数反向"),
        ]
        
        def trace_handler(prof):
            # 简化版 trace_handler - 只保存 trace 文件
            output_file = os.path.join(PROFILER_OUTPUT_DIR, f"trace_bf16_rank{rank}.json")
            prof.export_chrome_trace(output_file)
            print(f"\n[Profiler-BF16] Chrome Trace 已保存: {output_file}")
            print("[Profiler-BF16] 使用 Chrome 浏览器打开 chrome://tracing 加载此文件")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            
            # 以下为简化的统计
            backward_times = {
                'linear_bwd': 0,
                'flash_attn_bwd': 0,
                'norm_bwd': 0,
                'activation_bwd': 0,
                'other_bwd': 0,
            }
            
            # ==================== Forward 时间表 ====================
            print("\n")
            print("=" * 110)
            print("          📊 BF16 性能分析报告 - Forward + Backward (单 iteration)")
            print("=" * 110)
            
            # ===== FORWARD PASS =====
            print("\n" + "=" * 110)
            print("                              ⏩ FORWARD PASS")
            print("=" * 110)
            print()
            print("输入: x (视频 latent), timestep, context (文本)")
            print("      │")
            print("      ▼")
            print("┌" + "─" * 98 + "┐")
            print("│                              前置处理                                                          │")
            print("│" + "─" * 98 + "│")
            
            fwd_total = 0
            section_times = {'preprocessing': 0, 'self_attn': 0, 'cross_attn': 0, 'ffn': 0, 'postprocessing': 0}
            
            for key, display_name, desc in EXECUTION_ORDER_FWD[:4]:
                t = timings.get(key, 0)
                fwd_total += t
                section_times['preprocessing'] += t
                print(f"│  {display_name:<28} │ {desc:<25} │ {t:>10.2f} ms │   │")
            
            print("└" + "─" * 98 + "┘")
            print("      │")
            print("      ▼")
            print("┌" + "─" * 98 + "┐")
            print("│                              DiTBlock × 32                                                     │")
            print("│" + "─" * 98 + "│")
            print("│  ┌────────────────────── Self-Attention ──────────────────────┐                                │")
            
            for key, display_name, desc in EXECUTION_ORDER_FWD[4:11]:
                t = timings.get(key, 0)
                fwd_total += t
                section_times['self_attn'] += t
                print(f"│  │ {display_name:<26} │ {desc:<23} │ {t:>8.2f} ms │   │      │")
            
            print("│  └────────────────────────────────────────────────────────────┘                                │")
            print("│  ┌────────────────────── Cross-Attention ─────────────────────┐                                │")
            
            for key, display_name, desc in EXECUTION_ORDER_FWD[11:15]:
                t = timings.get(key, 0)
                fwd_total += t
                section_times['cross_attn'] += t
                print(f"│  │ {display_name:<26} │ {desc:<23} │ {t:>8.2f} ms │   │      │")
            
            print("│  └────────────────────────────────────────────────────────────┘                                │")
            print("│  ┌────────────────────── FFN ─────────────────────────────────┐                                │")
            
            for key, display_name, desc in EXECUTION_ORDER_FWD[15:16]:
                t = timings.get(key, 0)
                fwd_total += t
                section_times['ffn'] += t
                print(f"│  │ {display_name:<26} │ {desc:<23} │ {t:>8.2f} ms │   │      │")
            
            print("│  └────────────────────────────────────────────────────────────┘                                │")
            print("└" + "─" * 98 + "┘")
            print("      │")
            print("      ▼")
            print("┌" + "─" * 98 + "┐")
            
            for key, display_name, desc in EXECUTION_ORDER_FWD[16:]:
                t = timings.get(key, 0)
                fwd_total += t
                section_times['postprocessing'] += t
                print(f"│  {display_name:<28} │ {desc:<25} │ {t:>10.2f} ms │   │")
            
            print("└" + "─" * 98 + "┘")
            print("      │")
            print("      ▼")
            print("输出: loss")
            
            # ===== BACKWARD PASS =====
            print("\n" + "=" * 110)
            print("                              ⏪ BACKWARD PASS")
            print("=" * 110)
            print()
            print("loss.backward()")
            print("      │")
            print("      ▼")
            print("┌" + "─" * 98 + "┐")
            print("│                              梯度反向传播                                                      │")
            print("│" + "─" * 98 + "│")
            
            bwd_total = sum(backward_times.values())
            
            bwd_items = [
                ('linear_bwd', 'Linear Backward (GEMM)', '权重梯度计算'),
                ('flash_attn_bwd', 'FlashAttn Backward', 'Attention 梯度'),
                ('norm_bwd', 'Norm Backward', '归一化梯度'),
                ('activation_bwd', 'Activation Backward', '激活函数梯度'),
                ('other_bwd', 'Other Backward', '其他梯度'),
            ]
            
            for key, display_name, desc in bwd_items:
                t = backward_times[key]
                print(f"│  {display_name:<28} │ {desc:<25} │ {t:>10.2f} ms │   │")
            
            print("└" + "─" * 98 + "┘")
            print("      │")
            print("      ▼")
            print("梯度更新完成")
            print()
            
            # ==================== 总汇总 ====================
            total_time = fwd_total + bwd_total
            
            print("=" * 110)
            print("                         📈 Forward + Backward 汇总")
            print("=" * 110)
            print(f"{'阶段':<25} {'时间 (ms)':>15} {'占比':>12} {'说明':<30}")
            print("-" * 110)
            
            # Forward 分组
            print(f"{'[Forward]':<25} {fwd_total:>15.2f} {fwd_total/total_time*100 if total_time > 0 else 0:>11.1f}%")
            for name, display in [('preprocessing', '  ├─ 前置处理'), ('self_attn', '  ├─ Self-Attention'), 
                                  ('cross_attn', '  ├─ Cross-Attention'), ('ffn', '  ├─ FFN'), 
                                  ('postprocessing', '  └─ 后处理')]:
                t = section_times[name]
                pct = t / total_time * 100 if total_time > 0 else 0
                print(f"{display:<25} {t:>15.2f} {pct:>11.1f}%")
            
            # Backward 分组
            print(f"{'[Backward]':<25} {bwd_total:>15.2f} {bwd_total/total_time*100 if total_time > 0 else 0:>11.1f}%")
            for key, display_name, desc in bwd_items:
                t = backward_times[key]
                pct = t / total_time * 100 if total_time > 0 else 0
                print(f"  {'├─' if key != 'other_bwd' else '└─'} {display_name:<21} {t:>15.2f} {pct:>11.1f}%")
            
            print("-" * 110)
            print(f"{'总计':<25} {total_time:>15.2f} {'100.0':>11}%")
            print("=" * 110)
            print()
            
            # ==================== 保存 CSV 供后续对比 ====================
            csv_file = os.path.join(PROFILER_OUTPUT_DIR, f"timings_bf16_rank{rank}.csv")
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write("phase,component,display_name,description,time_ms\n")
                for key, display_name, desc in EXECUTION_ORDER_FWD:
                    t = timings.get(key, 0)
                    f.write(f"forward,{key},{display_name},{desc},{t:.4f}\n")
                for key, display_name, desc in bwd_items:
                    t = backward_times[key]
                    f.write(f"backward,{key},{display_name},{desc},{t:.4f}\n")
            print(f"[Profiler-BF16] 时间统计已保存: {csv_file}")
            print()
            
            # ==================== 原始详细报告 ====================
            print("=" * 80)
            print("[Profiler-BF16] 原始详细报告 (按 CUDA 时间排序，前 50 项)")
            print("=" * 80)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
        
        pytorch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=PROFILER_WAIT_STEPS, 
                warmup=PROFILER_WARMUP_STEPS, 
                active=PROFILER_ACTIVE_STEPS, 
                repeat=1
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,  # 启用 FLOPs 统计
            with_modules=True,  # 启用模块级别统计
        )
        pytorch_profiler.__enter__()
        print(f"[Profiler-BF16] 已启用 PyTorch Profiler (单 iter 模式: wait={PROFILER_WAIT_STEPS}, warmup={PROFILER_WARMUP_STEPS}, active={PROFILER_ACTIVE_STEPS})")
    # ======================================================

    loss_buffer = []
    for epoch_id in range(start_epoch, num_epochs):
        print("epoch:",epoch_id)

        train_dataloader = dataloader if skipped_dataloader is None else skipped_dataloader
        
        data_iter = iter(train_dataloader)
        
        for idx in range(len(train_dataloader)):
        # for data in train_dataloader:

            timer_recorder.start('full_iter')  

            timer_recorder.start('data_iter') 
            start_time = time.time()
            data = next(data_iter)
            end_time = time.time()
            timer_recorder.end('data_iter')

            if (end_time - start_time) * 1000 > 50:
                ### 超过100ms的fetch需要打印日志
                print(f"rank:{rank} data fetch time is too long: {(end_time - start_time) * 1000} milliseconds")
            
            if timer_recorder.timer_record_enable:
                real_dura = timer_recorder.records['data_iter'][-1]['elapsed']
                if real_dura * 1000 > 50:
                    ### 超过100ms的fetch需要打印日志
                    print(f"rank:{rank} data fetch time is too long: {real_dura * 1000:0.3f} milliseconds")

            # print('dbg dataloader: data {}'.format(data))
            if 'video' in data:
                for video in data["video"]:
                    if len(video) == 0:
                        print('Error: empty video, skip...')
                        continue

            if idx == 8 and os.environ.get("PROFILE") and (torch.distributed.get_rank()==0 if torch.distributed.is_initialized() else 1):
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA,],on_trace_ready=trace_handler_f) as p:
                    with accelerator.accumulate(model):
                        
                        timer_recorder.start('zero_grad')
                        optimizer.zero_grad()
                        timer_recorder.end('zero_grad')

                        timer_recorder.start('forward_model_all')
                        output_batch, loss = model(data)
                        timer_recorder.end('forward_model_all')

                        timer_recorder.start('backward')
                        # [Profiler] 已禁用测速标记
                        accelerator.backward(loss)
                        timer_recorder.end('backward')

                        timer_recorder.start('opt_step')
                        optimizer.step()
                        timer_recorder.end('opt_step')

                        model_logger.on_step_end(accelerator, model, save_steps)
                        # log
                        loss_gathered = accelerator.gather_for_metrics(loss.detach()).mean().item()
                        # loss_gathered = loss.item()

                        if len(loss_buffer) > 1000:
                            loss_buffer = loss_buffer[1:]
                        loss_buffer.append(loss_gathered)
                        loss_smooth = sum(loss_buffer) / len(loss_buffer)

                        timer_recorder.end('full_iter')  

                        if timer_recorder.timer_record_enable:

                            real_dura_forward = timer_recorder.records['forward_model_all'][-1]['elapsed']
                            real_dura_backward = timer_recorder.records['backward'][-1]['elapsed']

                            iter_dura = timer_recorder.records['full_iter'][-1]['elapsed']

                            time_speed_str = f'|| speed: {iter_dura:0.3f}s/iter forward: {real_dura_forward:0.2f}s backward: {real_dura_backward:0.2f}s '
                        else:
                            time_speed_str = ''

                        if accelerator.local_process_index == 0:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] rank: {rank}" 
                                  f"epoch {epoch_id}  step {model_logger.num_steps % len(train_dataloader)}/{len(train_dataloader)}  loss: {loss_gathered:.4f} loss-smooth: {loss_smooth:0.4f} "
                                  f"lr: {scheduler.get_last_lr()[0]:.2e} {time_speed_str} "
                                  
                                )
                            
                        scheduler.step()

                        # ========== Profiler step (BF16) ==========
                        if pytorch_profiler is not None:
                            pytorch_profiler.step()
                            # 分析完成后自动停止
                            if model_logger.num_steps >= PROFILER_WAIT_STEPS + PROFILER_WARMUP_STEPS + PROFILER_ACTIVE_STEPS:
                                pytorch_profiler.__exit__(None, None, None)
                                pytorch_profiler = None
                                ENABLE_PROFILER = False
                                print("[Profiler-BF16] 分析完成，已禁用")
            else:
                with accelerator.accumulate(model):
                    
                    timer_recorder.start('zero_grad')
                    optimizer.zero_grad()
                    timer_recorder.end('zero_grad')

                    timer_recorder.start('forward_model_all')
                    output_batch, loss = model(data)
                    timer_recorder.end('forward_model_all')

                    timer_recorder.start('backward')
                    # [Profiler] 已禁用测速标记
                    accelerator.backward(loss)
                    timer_recorder.end('backward')

                    timer_recorder.start('opt_step')
                    optimizer.step()
                    timer_recorder.end('opt_step')

                    model_logger.on_step_end(accelerator, model, save_steps)
                    # log
                    loss_gathered = accelerator.gather_for_metrics(loss.detach()).mean().item()
                    # loss_gathered = loss.item()

                    if len(loss_buffer) > 1000:
                        loss_buffer = loss_buffer[1:]
                    loss_buffer.append(loss_gathered)
                    loss_smooth = sum(loss_buffer) / len(loss_buffer)

                    timer_recorder.end('full_iter')  

                    if timer_recorder.timer_record_enable:

                        real_dura_forward = timer_recorder.records['forward_model_all'][-1]['elapsed']
                        real_dura_backward = timer_recorder.records['backward'][-1]['elapsed']

                        iter_dura = timer_recorder.records['full_iter'][-1]['elapsed']

                        time_speed_str = f'|| speed: {iter_dura:0.3f}s/iter forward: {real_dura_forward:0.2f}s backward: {real_dura_backward:0.2f}s '
                    else:
                        time_speed_str = ''

                    if accelerator.local_process_index == 0:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] rank: {rank}" 
                              f"epoch {epoch_id}  step {model_logger.num_steps % len(train_dataloader)}/{len(train_dataloader)}  loss: {loss_gathered:.4f} loss-smooth: {loss_smooth:0.4f} "
                              f"lr: {scheduler.get_last_lr()[0]:.2e} {time_speed_str} "
                              
                            )
                        
                    scheduler.step()

                    # ========== Profiler step (BF16) ==========
                    if pytorch_profiler is not None:
                        pytorch_profiler.step()
                        # 分析完成后自动停止
                        if model_logger.num_steps >= PROFILER_WAIT_STEPS + PROFILER_WARMUP_STEPS + PROFILER_ACTIVE_STEPS:
                            pytorch_profiler.__exit__(None, None, None)
                            pytorch_profiler = None
                            ENABLE_PROFILER = False
                            print("[Profiler-BF16] 分析完成，已禁用")
                    # ==========================================

                    # save_dir = f'{work_dir}/profile_files'
                    # os.makedirs(save_dir, exist_ok=True)

                    # save_path = '{}/iter_{:04d}_rank_{:03d}.pkl'.format(save_dir, idx, rank)
                    # with open(save_path, 'wb') as f:
                    #     pickle.dump(timer_recorder.records, f)
                    # timer_recorder.records = {}

        if skipped_dataloader is not None:
            skipped_dataloader = None
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)



if __name__ == '__main__':

    train_main()

