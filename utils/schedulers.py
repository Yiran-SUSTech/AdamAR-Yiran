import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

total_training_steps = 100000 # 示例值，请替换为你的实际值
warmup_steps = int(total_training_steps * 0.1) # 通常为总步数的5% - 10%

def lr_lambda(current_step: int):
    if current_step < warmup_steps:
        # 线性预热阶段：从0到1线性增长
        return float(current_step) / float(max(1, warmup_steps))
    else:
        # 余弦退火阶段：从1下降到0
        progress = float(current_step - warmup_steps) / float(max(1, total_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    
# 创建调度器
# LambdaLR 会根据lr_lambda函数调整学习率
scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# 训练循环中：
for epoch in range(start_epoch+1, args.epochs+1):
    for step, (data, target) in enumerate(train_device_loader):
        optimizer.zero_grad()
        # ... 前向传播和反向传播 ...
        xm.optimizer_step(optimizer)
        scheduler.step() # 在每个训练步调用 scheduler.step() 来更新学习率
        # ... 日志记录 ...