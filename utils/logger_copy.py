# import logging
# import os
# import torch_xla.core.xla_model as xm

# def create_logger(logging_dir, rank):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)

#     if rank == 0:
#         try:
#             os.makedirs(logging_dir, exist_ok=True)
#             log_file = os.path.join(logging_dir, "log.txt")
            
#             # 清除现有handler（避免重复）
#             for h in logger.handlers[:]:
#                 logger.removeHandler(h)
            
#             # 设置文件和控制台输出
#             file_handler = logging.FileHandler(log_file, mode='a')  # 追加模式
#             file_handler.setFormatter(logging.Formatter(
#                 '[%(asctime)s] %(message)s',
#                 datefmt='%Y-%m-%d %H:%M:%S'
#             ))
#             logger.addHandler(file_handler)

#             console_handler = logging.StreamHandler()
#             console_handler.setFormatter(logging.Formatter('\033[34m%(message)s\033[0m'))
#             logger.addHandler(console_handler)
            
#             logger.propagate = False  # 防止重复日志
#             # logger.info(f"Rank {rank} initialized logger")  # 测试日志
#         except Exception as e:
#             print(f"Logger初始化失败: {str(e)}", flush=True)
#             raise
#     else:
#         logger.addHandler(logging.NullHandler())
    
#     return logger

import logging
import os
import torch_xla.core.xla_model as xm

def create_logger(logging_dir):
    """为每个TPU核心创建独立的日志文件"""
    rank = xm.get_ordinal()  # 获取当前核心的rank
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)
    
    # 确保日志目录存在
    os.makedirs(logging_dir, exist_ok=True)
    
    # 为每个核心创建独立的日志文件
    log_file = os.path.join(logging_dir, f"rank_{rank}.log")
    
    try:
        # 清除现有handler（避免重复）
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 文件handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] [RANK %(rank)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
        
        # 控制台handler（可选）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            f'[RANK {rank}] %(message)s'
        ))
        logger.addHandler(console_handler)
        
        # 添加rank信息到日志记录
        old_factory = logging.getLogRecordFactory()
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.rank = rank  # 添加rank属性
            return record
        logging.setLogRecordFactory(record_factory)
    except Exception as e:
        print(f"Logger初始化失败: {str(e)}", flush=True)
        raise
    
    return logger