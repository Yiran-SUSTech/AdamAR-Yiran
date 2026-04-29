import os
import torch
import subprocess
from pathlib import Path


def detect_platform():
    if not torch.cuda.is_available():
        return 'cpu'
    
    try:
        result = subprocess.run(
            ['mx-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'MetaX' in result.stdout:
            return 'metax'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'GPU' in result.stdout:
            return 'nvidia'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    gpu_name = torch.cuda.get_device_name(0)
    if 'MetaX' in gpu_name or 'C500' in gpu_name:
        return 'metax'
    
    return 'nvidia'


def get_comm_backend(platform=None):
    if platform is None:
        platform = detect_platform()
    
    # MetaX PyTorch uses NCCL backend API (implemented by MCCL underneath)
    if platform == 'metax':
        return 'nccl'
    elif platform == 'nvidia':
        return 'nccl'
    else:
        return 'gloo'


def configure_platform(platform=None):
    if platform is None:
        platform = detect_platform()
    
    if platform == 'metax':
        os.environ['MACA_VISIBLE_DEVICES'] = os.environ.get(
            'CUDA_VISIBLE_DEVICES', 
            os.environ.get('MACA_VISIBLE_DEVICES', '')
        )
        
        if 'CUDA_VISIBLE_DEVICES' not in os.environ and 'MACA_VISIBLE_DEVICES' in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['MACA_VISIBLE_DEVICES']
        
        os.environ['MCCL_SOCKET_IFNAME'] = os.environ.get('MCCL_SOCKET_IFNAME', 'eth0')
        
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        return {
            'platform': 'metax',
            'backend': 'nccl',
            'tf32_enabled': False,
            'compile_supported': False,
        }
    
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        return {
            'platform': 'nvidia',
            'backend': 'nccl',
            'tf32_enabled': True,
            'compile_supported': True,
        }
