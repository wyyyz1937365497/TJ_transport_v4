"""
工具函数模块
"""

import os
import json
import time
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import logging


def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别

    Returns:
        logger: 日志记录器
    """
    os.makedirs(log_dir, exist_ok=True)

    # 创建logger
    logger = logging.getLogger('TrafficControl')
    logger.setLevel(log_level)

    # 文件处理器
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_config(
    config: Dict[str, Any],
    filepath: str
):
    """
    保存配置到JSON文件

    Args:
        config: 配置字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 配置已保存: {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """
    从JSON文件加载配置

    Args:
        filepath: 配置文件路径

    Returns:
        config: 配置字典
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config


def set_seed(seed: int):
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # CUDA确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: PyTorch模型

    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(
    prefer_cuda: bool = True,
    cuda_id: int = 0
) -> torch.device:
    """
    获取计算设备

    Args:
        prefer_cuda: 是否优先使用CUDA
        cuda_id: CUDA设备ID

    Returns:
        device: torch.device对象
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{cuda_id}')
        print(f"✅ 使用CUDA: {torch.cuda.get_device_name(cuda_id)}")
    else:
        device = torch.device('cpu')
        print(f"✅ 使用CPU")

    return device


def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def print_model_summary(model: torch.nn.Module):
    """
    打印模型摘要

    Args:
        model: PyTorch模型
    """
    print("\n" + "="*70)
    print("模型摘要")
    print("="*70)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

        print(f"{name:50s} {list(param.shape):30s} {num_params:>10,}")

    print("="*70)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")
    print("="*70 + "\n")


class EarlyStopping:
    """
    早停机制
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Args:
            patience: 容忍epoch数
            min_delta: 最小变化量
            mode: 'min' 或 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前得分

        Returns:
            是否早停
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AverageMeter:
    """
    平均值计算器
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新

        Args:
            val: 新值
            n: 数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载检查点

    Args:
        model: 模型
        checkpoint_path: 检查点路径
        optimizer: 优化器（可选）
        device: 设备（可选）

    Returns:
        checkpoint信息
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', -1)
    loss = checkpoint.get('loss', None)

    print(f"✅ 检查点已加载: {checkpoint_path}")
    print(f"   - Epoch: {epoch}")
    if loss is not None:
        print(f"   - Loss: {loss:.6f}")

    return checkpoint


def save_checkpoint(
    model: torch.nn.Module,
    filepath: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss: Optional[float] = None,
    **kwargs
):
    """
    保存检查点

    Args:
        model: 模型
        filepath: 保存路径
        epoch: 当前epoch
        optimizer: 优化器（可选）
        loss: 损失（可选）
        **kwargs: 其他要保存的信息
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if loss is not None:
        checkpoint['loss'] = loss

    checkpoint.update(kwargs)

    torch.save(checkpoint, filepath)
    print(f"✅ 检查点已保存: {filepath}")
