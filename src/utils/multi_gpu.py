"""
多GPU训练工具
支持 DataParallel 和 DistributedDataParallel
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import os


def setup_multi_gpu_training(
    model: nn.Module,
    device_ids: Optional[list] = None,
    output_device: Optional[torch.device] = None
) -> nn.Module:
    """
    设置多GPU训练

    Args:
        model: PyTorch模型
        device_ids: GPU设备ID列表（如[0, 1]）
        output_device: 输出设备

    Returns:
        支持多GPU的模型
    """
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用单CPU训练")
        return model.to(torch.device('cpu'))

    # 自动检测所有可用的GPU
    if device_ids is None:
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print("⚠️  没有检测到CUDA设备")
            return model.to(torch.device('cpu'))
        device_ids = list(range(device_count))

    if len(device_ids) == 1:
        # 单GPU
        print(f"✅ 使用单GPU训练: {torch.cuda.get_device_name(0)}")
        return model.to(torch.device(f'cuda:{device_ids[0]}'))

    # 多GPU
    print(f"🚀 启用多GPU训练 ({len(device_ids)} 张显卡):")
    for i, device_id in enumerate(device_ids):
        print(f"   GPU {i}: {torch.cuda.get_device_name(device_id)}")

    # 使用DataParallel
    model = model.to(torch.device(f'cuda:{device_ids[0]}'))
    model = nn.DataParallel(model, device_ids=device_ids, output_device=output_device)

    # 显示信息
    total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in device_ids) / (1024**3)
    print(f"   总显存: {total_memory:.1f} GB")

    return model


def adjust_hyperparameters_for_multi_gpu(
    base_config: Dict[str, Any],
    num_gpus: int
) -> Dict[str, Any]:
    """
    根据GPU数量调整超参数

    Args:
        base_config: 基础配置
        num_gpus: GPU数量

    Returns:
        调整后的配置
    """
    config = base_config.copy()

    # Batch size线性缩放
    # 如果原来是64，2个GPU就变成128
    if 'batch_size' in config:
        original_batch_size = config['batch_size']
        scaled_batch_size = original_batch_size * num_gpus
        config['batch_size'] = scaled_batch_size
        print(f"📊 Batch size调整: {original_batch_size} → {scaled_batch_size} (x{num_gpus})")

    # 学习率线性缩放
    # 如果原来是1e-4，2个GPU就变成2e-4
    if 'lr' in config:
        original_lr = config['lr']
        scaled_lr = original_lr * num_gpus
        config['lr'] = scaled_lr
        print(f"📈 学习率调整: {original_lr:.6f} → {scaled_lr:.6f} (x{num_gpus})")

    return config


class MultiGPUTrainer:
    """
    多GPU训练器包装类
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device_ids: Optional[list] = None
    ):
        """
        Args:
            model: PyTorch模型
            config: 训练配置
            device_ids: GPU设备ID列表
        """
        self.config = config
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        if device_ids is None:
            device_ids = list(range(self.num_gpus))

        # 设置多GPU模型
        self.model = setup_multi_gpu_training(model, device_ids=device_ids)

        # 调整超参数
        if self.num_gpus > 1:
            print(f"\n🔧 为多GPU训练调整超参数...")
            self.config = adjust_hyperparameters_for_multi_gpu(config, self.num_gpus)

        # 设备
        self.device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')

    def get_model(self):
        """获取多GPU模型"""
        return self.model

    def get_device(self):
        """获取主设备"""
        return self.device

    def get_num_gpus(self):
        """获取GPU数量"""
        return self.num_gpus


def print_gpu_info():
    """打印GPU信息"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用")
        return

    print("="*70)
    print("🎮 GPU 信息")
    print("="*70)

    device_count = torch.cuda.device_count()
    print(f"检测到 {device_count} 张CUDA设备:")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 显存: {props.total_memory / (1024**3):.1f} GB")
        print(f"  - 计算能力: {props.major}.{props.minor}")
        print(f"  - 多处理器数: {props.multi_processor_count}")

    print("="*70)


def monitor_gpu_usage():
    """监控GPU使用情况"""
    if not torch.cuda.is_available():
        return

    print("\n📊 GPU使用情况:")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        utilization = (allocated / total * 100) if total > 0 else 0

        print(f"  GPU {i}: {utilization:.1f}% 显存使用 "
              f"({allocated:.1f}GB / {total:.1f}GB, "
              f"保留: {reserved:.1f}GB)")
