"""
多GPU训练和进度条工具模块

提供：
1. GPU配置和检测
2. DataParallel包装器
3. tqdm进度条集成
4. 训练统计和监控
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from typing import Dict, Any, Optional
import time


class MultiGPUManager:
    """多GPU训练管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多GPU管理器

        Args:
            config: 训练配置
        """
        self.config = config or {}
        self.use_cuda = torch.cuda.is_available()
        self.num_gpus = torch.cuda.device_count() if self.use_cuda else 0

        # 检查配置文件中的device设置
        device_config = self.config.get('device', 'cuda')
        if isinstance(device_config, str):
            # 如果是字符串（如'cuda'），使用单GPU
            self.multi_gpu = False
        elif isinstance(device_config, dict) and device_config.get('multi_gpu', False):
            # 只有明确指定multi_gpu=True时才启用
            self.multi_gpu = True
        else:
            # 默认使用单GPU
            self.multi_gpu = False

        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')

        # 打印GPU信息
        self._print_gpu_info()

    def _print_gpu_info(self):
        """打印GPU配置信息"""
        print("\n" + "="*60)
        print("[GPU Configuration]")
        print("="*60)
        print(f"  CUDA Available: {self.use_cuda}")
        print(f"  GPU Count: {self.num_gpus}")
        print(f"  Multi-GPU Mode: {self.multi_gpu}")
        print(f"  Default Device: {self.device}")

        if self.use_cuda:
            print(f"\n[GPU Details]")
            for i in range(self.num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")

            if not self.multi_gpu and self.num_gpus > 1:
                print(f"\n[INFO] Using single GPU mode (cuda:0) despite {self.num_gpus} GPUs available")
                print(f"[INFO] This optimizes for speed by avoiding cross-GPU communication")
        print("="*60 + "\n")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        包装模型以支持多GPU训练

        Args:
            model: PyTorch模型

        Returns:
            包装后的模型
        """
        if self.multi_gpu:
            print(f"[GPU] Wrapping model with DataParallel on {self.num_gpus} GPUs")
            model = DataParallel(model)
            print(f"[OK] Model wrapped successfully")

            # 打印GPU分配
            print(f"[GPU] Each GPU will process {self.num_gpus} GPU(s)")
        else:
            print(f"[GPU] Using single GPU: {self.device}")

        return model

    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """
        获取有效的batch size（考虑多GPU）

        Args:
            base_batch_size: 基础batch size

        Returns:
            有效的batch size
        """
        if self.multi_gpu:
            effective = base_batch_size * self.num_gpus
            print(f"[GPU] Effective batch size: {effective} "
                  f"({base_batch_size} per GPU × {self.num_gpus} GPUs)")
            return effective
        return base_batch_size

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        移除DataParallel包装

        Args:
            model: 可能被DataParallel包装的模型

        Returns:
            原始模型
        """
        if isinstance(model, DataParallel):
            return model.module
        return model

    def get_model_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        获取模型状态字典（自动处理DataParallel）

        Args:
            model: PyTorch模型

        Returns:
            模型状态字典
        """
        return self.unwrap_model(model).state_dict()


class ProgressTracker:
    """训练进度跟踪器"""

    def __init__(self, total_epochs: int, num_batches_per_epoch: int,
                 phase_name: str = "Training"):
        """
        初始化进度跟踪器

        Args:
            total_epochs: 总epoch数
            num_batches_per_epoch: 每个epoch的batch数
            phase_name: 阶段名称
        """
        self.total_epochs = total_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.phase_name = phase_name

        # 创建进度条
        self.epoch_pbar = tqdm(total=total_epochs, desc=f"[{phase_name}] Epochs",
                              unit="epoch", position=0, dynamic_ncols=True)
        self.batch_pbar = None
        self.current_epoch = 0

        # 统计信息
        self.best_metric = float('inf')  # 假设是loss，越小越好
        self.epoch_losses = []
        self.batch_times = []

    def start_epoch(self, epoch: int):
        """
        开始新的epoch

        Args:
            epoch: 当前epoch编号（从0开始）
        """
        self.current_epoch = epoch
        self.batch_pbar = tqdm(total=self.num_batches_per_epoch,
                              desc=f"  Epoch {epoch+1}/{self.total_epochs}",
                              unit="batch", leave=False, position=1,
                              dynamic_ncols=True)
        self.epoch_start_time = time.time()

    def update_batch(self, batch_idx: int, loss: float,
                    extra_metrics: Optional[Dict[str, float]] = None):
        """
        更新batch进度

        Args:
            batch_idx: 当前batch编号
            loss: 当前batch loss
            extra_metrics: 额外的指标
        """
        if self.batch_pbar:
            # 计算平均loss
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else loss

            # 更新进度条
            postfix = {
                'loss': f'{loss:.4f}',
                'avg': f'{avg_loss:.4f}'
            }

            if extra_metrics:
                postfix.update(extra_metrics)

            self.batch_pbar.set_postfix(postfix)
            self.batch_pbar.update(1)

            # 记录loss
            self.epoch_losses.append(loss)

    def end_epoch(self, avg_loss: float,
                  extra_metrics: Optional[Dict[str, float]] = None,
                  is_best: bool = False):
        """
        结束epoch

        Args:
            avg_loss: 平均loss
            extra_metrics: 额外的指标
            is_best: 是否是最佳模型
        """
        epoch_time = time.time() - self.epoch_start_time
        self.batch_times.append(epoch_time)

        if self.batch_pbar:
            self.batch_pbar.close()

        # 更新epoch进度条
        postfix = {
            'loss': f'{avg_loss:.4f}',
            'time': f'{epoch_time:.1f}s'
        }

        if is_best:
            postfix['best'] = f'{avg_loss:.4f} ✓'
            self.best_metric = avg_loss
        else:
            postfix['best'] = f'{self.best_metric:.4f}'

        if extra_metrics:
            postfix.update(extra_metrics)

        self.epoch_pbar.set_postfix(postfix)
        self.epoch_pbar.update(1)

        # 清空epoch loss记录
        self.epoch_losses = []

    def close(self):
        """关闭所有进度条"""
        if self.batch_pbar:
            self.batch_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()

        # 打印统计信息
        if self.batch_times:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            print(f"\n[STATS] Average epoch time: {avg_time:.2f}s")
            print(f"[STATS] Total training time: {sum(self.batch_times):.2f}s")

    def __del__(self):
        """析构函数，确保进度条关闭"""
        self.close()


class DataCollectionProgress:
    """数据收集进度跟踪器"""

    def __init__(self, num_workers: int):
        """
        初始化数据收集进度跟踪器

        Args:
            num_workers: 工作进程数
        """
        self.num_workers = num_workers
        self.pbar = tqdm(total=num_workers, desc="  Collecting",
                        unit="worker", leave=False, position=2,
                        dynamic_ncols=True)
        self.collected_samples = 0
        self.worker_results = {}

    def update_worker(self, worker_id: int, num_samples: int):
        """
        更新worker进度

        Args:
            worker_id: worker ID
            num_samples: 收集的样本数
        """
        self.collected_samples += num_samples
        self.worker_results[worker_id] = num_samples

        self.pbar.set_postfix({
            'worker': f'{worker_id}',
            'samples': f'{num_samples}',
            'total': f'{self.collected_samples}'
        })
        self.pbar.update(1)

    def close(self):
        """关闭进度条"""
        if self.pbar:
            self.pbar.close()

        # 打印统计
        if self.worker_results:
            print(f"\n  [OK] Collection Summary:")
            for worker_id, num_samples in sorted(self.worker_results.items()):
                print(f"      Worker {worker_id}: {num_samples} samples")
            print(f"      Total: {self.collected_samples} samples")

    def __del__(self):
        """析构函数"""
        self.close()


class TrainingMetrics:
    """训练指标记录器"""

    def __init__(self):
        """初始化指标记录器"""
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'memory_used': []
        }

    def update(self, epoch: int, train_loss: float,
               val_loss: Optional[float] = None,
               learning_rate: Optional[float] = None,
               epoch_time: Optional[float] = None):
        """
        更新指标

        Args:
            epoch: 当前epoch
            train_loss: 训练loss
            val_loss: 验证loss
            learning_rate: 学习率
            epoch_time: epoch耗时
        """
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)

        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)

        if learning_rate is not None:
            self.metrics['learning_rate'].append(learning_rate)

        if epoch_time is not None:
            self.metrics['epoch_time'].append(epoch_time)

        # 记录GPU内存
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            self.metrics['memory_used'].append(memory_used)

    def get_summary(self) -> Dict[str, Any]:
        """
        获取指标摘要

        Returns:
            指标摘要字典
        """
        summary = {}

        if self.metrics['train_loss']:
            summary['final_train_loss'] = self.metrics['train_loss'][-1]
            summary['best_train_loss'] = min(self.metrics['train_loss'])
            summary['avg_train_loss'] = sum(self.metrics['train_loss']) / len(self.metrics['train_loss'])

        if self.metrics['val_loss']:
            summary['final_val_loss'] = self.metrics['val_loss'][-1]
            summary['best_val_loss'] = min(self.metrics['val_loss'])

        if self.metrics['epoch_time']:
            summary['total_time'] = sum(self.metrics['epoch_time'])
            summary['avg_epoch_time'] = summary['total_time'] / len(self.metrics['epoch_time'])

        if self.metrics['memory_used']:
            summary['max_memory_used'] = max(self.metrics['memory_used'])

        return summary

    def print_summary(self):
        """打印指标摘要"""
        summary = self.get_summary()

        print("\n" + "="*60)
        print("[Training Summary]")
        print("="*60)

        if 'final_train_loss' in summary:
            print(f"  Final Train Loss: {summary['final_train_loss']:.4f}")
            print(f"  Best Train Loss: {summary['best_train_loss']:.4f}")
            print(f"  Avg Train Loss: {summary['avg_train_loss']:.4f}")

        if 'final_val_loss' in summary:
            print(f"  Final Val Loss: {summary['final_val_loss']:.4f}")
            print(f"  Best Val Loss: {summary['best_val_loss']:.4f}")

        if 'total_time' in summary:
            print(f"  Total Training Time: {summary['total_time']:.2f}s")
            print(f"  Avg Epoch Time: {summary['avg_epoch_time']:.2f}s")

        if 'max_memory_used' in summary:
            print(f"  Max GPU Memory Used: {summary['max_memory_used']:.2f} GB")

        print("="*60 + "\n")


def create_multi_gpu_manager(config: Optional[Dict[str, Any]] = None) -> MultiGPUManager:
    """
    创建多GPU管理器的便捷函数

    Args:
        config: 训练配置

    Returns:
        MultiGPUManager实例
    """
    return MultiGPUManager(config)


def create_progress_tracker(total_epochs: int, num_batches_per_epoch: int,
                            phase_name: str = "Training") -> ProgressTracker:
    """
    创建进度跟踪器的便捷函数

    Args:
        total_epochs: 总epoch数
        num_batches_per_epoch: 每个epoch的batch数
        phase_name: 阶段名称

    Returns:
        ProgressTracker实例
    """
    return ProgressTracker(total_epochs, num_batches_per_epoch, phase_name)


def create_data_collection_progress(num_workers: int) -> DataCollectionProgress:
    """
    创建数据收集进度跟踪器的便捷函数

    Args:
        num_workers: 工作进程数

    Returns:
        DataCollectionProgress实例
    """
    return DataCollectionProgress(num_workers)
