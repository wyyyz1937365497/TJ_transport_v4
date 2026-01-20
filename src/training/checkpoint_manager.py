"""
阶段检查点管理器 - 自动保存和加载各阶段训练权重

功能：
1. 自动检查各阶段权重是否存在
2. 训练完成后自动保存当前阶段权重
3. 支持跳过已完成阶段
4. 提供权重状态查询

使用方法：
    from src.training.checkpoint_manager import PhaseCheckpointManager

    manager = PhaseCheckpointManager(checkpoint_dir='checkpoints/competition')

    # 检查阶段状态
    if manager.is_phase_completed('phase1'):
        print("Phase 1 已完成，跳过")
        checkpoint = manager.get_checkpoint_path('phase1')

    # 保存阶段权重
    manager.save_phase_checkpoint('phase1', model, optimizer, metrics)

    # 获取权重路径
    path = manager.get_checkpoint_path('phase2')
"""

import os
import torch
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json


class PhaseCheckpointManager:
    """
    阶段检查点管理器

    管理多阶段训练的检查点，支持：
    - 检查阶段是否完成
    - 自动保存阶段权重
    - 加载阶段权重
    - 生成训练报告
    """

    # 阶段定义
    PHASES = {
        'phase1': {
            'name': 'World Model预训练',
            'checkpoint_name': 'world_model_final.pth',
            'required_for': ['phase2', 'phase3', 'phase4']
        },
        'phase2': {
            'name': 'PPO策略训练',
            'checkpoint_name': 'ppo_policy.pth',
            'required_for': ['phase3', 'phase4']
        },
        'phase3': {
            'name': '端到端微调',
            'checkpoint_name': 'e2e_phase3.pth',
            'required_for': ['phase4']
        },
        'phase4': {
            'name': '约束优化训练',
            'checkpoint_name': 'final_model_competition.pth',
            'required_for': []
        }
    }

    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: 检查点根目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_phase_dir(self, phase: str) -> Path:
        """获取阶段检查点目录"""
        phase_dir = self.checkpoint_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def get_checkpoint_path(self, phase: str) -> Optional[str]:
        """
        获取阶段检查点路径

        Args:
            phase: 阶段名称 (phase1, phase2, phase3, phase4)

        Returns:
            检查点文件路径，如果不存在则返回None
        """
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase: {phase}")

        phase_dir = self.get_phase_dir(phase)
        checkpoint_name = self.PHASES[phase]['checkpoint_name']
        checkpoint_path = phase_dir / checkpoint_name

        if checkpoint_path.exists():
            return str(checkpoint_path)
        return None

    def is_phase_completed(self, phase: str) -> bool:
        """
        检查阶段是否完成（检查点是否存在）

        Args:
            phase: 阶段名称

        Returns:
            True if phase is completed
        """
        return self.get_checkpoint_path(phase) is not None

    def save_phase_checkpoint(
        self,
        phase: str,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        保存阶段检查点

        Args:
            phase: 阶段名称
            model: 模型对象
            optimizer: 优化器（可选）
            metrics: 训练指标（可选）
            additional_data: 额外数据（可选）

        Returns:
            保存的检查点路径
        """
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase: {phase}")

        phase_dir = self.get_phase_dir(phase)
        checkpoint_name = self.PHASES[phase]['checkpoint_name']
        checkpoint_path = phase_dir / checkpoint_name

        # 准备保存数据
        save_data = {
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
        }

        # 添加额外数据
        if additional_data:
            save_data.update(additional_data)

        # 统一使用 PyTorch 标准格式保存所有阶段
        if optimizer is not None:
            save_data['optimizer_state_dict'] = optimizer.state_dict()

        # 保存模型
        if hasattr(model, 'state_dict'):
            save_data['model_state_dict'] = model.state_dict()
        else:
            raise ValueError(f"Model for {phase} must have state_dict() method")

        torch.save(save_data, checkpoint_path)
        print(f"[OK] Phase {phase} saved to: {checkpoint_path}")

        # 保存元数据
        self._save_metadata(phase, metrics)

        return str(checkpoint_path)

    def load_phase_checkpoint(
        self,
        phase: str,
        model: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
    ) -> Dict[str, Any]:
        """
        加载阶段检查点

        Args:
            phase: 阶段名称
            model: 模型对象
            optimizer: 优化器（可选）
            device: 设备

        Returns:
            包含metrics的字典
        """
        checkpoint_path = self.get_checkpoint_path(phase)
        if checkpoint_path is None:
            raise FileNotFoundError(f"Checkpoint for {phase} not found")

        print(f"[LOAD] Loading {phase} from: {checkpoint_path}")

        # 统一使用 PyTorch 标准格式加载所有阶段
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 加载模型权重
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Phase {phase} model weights loaded")
        else:
            raise ValueError(f"Model for {phase} must have load_state_dict() method")

        # 加载优化器
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[OK] Phase {phase} optimizer loaded")

        return checkpoint

    def _save_metadata(self, phase: str, metrics: Optional[Dict[str, Any]]):
        """保存阶段元数据"""
        metadata_path = self.get_phase_dir(phase) / 'metadata.json'

        metadata = {
            'phase': phase,
            'phase_name': self.PHASES[phase]['name'],
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {},
            'checkpoint_exists': True
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_training_status(self) -> Dict[str, Any]:
        """
        获取训练状态报告

        Returns:
            包含各阶段状态的字典
        """
        status = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'phases': {}
        }

        for phase_key, phase_info in self.PHASES.items():
            phase_dir = self.get_phase_dir(phase_key)
            checkpoint_path = phase_dir / phase_info['checkpoint_name']
            metadata_path = phase_dir / 'metadata.json'

            phase_status = {
                'name': phase_info['name'],
                'completed': checkpoint_path.exists(),
                'checkpoint_path': str(checkpoint_path) if checkpoint_path.exists() else None,
                'required_for': phase_info['required_for']
            }

            # 尝试加载元数据
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        phase_status['timestamp'] = metadata.get('timestamp')
                        phase_status['metrics'] = metadata.get('metrics', {})
                except Exception as e:
                    phase_status['metadata_error'] = str(e)

            status['phases'][phase_key] = phase_status

        return status

    def print_training_status(self):
        """打印训练状态"""
        status = self.get_training_status()

        print("\n" + "="*80)
        print("训练状态报告 (Training Status)")
        print("="*80)

        for phase_key, phase_status in status['phases'].items():
            # 状态图标
            status_icon = "✅" if phase_status['completed'] else "⏳"

            print(f"\n{status_icon} {phase_key.upper()}: {phase_status['name']}")

            if phase_status['completed']:
                print(f"    状态: 已完成")
                print(f"    检查点: {phase_status['checkpoint_path']}")
                if 'timestamp' in phase_status:
                    print(f"    时间: {phase_status['timestamp']}")
                if 'metrics' in phase_status and phase_status['metrics']:
                    print(f"    指标: {phase_status['metrics']}")
            else:
                print(f"    状态: 未完成")
                required = phase_status.get('required_for', [])
                if required:
                    print(f"    前置要求: 无")
                else:
                    print(f"    前置要求: Phase {int(phase_key[-1]) - 1}")

        print("\n" + "="*80)

    def get_next_phase(self, current_phase: Optional[str] = None) -> Optional[str]:
        """
        获取下一个未完成的阶段

        Args:
            current_phase: 当前阶段（可选）

        Returns:
            下一个阶段的名称，如果全部完成则返回None
        """
        phase_keys = list(self.PHASES.keys())

        if current_phase:
            current_idx = phase_keys.index(current_phase)
            next_idx = current_idx + 1
        else:
            # 找到第一个未完成的阶段
            for i, phase_key in enumerate(phase_keys):
                if not self.is_phase_completed(phase_key):
                    return phase_key
            return None

        if next_idx < len(phase_keys):
            return phase_keys[next_idx]
        return None

    def can_skip_phase(self, phase: str) -> Tuple[bool, Optional[str]]:
        """
        检查是否可以跳过某个阶段

        Args:
            phase: 阶段名称

        Returns:
            (can_skip, checkpoint_path)
        """
        checkpoint_path = self.get_checkpoint_path(phase)
        if checkpoint_path:
            return True, checkpoint_path
        return False, None

    def clean_phase(self, phase: str, backup: bool = True):
        """
        清理某个阶段的检查点

        Args:
            phase: 阶段名称
            backup: 是否创建备份
        """
        if phase not in self.PHASES:
            raise ValueError(f"Unknown phase: {phase}")

        phase_dir = self.get_phase_dir(phase)
        checkpoint_name = self.PHASES[phase]['checkpoint_name']
        checkpoint_path = phase_dir / checkpoint_name

        if checkpoint_path.exists():
            if backup:
                # 创建备份
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = phase_dir / f"{checkpoint_name}.backup_{timestamp}"
                shutil.copy(str(checkpoint_path), str(backup_path))
                print(f"[BACKUP] Created backup: {backup_path}")

            # 删除检查点
            checkpoint_path.unlink()
            print(f"[CLEAN] Removed checkpoint: {checkpoint_path}")

            # 删除元数据
            metadata_path = phase_dir / 'metadata.json'
            if metadata_path.exists():
                metadata_path.unlink()


def create_checkpoint_manager(checkpoint_dir: str) -> PhaseCheckpointManager:
    """工厂函数：创建检查点管理器"""
    return PhaseCheckpointManager(checkpoint_dir)


# 便捷函数
def get_phase_checkpoint_path(checkpoint_dir: str, phase: str) -> Optional[str]:
    """获取阶段检查点路径（便捷函数）"""
    manager = PhaseCheckpointManager(checkpoint_dir)
    return manager.get_checkpoint_path(phase)


def is_phase_completed(checkpoint_dir: str, phase: str) -> bool:
    """检查阶段是否完成（便捷函数）"""
    manager = PhaseCheckpointManager(checkpoint_dir)
    return manager.is_phase_completed(phase)


def print_training_status(checkpoint_dir: str):
    """打印训练状态（便捷函数）"""
    manager = PhaseCheckpointManager(checkpoint_dir)
    manager.print_training_status()
