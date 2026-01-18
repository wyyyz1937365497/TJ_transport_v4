"""
训练模块包

包含：
1. WorldModelTrainer - Phase 1 世界模型预训练
2. CustomPPOTrainer - Phase 2 PPO训练器（GPU优化）
3. PhaseCheckpointManager - 阶段检查点管理器
"""

from .world_model_train_v4 import WorldModelTrainer
from .custom_ppo_trainer import CustomPPOTrainer, GPURolloutBuffer
from .checkpoint_manager import PhaseCheckpointManager

__all__ = [
    'WorldModelTrainer',
    'CustomPPOTrainer',
    'GPURolloutBuffer',
    'PhaseCheckpointManager',
]
