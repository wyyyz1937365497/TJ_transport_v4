"""
训练模块包

本模块包含所有训练相关的组件，包括：
1. 训练器（Trainers）：CustomPPOTrainer（v4）、V5PPOTrainer（v5）
2. 奖励计算（Rewards）：OCRRewardCalculator（基于官方评测公式）
3. 检查点管理（Checkpoint）：CheckpointManager
4. 训练增强（Enhancements）：课程学习、优先经验回放、失败案例库
5. 数据缓冲区（Buffers）：GPURolloutBuffer、V5RolloutBuffer
6. 多GPU工具（Multi-GPU）：设备映射和配置

使用示例：
    # v4架构
    from src.training import CustomPPOTrainer
    trainer = CustomPPOTrainer(env, policy, config, checkpoint_dir, device)

    # v5架构
    from src.training import V5PPOTrainer
    trainer = V5PPOTrainer(env, policy, config, checkpoint_dir, device)

    # OCR奖励计算
    from src.training import create_ocr_reward_calculator
    calculator = create_ocr_reward_calculator(config=config)

文档：参见 README.md
"""

# ========== 训练器（Trainers） ==========
from .custom_ppo_trainer import CustomPPOTrainer
from .v5_ppo_trainer import V5PPOTrainer, create_v5_ppo_trainer

# ========== 奖励计算（Rewards） ==========
from .ocr_rewards import (
    OCRRewardCalculator,
    EpisodeStatistics,
    BaselineStatisticsCollector,
    create_ocr_reward_calculator
)

# ========== 检查点管理（Checkpoint） ==========
from .checkpoint_manager import (
    PhaseCheckpointManager,
    create_checkpoint_manager
)

# ========== 训练增强（Enhancements） ==========
from .train_enhancements import (
    CurriculumManager,
    PrioritizedReplayBuffer,
    FailureCaseBank,
    EnhancedTrainingManager,
    DifficultyLevel,
    Transition,
    FailureCase,
    create_enhanced_training_manager,
    save_failure_cases,
    load_failure_cases
)

# ========== 数据缓冲区（Buffers） ==========
from .custom_ppo_trainer import GPURolloutBuffer
from .v5_rollout_buffer import V5RolloutBuffer

# ========== 多GPU工具（Multi-GPU） ==========
from .multi_gpu_utils import create_multi_gpu_manager

# ========== 导出列表 ==========
__all__ = [
    # 训练器
    'CustomPPOTrainer',
    'V5PPOTrainer',
    'create_v5_ppo_trainer',

    # 奖励计算
    'OCRRewardCalculator',
    'EpisodeStatistics',
    'BaselineStatisticsCollector',
    'create_ocr_reward_calculator',

    # 检查点管理
    'PhaseCheckpointManager',
    'create_checkpoint_manager',

    # 训练增强
    'CurriculumManager',
    'PrioritizedReplayBuffer',
    'FailureCaseBank',
    'EnhancedTrainingManager',
    'DifficultyLevel',
    'Transition',
    'FailureCase',
    'create_enhanced_training_manager',
    'save_failure_cases',
    'load_failure_cases',

    # 数据缓冲区
    'GPURolloutBuffer',
    'V5RolloutBuffer',

    # 多GPU工具
    'create_multi_gpu_manager',
]

# ========== 版本信息 ==========
__version__ = '1.1.0'
__author__ = 'Claude (Anthropic)'
__date__ = '2025-01-20'
