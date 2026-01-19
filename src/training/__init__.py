"""
训练增强模块包

包含：
1. 课程学习（Curriculum Learning）
2. 优先经验回放（Prioritized Experience Replay）
3. 失败案例库（Failure Case Bank）
4. 集成训练管理器
"""

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

__all__ = [
    'CurriculumManager',
    'PrioritizedReplayBuffer',
    'FailureCaseBank',
    'EnhancedTrainingManager',
    'DifficultyLevel',
    'Transition',
    'FailureCase',
    'create_enhanced_training_manager',
    'save_failure_cases',
    'load_failure_cases'
]
