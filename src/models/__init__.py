"""
神经网络模型模块（已移除SB3依赖）

主要组件：
- v4_architecture.py: 核心架构模块（GNN、RSSM、控制器等）
- ideal_policy_v4.py: 独立PPO策略网络（无SB3依赖）
"""

from .v4_architecture import (
    RiskSensitiveGNN,
    MultiScaleRSSM,
    EnhancedDynamicWeightGating,
    EnhancedInfluenceBasedController,
    LagrangianOptimizer,
    IdealTrafficControllerV4
)

from .ideal_policy_v4 import (
    IdealTrafficPolicyV4,
    DiagonalGaussianDistribution,
    compute_risk_features_jit,
    safe_item,
    create_policy_v4,
    create_ideal_traffic_policy_v4  # 兼容工厂函数（返回类）
)

# 导出所有模块
__all__ = [
    # 核心架构
    'RiskSensitiveGNN',
    'MultiScaleRSSM',
    'EnhancedDynamicWeightGating',
    'EnhancedInfluenceBasedController',
    'LagrangianOptimizer',
    'IdealTrafficControllerV4',

    # PPO策略网络
    'IdealTrafficPolicyV4',
    'DiagonalGaussianDistribution',
    'compute_risk_features_jit',
    'safe_item',
    'create_policy_v4',
    'create_ideal_traffic_policy_v4',  # 兼容性别名
]
