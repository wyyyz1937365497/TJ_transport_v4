"""
神经网络模型模块（v5轻量级架构）

主要组件：
- v5_lightweight.py: 轻量级OCR-GNN策略网络（专为初赛优化）
"""

from .v5_lightweight import (
    LightweightGraphConvolution,
    VehicleInfluenceScorer,
    LightweightOCRGNN,
    LightweightPolicyV5,
    create_lightweight_policy_v5,
)

# 导出所有模块
__all__ = [
    # 轻量级OCR-GNN
    'LightweightGraphConvolution',
    'VehicleInfluenceScorer',
    'LightweightOCRGNN',
    'LightweightPolicyV5',
    'create_lightweight_policy_v5',
]
