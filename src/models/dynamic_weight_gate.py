"""
动态权重门控（Dynamic Weight Gate）

功能：
1. 根据场景状态动态调整奖励权重
2. 元学习机制，端到端训练
3. 自动平衡效率、稳定性、成本三个目标

核心思想：
- 不同场景需要不同的优化策略
- 拥堵时：更关注效率 (w_eff ↑)
- 平稳时：更关注稳定性 (w_stab ↑)
- 高成本时：更关注成本 (w_cost ↑)

权重归一化：Softmax确保 w_eff + w_stab + w_cost = 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DynamicWeightGate(nn.Module):
    """
    动态权重门控网络

    输入：全局状态特征
    输出：三个目标的动态权重 [w_efficiency, w_stability, w_cost]

    架构：
    1. 特征提取：2层MLP
    2. 权重生成：Linear → Softmax
    """

    def __init__(
        self,
        global_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            global_dim: 全局状态特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()

        self.global_dim = global_dim
        self.hidden_dim = hidden_dim

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 权重生成头
        self.weight_head = nn.Linear(hidden_dim, 3)  # [w_eff, w_stab, w_cost]

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            global_state: [B, G] 全局状态特征

        Returns:
            weights: [B, 3] 动态权重 [w_efficiency, w_stability, w_cost]
                     归一化：sum(weights, dim=-1) = 1
        """
        # 特征提取
        features = self.feature_extractor(global_state)  # [B, hidden_dim]

        # 生成权重logits
        weight_logits = self.weight_head(features)  # [B, 3]

        # Softmax归一化
        weights = F.softmax(weight_logits, dim=-1)  # [B, 3]

        return weights

    def get_weight_dict(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取权重字典（便于使用）

        Args:
            global_state: [B, G] 全局状态特征

        Returns:
            Dict: {
                'w_efficiency': [B],
                'w_stability': [B],
                'w_cost': [B]
            }
        """
        weights = self.forward(global_state)  # [B, 3]

        return {
            'w_efficiency': weights[:, 0],
            'w_stability': weights[:, 1],
            'w_cost': weights[:, 2]
        }

    def compute_weighted_reward(
        self,
        global_state: torch.Tensor,
        reward_efficiency: torch.Tensor,
        reward_stability: torch.Tensor,
        reward_cost: torch.Tensor
    ) -> torch.Tensor:
        """
        计算加权总奖励

        Args:
            global_state: [B, G] 全局状态特征
            reward_efficiency: [B] 效率奖励
            reward_stability: [B] 稳定性奖励
            reward_cost: [B] 成本奖励（负成本）

        Returns:
            weighted_reward: [B] 加权总奖励
        """
        weights = self.forward(global_state)  # [B, 3]

        # 加权求和
        weighted_reward = (
            weights[:, 0] * reward_efficiency +
            weights[:, 1] * reward_stability +
            weights[:, 2] * reward_cost
        )

        return weighted_reward


class AdaptiveWeightGate(DynamicWeightGate):
    """
    自适应权重门控（增强版）

    在DynamicWeightGate基础上增加：
    1. 显式的拥堵状态感知
    2. 温度参数控制权重分布锐度
    3. 历史平滑机制
    """

    def __init__(
        self,
        global_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.1,
        temperature: float = 1.0,
        smoothing_coef: float = 0.9
    ):
        """
        Args:
            global_dim: 全局状态特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout率
            temperature: Softmax温度（越小越锐利）
            smoothing_coef: 历史平滑系数
        """
        super().__init__(global_dim, hidden_dim, dropout)

        self.temperature = temperature
        self.smoothing_coef = smoothing_coef

        # 注册历史权重缓冲区
        self.register_buffer(
            'prev_weights',
            torch.ones(1, 3) / 3.0  # 初始化为均匀分布
        )

    def forward(self, global_state: torch.Tensor, use_smoothing: bool = True) -> torch.Tensor:
        """
        前向传播（带温度和平滑）

        Args:
            global_state: [B, G] 全局状态特征
            use_smoothing: 是否使用历史平滑

        Returns:
            weights: [B, 3] 动态权重
        """
        # 特征提取
        features = self.feature_extractor(global_state)  # [B, hidden_dim]

        # 生成权重logits
        weight_logits = self.weight_head(features)  # [B, 3]

        # 应用温度
        weight_logits = weight_logits / self.temperature

        # Softmax归一化
        weights = F.softmax(weight_logits, dim=-1)  # [B, 3]

        # 历史平滑（可选）
        if use_smoothing and self.training:
            weights = (
                self.smoothing_coef * self.prev_weights +
                (1 - self.smoothing_coef) * weights.detach()
            )

            # 更新历史
            self.prev_weights = weights.mean(dim=0, keepdim=True)

        return weights

    def set_temperature(self, temperature: float):
        """动态调整温度"""
        self.temperature = temperature


def create_dynamic_weight_gate(
    global_dim: int = 64,
    hidden_dim: int = 32,
    dropout: float = 0.1,
    adaptive: bool = False,
    device: str = 'cuda'
) -> nn.Module:
    """
    创建动态权重门控的工厂函数

    Args:
        global_dim: 全局状态特征维度
        hidden_dim: 隐藏层维度
        dropout: Dropout率
        adaptive: 是否使用自适应版本
        device: 设备

    Returns:
        gate: DynamicWeightGate或AdaptiveWeightGate实例
    """
    if adaptive:
        model = AdaptiveWeightGate(
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        model = DynamicWeightGate(
            global_dim=global_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    return model.to(device)


# 预设权重配置（用于对比实验）
PRESET_WEIGHTS = {
    'balanced': torch.tensor([0.33, 0.33, 0.34]),     # 平衡
    'efficiency_focused': torch.tensor([0.6, 0.2, 0.2]),  # 效率优先
    'stability_focused': torch.tensor([0.2, 0.6, 0.2]),   # 稳定性优先
    'cost_focused': torch.tensor([0.2, 0.2, 0.6])        # 成本优先
}


class FixedWeightGate(nn.Module):
    """
    固定权重门控（Baseline）

    用于对比实验，验证动态权重门控的有效性
    """

    def __init__(self, preset: str = 'balanced'):
        """
        Args:
            preset: 预设配置名称
        """
        super().__init__()

        assert preset in PRESET_WEIGHTS, f"未知预设: {preset}"

        # 注册固定权重为缓冲区（不可训练）
        self.register_buffer('weights', PRESET_WEIGHTS[preset])

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播（返回固定权重）

        Args:
            global_state: [B, G] 全局状态特征（忽略）

        Returns:
            weights: [1, 3] 固定权重（广播到[B, 3]）
        """
        batch_size = global_state.size(0)
        return self.weights.unsqueeze(0).expand(batch_size, 3)
