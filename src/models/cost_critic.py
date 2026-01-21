"""
成本评论家（Cost Critic）- 约束优化模块

功能：
1. 预测干预成本（不同于传统的价值函数）
2. 支持拉格朗日松弛进行约束优化
3. 自适应调整拉格朗日乘数

核心思想：
- 传统RL只优化奖励 R(s,a)
- 约束RL需要同时优化：max E[R] s.t. E[C] ≤ threshold
- 使用拉格朗日松弛：L = R - λ * max(0, C - threshold)

参考文献：
- Achiam et al. (2017) "Constrained Policy Optimization"
- Stooke et al. (2020) "Constrained Policy Optimization: A Approximate Gradient Approach"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CostCritic(nn.Module):
    """
    成本评论家网络

    双头架构：
    1. Value头：预测状态价值 V(s)
    2. Cost头：预测成本 C(s)

    训练方式：
    - Value头：标准TD error
    - Cost头：类似TD error，但针对成本信号
    - 拉格朗日乘数λ自适应更新
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        global_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 车辆嵌入维度
            global_dim: 全局特征维度
            dropout: Dropout率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.global_dim = global_dim

        # 共享特征提取器
        self.shared_features = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Value头（预测价值）
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Cost头（预测成本）
        self.cost_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # 确保成本为正
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        global_state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            embeddings: [B, N, D] 车辆嵌入（池化后）
            global_state: [B, G] 全局状态特征

        Returns:
            Dict:
                'value': [B, 1] 状态价值
                'cost': [B, 1] 预期成本
        """
        # 全局池化车辆嵌入
        if embeddings.dim() == 3:  # [B, N, D]
            pooled_emb = embeddings.mean(dim=1)  # [B, D]
        else:  # [B, D]
            pooled_emb = embeddings

        # 拼接全局状态
        combined = torch.cat([pooled_emb, global_state], dim=-1)  # [B, D+G]

        # 共享特征提取
        shared_feat = self.shared_features(combined)  # [B, 128]

        # Value预测
        value = self.value_head(shared_feat)  # [B, 1]

        # Cost预测
        cost = self.cost_head(shared_feat)  # [B, 1]

        return {
            'value': value,
            'cost': cost
        }

    def compute_cost_loss(
        self,
        pred_costs: torch.Tensor,
        target_costs: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        计算成本预测损失（类似MSE）

        Args:
            pred_costs: [B, 1] 预测成本
            target_costs: [B, 1] 目标成本（实际发生的成本）
            mask: [B, 1] 掩码（可选，用于标记有效样本）

        Returns:
            loss: 标量损失
        """
        if mask is not None:
            pred_costs = pred_costs * mask
            target_costs = target_costs * mask

        loss = F.mse_loss(pred_costs, target_costs)

        return loss


class LagrangianOptimizer:
    """
    拉格朗日乘数优化器

    功能：
    1. 自适应调整λ
    2. 确保约束满足
    3. 支持对偶梯度下降

    更新规则：
    λ = max(0, λ + λ_lr * (C - threshold))
    """

    def __init__(
        self,
        initial_lambda: float = 0.1,
        lambda_lr: float = 0.01,
        cost_threshold: float = 0.5
    ):
        """
        Args:
            initial_lambda: 初始拉格朗日乘数
            lambda_lr: λ学习率
            cost_threshold: 成本阈值（约束上界）
        """
        self.lambda_lr = lambda_lr
        self.cost_threshold = cost_threshold

        # 注册λ为可学习参数
        self.lambda_param = nn.Parameter(torch.tensor(initial_lambda))

    def compute_lagrangian_loss(
        self,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        cost_threshold: float = None
    ) -> torch.Tensor:
        """
        计算拉格朗日损失

        L = -R + λ * max(0, C - threshold)

        Args:
            rewards: [B] 奖励
            costs: [B] 成本
            cost_threshold: 成本阈值（如果为None，使用self.cost_threshold）

        Returns:
            lagrangian_loss: [B] 拉格朗日损失
        """
        if cost_threshold is None:
            cost_threshold = self.cost_threshold

        # 约束违反：max(0, C - threshold)
        constraint_violation = F.relu(costs - cost_threshold)

        # 拉格朗日损失
        # 注意：我们最大化奖励，所以是 -R
        lagrangian_loss = -rewards + self.lambda_param * constraint_violation

        return lagrangian_loss

    def update_lambda(
        self,
        mean_cost: float,
        cost_threshold: float = None
    ) -> Dict[str, float]:
        """
        更新拉格朗日乘数λ

        更新规则：λ = max(0, λ + λ_lr * (mean_cost - threshold))

        Args:
            mean_cost: 平均成本
            cost_threshold: 成本阈值

        Returns:
            Dict: 更新信息
        """
        if cost_threshold is None:
            cost_threshold = self.cost_threshold

        # 计算梯度（手动实现）
        with torch.no_grad():
            # 计算约束违反
            constraint_violation = max(0, mean_cost - cost_threshold)

            # 更新λ
            old_lambda = self.lambda_param.item()
            new_lambda = max(0, old_lambda + self.lambda_lr * constraint_violation)
            self.lambda_param.data.fill_(new_lambda)

        return {
            'old_lambda': old_lambda,
            'new_lambda': new_lambda,
            'constraint_violation': constraint_violation
        }

    def get_lambda(self) -> float:
        """获取当前λ值"""
        return self.lambda_param.item()


class ConstrainedPPOOptimizer:
    """
    约束PPO优化器

    将CostCritic集成到PPO训练循环中
    """

    def __init__(
        self,
        cost_critic: CostCritic,
        lagrangian: LagrangianOptimizer,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        cost_loss_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        """
        Args:
            cost_critic: 成本评论家
            lagrangian: 拉格朗日优化器
            clip_param: PPO裁剪参数
            value_loss_coef: 价值损失系数
            cost_loss_coef: 成本损失系数
            entropy_coef: 熵正则化系数
        """
        self.cost_critic = cost_critic
        self.lagrangian = lagrangian
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.cost_loss_coef = cost_loss_coef
        self.entropy_coef = entropy_coef

    def compute_constrained_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        costs: torch.Tensor,
        policy_fn,
        clip_param: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算约束PPO损失

        Args:
            obs: [B, obs_dim] 观测
            actions: [B, action_dim] 动作
            old_log_probs: [B] 旧策略log概率
            advantages: [B] 优势函数
            returns: [B] 回报
            costs: [B] 成本
            policy_fn: 策略网络forward函数
            clip_param: 裁剪参数（可选）

        Returns:
            Dict: 各项损失
        """
        if clip_param is None:
            clip_param = self.clip_param

        # 重新计算当前策略
        outputs = policy_fn(obs)
        new_log_probs = outputs['log_prob']  # [B]
        entropy = outputs.get('entropy', torch.tensor(0.0))  # [B] or scalar

        # 计算ratio
        ratio = torch.exp(new_log_probs - old_log_probs)  # [B]

        # PPO clip损失
        surr1 = ratio * advantages  # [B]
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages  # [B]
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar

        # Value loss（使用CostCritic的value头）
        # 注意：这里需要传入global_state，暂时使用简单的mean pooling
        embeddings = outputs.get('embeddings', None)
        if embeddings is not None:
            global_state = embeddings.mean(dim=1) if embeddings.dim() == 3 else embeddings
            critic_outputs = self.cost_critic(embeddings, global_state)
            value_pred = critic_outputs['value']  # [B, 1]
            value_loss = F.mse_loss(value_pred.squeeze(1), returns)  # scalar
        else:
            value_loss = torch.tensor(0.0)

        # Cost loss
        if embeddings is not None:
            cost_pred = critic_outputs['cost']  # [B, 1]
            # 成本目标通常是0或1（是否违反约束）
            cost_targets = (costs > self.lagrangian.cost_threshold).float().unsqueeze(1)
            cost_loss = F.mse_loss(cost_pred, cost_targets)  # scalar
        else:
            cost_loss = torch.tensor(0.0)

        # 熵正则化
        if entropy.dim() > 0:
            entropy_loss = -entropy.mean()
        else:
            entropy_loss = -entropy

        # 总损失
        total_loss = (
            policy_loss +
            self.value_loss_coef * value_loss +
            self.cost_loss_coef * cost_loss +
            self.entropy_coef * entropy_loss
        )

        return {
            'total': total_loss,
            'policy': policy_loss,
            'value': value_loss,
            'cost': cost_loss,
            'entropy': entropy_loss
        }


def create_cost_critic(
    hidden_dim: int = 64,
    global_dim: int = 64,
    dropout: float = 0.1,
    device: str = 'cuda'
) -> CostCritic:
    """
    创建成本评论家的工厂函数

    Args:
        hidden_dim: 车辆嵌入维度
        global_dim: 全局特征维度
        dropout: Dropout率
        device: 设备

    Returns:
        cost_critic: CostCritic实例
    """
    model = CostCritic(
        hidden_dim=hidden_dim,
        global_dim=global_dim,
        dropout=dropout
    )

    return model.to(device)


def create_lagrangian_optimizer(
    initial_lambda: float = 0.1,
    lambda_lr: float = 0.01,
    cost_threshold: float = 0.5
) -> LagrangianOptimizer:
    """
    创建拉格朗日优化器的工厂函数

    Args:
        initial_lambda: 初始拉格朗日乘数
        lambda_lr: λ学习率
        cost_threshold: 成本阈值

    Returns:
        lagrangian: LagrangianOptimizer实例
    """
    return LagrangianOptimizer(
        initial_lambda=initial_lambda,
        lambda_lr=lambda_lr,
        cost_threshold=cost_threshold
    )
