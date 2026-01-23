"""
Simplified ICV Policy Network (OCR-MAX Architecture)

核心改进：
1. 移除SparseGate - 直接控制所有环境选择的ICV
2. 新增MultiViewGraphAttention - 3视角注意力机制
3. 保留RiskSensitiveGNN - 风险感知编码
4. 保留ValueHead - 状态价值估计
5. 保留SafetyShield - 轻量级安全保障（集成到推理模式）

相比JointICVPolicy的简化：
- ❌ 移除: ImportancePredictor, SparseGate, HierarchicalPooling
- ❌ 移除: WorldModel, CostCritic, DynamicWeightGate
- ✨ 新增: MultiViewGraphAttention
- ✅ 保留: RiskSensitiveGNN, ValueHead, SafetyShield
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional

# 从joint_icv_policy复用的组件
from src.models.joint_icv_policy import (
    RiskSensitiveGNN,
    RiskSensitiveGNNLayer,
    ValueHead
)

# 新增的MultiViewGraphAttention
from src.models.multi_view_attention import MultiViewGraphAttention


class SimplifiedPolicyHead(nn.Module):
    """
    简化的策略头（不使用selection_mask）

    输出：
    - 加速度：[-3, 2] m/s²
    - 换道概率：[0, 1]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # 移除+1，不再使用selection_mask
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 2)  # 2 actions
        )

        # 可学习的log_std（高斯策略）
        self.log_std = nn.Parameter(torch.zeros(2))

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for module in self.actor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            embeddings: [B, N, D] Multi-View Attention后的嵌入

        Returns:
            outputs: {
                'actions': [B, N, 2] (加速度, 换道概率),
                'action_mean': [B, N, 2],
                'log_std': [B, 2]
            }
        """
        # 策略输出（使用tanh确保输出在[-1, 1]范围）
        action_mean = torch.tanh(self.actor(embeddings))  # [B, N, 2]

        # 加速度：[-1, 1] → [-3, 2]
        accel = action_mean[:, :, 0:1]
        accel = (accel + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)
        accel = torch.clamp(accel, -3.0, 2.0)

        # 换道：[-1, 1] → [0, 1]
        lane_change = action_mean[:, :, 1:2]
        lane_change = (lane_change + 1.0) / 2.0
        lane_change = torch.clamp(lane_change, 0.0, 1.0)

        actions = torch.cat([accel, lane_change], dim=-1)  # [B, N, 2]

        # log_std广播到所有车辆
        B, N, _ = embeddings.shape
        log_std_broadcast = self.log_std.unsqueeze(0).unsqueeze(0).expand(B, N, 2)

        return {
            'actions': actions,
            'action_mean': action_mean,
            'log_std': log_std_broadcast
        }


class SimplifiedICVPolicy(nn.Module):
    """
    简化版ICV策略网络（OCR-MAX架构）

    核心改进：
    1. 直接控制所有ICV（不使用SparseGate筛选）
    2. Multi-View Graph Attention（3视角）
    3. 简化架构（移除不必要的组件）

    Args:
        obs_dim (int): 观测维度
        node_dim (int): 车辆状态维度
        hidden_dim (int): 隐藏层维度
        num_layers (int): GNN层数
        num_vehicles (int): 最大车辆数
        device (str): 设备
    """

    def __init__(
        self,
        obs_dim: int = 321,
        node_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_vehicles: int = 32,
        device: str = 'cuda',
        use_safety_shield: bool = True
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_vehicles = num_vehicles
        self.device = device

        # ========== 保留的核心组件 ==========

        # 1. GNN编码器（风险感知）
        self.gnn_encoder = RiskSensitiveGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
            interaction_radius=0.15
        )

        # 2. Multi-View Graph Attention（新增）
        self.multi_view_attn = MultiViewGraphAttention(
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )

        # 3. 简化的策略头（不使用selection_mask）
        self.policy_head = SimplifiedPolicyHead(hidden_dim=hidden_dim)

        # 4. 价值头
        self.value_head = ValueHead(hidden_dim=hidden_dim)

        # 5. SafetyShield（可选，推理时使用）
        self.use_safety_shield = use_safety_shield
        if use_safety_shield:
            from src.models.safety_shield import SafetyShield
            self.safety_shield = SafetyShield()

        self.training = True

    def _parse_observation(
        self,
        obs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        解析观测张量

        观测格式：[B, obs_dim]
        其中 obs_dim = max_vehicles * node_dim + 32 + 1
        - max_vehicles * node_dim: 车辆状态
        - + 32: global_stats
        - + 1: num_vehicles

        Args:
            obs: [B, obs_dim] 扁平化观测

        Returns:
            components: {
                'vehicle_states': [B, N, node_dim],
                'global_stats': [B, 32],
                'num_vehicles': [B, 1]
            }
        """
        B = obs.shape[0]

        # 提取车辆状态
        vehicle_features_size = self.num_vehicles * self.node_dim
        vehicle_states = obs[:, :vehicle_features_size]  # [B, N*node_dim]
        vehicle_states = vehicle_states.view(B, self.num_vehicles, self.node_dim)  # [B, N, node_dim]

        # 剩余部分
        remaining = obs[:, vehicle_features_size:]  # [B, 33]

        # 分离global_stats (32维) 和 num_vehicles (1维)
        global_stats = remaining[:, :32]  # [B, 32]
        num_vehicles = remaining[:, 32:33]  # [B, 1]

        return {
            'vehicle_states': vehicle_states,
            'global_stats': global_stats,
            'num_vehicles': num_vehicles  # [B, 1]
        }

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            obs: [B, obs_dim] 扁平化观测
            deterministic: 是否确定性输出（推理时）

        Returns:
            outputs: {
                'actions': [B, N*2] 展平的动作,
                'value': [B, 1] 状态价值,
                'log_prob': [B] 对数概率,
                'entropy': [B] 熵
            }
        """
        B = obs.shape[0]
        device = obs.device

        # 1. 解析观测
        components = self._parse_observation(obs)
        vehicle_states = components['vehicle_states']  # [B, N, 9]

        # 2. GNN编码
        gnn_out = self.gnn_encoder(vehicle_states)
        embeddings = gnn_out['embeddings']  # [B, N, D]

        # 3. Multi-View Graph Attention
        attended_embeddings = self.multi_view_attn(
            embeddings,
            vehicle_states
        )  # [B, N, D]

        # 4. 策略输出（直接输出所有车辆动作，不过滤）
        policy_outputs = self.policy_head(attended_embeddings)
        actions = policy_outputs['actions']  # [B, N, 2] 已变换到正确范围
        action_mean = policy_outputs['action_mean']  # [B, N, 2] 原始网络输出[-1,1]
        log_std = policy_outputs['log_std']  # [B, N, 2]

        # 5. 价值估计
        value = self.value_head(attended_embeddings)  # [B, 1]

        # 6. 计算log_prob和entropy（用于PPO训练）
        # 注意：log_prob和entropy基于原始action_mean（[-1,1]范围）计算
        if deterministic:
            # 确定性模式：直接使用变换后的actions
            actions_out = actions
            # 确定性模式：log_prob为0（熵为0）
            log_prob_per_dim = torch.zeros_like(action_mean)  # [B, N, 2]
        else:
            # 随机模式：对原始action_mean添加噪声，然后变换
            std = torch.exp(log_std.clamp(-5.0, 2.0))  # [B, N, 2]
            noise = torch.randn_like(action_mean)
            noisy_action_mean = action_mean + std * noise

            # 关键：clamp到[-1, 1]确保变换后不会超出范围
            noisy_action_mean = torch.clamp(noisy_action_mean, -1.0, 1.0)

            # 变换到正确范围
            # 加速度：[-1, 1] → [-3, 2]
            accel = noisy_action_mean[:, :, 0:1]
            accel = (accel + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)
            accel = torch.clamp(accel, -3.0, 2.0)

            # 换道：[-1, 1] → [0, 1]
            lane_change = noisy_action_mean[:, :, 1:2]
            lane_change = (lane_change + 1.0) / 2.0
            lane_change = torch.clamp(lane_change, 0.0, 1.0)

            actions_out = torch.cat([accel, lane_change], dim=-1)  # [B, N, 2]

            # 计算log_prob（高斯分布）- 使用原始[-1,1]空间
            # log_prob = -0.5 * (((sample - mean) / std)^2 + 2*log_std + log(2π))
            log_prob_per_dim = -0.5 * (
                ((noisy_action_mean - action_mean) / (std + 1e-6)) ** 2 +
                2 * log_std +
                np.log(2 * np.pi)
            )  # [B, N, 2]

        # 对所有车辆和动作维度求和
        log_prob = log_prob_per_dim.sum(dim=-1).sum(dim=-1)  # [B]

        # 计算熵
        # entropy = 0.5 * (1 + log(2π * std^2))
        entropy_per_dim = 0.5 * (
            1.0 +
            2 * log_std.clamp(-5.0, 2.0) +
            np.log(2 * np.pi)
        )  # [B, N, 2]

        # 对所有车辆和动作维度求平均
        entropy = entropy_per_dim.mean(dim=-1).mean(dim=-1)  # [B]

        # 7. SafetyShield过滤（推理时）
        actions_to_filter = actions_out  # [B, N, 2]
        if not self.training and self.use_safety_shield:
            # SafetyShield期望PyTorch张量输入
            safe_output = self.safety_shield.filter_actions(
                actions_to_filter,
                vehicle_states
            )
            actions_out = safe_output['safe_actions']

        # 8. 展平动作（格式：[accel_0, accel_1, ..., accel_N-1, lane_0, lane_1, ..., lane_N-1]）
        # 先transpose: [B, N, 2] -> [B, 2, N]
        actions_transposed = actions_out.transpose(1, 2)  # [B, 2, N]
        # 再flatten: [B, 2, N] -> [B, 2*N]
        actions_flat = actions_transposed.contiguous().view(B, -1)  # [B, 2*N]

        return {
            'actions': actions_flat,
            'value': value,
            'log_prob': log_prob,
            'entropy': entropy
        }

    def train_mode(self):
        """设置为训练模式"""
        self.training = True
        self.train()  # 使用train()而不是eval()

    def eval_mode(self):
        """设置为评估模式"""
        self.training = False
        self.eval()


# 测试代码
if __name__ == '__main__':
    # 创建策略网络
    policy = SimplifiedICVPolicy(
        obs_dim=321,
        node_dim=9,
        hidden_dim=128,
        num_layers=3,
        num_vehicles=32,
        device='cpu'
    )

    # 测试前向传播
    batch_size = 2
    obs = torch.randn(batch_size, 321)

    # 训练模式
    policy.train_mode()
    outputs_train = policy(obs, deterministic=False)

    print("=== 训练模式输出 ===")
    print(f"Actions shape: {outputs_train['actions'].shape}")
    print(f"Value shape: {outputs_train['value'].shape}")
    print(f"Log prob shape: {outputs_train['log_prob'].shape}")
    print(f"Entropy shape: {outputs_train['entropy'].shape}")
    print(f"Sample actions (first 5): {outputs_train['actions'][0, :5]}")
    print(f"Sample value: {outputs_train['value'][0, 0].item():.4f}")
    print(f"Sample entropy: {outputs_train['entropy'][0].item():.4f}")

    # 评估模式
    policy.eval_mode()
    outputs_eval = policy(obs, deterministic=True)

    print("\n=== 评估模式输出 ===")
    print(f"Actions shape: {outputs_eval['actions'].shape}")
    print(f"Value shape: {outputs_eval['value'].shape}")
    print(f"Sample actions (first 5): {outputs_eval['actions'][0, :5]}")

    # 参数量统计
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    print(f"\n=== 参数统计 ===")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
