"""
Influence-Driven Controller - 影响力驱动控制器
功能：选择Top-K关键车辆并生成控制动作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class InfluenceDrivenController(nn.Module):
    """
    影响力驱动控制器

    功能：
    1. 计算每辆ICV车辆的影响力得分
    2. 选择Top-K最具影响力的车辆
    3. 为选中车辆生成控制动作（加速度、换道）
    4. 估计价值和成本
    """

    def __init__(
        self,
        gnn_dim: int = 256,
        world_dim: int = 256,
        global_dim: int = 16,
        hidden_dim: int = 128,
        action_dim: int = 2,
        top_k: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()

        self.gnn_dim = gnn_dim
        self.world_dim = world_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.top_k = top_k

        # 全局上下文编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.LayerNorm(64)
        )

        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim + 64 + world_dim, 384),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(384),
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 影响力评分网络
        self.influence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 归一化到[0,1]
        )

        # 动作生成网络
        self.action_generator = nn.ModuleDict({
            'acceleration': nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # [-1,1] -> 映射到[-3,2] m/s²
            ),
            'lane_change': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # [0,1] 换道概率
            )
        })

        # 价值网络（状态价值）
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 成本价值网络（约束价值）
        self.cost_value_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # 优势网络（用于PPO）
        self.advantage_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )

    def forward(
        self,
        gnn_embedding: torch.Tensor,
        world_predictions: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            gnn_embedding: [N, gnn_dim] GNN嵌入
            world_predictions: [N, T, D] 世界模型预测
            global_metrics: [B, global_dim] 全局交通指标
            vehicle_ids: [N] 车辆ID列表
            is_icv: [N] 是否为智能网联车

        Returns:
            output: 包含选中车辆、动作、价值等
        """
        batch_size = gnn_embedding.size(0)

        # 1. 处理全局特征
        global_features = self.global_encoder(global_metrics)  # [B, 64]

        # 扩展全局特征到所有车辆
        if global_features.size(0) == 1:
            global_features_expanded = global_features.repeat(batch_size, 1)
        else:
            # 假设batch_size = B * N_per_batch
            global_features_expanded = global_features.repeat_interleave(
                batch_size // global_features.size(0), dim=0
            )

        # 2. 融合特征
        if world_predictions.dim() == 3:
            # [N, T, D] -> [N, D]
            avg_world_pred = world_predictions.mean(dim=1)
        else:
            avg_world_pred = world_predictions

        fused_input = torch.cat([
            gnn_embedding,
            global_features_expanded,
            avg_world_pred
        ], dim=1)  # [N, gnn_dim + 64 + world_dim]

        fused_features = self.fusion_layer(fused_input)  # [N, hidden_dim]

        # 3. 计算ICV车辆影响力
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]

        if len(icv_indices) == 0:
            return {
                'selected_vehicle_ids': [],
                'selected_indices': [],
                'raw_actions': torch.zeros(0, self.action_dim, device=gnn_embedding.device),
                'influence_scores': torch.zeros(0, device=gnn_embedding.device),
                'value_estimates': torch.zeros(0, device=gnn_embedding.device),
                'cost_estimates': torch.zeros(0, device=gnn_embedding.device),
                'advantage_estimates': torch.zeros(0, device=gnn_embedding.device),
                'action_probs': torch.zeros(0, self.action_dim, device=gnn_embedding.device)
            }

        # 提取ICV特征
        icv_features = fused_features[icv_mask]  # [N_icv, hidden_dim]

        # 计算影响力得分
        influence_scores = self.influence_scorer(icv_features).squeeze(-1)  # [N_icv]

        # 4. 选择Top-K
        k = min(self.top_k, len(icv_indices))
        top_k_scores, top_k_local_indices = torch.topk(
            influence_scores, k, largest=True, sorted=True
        )
        selected_indices = icv_indices[top_k_local_indices]  # 全局索引
        selected_vehicle_ids = [vehicle_ids[i] for i in selected_indices.cpu().numpy()]

        # 5. 生成动作
        selected_features = fused_features[selected_indices]  # [K, hidden_dim]

        accel_actions = self.action_generator['acceleration'](selected_features)  # [K, 1]
        lane_actions = self.action_generator['lane_change'](selected_features)  # [K, 1]

        raw_actions = torch.cat([accel_actions, lane_actions], dim=1)  # [K, 2]

        # 6. 价值估计
        value_estimates = self.value_network(fused_features).squeeze(-1)  # [N]
        cost_estimates = self.cost_value_network(fused_features).squeeze(-1)  # [N]
        advantage_estimates = self.advantage_network(fused_features).squeeze(-1)  # [N]

        # 7. 动作概率（用于PPO）
        action_probs = torch.zeros_like(raw_actions)  # [K, 2]
        action_probs[:, 0] = (accel_actions.squeeze(-1) + 1) / 2  # 映射到[0,1]
        action_probs[:, 1] = lane_actions.squeeze(-1)

        return {
            'selected_vehicle_ids': selected_vehicle_ids,
            'selected_indices': selected_indices.cpu().numpy().tolist(),
            'raw_actions': raw_actions,
            'influence_scores': influence_scores,
            'top_k_scores': top_k_scores,
            'value_estimates': value_estimates,
            'cost_estimates': cost_estimates,
            'advantage_estimates': advantage_estimates,
            'action_probs': action_probs,
            'fused_features': fused_features
        }

    def select_actions(
        self,
        gnn_embedding: torch.Tensor,
        world_predictions: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, any]:
        """
        选择动作（训练/推理）

        Args:
            deterministic: 是否使用确定性策略

        Returns:
            包含动作的字典
        """
        output = self.forward(
            gnn_embedding, world_predictions, global_metrics,
            vehicle_ids, is_icv
        )

        if deterministic:
            # 确定性动作：直接使用网络输出
            actions = output['raw_actions']
        else:
            # 随机动作：添加噪声（探索）
            actions = output['raw_actions'].clone()
            if actions.size(0) > 0:
                # 加速度添加高斯噪声
                accel_noise = torch.randn_like(actions[:, 0:1]) * 0.1
                actions[:, 0:1] = torch.clamp(actions[:, 0:1] + accel_noise, -1, 1)

                # 换道保持概率（不做随机）

        output['actions'] = actions
        return output

    def compute_action_cost(
        self,
        actions: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 5.0
    ) -> torch.Tensor:
        """
        计算动作成本

        Args:
            actions: [N, 2] (加速度, 换道概率)
            alpha: 加速度成本系数
            beta: 换道成本系数

        Returns:
            cost: [N] 总成本
        """
        accel_cost = alpha * torch.abs(actions[:, 0])  # 加速度成本

        # 换道成本（二值化）
        lane_change_binary = (actions[:, 1] > 0.5).float()
        lane_change_cost = beta * lane_change_binary

        total_cost = accel_cost + lane_change_cost

        return total_cost


class ValueNetwork(nn.Module):
    """
    独立的价值网络（用于critic）
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算状态价值

        Args:
            x: [N, input_dim] 状态特征

        Returns:
            value: [N, 1] 状态价值
        """
        return self.network(x)


class CostNetwork(nn.Module):
    """
    成本网络（预测约束违反）
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保非负
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测成本

        Args:
            x: [N, input_dim] 状态-动作对

        Returns:
            cost: [N, 1] 预测成本
        """
        return self.network(x)
