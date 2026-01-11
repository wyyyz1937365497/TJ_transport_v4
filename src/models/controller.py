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
        is_icv: torch.Tensor,
        traffic_predictions: Optional[Dict[str, any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（集成交通流预测）

        Args:
            gnn_embedding: [N, gnn_dim] GNN嵌入
            world_predictions: [N, T, D] 世界模型预测
            global_metrics: [B, global_dim] 全局交通指标
            vehicle_ids: [N] 车辆ID列表
            is_icv: [N] 是否为智能网联车
            traffic_predictions: 交通流预测结果（可选）
                - future_speeds: [B, N, F] 未来速度预测
                - congestion_prob: [B, N, 3] 拥堵概率
                - risk_vehicles: List[int] 高风险车辆索引
                - recommendations: Dict 控制建议

        Returns:
            output: 包含选中车辆、动作、价值等
        """
        batch_size = gnn_embedding.size(0)
        device = gnn_embedding.device

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

        # 3. 融合交通流预测信息
        if traffic_predictions is not None:
            fused_features = self._integrate_traffic_predictions(
                fused_features,
                traffic_predictions,
                vehicle_ids,
                batch_size,
                device
            )

        # 3. 计算ICV车辆影响力
        icv_mask = is_icv.bool()
        icv_indices = torch.where(icv_mask)[0]

        if len(icv_indices) == 0:
            return {
                'selected_vehicle_ids': [],
                'selected_indices': [],
                'raw_actions': torch.zeros(0, self.action_dim, device=device),
                'influence_scores': torch.zeros(0, device=device),
                'value_estimates': torch.zeros(0, device=device),
                'cost_estimates': torch.zeros(0, device=device),
                'advantage_estimates': torch.zeros(0, device=device),
                'action_probs': torch.zeros(0, self.action_dim, device=device),
                'prediction_info': {} if traffic_predictions is None else traffic_predictions
            }

        # 提取ICV特征
        icv_features = fused_features[icv_mask]  # [N_icv, hidden_dim]

        # 计算影响力得分
        influence_scores = self.influence_scorer(icv_features).squeeze(-1)  # [N_icv]

        # 4. 如果有预测信息，调整风险车辆的影响力权重
        if traffic_predictions is not None:
            influence_scores = self._adjust_influence_with_predictions(
                influence_scores,
                icv_indices,
                traffic_predictions,
                device
            )

        # 5. 选择Top-K
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
            'fused_features': fused_features,
            'prediction_info': {} if traffic_predictions is None else traffic_predictions
        }

    def select_actions(
        self,
        gnn_embedding: torch.Tensor,
        world_predictions: torch.Tensor,
        global_metrics: torch.Tensor,
        vehicle_ids: List[str],
        is_icv: torch.Tensor,
        deterministic: bool = False,
        traffic_predictions: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        选择动作（训练/推理）

        Args:
            deterministic: 是否使用确定性策略
            traffic_predictions: 交通流预测结果（可选）

        Returns:
            包含动作的字典
        """
        output = self.forward(
            gnn_embedding, world_predictions, global_metrics,
            vehicle_ids, is_icv, traffic_predictions
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

    def _integrate_traffic_predictions(
        self,
        fused_features: torch.Tensor,
        traffic_predictions: Dict[str, any],
        vehicle_ids: List[str],
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        融合交通流预测信息到特征中

        Args:
            fused_features: [N, hidden_dim] 融合特征
            traffic_predictions: 交通流预测结果
            vehicle_ids: 车辆ID列表
            batch_size: 批次大小
            device: 设备

        Returns:
            enhanced_features: [N, hidden_dim] 增强后的特征
        """
        # 1. 提取风险车辆索引
        risk_vehicles = traffic_predictions.get('risk_vehicles', [])

        if not risk_vehicles:
            return fused_features

        # 2. 为风险车辆创建增强向量
        risk_mask = torch.zeros(batch_size, device=device)
        for idx in risk_vehicles:
            if 0 <= idx < batch_size:
                risk_mask[idx] = 1.0

        # 3. 提取拥堵概率
        congestion_prob = traffic_predictions.get('congestion_prob', None)
        if congestion_prob is not None and congestion_prob.size(0) > 0:
            # congestion_prob: [B, N, 3] -> 取最大拥堵概率
            if congestion_prob.dim() == 3:
                max_congestion = congestion_prob[0, :, 2]  # 严重拥堵概率
            else:
                max_congestion = torch.zeros(batch_size, device=device)

            # 4. 融合风险信息到特征
            # 为风险车辆增加特征权重
            risk_enhancement = risk_mask.unsqueeze(1) * 0.2  # 增加20%权重
            congestion_enhancement = max_congestion.unsqueeze(1) * 0.1  # 拥堵增加10%权重

            # 应用增强
            enhanced_features = fused_features * (1.0 + risk_enhancement + congestion_enhancement)

            return enhanced_features

        return fused_features

    def _adjust_influence_with_predictions(
        self,
        influence_scores: torch.Tensor,
        icv_indices: torch.Tensor,
        traffic_predictions: Dict[str, any],
        device: torch.device
    ) -> torch.Tensor:
        """
        根据交通流预测调整影响力得分

        策略:
        1. 如果ICV车辆是风险车辆，增加其影响力权重
        2. 如果车辆接近瓶颈区域，适当增加权重
        3. 如果预测未来拥堵严重，增加预防性控制的权重

        Args:
            influence_scores: [N_icv] ICV影响力得分
            icv_indices: [N_icv] ICV全局索引
            traffic_predictions: 交通流预测结果
            device: 设备

        Returns:
            adjusted_scores: [N_icv] 调整后的影响力得分
        """
        adjusted_scores = influence_scores.clone()

        # 1. 提取风险车辆
        risk_vehicles = traffic_predictions.get('risk_vehicles', [])

        # 2. 获取推荐的控制车辆
        recommendations = traffic_predictions.get('recommendations', {})
        priority_vehicles = recommendations.get('priority_vehicles', [])

        # 3. 为风险车辆增加影响力权重
        if risk_vehicles:
            for local_idx, global_idx in enumerate(icv_indices):
                if global_idx in risk_vehicles:
                    # 风险车辆影响力增加30%
                    adjusted_scores[local_idx] *= 1.3

        # 4. 为优先车辆（预测器推荐的）增加额外权重
        if priority_vehicles:
            for local_idx, global_idx in enumerate(icv_indices):
                if global_idx in priority_vehicles:
                    # 推荐车辆影响力再增加20%
                    adjusted_scores[local_idx] *= 1.2

        # 5. 归一化到[0,1]
        if adjusted_scores.max() > 1.0:
            adjusted_scores = adjusted_scores / adjusted_scores.max()

        return adjusted_scores


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
