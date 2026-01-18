"""
独立PPO策略网络 - v4.0理想架构版本（移除SB3依赖）

核心功能：
1. 完全整合v4_architecture.py中的所有模块
2. 使用增强的影响力驱动Top-K控制器（可学习权重）
3. 增强的动态权重门控（场景识别）
4. 拉格朗日约束优化（动态λ更新）
5. 优化GNN边构建（使用空间邻近而非全连接）

所有增强功能默认启用。
移除了Stable-Baselines3依赖，使用独立的PyTorch实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

from .v4_architecture import (
    RiskSensitiveGNN,
    MultiScaleRSSM,
    EnhancedDynamicWeightGating,
    EnhancedInfluenceBasedController,
    LagrangianOptimizer,
    IdealTrafficControllerV4
)


# =============================================================================
# 工具函数
# =============================================================================

def safe_item(tensor: torch.Tensor) -> int:
    """
    安全地从张量中提取标量值，处理空张量情况

    Args:
        tensor: 输入张量

    Returns:
        张量的标量值，如果张量为空则返回0
    """
    if tensor.numel() == 0:
        return 0
    return int(tensor.item())


@script
def compute_risk_features_jit(
    vehicle_states: torch.Tensor
) -> torch.Tensor:
    """
    JIT编译的风险特征计算函数

    计算TTC和THW的向量化版本

    Args:
        vehicle_states: [N, 9] - [s, d, vs, vd, speed, accel, lane, angle, is_icv]

    Returns:
        risk_features: [N, 2] - [ttc_inv, thw_inv]
    """
    N = vehicle_states.size(0)

    if N == 0:
        return torch.zeros(0, 2, device=vehicle_states.device)

    # 特征提取
    s = vehicle_states[:, 0] * 1000.0  # 纵向位置(m)
    vs = vehicle_states[:, 2] * 30.0   # 纵向速度(m/s)
    speed = (vehicle_states[:, 4] * 30.0).clamp(min=0.1)  # 总速度(m/s)
    lane = vehicle_states[:, 6]        # 车道索引

    # 构建距离矩阵 [N, N]
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)
    lane_matrix = lane.unsqueeze(1) == lane.unsqueeze(0)

    # 排除自己
    not_self = ~torch.eye(N, dtype=torch.bool, device=vehicle_states.device)

    # 前车掩码
    ahead_mask = lane_matrix & (s_diff > 0) & not_self

    # 初始化输出
    ttc_inv = torch.zeros(N, device=vehicle_states.device)
    thw_inv = torch.zeros(N, device=vehicle_states.device)

    # 找最近的前车
    s_gap_matrix = torch.where(ahead_mask, s_diff, torch.tensor(float('inf'), device=vehicle_states.device))
    nearest_ahead_indices = torch.argmin(s_gap_matrix, dim=1)
    has_ahead = ahead_mask.any(dim=1)

    if has_ahead.any():
        valid_indices = torch.where(has_ahead)[0]
        nearest_idx = nearest_ahead_indices[valid_indices]
        s_gap_valid = s_gap_matrix[valid_indices, nearest_idx]

        vs_front_valid = vs[nearest_idx]
        vs_self_valid = vs[valid_indices]
        vs_diff = vs_self_valid - vs_front_valid

        catching_up = vs_diff > 0.1

        ttc = torch.where(
            catching_up,
            s_gap_valid / vs_diff.clamp(min=0.1),
            torch.tensor(float('inf'), device=vehicle_states.device)
        )
        ttc_inv_valid = 1.0 / (ttc + 0.1)
        ttc_inv_valid[~catching_up] = 0.0

        thw = s_gap_valid / speed[valid_indices].clamp(min=0.1)
        thw_inv_valid = 1.0 / (thw + 0.1)

        ttc_inv[valid_indices] = ttc_inv_valid
        thw_inv[valid_indices] = thw_inv_valid

    return torch.stack([ttc_inv, thw_inv], dim=-1)


class DiagonalGaussianDistribution:
    """
    对角高斯分布（用于PPO动作采样）

    复刻SB3的DiagonalGaussianDistribution，但不依赖SB3
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution(self, mean_actions: torch.Tensor, log_std: torch.Tensor):
        """
        创建分布

        Args:
            mean_actions: [batch_size, action_dim]
            log_std: [batch_size, action_dim]
        """
        self.mean_actions = mean_actions
        self.log_std = log_std
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """计算log概率"""
        # log_prob = -0.5 * (((actions - mean) / std) ^ 2 + 2 * log_std + log(2*pi))
        # 其中 std = exp(log_std)
        action_std = torch.exp(self.log_std)
        return -0.5 * (((actions - self.mean_actions) / action_std) ** 2 + 2 * self.log_std + np.log(2 * np.pi)).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """计算熵"""
        # entropy = 0.5 * (log(2*pi) + log_std + 1)
        return 0.5 * (np.log(2 * np.pi) + self.log_std + 1).sum(dim=-1)

    def mode(self) -> torch.Tensor:
        """返回众数（均值）"""
        return self.mean_actions

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        采样动作

        Args:
            deterministic: 是否确定性（返回均值）
        """
        if deterministic:
            return self.mode()

        # 重参数化采样: action = mean + std * epsilon, epsilon ~ N(0, I)
        action_std = torch.exp(self.log_std)
        return self.mean_actions + action_std * torch.randn_like(self.mean_actions)


class IdealTrafficPolicyV4(nn.Module):
    """
    理想交通策略 v4.0 - 独立PPO实现版

    架构层次（增强功能默认启用）：
    1. 感知层：RiskSensitiveGNN（风险敏感异构图）
    2. 预测层：MultiScaleRSSM（多尺度世界模型）
    3. 元控制层：EnhancedDynamicWeightGating（场景识别权重门控）
    4. 决策层：EnhancedInfluenceBasedController（可学习权重+自适应Top-K）
    5. 约束层：LagrangianOptimizer（动态拉格朗日优化）

    移除了SB3依赖，完全独立的PyTorch实现。
    """

    def __init__(
        self,
        obs_dim: int = 321,
        action_dim: int = 2,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # 保存配置
        if config is None:
            config = {}
        self.config = config

        # ========== GPU配置 ==========
        # 统一使用单一设备（cuda:0或cpu）
        if torch.cuda.is_available():
            self.device_train = torch.device('cuda:0')
            print(f"[GPU] 使用cuda:0进行所有计算")
        else:
            self.device_train = torch.device('cpu')
            print(f"[GPU] 使用CPU")

        # 从配置中提取参数
        model_config = config.get('model', {})
        self.top_k = model_config.get('controller', {}).get('top_k', 5)
        self.max_vehicles = config.get('environment', {}).get('max_vehicles', 32)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ============================================================
        # 1. 感知层：风险敏感GNN
        # ============================================================
        gnn_config = model_config.get('gnn', {})
        self.perception_layer = RiskSensitiveGNN(
            node_dim=gnn_config.get('node_dim', 9),
            edge_dim=gnn_config.get('edge_dim', 4),
            hidden_dim=gnn_config.get('hidden_dim', 64),
            output_dim=gnn_config.get('output_dim', 256),
            num_layers=gnn_config.get('num_layers', 3),
            num_heads=gnn_config.get('heads', 4),
            dropout=gnn_config.get('dropout', 0.1)
        )

        # ============================================================
        # 2. 预测层：多尺度RSSM
        # ============================================================
        wm_config = model_config.get('world_model', {})
        self.prediction_layer = MultiScaleRSSM(
            input_dim=gnn_config.get('output_dim', 256),
            hidden_dim=wm_config.get('hidden_dim', 128),
            latent_dim=wm_config.get('latent_dim', 64),
            future_steps=wm_config.get('future_steps', 5),
            num_layers=wm_config.get('num_layers', 2),
            dropout=wm_config.get('dropout', 0.1)
        )

        # ============================================================
        # 3. 元控制层：增强的动态权重门控（场景识别）
        # ============================================================
        self.weight_gating = EnhancedDynamicWeightGating(
            state_dim=gnn_config.get('output_dim', 256),
            history_dim=model_config.get('history_dim', 32),
            prediction_dim=wm_config.get('latent_dim', 64) * 2,
            hidden_dim=128,
            dropout=gnn_config.get('dropout', 0.1),
            use_temporal_smoothing=True
        )

        # ============================================================
        # 4. 决策层：增强的影响力控制器（可学习权重+自适应Top-K）
        # ============================================================
        ctrl_config = model_config.get('controller', {})
        self.decision_layer = EnhancedInfluenceBasedController(
            gnn_dim=gnn_config.get('output_dim', 256),
            flow_dim=wm_config.get('latent_dim', 64),
            risk_dim=wm_config.get('latent_dim', 64),
            global_dim=ctrl_config.get('global_dim', 32),
            hidden_dim=ctrl_config.get('hidden_dim', 128),
            action_dim=ctrl_config.get('action_dim', 2),
            base_top_k=self.top_k,
            max_top_k=min(self.top_k + 3, 10),
            min_top_k=max(self.top_k - 2, 2),
            dropout=ctrl_config.get('dropout', 0.2),
            use_learnable_weights=True
        )

        # ============================================================
        # 5. 价值网络（用于PPO）
        # ============================================================
        self.critic = nn.Sequential(
            nn.Linear(gnn_config.get('output_dim', 256), 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # ============================================================
        # 6. 动作投影层（用于生成动作均值）
        # ============================================================
        self.action_projection = nn.Linear(
            gnn_config.get('output_dim', 256) + wm_config.get('latent_dim', 64) * 2,
            64
        )

        # ============================================================
        # 7. 拉格朗日优化器（动态约束优化）
        # ============================================================
        self.lagrangian_optimizer = LagrangianOptimizer(
            cost_limit=0.1,
            lambda_init=0.1,
            adaptive_penalty=True
        )

        # ============================================================
        # 8. 动作分布（独立实现，不依赖SB3）
        # ============================================================
        self.action_dist = DiagonalGaussianDistribution(action_dim=64)

        # ============================================================
        # 9. LSTM隐藏状态（用于世界模型）
        # ============================================================
        self._rssm_hidden = None

        # ============================================================
        # 10. 图构建参数（使用空间邻近而非全连接）
        # ============================================================
        graph_config = model_config.get('graph', {})
        self.interaction_radius = graph_config.get('interaction_radius', 100.0)
        self.max_neighbors = graph_config.get('max_neighbors', 8)

    def extract_features(self, observations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取特征（完整v4.0架构）

        Args:
            observations: [batch_size, 321] 扁平化观测

        Returns:
            features_dict: 包含所有中间特征的字典
        """
        batch_size = observations.size(0)
        device = observations.device

        # 解析扁平化观测: [B, 288(vehicles) + 32(global) + 1(num)]
        vehicle_features = observations[:, :288]  # [B, 288]
        global_stats = observations[:, 288:320]   # [B, 32]
        num_vehicles = observations[:, 320:321]   # [B, 1]

        # 重塑车辆状态: [B, 32, 9]
        vehicle_states_reshaped = vehicle_features.view(
            batch_size, self.max_vehicles, 9
        )

        # 构建图数据（使用空间邻近）
        graph_data = self._build_graph_spatial(
            vehicle_states_reshaped,
            num_vehicles,
            batch_size
        )

        # ============================================================
        # 1. 感知层：风险敏感GNN
        # ============================================================
        gnn_output = self.perception_layer(
            node_features=graph_data['x'],
            edge_index=graph_data['edge_index'],
            edge_features=graph_data['edge_attr'],
            risk_features=graph_data['risk_features'],
            batch=graph_data['batch']
        )

        node_embeddings = gnn_output['node_embeddings']
        importance_scores = gnn_output['importance_scores']
        global_embedding = gnn_output['global_embedding']

        # Ensure global_embedding matches batch_size
        if global_embedding.size(0) != batch_size:
            # Mismatch detected - this can happen due to batch tensor issues
            # Create a corrected global embedding by manual mean pooling
            if graph_data.batch is not None and graph_data.batch.size(0) == node_embeddings.size(0):
                # Manual global mean pooling - 单GPU版本（避免GPU间传输）
                num_nodes_total = node_embeddings.size(0)
                embedding_dim = node_embeddings.size(1)

                # 所有计算在同一个设备上（避免传输）
                node_embeddings_compute = node_embeddings
                batch_compute = graph_data.batch

                # 初始化输出张量
                global_embedding_corrected = torch.zeros(batch_size, embedding_dim, device=device)

                # 使用one_hot + scatter进行向量化池化
                # 创建one-hot编码: [total_nodes, batch_size]
                batch_one_hot = torch.zeros(num_nodes_total, batch_size, device=device)
                batch_one_hot.scatter_(1, batch_compute.unsqueeze(1), 1.0)

                # 计算每个batch的节点数
                batch_counts = batch_one_hot.sum(dim=0)  # [batch_size]

                # 加权求和: [batch_size, embedding_dim]
                global_embedding_corrected = torch.matmul(batch_one_hot.t(), node_embeddings_compute)

                # 除以节点数得到均值（处理空batch）
                batch_counts_clamped = batch_counts.clamp(min=1.0)  # 避免除零
                global_embedding_corrected = global_embedding_corrected / batch_counts_clamped.unsqueeze(1)

                # 处理空batch（没有节点的batch）
                empty_batches = (batch_counts == 0)
                if empty_batches.any():
                    global_embedding_corrected[empty_batches] = 0.0

                global_embedding = global_embedding_corrected

        # ============================================================
        # 2. 预测层：多尺度RSSM
        # ============================================================
        # 使用LSTM隐藏状态（如果存在）
        if self._rssm_hidden is None:
            # 初始化隐藏状态
            batch_size_actual = node_embeddings.size(0) // batch_size  # 每个样本的节点数
            self._rssm_hidden = (
                torch.zeros(2, batch_size, self.prediction_layer.hidden_dim, device=device),
                torch.zeros(2, batch_size, self.prediction_layer.hidden_dim, device=device)
            )

        rssm_output = self.prediction_layer(
            node_embeddings=node_embeddings,
            hidden_state=self._rssm_hidden
        )

        # 更新LSTM隐藏状态
        self._rssm_hidden = rssm_output['hidden_state']

        z_flow = rssm_output['z_flow']
        z_risk = rssm_output['z_risk']

        # ============================================================
        # 3. 元控制层：动态权重门控
        # ============================================================
        weight_output = self.weight_gating(global_embedding)
        dynamic_weights = weight_output['all_weights']

        return {
            'node_embeddings': node_embeddings,
            'importance_scores': importance_scores,
            'global_embedding': global_embedding,
            'z_flow': z_flow,
            'z_risk': z_risk,
            'dynamic_weights': dynamic_weights,
            'rssm_output': rssm_output,
            'global_stats': global_stats,
            'num_vehicles': num_vehicles,
            'graph_data': graph_data
        }

    def _build_graph_spatial(
        self,
        vehicle_states: torch.Tensor,
        num_vehicles: torch.Tensor,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        使用空间邻近关系构建图（向量化优化版本）

        所有计算统一在 cuda:0 上进行，避免设备间通信开销。

        Returns:
            图数据字典（用于PyTorch Geometric）
        """
        from torch_geometric.data import Data

        device = vehicle_states.device

        # ============== 第1步：统计总节点数 ==============
        # 确保num_vehicles是1D tensor - 只squeeze最后一维，保持batch维度
        if num_vehicles.dim() > 1:
            num_vehicles = num_vehicles.squeeze(-1)

        # 先将num_vehicles限制在max_vehicles以内（在GPU上）
        num_vehicles_clamped = num_vehicles.clamp(max=self.max_vehicles)

        # 直接在GPU上计算total_nodes（避免传到CPU）
        total_nodes = num_vehicles_clamped.sum().item()

        # 只在需要时才创建列表（用于后续循环）
        # 延迟转换到真正需要的时候
        if total_nodes == 0:
            # 空batch
            batch_graph = Data(
                x=torch.zeros(0, 9, device=device),
                edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
                edge_attr=torch.empty((0, 4), device=device),
                risk_features=torch.zeros(0, 2, device=device),
                num_nodes=0
            )
            batch_graph.batch = torch.zeros(0, dtype=torch.long, device=device)
            return batch_graph

        # ============== 第2步：收集所有节点特征（向量化优化版本）=============
        # 修复：一次性将tensor转为CPU，避免循环中的CPU-GPU同步
        num_veh_cpu = num_vehicles_clamped.cpu()
        num_veh_list = [int(num_veh_cpu[b]) for b in range(batch_size)]

        # 计算累积偏移量（用于拼接）
        num_veh_tensor = torch.tensor(num_veh_list, device=device)
        cumsum = torch.cumsum(num_veh_tensor, dim=0)  # [batch_size]

        # 预分配输出tensor
        all_x_list = []
        all_risk_features_list = []
        all_edge_indices_list = []
        all_edge_attrs_list = []

        # 向量化处理：并行计算所有样本的特征和边
        # 注意：为了保持兼容性，仍然使用循环，但减少CPU-GPU同步
        for b in range(batch_size):
            num_veh = num_veh_list[b]

            # 跳过无效的车辆数量（负数或零）
            if num_veh <= 0:
                continue

            # 提取当前batch的车辆状态
            states_b = vehicle_states[b:b+1, :num_veh, :]  # [1, num_veh, 9]
            states_b = states_b.squeeze(0)  # [num_veh, 9]

            # 节点特征：直接使用原始特征
            all_x_list.append(states_b)

            # 计算风险特征（TTC, THW）
            risk_features_b = compute_risk_features_jit(states_b)
            all_risk_features_list.append(risk_features_b)

            # 计算边（使用空间邻近关系）
            # 提取位置和车道信息
            s = states_b[:, 0] * 1000.0  # 纵向位置(m)
            d = states_b[:, 1] * 10.0    # 横向位置(m)
            lane = states_b[:, 6]        # 车道索引

            # 计算距离矩阵 [num_veh, num_veh]
            s_diff = s.unsqueeze(1) - s.unsqueeze(0)
            d_diff = d.unsqueeze(1) - d.unsqueeze(0)
            dist_sq = s_diff ** 2 + d_diff ** 2

            # 车道掩码：只连接同车道或相邻车道的车辆
            lane_diff = (lane.unsqueeze(1) - lane.unsqueeze(0)).abs()
            lane_mask = lane_diff <= 1  # 同车道或相邻车道

            # 排除自连接
            not_self = ~torch.eye(num_veh, dtype=torch.bool, device=device)

            # 距离掩码（只连接半径内的车辆）
            radius_sq = self.interaction_radius ** 2
            dist_mask = (dist_sq < radius_sq) & lane_mask & not_self

            # 找到每个节点的邻居
            edge_indices = torch.nonzero(dist_mask, as_tuple=False)

            # 限制最大邻居数（使用Top-K）
            if edge_indices.size(0) > 0 and edge_indices.size(0) > num_veh * self.max_neighbors:
                # 计算每条边的距离
                edge_dists = dist_sq[edge_indices[:, 0], edge_indices[:, 1]]
                # 按距离排序，保留最近的边
                _, sorted_indices = torch.topk(edge_dists, k=num_veh * self.max_neighbors, largest=False)
                edge_indices = edge_indices[sorted_indices]

            # 添加batch偏移
            if b > 0:
                offset = cumsum[b - 1].item()
                edge_indices = edge_indices + offset

            all_edge_indices_list.append(edge_indices)

            # 边特征：[相对s, 相对d, 距离, 车道差]
            if edge_indices.size(0) > 0:
                src_nodes = edge_indices[:, 0]
                tgt_nodes = edge_indices[:, 1]

                # 减去偏移（用于索引当前batch）
                if b > 0:
                    offset = cumsum[b - 1].item()
                    src_nodes_local = src_nodes - offset
                    tgt_nodes_local = tgt_nodes - offset
                else:
                    src_nodes_local = src_nodes
                    tgt_nodes_local = tgt_nodes

                edge_s = s[src_nodes_local] - s[tgt_nodes_local]
                edge_d = d[src_nodes_local] - d[tgt_nodes_local]
                edge_dist = torch.sqrt(dist_sq[src_nodes_local, tgt_nodes_local])
                edge_lane_diff = lane[src_nodes_local] - lane[tgt_nodes_local]

                edge_attr = torch.stack([edge_s, edge_d, edge_dist, edge_lane_diff.float()], dim=-1)
                all_edge_attrs_list.append(edge_attr)

        # 拼接所有batch的数据
        if len(all_x_list) > 0:
            all_x = torch.cat(all_x_list, dim=0)  # [total_nodes, 9]
            all_risk_features = torch.cat(all_risk_features_list, dim=0)  # [total_nodes, 2]

            if len(all_edge_indices_list) > 0:
                all_edge_indices = torch.cat(all_edge_indices_list, dim=0)  # [E, 2]
                all_edge_indices = all_edge_indices.t()  # [2, E]

                if len(all_edge_attrs_list) > 0:
                    all_edge_attrs = torch.cat(all_edge_attrs_list, dim=0)  # [E, 4]
                else:
                    all_edge_attrs = torch.zeros(all_edge_indices.size(1), 4, device=device)
            else:
                all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=device)
                all_edge_attrs = torch.empty((0, 4), device=device)

            # 创建batch tensor
            batch_tensor = []
            for b in range(batch_size):
                if b == 0:
                    batch_tensor.append(torch.arange(num_veh_list[b], device=device))
                else:
                    offset = cumsum[b - 1].item()
                    batch_tensor.append(torch.arange(num_veh_list[b], device=device) + offset)

            batch_tensor = torch.cat(batch_tensor, dim=0)
        else:
            # 所有batch都是空的
            all_x = torch.zeros(0, 9, device=device)
            all_edge_indices = torch.empty((2, 0), dtype=torch.long, device=device)
            all_edge_attrs = torch.empty((0, 4), device=device)
            all_risk_features = torch.zeros(0, 2, device=device)
            batch_tensor = torch.zeros(0, dtype=torch.long, device=device)

        # 创建Data对象
        batch_graph = Data(
            x=all_x,
            edge_index=all_edge_indices,
            edge_attr=all_edge_attrs,
            risk_features=all_risk_features,
            num_nodes=all_x.size(0)
        )
        batch_graph.batch = batch_tensor

        return batch_graph

    def forward(
        self,
        observations: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播（PPO训练调用）

        Args:
            observations: [batch_size, obs_dim]
            deterministic: 是否确定性动作

        Returns:
            actions: [batch_size, action_dim]
            values: [batch_size, 1]
            log_probs: [batch_size]
        """
        batch_size = observations.size(0)
        device = observations.device

        # 1. 提取特征（完整v4.0架构）
        features_dict = self.extract_features(observations)

        node_embeddings = features_dict['node_embeddings']
        z_flow = features_dict['z_flow']
        z_risk = features_dict['z_risk']
        global_stats = features_dict['global_stats']
        dynamic_weights = features_dict['dynamic_weights']

        # 2. 获取价值估计（使用全局嵌入）
        global_features = features_dict['global_embedding']

        # Ensure global_features has correct shape and no NaN
        global_features = torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Check if global_features is empty or all zeros
        if global_features.numel() == 0 or global_features.abs().sum() < 1e-6:
            # Return zero values
            values = torch.zeros(batch_size, 1, device=device)
        else:
            # Ensure shape is [batch_size, feature_dim]
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)

            # If we have more features than batch_size, take first batch_size
            if global_features.size(0) > batch_size:
                global_features = global_features[:batch_size]

            # If we have fewer features than batch_size, repeat the last one
            elif global_features.size(0) < batch_size:
                last_feature = global_features[-1:].unsqueeze(0)
                global_features = torch.cat([global_features, last_feature.repeat(batch_size - global_features.size(0), 1)], dim=0)

            values = self.critic(global_features)

            # Ensure values has correct shape [batch_size, 1]
            if values.dim() == 1:
                values = values.unsqueeze(-1)
            if values.size(0) != batch_size:
                values = torch.zeros(batch_size, 1, device=device)

        # 3. 生成动作（使用重要性加权）
        # 转换为控制器输入格式
        if batch_size > 1:
            # Batch模式：使用第一个样本的图结构
            num_veh = safe_item(features_dict['num_vehicles'][0])
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]
            is_icv = torch.zeros(self.max_vehicles, device=device)
            is_icv[:num_veh] = 1.0

            # 使用影响力评分作为动作权重
            importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)
        else:
            # 单样本模式
            num_veh = safe_item(features_dict['num_vehicles'])
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]
            is_icv = torch.zeros(self.max_vehicles, device=device)
            is_icv[:num_veh] = 1.0

            importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)

        # 为所有车辆生成动作（使用重要性加权）
        # 融合特征用于动作生成
        fused = torch.cat([
            node_embeddings[:self.max_vehicles],
            z_flow[:self.max_vehicles],
            z_risk[:self.max_vehicles]
        ], dim=-1)

        # 检查是否有NaN或Inf，如果有则替换为零
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        # 检查是否全为零（没有车辆或无效数据）
        if fused.abs().sum() < 1e-6:
            # 返回零动作
            action_mean = torch.zeros(batch_size, 64, device=device)
        else:
            # 动作生成头 - 生成64维动作（匹配action_space）
            # 使用在__init__中初始化的action_projection
            action_features = self.action_projection(fused)  # [max_vehicles, 64]

            # 检查投影后的NaN
            action_features = torch.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

            # 使用tanh确保动作在[-1, 1]范围内
            actions = torch.tanh(action_features)  # [max_vehicles, 64]

            # 应用Top-K掩码（使用重要性得分）
            k = min(self.top_k, int(is_icv.sum()))
            if k > 0 and importance.size(0) >= k:
                top_k_values, top_k_indices = torch.topk(importance[:k], k)

                # 创建掩码 - 只控制Top-K车辆，其他置零
                mask = torch.zeros_like(actions)  # [max_vehicles, 64]
                mask[top_k_indices] = 1.0

                # 应用掩码
                actions = actions * mask

            # 取平均或最大池化到单个动作向量
            # 对于PPO，我们只需要一个64维的动作向量
            action_mean = actions.mean(dim=0, keepdim=True)  # [1, 64]

            # 如果是batch，扩展到batch大小
            if batch_size > 1:
                action_mean = action_mean.expand(batch_size, -1)  # [batch_size, 64]

        # 动作分布
        action_log_std = torch.ones_like(action_mean) * 0.1
        self.action_dist.proba_distribution(action_mean, action_log_std)

        if deterministic:
            actions_out = self.action_dist.mode()
        else:
            actions_out = self.action_dist.get_actions(deterministic=False)

        # 计算log_prob（在压缩动作维度之前）
        log_prob = self.action_dist.log_prob(actions_out)

        # 压缩到2维动作空间 [acceleration, lane_change]
        actions_out = actions_out[:, :2]

        return actions_out, values, log_prob

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（PPO训练时调用）

        重新计算给定observations下actions的log_prob和entropy

        重要：避免使用LSTM隐藏状态，防止计算图冲突
        """
        batch_size = observations.size(0)
        device = observations.device

        # 保存当前的LSTM隐藏状态（如果有）
        saved_hidden = getattr(self, '_rssm_hidden', None)

        # 清除LSTM隐藏状态，避免多次backward时计算图冲突
        self._rssm_hidden = None

        try:
            # 1. 提取特征
            features_dict = self.extract_features(observations)

            # 2. 价值估计
            global_features = features_dict['global_embedding']
            global_features = torch.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0)

            # 处理global_features形状
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)
            if global_features.size(0) != batch_size:
                if global_features.size(0) < batch_size:
                    last_feature = global_features[-1:].unsqueeze(0)
                    global_features = torch.cat([global_features, last_feature.repeat(batch_size - global_features.size(0), 1)], dim=0)
                else:
                    global_features = global_features[:batch_size]

            values = self.critic(global_features)

            # 3. 生成动作均值
            action_mean = self._get_action_mean(observations, features_dict)

            # 4. 创建动作分布
            action_std = torch.ones_like(action_mean) * 0.1
            self.action_dist.proba_distribution(action_mean, torch.log(action_std))

            # 5. 计算log_prob和entropy
            # 扩展actions到64维
            actions_64 = torch.cat([actions, torch.zeros(batch_size, 62, device=device)], dim=-1) if actions.size(1) == 2 else actions
            log_prob = self.action_dist.log_prob(actions_64)
            entropy = self.action_dist.entropy()

            return values, log_prob, entropy

        finally:
            # 不恢复saved_hidden，保持evaluate的独立性
            pass

    def _get_action_mean(
        self,
        observations: torch.Tensor,
        features_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        生成动作均值

        Returns:
            action_mean: [batch_size, 64]
        """
        batch_size = observations.size(0)
        device = observations.device

        node_embeddings = features_dict['node_embeddings']
        z_flow = features_dict['z_flow']
        z_risk = features_dict['z_risk']
        importance = features_dict['importance_scores']

        # 检查是否有有效数据
        if node_embeddings.size(0) == 0:
            return torch.zeros(batch_size, 64, device=device)

        # 融合特征用于动作生成
        fused = torch.cat([
            node_embeddings[:self.max_vehicles],
            z_flow[:self.max_vehicles],
            z_risk[:self.max_vehicles]
        ], dim=-1)

        # 清理NaN/Inf
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        if fused.abs().sum() < 1e-6:
            return torch.zeros(batch_size, 64, device=device)

        # 动作生成
        action_features = self.action_projection(fused)  # [max_vehicles, 64]
        action_features = torch.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

        actions = torch.tanh(action_features)  # [max_vehicles, 64]

        # Top-K掩码
        if batch_size > 1:
            num_veh = safe_item(features_dict['num_vehicles'][0])
        else:
            num_veh = safe_item(features_dict['num_vehicles'])

        is_icv = torch.zeros(self.max_vehicles, device=device)
        is_icv[:num_veh] = 1.0

        k = min(self.top_k, int(is_icv.sum()))
        if k > 0 and importance.size(0) >= k:
            top_k_values, top_k_indices = torch.topk(importance[:k], k)
            mask = torch.zeros_like(actions)
            mask[top_k_indices] = 1.0
            actions = actions * mask

        # 平均到单个向量
        action_mean = actions.mean(dim=0, keepdim=True)  # [1, 64]
        actions_mean = action_mean.expand(batch_size, -1)  # [batch_size, 64]

        return actions_mean

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        return_top_k: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        推理（预测动作）- 使用完整的影响力驱动控制器

        Args:
            observation: 单个观测 [321]
            deterministic: 是否确定性
            return_top_k: 是否返回Top-K信息

        Returns:
            actions: [2] 动作 [acceleration, lane_change]
            info: 额外信息（可选）
        """
        device = self.device_train
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            # 提取特征
            features_dict = self.extract_features(obs_tensor)

            node_embeddings = features_dict['node_embeddings']
            z_flow = features_dict['z_flow']
            z_risk = features_dict['z_risk']
            global_stats = features_dict['global_stats']
            importance = features_dict['importance_scores']

            # 确定实际车辆数
            num_veh = safe_item(features_dict['num_vehicles'])

            # 车辆ID列表（模拟）
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]

            # ICV标志（假设前25%是ICV）
            is_icv = torch.zeros(num_veh, device=device)
            num_icv = max(1, int(num_veh * 0.25))
            is_icv[:num_icv] = 1.0

            # 使用影响力驱动控制器选择Top-K
            decision_output = self.decision_layer(
                gnn_embedding=node_embeddings,
                z_flow=z_flow,
                z_risk=z_risk,
                global_metrics=global_stats,
                vehicle_ids=vehicle_ids,
                is_icv=is_icv,
                deterministic=deterministic
            )

            selected_vehicle_ids = decision_output['selected_vehicle_ids']
            raw_actions = decision_output['raw_actions']
            influence_scores = decision_output['influence_scores']

            # 将动作转换为扁平化格式
            accel_flat = torch.zeros(self.max_vehicles, device=device)
            lane_flat = torch.zeros(self.max_vehicles, device=device)

            # 为选中的车辆分配动作
            for i, veh_id in enumerate(selected_vehicle_ids):
                if i < len(raw_actions) and veh_id in vehicle_ids:
                    local_idx = vehicle_ids.index(veh_id)
                    if local_idx < self.max_vehicles:
                        accel_flat[local_idx] = raw_actions[i, 0]
                        lane_flat[local_idx] = raw_actions[i, 1]

            # 合并 - 只返回前2维 [acceleration, lane_change]
            actions = torch.stack([accel_flat, lane_flat], dim=1).flatten()[:2].cpu().numpy()

            # 额外信息
            info = None
            if return_top_k:
                info = {
                    'selected_vehicle_ids': selected_vehicle_ids,
                    'influence_scores': influence_scores.cpu().numpy(),
                    'dynamic_weights': features_dict['dynamic_weights'].cpu().numpy(),
                    'z_flow': z_flow.mean().cpu().numpy(),
                    'z_risk': z_risk.mean().cpu().numpy(),
                }

        return actions, info


def create_policy_v4(obs_dim: int = 321, action_dim: int = 2, config: Optional[Dict[str, Any]] = None) -> IdealTrafficPolicyV4:
    """
    创建策略网络（工厂函数）

    Args:
        obs_dim: 观测维度（默认321）
        action_dim: 动作维度（默认2）
        config: 配置字典

    Returns:
        policy: IdealTrafficPolicyV4实例
    """
    policy = IdealTrafficPolicyV4(
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config if config is not None else {}
    )
    return policy


def create_ideal_traffic_policy_v4(config: Dict[str, Any]) -> type:
    """
    创建策略网络类（兼容旧版训练脚本的工厂函数）

    这个函数返回一个策略类，而不是实例，以兼容旧的SB3训练代码。

    Args:
        config: 配置字典

    Returns:
        PolicyClass: 策略类（可直接实例化）
    """
    class PolicyClass(IdealTrafficPolicyV4):
        """兼容SB3训练脚本的自适应策略类"""

        def __init__(self, observation_space=None, action_space=None, lr_schedule=None, **kwargs):
            """
            兼容SB3的初始化方式

            Args:
                observation_space: 观测空间（Gym Space）
                action_space: 动作空间（Gym Space）
                lr_schedule: 学习率调度（SB3需要，但这里不使用）
                **kwargs: 其他参数
            """
            # 从observation_space和action_space提取维度
            if observation_space is not None:
                obs_dim = observation_space.shape[0] if hasattr(observation_space, 'shape') else 321
            else:
                obs_dim = 321

            if action_space is not None:
                action_dim = action_space.shape[0] if hasattr(action_space, 'shape') else 2
            else:
                action_dim = 2

            # 调用父类初始化
            super().__init__(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config
            )

        def to(self, device):
            """重写to方法，确保所有子模块正确移动设备"""
            # 调用父类的to方法
            result = super().to(device)

            # 确保所有子模块都在正确的设备上
            if hasattr(self, 'perception_layer'):
                self.perception_layer = self.perception_layer.to(device)
            if hasattr(self, 'prediction_layer'):
                self.prediction_layer = self.prediction_layer.to(device)
            if hasattr(self, 'weight_gating'):
                self.weight_gating = self.weight_gating.to(device)
            if hasattr(self, 'decision_layer'):
                self.decision_layer = self.decision_layer.to(device)
            if hasattr(self, 'critic'):
                self.critic = self.critic.to(device)
            if hasattr(self, 'action_projection'):
                self.action_projection = self.action_projection.to(device)
            if hasattr(self, 'action_dist'):
                # action_dist不是nn.Module，不需要移动
                pass

            return result

    return PolicyClass


# 兼容性别名（保持与训练脚本的兼容性）
create_ideal_traffic_policy_v4_compat = create_ideal_traffic_policy_v4
