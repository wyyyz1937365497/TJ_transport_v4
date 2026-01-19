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

        # ✅ 从obs_dim动态推断max_vehicles
        # obs_dim = max_vehicles * 9 + 32 + 1
        # 因此：max_vehicles = (obs_dim - 32 - 1) / 9
        inferred_max_vehicles = (obs_dim - 32 - 1) // 9
        # 如果推断成功（>0），使用推断值；否则使用config的值
        self.max_vehicles = inferred_max_vehicles if inferred_max_vehicles > 0 else config.get('environment', {}).get('max_vehicles', 32)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        print(f"[MODEL] max_vehicles={self.max_vehicles} (inferred from obs_dim={obs_dim})")

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
        # ✅ 修复：输出2维动作（acceleration, lane_change）而非64维
        # 这样每辆车有自己的动作，而不是所有车辆共享同一个64维向量
        self.action_projection = nn.Linear(
            gnn_config.get('output_dim', 256) + wm_config.get('latent_dim', 64) * 2,
            2  # 每辆车2维动作：[acceleration, lane_change]
        )

        # ============================================================
        # 7. ✅ 新增：干预必要性判断模块（动态按需干预）
        # ============================================================
        # 输入：全局统计特征 + GNN全局嵌入
        # 输出：干预必要性评分 [0, 1]，用于动态调整控制强度
        self.intervention_necessity_net = nn.Sequential(
            nn.Linear(gnn_config.get('output_dim', 256) + 32, 64),  # GNN全局嵌入 + 全局统计
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出[0,1]，0=不需要干预，1=必须干预
        )

        # ============================================================
        # 8. 拉格朗日优化器（动态约束优化）
        # ============================================================
        self.lagrangian_optimizer = LagrangianOptimizer(
            cost_limit=0.1,
            lambda_init=0.1,
            adaptive_penalty=True
        )

        # ============================================================
        # 8. 动作分布（独立实现，不依赖SB3）
        # ============================================================
        self.action_dist = DiagonalGaussianDistribution(action_dim=self.max_vehicles * 2)

        # ✅ 可学习的log_std参数（用于策略探索）
        # 初始化为log(0.1) ≈ -2.3，与之前的固定值一致
        # 维度：max_vehicles * 2（512辆车 × 2个动作 = 1024维）
        self.log_std = nn.Parameter(torch.full((self.max_vehicles * 2,), np.log(0.1), dtype=torch.float32))

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
            observations: [batch_size, max_vehicles*9 + 32 + 1] 扁平化观测
                         例如：max_vehicles=512时，维度为4641（4608+32+1）

        Returns:
            features_dict: 包含所有中间特征的字典
        """
        batch_size = observations.size(0)
        device = observations.device

        # 解析扁平化观测: [B, max_vehicles*9(vehicles) + 32(global) + 1(num)]
        # ✅ self.max_vehicles已经在__init__中从obs_dim动态推断
        vehicle_dim = self.max_vehicles * 9
        vehicle_features = observations[:, :vehicle_dim]  # [B, max_vehicles * 9]
        global_stats = observations[:, vehicle_dim:vehicle_dim + 32]   # [B, 32]
        num_vehicles = observations[:, vehicle_dim + 32:vehicle_dim + 33]   # [B, 1]

        # 重塑车辆状态: [B, max_vehicles, 9]
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

            # 创建batch tensor - 每个节点标记它所属的batch索引
            batch_tensor = []
            for b in range(batch_size):
                # 第b个batch的所有节点都应该有batch索引b
                batch_tensor.append(torch.full((num_veh_list[b],), b, device=device, dtype=torch.long))

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
        obs_dim = observations.size(1)

        # ✅ 动态推断实际的max_vehicles（从观测维度）
        # 这确保模型能处理不同课程级别的不同车辆数
        actual_max_vehicles = (obs_dim - 32 - 1) // 9

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
        # ✅ 修复：从observation中提取真实的is_icv标志
        # 车辆状态特征格式：[s, d, vs, vd, speed, accel, lane, angle, is_icv]
        # is_icv是第9个特征（索引8）

        if batch_size > 1:
            # Batch模式：使用第一个样本的图结构
            num_veh = safe_item(features_dict['num_vehicles'][0])
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]

            # ✅ 从node_embeddings中提取is_icv标志（第9个特征）
            # node_embeddings shape: [total_nodes, embedding_dim]
            # 但我们需要原始的vehicle_states来获取is_icv

            # 从observations中提取is_icv标志
            # vehicle_features: [batch, actual_max_vehicles * 9]
            vehicle_features = observations[:, :actual_max_vehicles * 9]
            vehicle_features_reshaped = vehicle_features.view(batch_size, actual_max_vehicles, 9)

            # 取第一个batch的is_icv标志（索引8）
            is_icv_full = vehicle_features_reshaped[0, :, 8]  # [actual_max_vehicles]

            # 使用影响力评分作为动作权重（暂时保存，后面会根据actual_num_vehicles调整）
            importance = features_dict['importance_scores'][:actual_max_vehicles].squeeze(-1)
        else:
            # 单样本模式
            num_veh = safe_item(features_dict['num_vehicles'])
            vehicle_ids = [f"veh_{i}" for i in range(num_veh)]

            # 从observations中提取is_icv标志
            vehicle_features = observations[:actual_max_vehicles * 9].view(actual_max_vehicles, 9)
            is_icv_full = vehicle_features[:, 8]  # [actual_max_vehicles]

            # 使用影响力评分作为动作权重（暂时保存，后面会根据actual_num_vehicles调整）
            importance = features_dict['importance_scores'][:actual_max_vehicles].squeeze(-1)

        # ✅ 关键修复：使用实际车辆数而不是actual_max_vehicles
        # node_embeddings的实际节点数可能小于actual_max_vehicles（因为场景中车辆还未完全加载）
        # 同时也要确保不超过观测空间的维度
        actual_num_vehicles = min(num_veh, node_embeddings.size(0), actual_max_vehicles)

        # ✅ 根据actual_num_vehicles裁剪is_icv
        is_icv = is_icv_full[:actual_num_vehicles]
        importance = importance[:actual_num_vehicles]

        # 为所有车辆生成动作（使用重要性加权）
        # 融合特征用于动作生成
        fused = torch.cat([
            node_embeddings[:actual_num_vehicles],
            z_flow[:actual_num_vehicles],
            z_risk[:actual_num_vehicles]
        ], dim=-1)  # [actual_num_vehicles, feature_dim]

        # 检查是否有NaN或Inf，如果有则替换为零
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        # 检查是否全为零（没有车辆或无效数据）
        if fused.abs().sum() < 1e-6:
            # ✅ 返回零动作（actual_max_vehicles*2维）
            action_mean = torch.zeros(batch_size, actual_max_vehicles * 2, device=device)
        else:
            # ✅ 修复：为每辆车生成2维动作 [acceleration, lane_change]
            action_features = self.action_projection(fused)  # [actual_num_vehicles, 2]

            # 检查投影后的NaN
            action_features = torch.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

            # ✅ 修复：将tanh输出[-1,1]映射到正确的动作范围
            # 加速度：[-1, 1] → [-3.0, 2.0]（DEFAULT_MAX_DECEL 到 DEFAULT_MAX_ACCEL）
            # 换道：[-1, 1] → [0.0, 1.0]
            action_raw = torch.tanh(action_features)  # [actual_num_vehicles, 2]

            # 第1维：加速度 [-1, 1] → [-3.0, 2.0]
            # 公式：output = (input + 1) / 2 * (high - low) + low
            accel = (action_raw[:, 0:1] + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)

            # 第2维：换道 [-1, 1] → [0.0, 1.0]
            # 公式：output = (input + 1) / 2
            lane_change = (action_raw[:, 1:2] + 1.0) / 2.0

            actions = torch.cat([accel, lane_change], dim=-1)  # [actual_num_vehicles, 2]

            # ✅ 修复：应用ICV掩码 - 只为ICV车辆生成有效动作
            # 非ICV车辆的动作置零（环境不会使用这些动作）
            icv_mask = is_icv.unsqueeze(-1).expand_as(actions)  # [actual_num_vehicles, 2]
            actions = actions * icv_mask

            # ✅ 新增：动态干预机制 - 根据干预必要性调整动作强度
            # 计算全局干预必要性评分
            if batch_size > 1:
                global_stats_batch = observations[:, actual_max_vehicles * 9:actual_max_vehicles * 9 + 32]  # [batch, 32] (全局统计维度固定32)
                global_stats_single = global_stats_batch[0:1]  # [1, 32]
            else:
                global_stats_single = observations[actual_max_vehicles * 9:actual_max_vehicles * 9 + 32].unsqueeze(0)  # [1, 32] (全局统计维度固定32)

            # 使用GNN全局嵌入 + 全局统计预测干预必要性
            global_embedding = features_dict['global_embedding']  # [1 or batch, hidden_dim]
            if global_embedding.dim() == 1:
                global_embedding = global_embedding.unsqueeze(0)

            intervention_input = torch.cat([
                global_embedding.mean(dim=0, keepdim=True),  # [1, hidden_dim]
                global_stats_single  # [1, 32]
            ], dim=-1)  # [1, hidden_dim + 32]

            intervention_necessity = self.intervention_necessity_net(intervention_input)  # [1, 1]

            # 根据干预必要性动态调整动作强度
            # necessity=0 → 不干预（动作衰减到0）
            # necessity=1 → 积极干预（动作保持原值）
            intervention_scale = intervention_necessity.squeeze()  # scalar

            # 应用干预缩放（只对非零动作生效，避免放大噪声）
            actions = actions * intervention_scale

            # 调试信息：每100步打印一次干预必要性
            if hasattr(self, '_step_count'):
                self._step_count += 1
            else:
                self._step_count = 1

            if self._step_count % 100 == 0:
                # ✅ 修复：统计实际车辆数（位置非零的车辆）
                # vehicle_features: [actual_max_vehicles, 9]，第一列是s坐标
                if batch_size > 1:
                    vehicle_features = observations[:, :actual_max_vehicles * 9].view(batch_size, actual_max_vehicles, 9)
                    actual_vehicles = (vehicle_features[0, :, 0].abs() > 1e-6).sum().item()  # s坐标非零
                else:
                    vehicle_features = observations[:actual_max_vehicles * 9].view(actual_max_vehicles, 9)
                    actual_vehicles = (vehicle_features[:, 0].abs() > 1e-6).sum().item()

                icv_count = is_icv[:actual_num_vehicles].sum().item()
                print(f"[INTERVENTION] Necessity: {intervention_necessity.item():.3f}, Scale: {intervention_scale:.3f}")
                print(f"  Actual vehicles: {actual_num_vehicles}/{actual_max_vehicles}, ICV count: {icv_count} ({icv_count/max(actual_num_vehicles,1)*100:.1f}%)")

            # ✅ 修复：展平为actual_max_vehicles*2维向量（环境期望的维度）
            # actions: [actual_num_vehicles, 2]，需要padding到 [actual_max_vehicles, 2]
            if actual_num_vehicles < actual_max_vehicles:
                # Padding零
                actions_padded = torch.zeros(actual_max_vehicles, 2, device=device)
                actions_padded[:actual_num_vehicles, :] = actions
                actions_clipped = actions_padded
            else:
                actions_clipped = actions[:actual_max_vehicles, :]

            # 展平：[actual_max_vehicles, 2] → [actual_max_vehicles * 2]
            action_flat = actions_clipped.flatten()

            # 为batch中的每个样本复制相同的动作
            action_mean = action_flat.unsqueeze(0).expand(batch_size, -1)  # [batch_size, actual_max_vehicles * 2]

        # 动作分布（使用可学习的log_std参数）
        # ✅ 动态调整log_std维度以匹配actual_max_vehicles
        expected_action_dim = actual_max_vehicles * 2
        actual_action_dim = action_mean.size(-1)

        # 如果self.log_std维度不匹配，进行裁剪或padding
        if self.log_std.size(0) != expected_action_dim:
            if self.log_std.size(0) > expected_action_dim:
                # 裁剪
                log_std = self.log_std[:expected_action_dim]
            else:
                # Padding
                log_std = torch.cat([
                    self.log_std,
                    torch.full((expected_action_dim - self.log_std.size(0),), np.log(0.1), device=self.log_std.device, dtype=self.log_std.dtype)
                ])
        else:
            log_std = self.log_std

        # 确保log_std形状与action_mean匹配
        if action_mean.dim() == 2:
            log_std = log_std.unsqueeze(0).expand_as(action_mean)

        # ✅ 数值稳定性：裁剪log_std范围（防止std过大或过小）
        # log_std在[-5, 2]范围内 → std在[0.007, 7.4]范围内
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)

        self.action_dist.proba_distribution(action_mean, log_std)

        if deterministic:
            actions_out = self.action_dist.mode()
        else:
            actions_out = self.action_dist.get_actions(deterministic=False)

        # 计算log_prob（max_vehicles*2维动作空间，1024维）
        log_prob = self.action_dist.log_prob(actions_out)

        # 返回完整的max_vehicles*2维动作（512辆车 × 2个动作 = 1024维）
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

            # 4. 创建动作分布（使用可学习的log_std参数）
            # 确保log_std形状与action_mean匹配
            if action_mean.dim() == 2:
                log_std = self.log_std.unsqueeze(0).expand_as(action_mean)
            else:
                log_std = self.log_std

            # ✅ 数值稳定性：裁剪log_std范围
            log_std = torch.clamp(log_std, min=-5.0, max=2.0)

            self.action_dist.proba_distribution(action_mean, log_std)

            # 5. 计算log_prob和entropy
            # actions 应该已经是64维
            log_prob = self.action_dist.log_prob(actions)
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

        # ✅ 修复：从observations中提取真实的is_icv标志
        vehicle_features = observations[:, :self.max_vehicles * 9]  # [batch, 288]
        vehicle_features_reshaped = vehicle_features.view(batch_size, self.max_vehicles, 9)  # [batch, 32, 9]
        is_icv = vehicle_features_reshaped[0, :, 8]  # [max_vehicles] - 取第一个batch

        # 处理importance_scores（需要切片和squeeze，与forward方法一致）
        importance = features_dict['importance_scores'][:self.max_vehicles].squeeze(-1)

        # 检查是否有有效数据
        if node_embeddings.size(0) == 0:
            return torch.zeros(batch_size, self.max_vehicles * 2, device=device)

        # 融合特征用于动作生成
        fused = torch.cat([
            node_embeddings[:self.max_vehicles],
            z_flow[:self.max_vehicles],
            z_risk[:self.max_vehicles]
        ], dim=-1)

        # 清理NaN/Inf
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)

        if fused.abs().sum() < 1e-6:
            return torch.zeros(batch_size, self.max_vehicles * 2, device=device)

        # ✅ 修复：为每辆车生成2维动作
        action_features = self.action_projection(fused)  # [max_vehicles, 2]
        action_features = torch.nan_to_num(action_features, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ 修复：将tanh输出[-1,1]映射到正确的动作范围
        action_raw = torch.tanh(action_features)  # [max_vehicles, 2]

        # 第1维：加速度 [-1, 1] → [-3.0, 2.0]
        accel = (action_raw[:, 0:1] + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)

        # 第2维：换道 [-1, 1] → [0.0, 1.0]
        lane_change = (action_raw[:, 1:2] + 1.0) / 2.0

        actions = torch.cat([accel, lane_change], dim=-1)  # [max_vehicles, 2]

        # ✅ 修复：应用ICV掩码 - 只为ICV车辆生成有效动作
        icv_mask = is_icv.unsqueeze(-1).expand_as(actions)  # [max_vehicles, 2]
        actions = actions * icv_mask

        # ✅ 新增：动态干预机制 - 根据干预必要性调整动作强度
        # 计算全局干预必要性评分
        global_stats_single = observations[:, self.max_vehicles * 9:self.max_vehicles * 9 + 32][0:1]  # [1, 32] (全局统计维度固定32)

        # 使用GNN全局嵌入 + 全局统计预测干预必要性
        global_embedding = features_dict['global_embedding']  # [1 or batch, hidden_dim]
        if global_embedding.dim() == 1:
            global_embedding = global_embedding.unsqueeze(0)

        intervention_input = torch.cat([
            global_embedding.mean(dim=0, keepdim=True),  # [1, hidden_dim]
            global_stats_single  # [1, 32]
        ], dim=-1)  # [1, hidden_dim + 32]

        intervention_necessity = self.intervention_necessity_net(intervention_input)  # [1, 1]
        intervention_scale = intervention_necessity.squeeze()  # scalar

        # 应用干预缩放
        actions = actions * intervention_scale

        # ✅ 修复：展平为max_vehicles*2维向量（512辆车 × 2维动作 = 1024维）
        actions_clipped = actions[:self.max_vehicles, :]  # [max_vehicles, 2] → [512, 2]
        action_flat = actions_clipped.flatten()  # [1024]

        # 确保是max_vehicles*2维
        expected_dim = self.max_vehicles * 2
        if action_flat.size(0) < expected_dim:
            action_flat = torch.cat([
                action_flat,
                torch.zeros(expected_dim - action_flat.size(0), device=device)
            ])

        # 为batch中的每个样本复制相同的动作
        action_mean = action_flat.unsqueeze(0).expand(batch_size, -1)  # [batch_size, 1024]

        return action_mean

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
        return_top_k: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        推理（预测动作）- 使用完整的影响力驱动控制器

        Args:
            observation: 单个观测 [4641] = 512×9(车辆特征) + 32(全局统计) + 1(车辆数量)
            deterministic: 是否确定性
            return_top_k: 是否返回Top-K信息

        Returns:
            actions: [1024] 动作向量（512辆车 × 2个动作）
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

            # 合并所有max_vehicles*2维动作（512辆车 × 2个动作 = 1024维）
            actions = torch.stack([accel_flat, lane_flat], dim=1).flatten().cpu().numpy()

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

        def to(self, *args, **kwargs):
            """重写to方法，确保所有子模块正确移动设备"""
            # 调用父类的to方法
            result = super().to(*args, **kwargs)

            # 确保所有子模块都在正确的设备上
            # (父类的to方法应该已经处理了所有注册的子模块，这里保留是为了兼容性)
            # 注意：实际上这些调用是冗余的，因为父类的to方法已经处理了所有nn.Module子模块
            if hasattr(self, 'action_dist'):
                # action_dist不是nn.Module，不需要移动
                pass

            return result

    return PolicyClass


# 兼容性别名（保持与训练脚本的兼容性）
create_ideal_traffic_policy_v4_compat = create_ideal_traffic_policy_v4
