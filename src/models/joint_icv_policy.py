"""
联合ICV评分与策略生成网络（v5.0）

核心创新：
1. 端到端可学习的车辆重要性评分
2. 可微分的稀疏门控机制（Gumbel-Softmax Top-K）
3. 联合PPO训练（importance + action）
4. 动态K值选择
5. 风险感知的GNN编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class RiskSensitiveGNNLayer(nn.Module):
    """
    风险感知的GNN层

    核心创新：
    - 边特征融入TTC（碰撞时间）倒数
    - 注意力偏向高风险交互
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # 风险感知的注意力权重
        self.risk_bias_weight = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, risk_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] 节点特征
            adj: [B, N, N] 邻接矩阵
            risk_features: [B, N, N] 风险特征（TTC倒数）

        Returns:
            out: [B, N, D] 更新后的节点特征
        """
        # 风险感知加权
        # TTC越小（风险越大），权重越高
        risk_weight = torch.sigmoid(self.risk_bias_weight * risk_features)  # [B, N, N]

        # 将风险权重应用到邻接矩阵
        weighted_adj = adj * risk_weight  # [B, N, N]

        # 消息传递：聚合邻居特征（使用风险加权邻接矩阵）
        messages = torch.bmm(weighted_adj, x)  # [B, N, D]

        # 线性变换
        out = self.linear(messages)

        # 归一化和激活
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class RiskSensitiveGNN(nn.Module):
    """
    风险敏感的图神经网络编码器

    输入：车辆状态 [B, N, 9]
    输出：节点嵌入 [B, N, D]
    """

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        interaction_radius: float = 0.15
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.interaction_radius = interaction_radius

        # 输入嵌入层
        self.input_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN层
        self.gnn_layers = nn.ModuleList([
            RiskSensitiveGNNLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

    def _build_adjacency_and_risk(
        self,
        vehicle_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建邻接矩阵和风险特征矩阵

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            adj: [B, N, N] 邻接矩阵
            risk_features: [B, N, N] 风险特征（TTC倒数）
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取位置和速度信息
        s = vehicle_states[:, :, 0]  # [B, N] 纵向位置
        d = vehicle_states[:, :, 1]  # [B, N] 横向位置
        lanes = vehicle_states[:, :, 6].long()  # [B, N] 车道索引
        vs = vehicle_states[:, :, 2]  # [B, N] 纵向速度

        # 计算距离矩阵
        s_diff = s.unsqueeze(2) - s.unsqueeze(1)  # [B, N, N]
        d_diff = d.unsqueeze(2) - d.unsqueeze(1)  # [B, N, N]
        lane_diff = (lanes.unsqueeze(2) - lanes.unsqueeze(1)).abs()  # [B, N, N]

        dist_sq = s_diff ** 2 + d_diff ** 2  # [B, N, N]
        dist = torch.sqrt(dist_sq + 1e-6)  # [B, N, N]

        # 计算相对速度（用于TTC）
        vs_diff = vs.unsqueeze(2) - vs.unsqueeze(1)  # [B, N, N]

        # ========== 邻接矩阵 ==========
        # 车道掩码：只连接同车道或相邻车道
        lane_mask = (lane_diff <= 1).float()  # [B, N, N]

        # 距离掩码：只连接半径内的车辆
        dist_mask = (dist < self.interaction_radius).float()  # [B, N, N]

        # 排除自连接
        not_self = 1.0 - torch.eye(N, device=device).unsqueeze(0)  # [B, N, N]

        # 有效车辆掩码
        valid_mask = (s > 0).float().unsqueeze(2)  # [B, N, 1]

        # 组合掩码
        mask = lane_mask * dist_mask * not_self  # [B, N, N]
        mask = mask * valid_mask * valid_mask.transpose(1, 2)  # [B, N, N]

        # 行归一化
        row_sum = mask.sum(dim=2, keepdim=True)  # [B, N, 1]
        adj = mask / (row_sum + 1e-6)  # [B, N, N]

        # ========== 风险特征（TTC倒数）==========
        # TTC = distance / relative_speed
        # 只有当后车（s_i < s_j）且速度更大（vs_i > vs_j）时才有碰撞风险
        approaching_mask = (s_diff < 0).float() * (vs_diff > 0).float()  # [B, N, N]

        # 避免除零
        relative_speed = torch.clamp(vs_diff, min=0.1)  # [B, N, N]

        # TTC（秒）
        ttc = dist / (relative_speed + 1e-6)  # [B, N, N]
        ttc = ttc * approaching_mask  # 只对逼近的车辆计算TTC
        ttc = ttc + (1.0 - approaching_mask) * 1000.0  # 非逼近车辆设为大TTC

        # TTC倒数（风险特征）
        risk_features = 1.0 / (ttc + 1e-6)  # [B, N, N]
        # 归一化到[0, 1]
        risk_features = torch.clamp(risk_features, max=1.0)

        return adj, risk_features

    def forward(self, vehicle_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            outputs: {
                'embeddings': [B, N, D] 节点嵌入,
                'adjacency_matrix': [B, N, N] 邻接矩阵,
                'risk_features': [B, N, N] 风险特征
            }
        """
        # 输入嵌入
        h = self.input_embedding(vehicle_states)  # [B, N, D]

        # 构建图
        adj, risk_features = self._build_adjacency_and_risk(vehicle_states)

        # GNN层（消息传递）
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj, risk_features)  # [B, N, D]

        return {
            'embeddings': h,
            'adjacency_matrix': adj,
            'risk_features': risk_features
        }


class ImportancePredictor(nn.Module):
    """
    重要性预测头（可学习）

    核心思想：
    - 不是规则，而是端到端学习
    - 通过PPO的policy gradient自动优化
    - 输入：GNN嵌入（包含全局信息）
    - 输出：重要性评分 [0, 1]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] GNN嵌入

        Returns:
            importance: [B, N, 1] 重要性评分 [0, 1]
        """
        importance = self.mlp(embeddings)  # [B, N, 1]
        importance = torch.sigmoid(importance)  # [0, 1]

        return importance


class SparseGate(nn.Module):
    """
    稀疏门控机制（可微分的Top-K）

    核心技术：
    - 训练时：Gumbel-Softmax（可微分）
    - 推理时：Hard Top-K（高效）
    - 直通估计器（Straight-Through Estimator）
    """

    def __init__(
        self,
        initial_k_ratio: float = 0.10,
        min_k: int = 5,
        max_k: int = 150
    ):
        super().__init__()

        self.k_ratio = initial_k_ratio
        self.min_k = min_k
        self.max_k = max_k

        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        importance_scores: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            importance_scores: [B, N, 1] 重要性评分
            training: 是否训练模式

        Returns:
            selection_mask: [B, N, 1] 选择掩码
            k: 实际选择的车辆数
        """
        B, N, _ = importance_scores.shape
        device = importance_scores.device

        # 计算动态K值
        k = int(self.k_ratio * N)
        k = max(k, self.min_k)
        k = min(k, self.max_k)

        if training:
            # ========== 训练模式：Gumbel-Softmax（可微分）==========

            # 1. 添加Gumbel噪声
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(importance_scores) + 1e-10) + 1e-10
            )
            logits = torch.log(importance_scores + 1e-10) + gumbel_noise

            # 2. 温度缩放
            temperature = torch.clamp(self.temperature, min=0.1, max=5.0)
            scaled_logits = logits / temperature

            # 3. Softmax归一化
            probs = torch.softmax(scaled_logits, dim=1)  # [B, N, 1]

            # 4. 直通估计器（Straight-Through Estimator）
            # 前向：hard top-k
            # 反向：soft梯度
            _, top_k_indices = torch.topk(
                importance_scores.squeeze(-1),
                k,
                dim=1
            )  # [B, k]

            hard_mask = torch.zeros_like(probs)  # [B, N, 1]
            hard_mask.scatter_(
                1,
                top_k_indices.unsqueeze(-1),
                1.0
            )  # [B, N, 1]

            # 直通估计器：前向hard，反向soft
            selection_mask = probs - probs.detach() + hard_mask

        else:
            # ========== 推理模式：Hard Top-K（高效）==========
            _, top_k_indices = torch.topk(
                importance_scores.squeeze(-1),
                k,
                dim=1
            )  # [B, k]

            selection_mask = torch.zeros_like(importance_scores)  # [B, N, 1]
            selection_mask.scatter_(
                1,
                top_k_indices.unsqueeze(-1),
                1.0
            )  # [B, N, 1]

        return selection_mask, k

    def set_k_ratio(self, k_ratio: float):
        """动态调整K值比例"""
        self.k_ratio = k_ratio


class PolicyHead(nn.Module):
    """
    策略头

    输出：
    - 加速度：[-3, 2] m/s²
    - 换道概率：[0, 1]
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 128),  # +1 for selection mask
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

    def forward(
        self,
        embeddings: torch.Tensor,
        selection_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] GNN嵌入
            selection_mask: [B, N, 1] 选择掩码

        Returns:
            actions: [B, N, 2] (加速度, 换道概率)
        """
        # 拼接嵌入和选择掩码
        features = torch.cat([embeddings, selection_mask], dim=-1)  # [B, N, D+1]

        # 策略输出
        action_mean = self.actor(features)  # [B, N, 2]

        # 加速度：[-1, 1] → [-3, 2]
        accel = action_mean[:, :, 0:1]
        accel = (accel + 1.0) / 2.0 * (2.0 - (-3.0)) + (-3.0)
        accel = torch.clamp(accel, -3.0, 2.0)

        # 换道：[-1, 1] → [0, 1]
        lane_change = action_mean[:, :, 1:2]
        lane_change = (lane_change + 1.0) / 2.0
        lane_change = torch.clamp(lane_change, 0.0, 1.0)

        actions = torch.cat([accel, lane_change], dim=-1)  # [B, N, 2]

        return actions


class ValueHead(nn.Module):
    """
    价值头

    输出状态价值估计
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] 节点嵌入

        Returns:
            value: [B, 1] 状态价值
        """
        # 全局平均池化
        global_features = embeddings.mean(dim=1)  # [B, D]

        # 价值估计
        value = self.value_net(global_features)  # [B, 1]

        return value


class HierarchicalPooling(nn.Module):
    """
    层次化图池化模块

    功能：将车辆级嵌入聚合成车道级、路段级、全局级表征

    聚合层次：
    1. Vehicle-level: 原始车辆嵌入 [B, N, D]
    2. Lane-level: 车道级聚合 [B, L, D]
    3. Section-level: 路段级聚合 [B, S, D]
    4. Global-level: 全局注意力池化 [B, D]

    核心创新：
    - 使用注意力机制而非简单mean pooling
    - 保留空间结构信息（车道、路段）
    - 可微分端到端训练
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_lanes: int = 4,
        num_sections: int = 5
    ):
        """
        Args:
            hidden_dim: 嵌入维度
            num_lanes: 车道数量（默认4：E1, E2, E3, E5）
            num_sections: 路段数量（分段聚合）
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_lanes = num_lanes
        self.num_sections = num_sections

        # 可学习的空嵌入（用于空车道/路段）
        self.empty_lane_embedding = nn.Parameter(torch.zeros(hidden_dim))
        self.empty_section_embedding = nn.Parameter(torch.zeros(hidden_dim))

        # 车道级注意力池化
        self.lane_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 车道级特征变换（可选）
        self.lane_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 路段级注意力池化
        self.section_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 路段级特征变换
        self.section_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 全局注意力池化
        self.global_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 全局特征变换
        self.global_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        层次化池化

        Args:
            embeddings: [B, N, D] 车辆嵌入
            vehicle_states: [B, N, 9] 车辆状态
                - 第0维：纵向位置 x (用于路段划分)
                - 第6维：车道索引 lane_id (用于车道聚合)
                - 其他维度：可用于高级特征

        Returns:
            Dict:
                'vehicle': [B, N, D] 原始车辆级
                'lane': [B, L, D] 车道级聚合
                'section': [B, S, D] 路段级聚合
                'global': [B, D] 全局池化
        """
        batch_size, num_vehicles, hidden_dim = embeddings.shape
        device = embeddings.device

        # ============================================
        # Level 1: Vehicle-level (原始嵌入)
        # ============================================
        vehicle_features = embeddings  # [B, N, D]

        # ============================================
        # Level 2: Lane-level 聚合
        # ============================================
        # 提取车道索引 (vehicle_states[:, :, 6])
        # 车道索引范围: [0, num_lanes-1]，需要处理无效值（-1）
        lane_indices = vehicle_states[:, :, 6].long()  # [B, N]
        valid_lane_mask = (lane_indices >= 0) & (lane_indices < self.num_lanes)  # [B, N]

        # 创建车道级聚合矩阵
        lane_features = []
        for b in range(batch_size):
            batch_lane_features = []
            for lane_id in range(self.num_lanes):
                # 找到该车道的所有车辆
                lane_mask = valid_lane_mask[b] & (lane_indices[b] == lane_id)

                if lane_mask.sum() > 0:
                    # 提取该车道的车辆嵌入
                    lane_vehicles = embeddings[b][lane_mask]  # [N_lane, D]

                    # 注意力权重
                    attn_scores = self.lane_attention(lane_vehicles)  # [N_lane, 1]
                    attn_weights = F.softmax(attn_scores, dim=0)  # [N_lane, 1]

                    # 加权聚合
                    lane_pooled = (attn_weights * lane_vehicles).sum(dim=0, keepdim=True)  # [1, D]
                else:
                    # 如果该车道的没有车辆，使用可学习的空嵌入
                    lane_pooled = self.empty_lane_embedding.unsqueeze(0)  # [D] -> [1, D]

                batch_lane_features.append(lane_pooled)

            # [L, D]
            batch_lane_features = torch.cat(batch_lane_features, dim=0)
            lane_features.append(batch_lane_features)

        # [B, L, D]
        lane_features = torch.stack(lane_features, dim=0)

        # 车道级特征变换
        lane_features = self.lane_transform(lane_features)  # [B, L, D]

        # ============================================
        # Level 3: Section-level 聚合
        # ============================================
        # 提取纵向位置 (vehicle_states[:, :, 0])
        # 位置范围: [0, 3000]（归一化后约为 [0, 1]）
        # 将道路分为 num_sections 个路段
        longitudinal_positions = vehicle_states[:, :, 0]  # [B, N]

        # 归一化位置到 [0, 1]（假设最大长度为3000m）
        max_length = 3000.0
        normalized_positions = longitudinal_positions / max_length  # [B, N]

        # 计算路段索引
        section_indices = (normalized_positions * self.num_sections).long()
        section_indices = torch.clamp(section_indices, 0, self.num_sections - 1)  # [B, N]

        # 创建路段级聚合矩阵
        section_features = []
        for b in range(batch_size):
            batch_section_features = []
            for section_id in range(self.num_sections):
                # 找到该路段的所有车辆
                section_mask = section_indices[b] == section_id

                if section_mask.sum() > 0:
                    # 提取该路段的车辆嵌入
                    section_vehicles = embeddings[b][section_mask]  # [N_section, D]

                    # 注意力权重
                    attn_scores = self.section_attention(section_vehicles)  # [N_section, 1]
                    attn_weights = F.softmax(attn_scores, dim=0)  # [N_section, 1]

                    # 加权聚合
                    section_pooled = (attn_weights * section_vehicles).sum(dim=0, keepdim=True)  # [1, D]
                else:
                    # 如果该路段没有车辆，使用可学习的空嵌入
                    section_pooled = self.empty_section_embedding.unsqueeze(0)  # [D] -> [1, D]

                batch_section_features.append(section_pooled)

            # [S, D]
            batch_section_features = torch.cat(batch_section_features, dim=0)
            section_features.append(batch_section_features)

        # [B, S, D]
        section_features = torch.stack(section_features, dim=0)

        # 路段级特征变换
        section_features = self.section_transform(section_features)  # [B, S, D]

        # ============================================
        # Level 4: Global-level 全局池化
        # ============================================
        # 使用所有车辆嵌入进行全局注意力池化
        attn_scores = self.global_attention(embeddings)  # [B, N, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]

        # 加权聚合
        global_features = (attn_weights * embeddings).sum(dim=1)  # [B, D]

        # 全局特征变换
        global_features = self.global_transform(global_features)  # [B, D]

        # ============================================
        # 返回所有层次的表征
        # ============================================
        return {
            'vehicle': vehicle_features,   # [B, N, D]
            'lane': lane_features,         # [B, L, D]
            'section': section_features,   # [B, S, D]
            'global': global_features      # [B, D]
        }


class JointICVPolicy(nn.Module):
    """
    联合ICV评分与策略生成网络（v5.0）

    核心创新：
    1. 端到端可学习的车辆重要性评分
    2. 可微分的稀疏门控机制
    3. 联合PPO训练（importance + action）
    4. 动态K值选择
    """

    def __init__(
        self,
        obs_dim: int = 321,  # max_vehicles(32) * 9 + 32(global) + 1(num)
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 3,
        initial_k_ratio: float = 0.10,
        device: str = 'cuda'
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # 推断max_vehicles
        self.max_vehicles = (obs_dim - 32 - 1) // 9

        # 1. GNN编码器
        self.gnn_encoder = RiskSensitiveGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.1
        )

        # 2. 重要性预测头
        self.importance_predictor = ImportancePredictor(
            hidden_dim=hidden_dim
        )

        # 3. 稀疏门控
        self.sparse_gate = SparseGate(
            initial_k_ratio=initial_k_ratio,
            min_k=5,
            max_k=int(0.25 * self.max_vehicles)  # 官方上限25%
        )

        # 4. 策略头
        self.policy_head = PolicyHead(
            hidden_dim=hidden_dim
        )

        # 5. 价值头
        self.value_head = ValueHead(
            hidden_dim=hidden_dim
        )

        # 6. 层次化池化（v5.0新增）
        self.hierarchical_pooling = HierarchicalPooling(
            hidden_dim=hidden_dim,
            num_lanes=4,      # E1, E2, E3, E5
            num_sections=5    # 5个路段
        )

        # 7. 动作分布参数（初始化为std=1.0以确保非负熵）
        self.log_std = nn.Parameter(torch.zeros(self.max_vehicles * 2))  # log(1.0) = 0

        # ========== v5.0 扩展组件（可选） ==========

        # 8. WorldModel（世界模型）
        self.use_world_model = False  # 默认禁用，通过配置启用
        self.world_model = None

        # 9. CostCritic（成本评论家）
        self.use_cost_critic = False  # 默认禁用
        self.cost_critic = None

        # 10. DynamicWeightGate（动态权重门控）
        self.use_dynamic_gate = False  # 默认禁用
        self.dynamic_gate = None

        # 11. SafetyShield（安全屏障）
        self.use_safety_shield = False  # 默认禁用
        self.safety_shield = None

        # ===========================================

    def enable_world_model(self, num_vehicles: int = 32, latent_dim: int = 64):
        """启用WorldModel"""
        from .world_model import WorldModel
        self.use_world_model = True
        self.world_model = WorldModel(
            hidden_dim=self.hidden_dim,
            latent_dim=latent_dim,
            num_layers=2,
            num_vehicles=num_vehicles
        ).to(self.device)

    def enable_cost_critic(self, global_dim: int = 64):
        """启用CostCritic"""
        from .cost_critic import CostCritic
        self.use_cost_critic = True
        self.cost_critic = CostCritic(
            hidden_dim=self.hidden_dim,
            global_dim=global_dim,
            dropout=0.1
        ).to(self.device)

    def enable_dynamic_gate(self, global_dim: int = 64):
        """启用DynamicWeightGate"""
        from .dynamic_weight_gate import DynamicWeightGate
        self.use_dynamic_gate = True
        self.dynamic_gate = DynamicWeightGate(
            global_dim=global_dim,
            hidden_dim=32,
            dropout=0.1
        ).to(self.device)

    def enable_safety_shield(self):
        """启用SafetyShield"""
        from .safety_shield import SafetyShield
        self.use_safety_shield = True
        self.safety_shield = SafetyShield(device='cpu')

    def _parse_observation(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        解析观测张量

        Args:
            obs: [B, obs_dim] 扁平化观测

        Returns:
            vehicle_states: [B, N, 9]
            global_stats: [B, 32]
            num_vehicles: int
        """
        B = obs.size(0)
        device = obs.device

        # 动态推断max_vehicles
        actual_max_vehicles = (obs.size(1) - 32 - 1) // 9

        # 车辆特征
        vehicle_dim = actual_max_vehicles * 9
        vehicle_features = obs[:, :vehicle_dim]  # [B, N*9]
        vehicle_states = vehicle_features.view(
            B, actual_max_vehicles, 9
        )  # [B, N, 9]

        # 全局统计
        global_stats = obs[:, vehicle_dim:vehicle_dim + 32]  # [B, 32]

        # 实际车辆数
        num_vehicles_tensor = obs[:, vehicle_dim + 32:vehicle_dim + 33]  # [B, 1]
        num_vehicles = int(num_vehicles_tensor[0, 0].item())

        return vehicle_states, global_stats, num_vehicles

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            obs: [B, obs_dim] 扁平化观测
            deterministic: 是否确定性动作

        Returns:
            outputs: {
                'actions': [B, N*2] 动作,
                'value': [B, 1] 价值,
                'log_prob': [B] log概率,
                'importance': [B, N, 1] 重要性评分,
                'selection': [B, N, 1] 选择掩码,
                'k': int 选择的车辆数
            }
        """
        B = obs.size(0)
        device = obs.device

        # 1. 解析观测
        vehicle_states, global_stats, num_vehicles = self._parse_observation(obs)

        # 2. GNN编码
        gnn_outputs = self.gnn_encoder(vehicle_states)
        embeddings = gnn_outputs['embeddings']  # [B, N, D]

        # 2.5. 层次化池化（v5.0新增）
        pooled_features = self.hierarchical_pooling(embeddings, vehicle_states)
        # pooled_features: {
        #     'vehicle': [B, N, D],
        #     'lane': [B, L, D],
        #     'section': [B, S, D],
        #     'global': [B, D]
        # }

        # 3. 预测重要性
        importance = self.importance_predictor(embeddings)  # [B, N, 1]

        # 4. 稀疏门控
        selection_mask, k = self.sparse_gate(
            importance,
            training=self.training
        )  # [B, N, 1], int

        # 5. 策略输出
        raw_actions = self.policy_head(embeddings, selection_mask)  # [B, N, 2]

        # 应用选择掩码
        actions = raw_actions * selection_mask  # [B, N, 2]

        # 展平动作
        actions_flat = actions.view(B, -1)  # [B, N*2]

        # 6. 价值估计
        value = self.value_head(embeddings)  # [B, 1]

        # ========== v5.0 扩展组件集成 ==========
        additional_outputs = {}

        # 6.1 WorldModel（如果启用）
        if self.use_world_model and self.world_model is not None:
            world_outputs = self.world_model(embeddings)
            additional_outputs['world_model'] = world_outputs
            # world_outputs包含: z_flow, z_risk, pred_next_states, risk_prob, hidden

        # 6.2 CostCritic（如果启用）
        if self.use_cost_critic and self.cost_critic is not None:
            global_state = pooled_features['global']  # [B, D]
            cost_outputs = self.cost_critic(embeddings, global_state)
            additional_outputs['cost_value'] = cost_outputs['value']  # [B, 1]
            additional_outputs['cost_cost'] = cost_outputs['cost']    # [B, 1]

        # 6.3 DynamicWeightGate（如果启用）
        if self.use_dynamic_gate and self.dynamic_gate is not None:
            global_state = pooled_features['global']  # [B, D]
            weights = self.dynamic_gate(global_state)  # [B, 3]
            additional_outputs['reward_weights'] = weights
            # weights: [w_efficiency, w_stability, w_cost]

        # 6.4 SafetyShield（如果启用且在推理模式）
        safety_reward = torch.zeros(B, device=self.device)
        if self.use_safety_shield and self.safety_shield is not None and not self.training:
            # 将actions转回[B, N, 2]形状
            actions_reshaped = actions  # [B, N, 2]
            # 安全屏障过滤（在CPU上执行）
            actions_cpu = actions_reshaped.cpu()
            vehicle_states_cpu = vehicle_states.cpu()
            shield_result = self.safety_shield.filter_actions(
                actions_cpu,
                vehicle_states_cpu,
                return_details=False
            )
            # 将过滤后的动作移回GPU并展平
            actions = shield_result['safe_actions'].to(device)  # [B, N, 2]
            actions_flat = actions.view(B, -1)  # [B, N*2] 重新展平
            safety_reward = shield_result['safety_reward'].to(device)  # [B]

        # ===========================================

        # 7. 计算log_prob和entropy（高斯策略）
        # 只为选中的ICV计算log_prob和entropy
        log_std = self.log_std.unsqueeze(0).expand_as(actions_flat)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)

        mean = actions_flat
        std = torch.exp(log_std)

        if deterministic:
            actions_out = mean
        else:
            actions_out = mean + std * torch.randn_like(mean)

        # 计算每个动作维度的log_prob
        log_prob_per_dim = -0.5 * (
            ((actions_out - mean) / (std + 1e-6)) ** 2 +
            2 * log_std +
            np.log(2 * np.pi)
        )  # [B, N*2]

        # 创建选择mask（从[B, N, 1]扩展到[B, N*2]）
        selection_mask_2d = selection_mask.squeeze(-1)  # [B, N]
        selection_mask_flat = selection_mask_2d.repeat(1, 2)  # [B, N*2] 每辆车2个动作

        # 只对选中的ICV计算log_prob和entropy
        masked_log_prob = log_prob_per_dim * selection_mask_flat  # [B, N*2]
        log_prob = masked_log_prob.sum(dim=-1)  # [B] 只对选中的维求和

        # 计算熵（只对选中的ICV）
        entropy_per_dim = 0.5 * (
            np.log(2 * np.pi) + 2 * log_std + 1
        )  # [B, N*2]
        masked_entropy = entropy_per_dim * selection_mask_flat  # [B, N*2]
        entropy = masked_entropy.sum(dim=-1)  # [B] 只对选中的维求和

        return {
            'actions': actions_out,
            'value': value,
            'log_prob': log_prob,
            'entropy': entropy,  # 新增：返回策略熵
            'importance': importance,
            'selection': selection_mask,
            'k': k,
            'embeddings': embeddings,
            'pooled_features': pooled_features,  # v5.0新增: 层次化特征
            'safety_reward': safety_reward,      # v5.0新增: 安全奖励
            **additional_outputs  # v5.0新增: 其他组件输出
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作（PPO训练时使用）

        重新计算给定obs下actions的log_prob和entropy
        """
        B = obs.size(0)

        # 解析观测
        vehicle_states, global_stats, num_vehicles = self._parse_observation(obs)

        # GNN编码
        gnn_outputs = self.gnn_encoder(vehicle_states)
        embeddings = gnn_outputs['embeddings']

        # 重要性
        importance = self.importance_predictor(embeddings)

        # 选择
        selection_mask, k = self.sparse_gate(importance, training=self.training)

        # 策略输出
        raw_actions = self.policy_head(embeddings, selection_mask)
        actions_flat = raw_actions.view(B, -1)

        # 计算log_prob和entropy（只对选中的ICV）
        log_std = self.log_std.unsqueeze(0).expand_as(actions_flat)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)

        mean = actions_flat
        std = torch.exp(log_std)

        # 计算每个动作维度的log_prob
        log_prob_per_dim = -0.5 * (
            ((actions - mean) / (std + 1e-6)) ** 2 +
            2 * log_std +
            np.log(2 * np.pi)
        )  # [B, N*2]

        # 创建选择mask（从[B, N, 1]扩展到[B, N*2]）
        selection_mask_2d = selection_mask.squeeze(-1)  # [B, N]
        selection_mask_flat = selection_mask_2d.repeat(1, 2)  # [B, N*2]

        # 只对选中的ICV计算log_prob和entropy
        masked_log_prob = log_prob_per_dim * selection_mask_flat
        log_prob = masked_log_prob.sum(dim=-1)

        entropy_per_dim = 0.5 * (
            np.log(2 * np.pi) + 2 * log_std + 1
        )
        masked_entropy = entropy_per_dim * selection_mask_flat
        entropy = masked_entropy.sum(dim=-1).mean()

        value = self.value_head(embeddings)

        return value, log_prob, entropy

    def set_k_ratio(self, k_ratio: float):
        """动态调整K值比例"""
        self.sparse_gate.set_k_ratio(k_ratio)

    def get_num_controlled(self, obs: torch.Tensor) -> int:
        """获取当前控制的车辆数"""
        with torch.no_grad():
            outputs = self.forward(obs, deterministic=True)
            return outputs['k']


def create_joint_icv_policy(
    obs_dim: int = 321,
    node_dim: int = 9,
    hidden_dim: int = 64,
    num_layers: int = 3,
    initial_k_ratio: float = 0.10,
    device: str = 'cuda'
) -> JointICVPolicy:
    """
    创建联合ICV策略网络的工厂函数

    Args:
        obs_dim: 观测维度
        node_dim: 节点特征维度
        hidden_dim: 隐藏层维度
        num_layers: GNN层数
        initial_k_ratio: 初始K值比例
        device: 设备

    Returns:
        policy: JointICVPolicy实例
    """
    policy = JointICVPolicy(
        obs_dim=obs_dim,
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        initial_k_ratio=initial_k_ratio,
        device=device
    )

    return policy.to(device)
