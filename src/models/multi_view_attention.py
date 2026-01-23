"""
Multi-View Graph Attention for Traffic Control

核心创新：
- 3视角注意力机制（Spatial/Interaction/Type）
- Spatial View：基于空间邻近性
- Interaction View：基于TTC交互强度
- Type View：基于ICV类型

论文参考：
- Multi-Class Traffic Assignment Using Multi-View Graph Attention (2024)
- GMAN: Graph Multi-Attention Network for Traffic Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiViewGraphAttention(nn.Module):
    """
    多视角图注意力机制

    3个视角:
    1. Spatial View: 空间邻近性(基于距离的高斯核)
    2. Interaction View: 交互强度(基于TTC)
    3. Type View: 车辆类型(ICV之间权重更高)

    Args:
        hidden_dim (int): 隐藏层维度
        num_heads (int): 注意力头数
        dropout (float): Dropout比例
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # 3个视角的投影层
        self.spatial_proj = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_proj = nn.Linear(hidden_dim, hidden_dim)
        self.type_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影（融合3个视角）
        self.out_proj = nn.Linear(hidden_dim * 3, hidden_dim)

        # 归一化
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 超参数
        self.spatial_sigma = 0.15  # 空间邻近性阈值(150米)
        self.interaction_radius = 0.15  # 交互半径
        self.ttc_threshold = 3.0  # TTC阈值(秒)

        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        for module in [self.spatial_proj, self.interaction_proj, self.type_proj, self.out_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, D] GNN编码后的车辆嵌入
            vehicle_states: [B, N, 9] 原始车辆状态

        Returns:
            attended_embeddings: [B, N, D] 多视角注意力后的嵌入
        """
        B, N, D = embeddings.shape
        device = embeddings.device

        # ========== View 1: Spatial Attention ==========
        # 基于空间邻近性
        spatial_adj = self._compute_spatial_adjacency(vehicle_states)  # [B, N, N]
        spatial_out = self._apply_attention(
            embeddings, spatial_adj, self.spatial_proj
        )  # [B, N, D]

        # ========== View 2: Interaction Attention ==========
        # 基于交互强度(TTC)
        interaction_adj, _ = self._compute_interaction_adjacency(vehicle_states)  # [B, N, N]
        interaction_out = self._apply_attention(
            embeddings, interaction_adj, self.interaction_proj
        )  # [B, N, D]

        # ========== View 3: Type Attention ==========
        # ICV vs 人类驾驶
        type_adj = self._compute_type_adjacency(vehicle_states)  # [B, N, N]
        type_out = self._apply_attention(
            embeddings, type_adj, self.type_proj
        )  # [B, N, D]

        # ========== 融合3个视角 ==========
        # 拼接
        multi_view_features = torch.cat([
            spatial_out,
            interaction_out,
            type_out
        ], dim=-1)  # [B, N, 3*D]

        # 投影回原始维度
        out = self.out_proj(multi_view_features)  # [B, N, D]

        # 残差连接 + 归一化
        out = self.norm(out + embeddings)
        out = self.dropout(out)

        return out

    def _compute_spatial_adjacency(self, vehicle_states: torch.Tensor) -> torch.Tensor:
        """
        计算空间邻近性邻接矩阵（基于距离的高斯核）

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            adj: [B, N, N] 空间邻接矩阵
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取位置
        s = vehicle_states[:, :, 0]  # [B, N] 纵向位置
        d = vehicle_states[:, :, 1]  # [B, N] 横向位置

        # 计算距离矩阵
        s_diff = s.unsqueeze(2) - s.unsqueeze(1)  # [B, N, N]
        d_diff = d.unsqueeze(2) - d.unsqueeze(1)  # [B, N, N]
        dist_sq = s_diff ** 2 + d_diff ** 2  # [B, N, N]

        # 高斯核
        adj = torch.exp(-dist_sq / (2 * self.spatial_sigma ** 2))  # [B, N, N]

        # 排除自连接
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)  # [B, N, N]
        adj = adj * mask

        # 有效车辆掩码
        valid_mask = (s > 0).float()  # [B, N]
        valid_adj = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)  # [B, N, N]
        adj = adj * valid_adj

        # 行归一化
        row_sum = adj.sum(dim=-1, keepdim=True)  # [B, N, 1]
        adj = adj / (row_sum + 1e-6)

        return adj

    def _compute_interaction_adjacency(
        self,
        vehicle_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算交互强度邻接矩阵（基于TTC）

        复用RiskSensitiveGNN的TTC计算逻辑

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            adj: [B, N, N] 交互邻接矩阵
            risk_features: [B, N, N] 风险特征（TTC倒数）
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取位置、速度、车道
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

        # 计算相对速度
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
        row_sum = mask.sum(dim=-1, keepdim=True)  # [B, N, 1]
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

    def _compute_type_adjacency(self, vehicle_states: torch.Tensor) -> torch.Tensor:
        """
        计算车辆类型邻接矩阵（ICV vs 人类驾驶）

        ICV之间的权重更高，促进协同

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            adj: [B, N, N] 类型邻接矩阵
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取车辆类型（第8维: is_icv）
        is_icv = vehicle_states[:, :, 8]  # [B, N]

        # ICV之间权重更高（协同）
        type_matrix = is_icv.unsqueeze(1) * is_icv.unsqueeze(2)  # [B, N, N]

        # 非ICV车辆之间也有连接，但权重较低
        type_matrix = type_matrix * 2.0 + 0.5  # ICV: 2.0, 其他: 0.5

        # 排除自连接
        mask = 1.0 - torch.eye(N, device=device).unsqueeze(0)  # [B, N, N]
        type_matrix = type_matrix * mask

        # 有效车辆掩码
        s = vehicle_states[:, :, 0]  # [B, N]
        valid_mask = (s > 0).float()  # [B, N]
        valid_adj = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)  # [B, N, N]
        type_matrix = type_matrix * valid_adj

        # 行归一化
        row_sum = type_matrix.sum(dim=-1, keepdim=True)  # [B, N, 1]
        type_adj = type_matrix / (row_sum + 1e-6)

        return type_adj

    def _apply_attention(
        self,
        embeddings: torch.Tensor,
        adj: torch.Tensor,
        proj: nn.Module
    ) -> torch.Tensor:
        """
        应用注意力机制

        Args:
            embeddings: [B, N, D] 输入嵌入
            adj: [B, N, N] 邻接矩阵
            proj: 投影层

        Returns:
            out: [B, N, D] 注意力输出
        """
        # 投影
        h = proj(embeddings)  # [B, N, D]

        # 加权聚合
        out = torch.bmm(adj, h)  # [B, N, D]

        return out
