"""
Risk-Sensitive Graph Neural Network - 感知层核心模块
功能：从交通图中提取风险敏感的特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
import numpy as np


class RiskSensitiveGNN(nn.Module):
    """
    风险敏感图神经网络

    特性：
    - 多头注意力机制捕捉车辆交互
    - 风险加权边特征
    - 残差连接和层归一化
    - 全局图级嵌入
    """

    def __init__(
        self,
        node_dim: int = 9,
        edge_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 256,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.heads = heads

        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 边特征编码器
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Dropout(dropout),
            nn.Linear(16, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )

        # 风险注意力机制
        self.risk_attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # GNN层堆叠
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            try:
                # 尝试创建支持边特征的 GAT 层
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=heads,
                        concat=False,
                        edge_dim=hidden_dim // 2,  # 某些版本可能不支持
                        dropout=dropout,
                        bias=True
                    )
                )
            except TypeError:
                # 如果不支持 edge_dim，则创建不带边特征的版本
                print(f"⚠️  警告: GATv2Conv 不支持 edge_dim，将使用不带边特征的版本")
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        bias=True
                    )
                )
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # 输出投影层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 全局池化（可选）
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            node_features: [N, node_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_features: [E, edge_dim] 边特征
            batch: [N] 批次索引（可选）

        Returns:
            dict: 包含节点嵌入和全局嵌入的字典
        """
        # 1. 编码节点和边特征
        node_emb = self.node_encoder(node_features)  # [N, hidden_dim]

        if edge_features.size(0) > 0:
            edge_emb = self.edge_encoder(edge_features)  # [E, hidden_dim//2]
        else:
            edge_emb = torch.zeros(0, self.hidden_dim // 2, device=node_features.device)

        # 2. 计算风险注意力权重
        risk_weights = None
        if edge_features.size(0) > 0:
            src_nodes = edge_index[0]  # [E]
            src_features = node_emb[src_nodes]  # [E, hidden_dim]

            risk_input = torch.cat([src_features, edge_emb], dim=1)
            risk_weights = self.risk_attention(risk_input)  # [E, 1]

            # 将风险权重融合到边特征中
            edge_emb = edge_emb * risk_weights

        # 3. GNN多层传播
        x = node_emb
        for i, gnn_layer in enumerate(self.gnn_layers):
            residual = x

            # GNN传播（兼容不同版本的PyTorch Geometric）
            try:
                # 尝试使用 edge_attr 参数
                if edge_features.size(0) > 0:
                    x = gnn_layer(x, edge_index, edge_attr=edge_emb)
                else:
                    x = gnn_layer(x, edge_index)
            except TypeError:
                # 如果不支持 edge_attr，则不使用边特征
                x = gnn_layer(x, edge_index)

            # 残差连接 + 层归一化
            x = self.layer_norms[i](x + residual)
            x = F.relu(x)

        # 4. 输出投影
        node_output = self.output_layer(x)  # [N, output_dim]

        # 5. 全局嵌入（如果需要）
        global_output = None
        if batch is not None:
            # 批次模式：全局池化
            global_output = global_mean_pool(node_output, batch)
        else:
            # 单图模式：平均所有节点
            global_output = node_output.mean(dim=0, keepdim=True)

        return {
            'node_embedding': node_output,
            'global_embedding': global_output,
            'risk_weights': risk_weights
        }

    def compute_risk_map(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算车辆风险图

        Args:
            node_features: [N, node_dim]
            edge_index: [2, E]
            edge_features: [E, edge_dim]

        Returns:
            risk_scores: [N] 每个节点的风险得分
        """
        with torch.no_grad():
            # 编码
            node_emb = self.node_encoder(node_features)

            if edge_features.size(0) > 0:
                edge_emb = self.edge_encoder(edge_features)
                src_nodes = edge_index[0]
                src_features = node_emb[src_nodes]

                # 计算边风险
                edge_risk = self.risk_attention(torch.cat([src_features, edge_emb], dim=1))

                # 聚合到节点
                node_risk = torch.zeros(node_features.size(0), 1, device=node_features.device)
                node_risk.scatter_add_(0, edge_index[0].unsqueeze(1), edge_risk)

                # 归一化
                node_degree = torch.zeros(node_features.size(0), 1, device=node_features.device)
                node_degree.scatter_add_(0, edge_index[0].unsqueeze(1), torch.ones_like(edge_risk))
                node_risk = node_risk / (node_degree + 1e-6)

                return node_risk.squeeze(-1)
            else:
                return torch.zeros(node_features.size(0), device=node_features.device)


class GraphBuilder:
    """
    交通图构建器

    功能：
    - 从车辆状态构建交通交互图
    - 连接相邻车辆（同车道前后车）
    - 连接潜在冲突车辆（相邻车道）
    """

    def __init__(
        self,
        interaction_radius: float = 100.0,
        max_neighbors: int = 8,
        lane_change_distance: float = 50.0
    ):
        self.interaction_radius = interaction_radius
        self.max_neighbors = max_neighbors
        self.lane_change_distance = lane_change_distance

    def build_graph(
        self,
        vehicle_states: Dict[str, Dict],
        icv_ids: set
    ) -> Data:
        """
        构建交通图

        Args:
            vehicle_states: {vehicle_id: {state_dict}}
            icv_ids: 智能网联车ID集合

        Returns:
            PyG Data对象
        """
        if not vehicle_states:
            # 空图
            return Data(
                x=torch.zeros(0, 9),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, 4)
            )

        # 1. 提取节点特征
        vehicle_ids = list(vehicle_states.keys())
        num_vehicles = len(vehicle_ids)

        node_features = []
        for vid in vehicle_ids:
            state = vehicle_states[vid]
            features = self._extract_node_features(state, icv_ids)
            node_features.append(features)

        node_features = torch.tensor(np.array(node_features), dtype=torch.float32)

        # 2. 构建边
        edge_indices = []
        edge_features = []

        for i, vid_i in enumerate(vehicle_ids):
            for j, vid_j in enumerate(vehicle_ids):
                if i == j:
                    continue

                state_i = vehicle_states[vid_i]
                state_j = vehicle_states[vid_j]

                # 检查是否应该连接
                should_connect, edge_feat = self._should_connect(state_i, state_j)

                if should_connect:
                    edge_indices.append([i, j])
                    edge_features.append(edge_feat)

        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_features), dtype=torch.float32)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_attr = torch.zeros(0, 4)

        # 3. 创建PyG Data对象
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        return graph

    def _extract_node_features(self, state: Dict, icv_ids: set) -> List[float]:
        """
        提取节点特征（9维）

        特征：
        1-3: 位置(x, y, z)
        4-5: 速度(vx, vy)
        6-7: 加速度(ax, ay)
        8: 车道索引
        9: 是否为ICV
        """
        return [
            state.get('x', 0.0),
            state.get('y', 0.0),
            state.get('z', 0.0),
            state.get('vx', 0.0),
            state.get('vy', 0.0),
            state.get('ax', 0.0),
            state.get('ay', 0.0),
            state.get('lane_index', 0.0),
            1.0 if state.get('id', '') in icv_ids else 0.0
        ]

    def _should_connect(self, state_i: Dict, state_j: Dict) -> Tuple[bool, List[float]]:
        """
        判断两个车辆是否应该连接

        边特征（4维）：
        1: 距离
        2: 相对速度
        3: 相对加速度
        4: 是否在同一车道
        """
        # 计算距离
        dx = state_j['x'] - state_i['x']
        dy = state_j['y'] - state_i['y']
        distance = np.sqrt(dx**2 + dy**2)

        if distance > self.interaction_radius:
            return False, []

        # 相对速度
        dvx = state_j['vx'] - state_i['vx']
        dvy = state_j['vy'] - state_i['vy']

        # 相对加速度
        dax = state_j['ax'] - state_i['ax']
        day = state_j['ay'] - state_i['ay']

        # 是否同一车道
        same_lane = 1.0 if state_i['lane_id'] == state_j['lane_id'] else 0.0

        # 连接条件：
        # 1. 同车道且距离 < 交互半径
        # 2. 相邻车道且距离 < 换道距离
        if same_lane > 0.5 or distance < self.lane_change_distance:
            edge_feat = [
                distance,
                np.sqrt(dvx**2 + dvy**2),
                np.sqrt(dax**2 + day**2),
                same_lane
            ]
            return True, edge_feat

        return False, []


def build_batch_graphs(
    vehicle_states_batch: List[Dict[str, Dict]],
    icv_ids_batch: List[set],
    graph_builder: GraphBuilder
) -> Batch:
    """
    批量构建图并打包成批次

    Args:
        vehicle_states_batch: 批次车辆状态列表
        icv_ids_batch: 批次ICV ID列表
        graph_builder: 图构建器

    Returns:
        批次图对象
    """
    graphs = []

    for vehicle_states, icv_ids in zip(vehicle_states_batch, icv_ids_batch):
        graph = graph_builder.build_graph(vehicle_states, icv_ids)
        graphs.append(graph)

    # 打包成批次
    batch = Batch.from_data_list(graphs)

    return batch
