"""
基于神经网络的智能ICV车辆评分系统

核心功能：
1. 使用GNN学习车辆之间的交互影响
2. 通过注意力机制计算车辆重要性评分
3. 端到端优化OCR目标

设计原则：
- 学习复杂交互模式（多车协同、连锁反应）
- 自适应权重（适应不同场景）
- 计算高效（轻量级GNN）
- 完全数据驱动（无规则依赖）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.v5_lightweight import LightweightGraphConvolution


class VehicleGraphBuilder:
    """
    车辆交互图构建器

    将车辆状态转换为图结构：
    - 节点：每辆车
    - 边：车辆之间的交互关系
    """

    def __init__(
        self,
        distance_threshold: float = 100.0,  # 距离阈值（米）
        max_neighbors: int = 8               # 最大邻居数
    ):
        self.distance_threshold = distance_threshold
        self.max_neighbors = max_neighbors

    def build_graph(
        self,
        vehicle_states: Dict[str, Dict],
        num_vehicles: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建车辆交互图

        Args:
            vehicle_states: 车辆状态字典 {veh_id: state_dict}
            num_vehicles: 车辆数量

        Returns:
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_dim] 边特征
        """
        vehicle_ids = list(vehicle_states.keys())
        edge_list = []
        edge_attr_list = []

        # 提取车辆位置信息
        positions = {}
        for veh_id in vehicle_ids:
            state = vehicle_states[veh_id]
            positions[veh_id] = np.array([state.get('x', 0), state.get('y', 0)])

        # 构建边（基于距离）
        for i, veh_id_i in enumerate(vehicle_ids):
            distances = []

            for j, veh_id_j in enumerate(vehicle_ids):
                if i == j:
                    continue

                # 计算距离
                dist = np.linalg.norm(positions[veh_id_i] - positions[veh_id_j])

                if dist < self.distance_threshold:
                    distances.append((j, dist))

            # 只保留最近的K个邻居
            distances.sort(key=lambda x: x[1])
            for j, dist in distances[:self.max_neighbors]:
                edge_list.append([i, j])
                edge_list.append([j, i])  # 无向图

                # 边特征：距离、相对速度等
                state_i = vehicle_states[veh_id_i]
                state_j = vehicle_states[vehicle_ids[j]]
                edge_features = self._compute_edge_features(state_i, state_j, dist)
                edge_attr_list.append(edge_features)
                edge_attr_list.append(edge_features)  # 对称边

        if len(edge_list) == 0:
            # 如果没有边，返回空张量
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        return edge_index, edge_attr

    def _compute_edge_features(
        self,
        state_i: Dict,
        state_j: Dict,
        distance: float
    ) -> List[float]:
        """
        计算边特征

        Args:
            state_i: 车辆i的状态
            state_j: 车辆j的状态
            distance: 两车距离

        Returns:
            边特征向量 [dist, speed_diff, lane_diff, accel_diff]
        """
        speed_i = state_i.get('speed', 0)
        speed_j = state_j.get('speed', 0)
        lane_i = state_i.get('lane_index', 0)
        lane_j = state_j.get('lane_index', 0)
        accel_i = state_i.get('acceleration', 0)
        accel_j = state_j.get('acceleration', 0)

        return [
            distance / 100.0,           # 归一化距离
            (speed_i - speed_j) / 20.0,  # 归一化速度差
            (lane_i - lane_j) / 10.0,    # 归一化车道差
            (accel_i - accel_j) / 5.0    # 归一化加速度差
        ]


class GraphAttentionScoring(nn.Module):
    """
    基于图注意力的车辆评分模块

    使用多头注意力机制学习车辆之间的交互影响
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        node_embeddings: torch.Tensor  # [N, hidden_dim]
    ) -> torch.Tensor:
        """
        计算车辆重要性评分

        Args:
            node_embeddings: 节点嵌入 [N, hidden_dim]

        Returns:
            scores: 重要性评分 [N, 1]
        """
        # 自注意力（每个车辆"关注"所有其他车辆）
        attn_output, attn_weights = self.attention(
            node_embeddings,
            node_embeddings,
            node_embeddings
        )

        # 聚合注意力信息
        scores = self.output_proj(attn_output)  # [N, 1]

        return scores.squeeze(-1)  # [N]


class NeuralVehicleScorer(nn.Module):
    """
    神经网络车辆评分器

    完整流程：
    1. 嵌入车辆特征
    2. GNN编码（学习交互）
    3. 注意力评分
    4. 输出重要性评分
    """

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # 特征嵌入
        self.embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GNN层（使用现有的轻量级卷积）
        self.gnn_layers = nn.ModuleList([
            LightweightGraphConvolution(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # 注意力评分
        self.attention_scorer = GraphAttentionScoring(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 最终评分头（归一化到0-1）
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        vehicle_features: torch.Tensor,     # [N, node_dim]
        edge_index: torch.Tensor,            # [2, E]
        edge_attr: Optional[torch.Tensor] = None  # [E, edge_dim]
    ) -> torch.Tensor:
        """
        计算车辆重要性评分

        Args:
            vehicle_features: 车辆特征 [N, node_dim]
            edge_index: 边索引 [2, E]
            edge_attr: 边特征 [E, edge_dim]（可选）

        Returns:
            scores: 重要性评分 [N]，范围0-1
        """
        # 1. 特征嵌入
        x = self.embedding(vehicle_features)  # [N, hidden_dim]

        # 2. GNN编码（传播邻居信息）
        # 检查边索引是否为空
        if edge_index.shape[1] == 0:
            # 如果没有边，直接使用嵌入特征
            for i in range(len(self.gnn_layers)):
                x = self.layer_norms[i](x)
                x = F.relu(x)
        else:
            for i, gnn_layer in enumerate(self.gnn_layers):
                residual = x
                x = gnn_layer(x, edge_index)
                x = self.layer_norms[i](x + residual)  # 残差连接
                x = F.relu(x)

        # 3. 注意力评分（可选，两种评分方式）
        # 方式1：基于注意力权重
        attn_scores = self.attention_scorer(x)  # [N]

        # 方式2：基于MLP
        mlp_scores = self.score_head(x).squeeze(-1)  # [N]

        # 混合两种评分（可学习）
        scores = 0.5 * attn_scores + 0.5 * mlp_scores

        return scores  # [N], 范围0-1


class NeuralICVScorer:
    """
    纯神经网络ICV车辆评分器

    完全基于GNN的评分系统，无需规则评分
    """

    def __init__(
        self,
        neural_scorer: NeuralVehicleScorer,
        device: str = 'cuda'
    ):
        self.neural_scorer = neural_scorer.to(device)
        self.device = device
        self.graph_builder = VehicleGraphBuilder()

        # 训练模式标志
        self.is_training = False

    def eval(self):
        """设置为评估模式"""
        self.neural_scorer.eval()
        self.is_training = False

    def train(self):
        """设置为训练模式"""
        self.neural_scorer.train()
        self.is_training = True

    @torch.no_grad()
    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        rule_scores: Optional[Dict[str, float]] = None  # 保留参数以兼容接口，但不使用
    ) -> Dict[str, float]:
        """
        计算神经网络评分

        Args:
            vehicle_states: 车辆状态字典
            rule_scores: 规则评分字典（已弃用，仅保留以兼容接口）

        Returns:
            final_scores: 最终评分字典 {veh_id: score}
        """
        if len(vehicle_states) == 0:
            return {}

        vehicle_ids = list(vehicle_states.keys())

        # 1. 提取特征
        vehicle_features = []
        for veh_id in vehicle_ids:
            state = vehicle_states[veh_id]
            features = self._extract_features(state)
            vehicle_features.append(features)

        vehicle_features = torch.tensor(
            vehicle_features,
            dtype=torch.float32,
            device=self.device
        )

        # 2. 构建图
        edge_index, edge_attr = self.graph_builder.build_graph(
            vehicle_states,
            len(vehicle_ids)
        )
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        # 3. 神经网络评分
        self.neural_scorer.eval()
        neural_scores = self.neural_scorer(
            vehicle_features,
            edge_index,
            edge_attr
        )  # [N], 范围0-1

        neural_scores = neural_scores.cpu().numpy()

        # 4. 归一化到0-53范围（与原规则评分范围一致）
        final_scores = {}
        for i, veh_id in enumerate(vehicle_ids):
            final_scores[veh_id] = neural_scores[i] * 53.0

        return final_scores

    def _extract_features(self, state: Dict) -> List[float]:
        """
        从状态字典提取特征向量（9维）

        Args:
            state: 车辆状态字典

        Returns:
            features: 特征向量 [9]
        """
        return [
            state.get('s', 0.0) / 1000.0,          # 纵向位置（归一化）
            state.get('d', 0.0) / 10.0,            # 横向位置（归一化）
            state.get('vs', 0.0) / 30.0,           # 纵向速度（归一化）
            state.get('vd', 0.0) / 10.0,           # 横向速度（归一化）
            state.get('speed', 0.0) / 30.0,        # 总速度（归一化）
            state.get('acceleration', 0.0) / 3.0,  # 加速度（归一化）
            state.get('lane_index', 0.0) / 10.0,   # 车道索引（归一化）
            state.get('angle', 0.0) / 360.0,       # 航向角（归一化）
            1.0 if state.get('is_icv', False) else 0.0  # 是否是ICV
        ]

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前训练轮数
            loss: 当前损失
        """
        torch.save({
            'epoch': epoch,
            'neural_scorer': self.neural_scorer.state_dict(),
            'loss': loss,
        }, path)
        print(f"✅ 检查点已保存: {path}")

    def load_checkpoint(self, path: str) -> Dict:
        """
        加载检查点

        Args:
            path: 检查点路径

        Returns:
            检查点信息字典
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.neural_scorer.load_state_dict(checkpoint['neural_scorer'])
        print(f"✅ 检查点已加载: {path} (epoch={checkpoint.get('epoch', 'N/A')}, loss={checkpoint.get('loss', 'N/A')})")
        return checkpoint


def create_neural_icv_scorer(
    node_dim: int = 9,
    hidden_dim: int = 64,
    num_layers: int = 3,
    num_heads: int = 4,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda'
) -> NeuralICVScorer:
    """
    创建纯神经网络ICV评分器

    Args:
        node_dim: 节点特征维度
        hidden_dim: 隐藏层维度
        num_layers: GNN层数
        num_heads: 注意力头数
        checkpoint_path: 预训练权重路径（可选）
        device: 设备

    Returns:
        NeuralICVScorer实例
    """
    # 创建神经网络评分器
    neural_scorer = NeuralVehicleScorer(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )

    # 加载预训练权重（如果有）
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            neural_scorer.load_state_dict(checkpoint['neural_scorer'])
            print(f"✅ 加载预训练权重: {checkpoint_path}")
        except Exception as e:
            print(f"⚠️  加载权重失败: {e}，使用随机初始化")

    # 创建评分器
    scorer = NeuralICVScorer(
        neural_scorer=neural_scorer,
        device=device
    )

    return scorer
