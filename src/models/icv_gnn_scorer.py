"""
ICV评分GNN模型

专门用于车辆重要性评分的图神经网络架构。
使用轻量级图卷积层和特征融合网络，输出[0,1]范围的车辆影响力评分。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class LightweightGraphConvolution(nn.Module):
    """
    轻量级图卷积层

    使用简化的消息传递机制，避免复杂的GNN库依赖。
    专门为ICV评分任务优化。
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [B, N, in_features] 节点特征
            adj: [B, N, N] 邻接矩阵（行归一化）

        Returns:
            out: [B, N, out_features] 更新后的节点特征
        """
        # 消息传递：聚合邻居特征
        # adj @ x: 对每个节点，聚合其邻居的特征
        messages = torch.bmm(adj, x)  # [B, N, in_features]

        # 线性变换
        out = self.linear(messages)  # [B, N, out_features]

        # 归一化和激活
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class ICVGraphScorer(nn.Module):
    """
    ICV评分GNN模型

    核心特性：
    1. 使用图卷积层捕捉车辆交互
    2. 基于空间距离构建图结构
    3. 多层特征融合网络
    4. 输出[0,1]范围的车辆影响力评分
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
        self.num_layers = num_layers
        self.interaction_radius = interaction_radius

        # ========== 输入嵌入层 ==========
        self.input_embedding = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========== GNN层 ==========
        self.gnn_layers = nn.ModuleList([
            LightweightGraphConvolution(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # ========== 特征融合网络（评分头）==========
        self.score_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # ========== 初始化参数 ==========
        self.reset_parameters()

    def reset_parameters(self):
        """重置所有参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_adjacency_matrix(
        self,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        构建邻接矩阵（基于空间距离和车道信息）

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            adj: [B, N, N] 邻接矩阵（行归一化）
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # 提取位置和车道信息
        s = vehicle_states[:, :, 0]  # [B, N] 纵向位置
        d = vehicle_states[:, :, 1]  # [B, N] 横向位置
        lanes = vehicle_states[:, :, 6].long()  # [B, N] 车道索引

        # ========== 计算距离矩阵 ==========
        s_diff = s.unsqueeze(2) - s.unsqueeze(1)  # [B, N, N]
        d_diff = d.unsqueeze(2) - d.unsqueeze(1)  # [B, N, N]
        lane_diff = (lanes.unsqueeze(2) - lanes.unsqueeze(1)).abs()  # [B, N, N]

        # 空间距离（欧几里得距离）
        dist_sq = s_diff ** 2 + d_diff ** 2  # [B, N, N]
        dist = torch.sqrt(dist_sq + 1e-6)  # [B, N, N]

        # ========== 构建连接掩码 ==========
        # 1. 车道掩码：只连接同车道或相邻车道
        lane_mask = (lane_diff <= 1).float()  # [B, N, N]

        # 2. 距离掩码：只连接半径内的车辆
        dist_mask = (dist < self.interaction_radius).float()  # [B, N, N]

        # 3. 排除自连接
        not_self = 1.0 - torch.eye(N, device=device).unsqueeze(0)  # [B, N, N]

        # 4. 有效车辆掩码（排除padding的车辆，位置为0的车辆）
        valid_mask = (s > 0).float().unsqueeze(2)  # [B, N, 1]

        # ========== 组合掩码 ==========
        mask = lane_mask * dist_mask * not_self  # [B, N, N]
        mask = mask * valid_mask * valid_mask.transpose(1, 2)  # [B, N, N]

        # ========== 归一化 ==========
        # 行归一化：每行和为1（消息传递的权重）
        row_sum = mask.sum(dim=2, keepdim=True)  # [B, N, 1]
        adj = mask / (row_sum + 1e-6)  # [B, N, N]

        return adj

    def forward(
        self,
        vehicle_states: torch.Tensor,
        return_debug_info: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            vehicle_states: [batch_size, max_vehicles, 9] 车辆状态
            return_debug_info: 是否返回调试信息

        Returns:
            outputs: 包含以下字段的字典
                - scores: [B, N, 1] 车辆影响力评分（[0, 1]）
                - node_embeddings: [B, N, hidden_dim] 节点嵌入
                - adjacency_matrix: [B, N, N] 邻接矩阵（可选）
        """
        B, N, _ = vehicle_states.shape
        device = vehicle_states.device

        # ========== 1. 输入嵌入 ==========
        h = self.input_embedding(vehicle_states)  # [B, N, hidden_dim]

        # ========== 2. 构建邻接矩阵 ==========
        adj = self._build_adjacency_matrix(vehicle_states)  # [B, N, N]

        # ========== 3. GNN层（消息传递）==========
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, adj)  # [B, N, hidden_dim]

        # ========== 4. 计算评分 ==========
        scores = self.score_fusion(h)  # [B, N, 1]

        # 使用sigmoid归一化到[0, 1]
        scores = torch.sigmoid(scores)  # [B, N, 1]

        # ========== 5. 准备输出 ==========
        outputs = {
            'scores': scores,  # [B, N, 1]
            'node_embeddings': h,  # [B, N, hidden_dim]
        }

        if return_debug_info:
            outputs['adjacency_matrix'] = adj  # [B, N, N]
            outputs['edge_count'] = adj.sum(dim=(1, 2))  # [B] 每个样本的边数

        return outputs

    def compute_scores(
        self,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        计算车辆评分（简化接口）

        Args:
            vehicle_states: [B, N, 9] 车辆状态

        Returns:
            scores: [B, N] 车辆评分（[0, 1]）
        """
        outputs = self.forward(vehicle_states, return_debug_info=False)
        scores = outputs['scores'].squeeze(-1)  # [B, N]
        return scores


class ICVGNNScorer:
    """
    ICV GNN评分器包装类

    提供与规则评分器一致的接口，方便统一调用。
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None
    ):
        """
        初始化ICV GNN评分器

        Args:
            config: 配置字典
            device: 设备（'cuda'或'cpu'）
            checkpoint_path: 预训练权重路径
        """
        self.config = config or {}
        self.device = device

        # 从配置中读取模型参数
        neural_config = self.config.get('neural_icv_scoring', {})
        model_config = self.config.get('model', {}).get('gnn', {})

        self.node_dim = neural_config.get('node_dim', 9)
        self.hidden_dim = neural_config.get('hidden_dim', 64)
        self.num_layers = neural_config.get('num_layers', 3)
        self.dropout = neural_config.get('dropout', 0.1)
        self.interaction_radius = neural_config.get('interaction_radius', 0.15)

        # 创建模型
        self.model = ICVGraphScorer(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            interaction_radius=self.interaction_radius
        ).to(self.device)

        # 加载预训练权重（如果有）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # 设置为评估模式
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """加载预训练权重"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[ICVGNNScorer] 成功加载权重: {checkpoint_path}")
        except Exception as e:
            print(f"[ICVGNNScorer] 警告：无法加载权重 {checkpoint_path}: {e}")
            print("[ICVGNNScorer] 将使用随机初始化的模型")

    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        context: Dict
    ) -> Dict[str, float]:
        """
        计算所有车辆的评分

        Args:
            vehicle_states: {veh_id: {s, d, vs, vd, speed, accel, lane, angle, ...}}
            context: {traci_lib, all_vehicle_ids, ...}

        Returns:
            scores: {veh_id: score} 评分范围 [0, 1]
        """
        all_vehicle_ids = context.get('all_vehicle_ids', [])

        if not all_vehicle_ids:
            return {}

        # 准备输入张量
        vehicle_tensor = self._prepare_vehicle_tensor(
            vehicle_states,
            all_vehicle_ids
        )  # [1, N, 9]

        # 移动到设备
        vehicle_tensor = vehicle_tensor.to(self.device)

        # 前向传播
        with torch.no_grad():
            outputs = self.model(vehicle_tensor, return_debug_info=False)
            scores = outputs['scores'].cpu().numpy()  # [1, N, 1] -> numpy

        # 转换为字典
        scores_dict = {}
        for i, veh_id in enumerate(all_vehicle_ids):
            if i < scores.shape[1]:
                scores_dict[veh_id] = float(scores[0, i, 0])

        return scores_dict

    def _prepare_vehicle_tensor(
        self,
        vehicle_states: Dict[str, Dict],
        all_vehicle_ids: List[str],
        max_vehicles: Optional[int] = None
    ) -> torch.Tensor:
        """
        将车辆状态字典转换为张量

        Args:
            vehicle_states: {veh_id: {s, d, vs, vd, speed, ...}}
            all_vehicle_ids: 车辆ID列表
            max_vehicles: 最大车辆数（None表示使用实际车辆数）

        Returns:
            vehicle_tensor: [1, N, 9] 车辆状态张量
        """
        if max_vehicles is None:
            max_vehicles = len(all_vehicle_ids)

        # 初始化张量
        vehicle_tensor = np.zeros((1, max_vehicles, 9), dtype=np.float32)

        for i, veh_id in enumerate(all_vehicle_ids):
            if i >= max_vehicles:
                break

            if veh_id not in vehicle_states:
                continue

            state = vehicle_states[veh_id]

            # 提取9维特征
            # [s, d, vs, vd, speed, accel, lane, angle, in_bottleneck]
            vehicle_tensor[0, i, 0] = state.get('s', 0.0)
            vehicle_tensor[0, i, 1] = state.get('d', 0.0)
            vehicle_tensor[0, i, 2] = state.get('vs', 0.0)
            vehicle_tensor[0, i, 3] = state.get('vd', 0.0)
            vehicle_tensor[0, i, 4] = state.get('speed', 0.0)
            vehicle_tensor[0, i, 5] = state.get('acceleration', 0.0)
            vehicle_tensor[0, i, 6] = state.get('lane_index', 0)
            vehicle_tensor[0, i, 7] = state.get('angle', 0.0)
            vehicle_tensor[0, i, 8] = 1.0 if state.get('in_bottleneck', False) else 0.0

        return torch.from_numpy(vehicle_tensor)

    def train_mode(self):
        """切换到训练模式"""
        self.model.train()

    def eval_mode(self):
        """切换到评估模式"""
        self.model.eval()

    def save_checkpoint(self, path: str, epoch: int = 0, optimizer=None):
        """保存模型权重"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, path)
        print(f"[ICVGNNScorer] 保存权重到: {path}")


def create_icv_gnn_scorer(
    config: Optional[Dict] = None,
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None
) -> ICVGNNScorer:
    """
    创建ICV GNN评分器的工厂函数

    Args:
        config: 配置字典
        device: 设备
        checkpoint_path: 预训练权重路径

    Returns:
        scorer: ICVGNNScorer实例
    """
    scorer = ICVGNNScorer(
        config=config,
        device=device,
        checkpoint_path=checkpoint_path
    )
    return scorer
