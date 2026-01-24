"""
ICV GNN车辆评分器

使用图神经网络对车辆进行重要性评分，用于选择需要控制的ICV车辆。
评分基于车辆的状态、位置、速度以及对交通流的影响。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# 导入已有的GNN组件
from src.models.joint_icv_policy import RiskSensitiveGNN, RiskSensitiveGNNLayer


class ICVScoringNetwork(nn.Module):
    """
    ICV评分网络

    使用GNN编码车辆状态，然后预测每个车辆的重要性评分。
    """

    def __init__(
        self,
        node_dim: int = 9,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        interaction_radius: float = 0.15
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # GNN编码器
        self.gnn = RiskSensitiveGNN(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            interaction_radius=interaction_radius
        )

        # 评分头（预测每个车辆的重要性）
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出[0, 1]范围的评分
        )

    def forward(
        self,
        vehicle_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            vehicle_states: [B, N, node_dim] 车辆状态
            mask: [B, N] bool mask (True表示有效车辆)

        Returns:
            scores: [B, N] 车辆重要性评分 [0, 1]
        """
        # GNN编码
        gnn_outputs = self.gnn(vehicle_states)
        embeddings = gnn_outputs['embeddings']  # [B, N, hidden_dim]

        # 预测评分
        scores = self.scoring_head(embeddings)  # [B, N, 1]
        scores = scores.squeeze(-1)  # [B, N]

        # 应用mask（如果有）
        if mask is not None:
            scores = scores * mask.float()

        return scores


class ICVGNNScorer:
    """
    基于GNN的ICV车辆评分器

    使用神经网络评估车辆重要性，用于选择需要控制的车辆。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = 'cuda',
        checkpoint_path: Optional[str] = None
    ):
        """
        初始化评分器

        Args:
            config: 配置字典
            device: 设备
            checkpoint_path: 预训练权重路径（可选）
        """
        self.config = config
        self.device = device

        # 神经网络配置
        neural_config = config.get('neural_icv_scoring', {})
        self.node_dim = neural_config.get('node_dim', 9)
        self.hidden_dim = neural_config.get('hidden_dim', 128)
        self.num_layers = neural_config.get('num_layers', 3)
        self.dropout = neural_config.get('dropout', 0.1)
        self.interaction_radius = neural_config.get('interaction_radius', 0.15)

        # 创建网络
        self.network = ICVScoringNetwork(
            node_dim=self.node_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            interaction_radius=self.interaction_radius
        ).to(device)

        # 加载权重（如果有）
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # 评估模式
        self.network.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载预训练权重

        Args:
            checkpoint_path: 权重文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载网络权重
        if 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.network.load_state_dict(checkpoint)

        print(f"[ICVGNNScorer] 已加载权重: {checkpoint_path}")

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

        if len(all_vehicle_ids) == 0:
            return {}

        # 准备输入张量
        max_vehicles = min(len(all_vehicle_ids), 32)

        # 归一化车辆状态
        vehicle_features = np.zeros((1, max_vehicles, self.node_dim), dtype=np.float32)

        valid_mask = np.zeros((1, max_vehicles), dtype=np.float32)

        for i, veh_id in enumerate(all_vehicle_ids[:max_vehicles]):
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]

                # 提取并归一化特征
                vehicle_features[0, i, 0] = state.get('s', 0.0) / 1000.0       # 纵向位置
                vehicle_features[0, i, 1] = state.get('d', 0.0) / 10.0         # 横向位置
                vehicle_features[0, i, 2] = state.get('vs', 0.0) / 30.0        # 纵向速度
                vehicle_features[0, i, 3] = state.get('vd', 0.0) / 10.0        # 横向速度
                vehicle_features[0, i, 4] = state.get('speed', 0.0) / 30.0     # 速度
                vehicle_features[0, i, 5] = state.get('acceleration', 0.0) / 3.0  # 加速度
                vehicle_features[0, i, 6] = state.get('lane_index', 0.0) / 10.0  # 车道
                vehicle_features[0, i, 7] = state.get('angle', 0.0) / 360.0     # 角度
                vehicle_features[0, i, 8] = 1.0  # 占位符

                valid_mask[0, i] = 1.0

        # 转换为张量
        vehicle_tensor = torch.from_numpy(vehicle_features).to(self.device)
        mask_tensor = torch.from_numpy(valid_mask).to(self.device)

        # 推理
        with torch.no_grad():
            scores = self.network(vehicle_tensor, mask_tensor)  # [1, N]

        # 转换为字典
        scores_np = scores[0].cpu().numpy()

        scores_dict = {}
        for i, veh_id in enumerate(all_vehicle_ids[:max_vehicles]):
            if valid_mask[0, i] > 0:
                scores_dict[veh_id] = float(scores_np[i])
            else:
                scores_dict[veh_id] = 0.0

        return scores_dict

    def get_top_k_vehicles(
        self,
        vehicle_states: Dict[str, Dict],
        context: Dict,
        k: int,
        min_score: float = 0.0
    ) -> List[str]:
        """
        获取评分最高的K个车辆

        Args:
            vehicle_states: 车辆状态字典
            context: 上下文信息
            k: 返回的车辆数量
            min_score: 最低评分阈值

        Returns:
            top_k_vehicles: 按评分排序的车辆ID列表
        """
        scores = self.compute_scores(vehicle_states, context)

        # 按评分排序
        sorted_vehicles = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 过滤低于阈值的车辆
        filtered_vehicles = [
            (veh_id, score) for veh_id, score in sorted_vehicles
            if score >= min_score
        ]

        # 取前K个
        top_k_vehicles = [
            veh_id for veh_id, score in filtered_vehicles[:k]
        ]

        return top_k_vehicles


def create_icv_gnn_scorer(
    config: Dict[str, Any],
    device: str = 'cuda',
    checkpoint_path: Optional[str] = None
) -> ICVGNNScorer:
    """
    创建ICV GNN评分器的工厂函数

    Args:
        config: 配置字典
        device: 设备（'cuda'或'cpu'）
        checkpoint_path: 预训练权重路径（可选）

    Returns:
        scorer: ICVGNNScorer实例
    """
    scorer = ICVGNNScorer(
        config=config,
        device=device,
        checkpoint_path=checkpoint_path
    )
    return scorer


# 测试代码
if __name__ == '__main__':
    print("Testing ICVGNNScorer...")

    # 创建配置
    config = {
        'neural_icv_scoring': {
            'node_dim': 9,
            'hidden_dim': 128,
            'num_layers': 3,
            'dropout': 0.1,
            'interaction_radius': 0.15
        }
    }

    # 创建评分器
    scorer = create_icv_gnn_scorer(config, device='cpu')

    # 创建模拟数据
    vehicle_states = {}
    for i in range(10):
        veh_id = f'veh_{i}'
        vehicle_states[veh_id] = {
            's': 1000.0 + i * 50,
            'd': 0.0,
            'vs': 20.0,
            'vd': 0.0,
            'speed': 20.0,
            'acceleration': 0.0,
            'lane_index': 0,
            'angle': 0.0
        }

    context = {
        'all_vehicle_ids': list(vehicle_states.keys())
    }

    # 计算评分
    scores = scorer.compute_scores(vehicle_states, context)

    print(f"\nVehicle scores:")
    for veh_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {veh_id}: {score:.4f}")

    # 测试top-k选择
    top_k = scorer.get_top_k_vehicles(vehicle_states, context, k=3)
    print(f"\nTop 3 vehicles: {top_k}")

    print("\n✓ All tests passed!")
