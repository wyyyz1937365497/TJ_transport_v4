"""
Joint ICV Policy组件 - 为向后兼容提供的组件

这些组件被SimplifiedICVPolicy使用，包括：
- RiskSensitiveGNN: 风险感知图神经网络
- RiskSensitiveGNNLayer: GNN层
- ValueHead: 价值头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class RiskSensitiveGNNLayer(nn.Module):
    """
    风险感知GNN层
    
    对每个车辆节点聚合其邻居信息，考虑距离和相对速度的风险权重
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, interaction_radius: float = 0.15):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.interaction_radius = interaction_radius
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),  # +2 for distance and relative speed
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, node_features: torch.Tensor, vehicle_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [B, N, hidden_dim] 节点特征
            vehicle_states: [B, N, 9] 车辆状态 (s, d, vs, vd, speed, acceleration, lane_index, angle, veh_length)
            
        Returns:
            updated_features: [B, N, hidden_dim] 更新后的节点特征
        """
        B, N, D = node_features.shape
        
        # 提取位置和速度
        positions = vehicle_states[:, :, 0:2]  # [B, N, 2] (s, d)
        speeds = vehicle_states[:, :, 4:5]     # [B, N, 1]
        
        # 计算成对距离 [B, N, N]
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
        distances = torch.norm(pos_diff, dim=-1)  # [B, N, N]
        
        # 计算相对速度 [B, N, N]
        speed_diff = speeds.unsqueeze(2) - speeds.unsqueeze(1)  # [B, N, N, 1]
        speed_diff = speed_diff.squeeze(-1)  # [B, N, N]
        
        # 构建邻接矩阵（基于距离阈值）
        # 归一化距离到[0, 1]范围
        max_dist = 2000.0  # 最大感知距离 (m)
        norm_distances = torch.clamp(distances / max_dist, 0.0, 1.0)
        
        # 只考虑距离小于阈值的邻居
        adjacency = (norm_distances < self.interaction_radius).float()  # [B, N, N]
        adjacency = adjacency * (1.0 - torch.eye(N, device=node_features.device).unsqueeze(0))  # 移除自连接
        
        # Message passing
        aggregated = []
        for b in range(B):
            batch_aggregated = []
            for i in range(N):
                # 获取邻居
                neighbors = adjacency[b, i]  # [N]
                neighbor_indices = torch.where(neighbors > 0)[0]
                
                if len(neighbor_indices) == 0:
                    # 没有邻居，保持当前特征
                    batch_aggregated.append(torch.zeros_like(node_features[b, i]))
                    continue
                
                # 邻居特征
                neighbor_features = node_features[b, neighbor_indices]  # [num_neighbors, D]
                
                # 当前节点特征重复
                current_features = node_features[b, i:i+1].expand(len(neighbor_indices), -1)  # [num_neighbors, D]
                
                # 边特征（距离和相对速度）
                edge_distances = norm_distances[b, i, neighbor_indices].unsqueeze(-1)  # [num_neighbors, 1]
                edge_speeds = speed_diff[b, i, neighbor_indices].unsqueeze(-1)  # [num_neighbors, 1]
                
                # 构建消息
                messages = torch.cat([
                    current_features,
                    neighbor_features,
                    edge_distances,
                    edge_speeds
                ], dim=-1)  # [num_neighbors, 2*D + 2]
                
                # 通过message网络
                messages = self.message_net(messages)  # [num_neighbors, D]
                
                # 聚合（平均）
                aggregated_msg = messages.mean(dim=0)  # [D]
                batch_aggregated.append(aggregated_msg)
            
            batch_aggregated = torch.stack(batch_aggregated, dim=0)  # [N, D]
            aggregated.append(batch_aggregated)
        
        aggregated = torch.stack(aggregated, dim=0)  # [B, N, D]
        
        # 更新节点特征
        combined = torch.cat([node_features, aggregated], dim=-1)  # [B, N, 2*D]
        updated_features = self.update_gate(combined)  # [B, N, D]
        
        # 残差连接
        output = node_features + updated_features
        
        return output


class RiskSensitiveGNN(nn.Module):
    """
    风险感知图神经网络
    
    多层GNN编码器，用于学习车辆之间的交互关系
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
        self.num_layers = num_layers
        
        # 节点特征编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            RiskSensitiveGNNLayer(
                hidden_dim=hidden_dim,
                dropout=dropout,
                interaction_radius=interaction_radius
            )
            for _ in range(num_layers)
        ])
        
    def forward(self, vehicle_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            vehicle_states: [B, N, node_dim] 车辆状态
            
        Returns:
            outputs: {
                'embeddings': [B, N, hidden_dim] 节点嵌入
            }
        """
        # 编码节点特征
        embeddings = self.node_encoder(vehicle_states)  # [B, N, hidden_dim]
        
        # 通过GNN层
        for layer in self.gnn_layers:
            embeddings = layer(embeddings, vehicle_states)
        
        return {
            'embeddings': embeddings
        }


class ValueHead(nn.Module):
    """
    价值头 - 估计状态价值
    
    用于PPO训练或批评者网络
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [B, N, hidden_dim] 节点嵌入
            
        Returns:
            value: [B, 1] 状态价值
        """
        # 池化所有车辆的嵌入
        pooled = embeddings.mean(dim=1)  # [B, hidden_dim]
        
        # 预测价值
        value = self.value_net(pooled)  # [B, 1]
        
        return value


# 测试代码
if __name__ == '__main__':
    # 测试RiskSensitiveGNN
    print("Testing RiskSensitiveGNN...")
    
    B, N, node_dim = 2, 32, 9
    vehicle_states = torch.randn(B, N, node_dim)
    
    gnn = RiskSensitiveGNN(
        node_dim=node_dim,
        hidden_dim=128,
        num_layers=3
    )
    
    outputs = gnn(vehicle_states)
    print(f"Embeddings shape: {outputs['embeddings'].shape}")
    
    # 测试ValueHead
    print("\nTesting ValueHead...")
    
    value_head = ValueHead(hidden_dim=128)
    value = value_head(outputs['embeddings'])
    print(f"Value shape: {value.shape}")
    
    print("\n✓ All tests passed!")
