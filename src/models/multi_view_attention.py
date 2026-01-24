"""
Multi-View Graph Attention机制

实现3视角注意力机制，用于捕捉车辆之间的多种交互关系：
1. 空间视角：基于位置距离的注意力
2. 速度视角：基于速度相似性的注意力  
3. 风险视角：基于碰撞风险的注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiViewGraphAttention(nn.Module):
    """
    多视角图注意力机制
    
    整合3种不同视角的注意力：
    - 空间注意力：捕捉位置关系
    - 速度注意力：捕捉速度协同
    - 风险注意力：捕捉安全关系
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
        
        # 空间视角
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 速度视角
        self.velocity_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 风险视角
        self.risk_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 视角融合
        self.view_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
    def _compute_spatial_mask(
        self,
        vehicle_states: torch.Tensor,
        threshold: float = 0.2
    ) -> torch.Tensor:
        """
        计算空间注意力mask（基于距离）
        
        Args:
            vehicle_states: [B, N, 9] 车辆状态
            threshold: 距离阈值（归一化）
            
        Returns:
            mask: [B, N, N] 注意力mask (True表示被mask掉)
        """
        B, N, _ = vehicle_states.shape
        
        # 提取位置 [B, N, 2] (s, d)
        positions = vehicle_states[:, :, 0:2]
        
        # 计算成对距离
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
        distances = torch.norm(pos_diff, dim=-1)  # [B, N, N]
        
        # 归一化距离
        max_dist = 2000.0  # 最大感知距离 (m)
        norm_distances = torch.clamp(distances / max_dist, 0.0, 1.0)
        
        # 创建mask：距离大于阈值的被mask掉
        mask = norm_distances > threshold
        
        return mask
    
    def _compute_velocity_mask(
        self,
        vehicle_states: torch.Tensor,
        threshold: float = 10.0
    ) -> torch.Tensor:
        """
        计算速度注意力mask（基于相对速度）
        
        Args:
            vehicle_states: [B, N, 9] 车辆状态
            threshold: 速度差异阈值 (m/s)
            
        Returns:
            mask: [B, N, N] 注意力mask
        """
        B, N, _ = vehicle_states.shape
        
        # 提取速度 [B, N, 1]
        speeds = vehicle_states[:, :, 4:5]
        
        # 计算相对速度
        speed_diff = torch.abs(speeds.unsqueeze(2) - speeds.unsqueeze(1))  # [B, N, N, 1]
        speed_diff = speed_diff.squeeze(-1)  # [B, N, N]
        
        # 创建mask：速度差异大于阈值的被mask掉
        mask = speed_diff > threshold
        
        return mask
    
    def _compute_risk_mask(
        self,
        vehicle_states: torch.Tensor,
        ttc_threshold: float = 5.0
    ) -> torch.Tensor:
        """
        计算风险注意力mask（基于TTC - Time To Collision）
        
        Args:
            vehicle_states: [B, N, 9] 车辆状态
            ttc_threshold: TTC阈值 (秒)
            
        Returns:
            mask: [B, N, N] 注意力mask
        """
        B, N, _ = vehicle_states.shape
        
        # 提取位置和速度
        positions = vehicle_states[:, :, 0:2]  # [B, N, 2] (s, d)
        speeds = vehicle_states[:, :, 4:5]     # [B, N, 1]
        
        # 计算相对位置和相对速度
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [B, N, N, 2]
        distances = torch.norm(pos_diff, dim=-1)  # [B, N, N]
        
        speed_diff = speeds.unsqueeze(2) - speeds.unsqueeze(1)  # [B, N, N, 1]
        speed_diff = speed_diff.squeeze(-1)  # [B, N, N]
        
        # 计算TTC
        # TTC = distance / relative_speed (只考虑接近的情况)
        ttc = torch.where(
            speed_diff > 0.1,  # 接近
            distances / (speed_diff + 1e-6),
            torch.full_like(distances, float('inf'))
        )
        
        # 创建mask：TTC大于阈值的被mask掉（低风险）
        mask = ttc > ttc_threshold
        
        return mask
    
    def forward(
        self,
        embeddings: torch.Tensor,
        vehicle_states: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            embeddings: [B, N, hidden_dim] GNN嵌入
            vehicle_states: [B, N, 9] 车辆状态
            
        Returns:
            output: [B, N, hidden_dim] 融合后的嵌入
        """
        B, N, D = embeddings.shape
        
        # 计算不同视角的注意力mask
        spatial_mask = self._compute_spatial_mask(vehicle_states)  # [B, N, N]
        velocity_mask = self._compute_velocity_mask(vehicle_states)  # [B, N, N]
        risk_mask = self._compute_risk_mask(vehicle_states)  # [B, N, N]
        
        # 空间视角注意力
        spatial_output, _ = self.spatial_attn(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            attn_mask=spatial_mask
        )  # [B, N, D]
        
        # 速度视角注意力
        velocity_output, _ = self.velocity_attn(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            attn_mask=velocity_mask
        )  # [B, N, D]
        
        # 风险视角注意力
        risk_output, _ = self.risk_attn(
            query=embeddings,
            key=embeddings,
            value=embeddings,
            attn_mask=risk_mask
        )  # [B, N, D]
        
        # 融合3个视角
        combined = torch.cat([spatial_output, velocity_output, risk_output], dim=-1)  # [B, N, 3*D]
        fused = self.view_fusion(combined)  # [B, N, D]
        
        # 输出投影 + 残差连接
        output = self.output_proj(fused) + embeddings  # [B, N, D]
        
        return output


# 测试代码
if __name__ == '__main__':
    print("Testing MultiViewGraphAttention...")
    
    B, N, hidden_dim = 2, 32, 128
    
    # 创建模块
    multi_view_attn = MultiViewGraphAttention(
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=0.1
    )
    
    # 创建测试数据
    embeddings = torch.randn(B, N, hidden_dim)
    vehicle_states = torch.randn(B, N, 9)
    
    # 前向传播
    output = multi_view_attn(embeddings, vehicle_states)
    
    print(f"Input shape: {embeddings.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == embeddings.shape, "Output shape mismatch!"
    
    print("\n✓ Test passed!")
