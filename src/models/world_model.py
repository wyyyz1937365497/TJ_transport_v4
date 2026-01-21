"""
世界模型（World Model）- RSSM (Recurrent State Space Model)

功能：预测未来交通状态演化
- 双头预测：流演化 + 风险演化
- 基于GRU的时序编码
- 端到端可训练

参考文献：
- Ha & Schmidhuber (2018) "World Models"
- Hafner et al. (2019) "Dreamer: Reinforcement Learning with Unsupervised World Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class FlowDecoder(nn.Module):
    """
    流解码器：预测交通流演化

    输入：潜在状态 z_flow
    输出：未来车辆状态（位置、速度、加速度）
    """

    def __init__(self, latent_dim: int = 64, num_vehicles: int = 32):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_vehicles = num_vehicles

        # 解码器：从潜在状态重构车辆状态
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_vehicles * 9)  # [B, N*9]
        )

    def forward(self, z_flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_flow: [B, latent_dim] 流潜在状态

        Returns:
            pred_states: [B, N, 9] 预测的车辆状态
        """
        B = z_flow.size(0)

        # 解码
        pred_flat = self.decoder(z_flow)  # [B, N*9]

        # 重塑
        pred_states = pred_flat.view(B, self.num_vehicles, 9)  # [B, N, 9]

        return pred_states


class RiskDecoder(nn.Module):
    """
    风险解码器：预测碰撞风险

    输入：潜在状态 z_risk
    输出：每辆车的碰撞风险概率
    """

    def __init__(self, latent_dim: int = 64, num_vehicles: int = 32):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_vehicles = num_vehicles

        # 解码器：从潜在状态预测风险
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_vehicles)  # [B, N]
        )

    def forward(self, z_risk: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_risk: [B, latent_dim] 风险潜在状态

        Returns:
            risk_prob: [B, N] 每辆车的碰撞风险概率 (sigmoid)
        """
        # 解码
        risk_logits = self.decoder(z_risk)  # [B, N]

        # Sigmoid激活（概率）
        risk_prob = torch.sigmoid(risk_logits)  # [B, N]

        return risk_prob


class RSSMEncoder(nn.Module):
    """
    RSSM编码器：将车辆嵌入编码为潜在状态

    使用2层GRU编码器，输出双头潜在状态
    """

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 64, num_layers: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # GRU编码器
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 双头投影：流头 + 风险头
        self.flow_projection = nn.Linear(latent_dim, latent_dim)
        self.risk_projection = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        prev_hidden: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: [B, N, D] 车辆嵌入（平均池化后作为输入）
            prev_hidden: [num_layers, B, latent_dim] 前一时刻的隐状态

        Returns:
            z_flow: [B, latent_dim] 流潜在状态
            z_risk: [B, latent_dim] 风险潜在状态
            hidden: [num_layers, B, latent_dim] 当前隐状态
        """
        B = embeddings.size(0)

        # 全局池化（将车辆嵌入聚合为单个向量）
        global_emb = embeddings.mean(dim=1)  # [B, D]

        # 添加序列维度（GRU需要 [B, T, D]）
        global_emb = global_emb.unsqueeze(1)  # [B, 1, D]

        # 初始化隐状态
        if prev_hidden is None:
            num_layers = self.num_layers
            device = embeddings.device
            prev_hidden = torch.zeros(num_layers, B, self.latent_dim, device=device)

        # GRU前向传播
        _, hidden = self.gru(global_emb, prev_hidden)  # hidden: [num_layers, B, latent_dim]

        # 提取最后一层的隐状态
        last_hidden = hidden[-1]  # [B, latent_dim]

        # 双头投影
        z_flow = self.flow_projection(last_hidden)  # [B, latent_dim]
        z_risk = self.risk_projection(last_hidden)  # [B, latent_dim]

        return z_flow, z_risk, hidden


class WorldModel(nn.Module):
    """
    世界模型（World Model）- 完整的RSSM实现

    核心功能：
    1. 编码当前状态为潜在状态（双头：流 + 风险）
    2. 预测未来车辆状态（流解码）
    3. 预测碰撞风险（风险解码）

    训练方式：
    - 监督学习：使用IDM人工驾驶轨迹
    - Loss = MSE(预测状态, 真实状态) + BCE(预测风险, 真实风险)

    使用方式：
    - Stage 1训练：使用世界观察者收集的数据训练
    - Stage 2/3：预训练的世界模型作为辅助任务
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        num_vehicles: int = 32,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 输入嵌入维度
            latent_dim: 潜在状态维度
            num_layers: GRU层数
            num_vehicles: 最大车辆数
            dropout: Dropout率
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_vehicles = num_vehicles

        # RSSM编码器
        self.encoder = RSSMEncoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        )

        # 流解码器
        self.flow_decoder = FlowDecoder(
            latent_dim=latent_dim,
            num_vehicles=num_vehicles
        )

        # 风险解码器
        self.risk_decoder = RiskDecoder(
            latent_dim=latent_dim,
            num_vehicles=num_vehicles
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        prev_hidden: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            embeddings: [B, N, D] 车辆嵌入
            prev_hidden: [num_layers, B, latent_dim] 前一时刻的隐状态

        Returns:
            Dict:
                'z_flow': [B, latent_dim] 流潜在状态
                'z_risk': [B, latent_dim] 风险潜在状态
                'pred_next_states': [B, N, 9] 预测的下一时刻车辆状态
                'risk_prob': [B, N] 碰撞风险概率
                'hidden': [num_layers, B, latent_dim] 当前隐状态
        """
        # 1. RSSM编码
        z_flow, z_risk, hidden = self.encoder(embeddings, prev_hidden)

        # 2. 流解码（预测未来状态）
        pred_next_states = self.flow_decoder(z_flow)  # [B, N, 9]

        # 3. 风险解码（预测碰撞风险）
        risk_prob = self.risk_decoder(z_risk)  # [B, N]

        return {
            'z_flow': z_flow,
            'z_risk': z_risk,
            'pred_next_states': pred_next_states,
            'risk_prob': risk_prob,
            'hidden': hidden
        }

    def compute_loss(
        self,
        pred_next_states: torch.Tensor,
        true_next_states: torch.Tensor,
        pred_risk_prob: torch.Tensor,
        true_risk_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算世界模型损失

        Args:
            pred_next_states: [B, N, 9] 预测的下一时刻状态
            true_next_states: [B, N, 9] 真实的下一时刻状态
            pred_risk_prob: [B, N] 预测的碰撞风险概率
            true_risk_labels: [B, N] 真实的碰撞风险标签（0或1）

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 1. 流预测损失（MSE）
        flow_loss = F.mse_loss(pred_next_states, true_next_states)

        # 2. 风险预测损失（BCE）
        # 使用二元交叉熵
        risk_loss = F.binary_cross_entropy(
            pred_risk_prob,
            true_risk_labels.float()
        )

        # 3. 总损失（加权组合）
        total_loss = flow_loss + 0.5 * risk_loss

        loss_dict = {
            'total': total_loss,
            'flow': flow_loss,
            'risk': risk_loss
        }

        return total_loss, loss_dict

    def initialize_hidden(self, batch_size: int, device: str = 'cuda') -> torch.Tensor:
        """
        初始化GRU隐状态

        Args:
            batch_size: 批量大小
            device: 设备

        Returns:
            hidden: [num_layers, B, latent_dim] 初始隐状态
        """
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.latent_dim,
            device=device
        )

    def get_trajectory_predictions(
        self,
        embeddings_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        获取轨迹预测（多步预测）

        Args:
            embeddings_sequence: [T, B, N, D] 嵌入序列

        Returns:
            Dict:
                'pred_states': [T, B, N, 9] 预测的状态序列
                'risk_probs': [T, B, N] 风险概率序列
        """
        T, B, N, D = embeddings_sequence.shape
        device = embeddings_sequence.device

        # 初始化隐状态
        hidden = self.initialize_hidden(B, device)

        # 存储预测结果
        pred_states_list = []
        risk_probs_list = []

        # 逐步预测
        for t in range(T):
            emb_t = embeddings_sequence[t]  # [B, N, D]

            # 前向传播
            outputs = self.forward(emb_t, hidden)

            # 更新隐状态
            hidden = outputs['hidden']

            # 存储预测
            pred_states_list.append(outputs['pred_next_states'])  # [B, N, 9]
            risk_probs_list.append(outputs['risk_prob'])  # [B, N]

        # 堆叠为序列
        pred_states = torch.stack(pred_states_list, dim=0)  # [T, B, N, 9]
        risk_probs = torch.stack(risk_probs_list, dim=0)  # [T, B, N]

        return {
            'pred_states': pred_states,
            'risk_probs': risk_probs
        }


def create_world_model(
    hidden_dim: int = 64,
    latent_dim: int = 64,
    num_layers: int = 2,
    num_vehicles: int = 32,
    device: str = 'cuda'
) -> WorldModel:
    """
    创建世界模型的工厂函数

    Args:
        hidden_dim: 输入嵌入维度
        latent_dim: 潜在状态维度
        num_layers: GRU层数
        num_vehicles: 最大车辆数
        device: 设备

    Returns:
        world_model: WorldModel实例
    """
    model = WorldModel(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        num_vehicles=num_vehicles
    )

    return model.to(device)
