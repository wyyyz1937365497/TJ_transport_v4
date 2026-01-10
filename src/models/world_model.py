"""
Progressive World Model - 渐进式世界模型
功能：预测交通系统的未来状态和风险演化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class ProgressiveWorldModel(nn.Module):
    """
    渐进式世界模型

    两阶段训练：
    - Phase 1: 基础动力学学习（预测下一时刻状态）
    - Phase 2: 风险演化学习（预测未来5步状态+冲突概率）

    特性：
    - LSTM时序建模
    - 多步预测解码器
    - 辅助冲突分类器
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        future_steps: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.current_phase = 1

        # 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 192),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(192),
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Phase 1: 基础动力学LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0.0
        )

        # Phase 2: 风险演化解码器（多步）
        self.risk_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 192),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(192),
                nn.Linear(192, input_dim + 1)  # 状态 + 冲突概率
            ) for _ in range(future_steps)
        ])

        # 辅助冲突分类器
        self.conflict_classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 速度预测头（辅助任务）
        self.speed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 位置预测头（辅助任务）
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # x, y
        )

    def set_phase(self, phase: int):
        """设置训练阶段"""
        if phase not in [1, 2]:
            raise ValueError(f"Invalid phase: {phase}. Must be 1 or 2.")
        self.current_phase = phase

    def forward(
        self,
        gnn_embedding: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            gnn_embedding: [N, input_dim] GNN输出嵌入
            hidden_state: LSTM隐藏状态（可选）

        Returns:
            dict: 包含预测结果的字典
        """
        batch_size = gnn_embedding.size(0)

        # 1. 共享编码
        encoded = self.shared_encoder(gnn_embedding)  # [N, hidden_dim]

        if self.current_phase == 1:
            # Phase 1: 基础动力学预测
            return self._phase1_forward(encoded, hidden_state)
        else:
            # Phase 2: 风险演化预测
            return self._phase2_forward(encoded, hidden_state)

    def _phase1_forward(
        self,
        encoded: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1前向传播：预测下一时刻状态
        """
        # LSTM处理
        lstm_input = encoded.unsqueeze(1)  # [N, 1, hidden_dim]

        if hidden_state is not None:
            lstm_output, hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, hidden_state = self.lstm(lstm_input)

        lstm_output = lstm_output.squeeze(1)  # [N, hidden_dim]

        # 预测下一状态
        next_state_pred = self.risk_decoders[0](lstm_output)  # [N, input_dim + 1]
        next_state = next_state_pred[:, :-1]  # [N, input_dim]
        conflict_prob = next_state_pred[:, -1:]  # [N, 1]

        # 辅助预测
        speed_pred = self.speed_predictor(lstm_output)  # [N, 1]
        position_pred = self.position_predictor(lstm_output)  # [N, 2]

        return {
            'next_state': next_state,
            'conflict_prob': conflict_prob,
            'speed_pred': speed_pred,
            'position_pred': position_pred,
            'hidden_state': hidden_state,
            'encoded_features': lstm_output
        }

    def _phase2_forward(
        self,
        encoded: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 2前向传播：预测未来5步状态
        """
        predictions = []
        conflict_probs = []

        current_encoded = encoded

        # 自回归多步预测
        for t in range(self.future_steps):
            # 添加时间步嵌入
            time_embedding = 0.1 * t / self.future_steps
            time_input = current_encoded + time_embedding * torch.ones_like(current_encoded)

            # 预测
            pred = self.risk_decoders[t](time_input)  # [N, input_dim + 1]

            state_pred = pred[:, :-1]  # [N, input_dim]
            conflict_pred = pred[:, -1:]  # [N, 1]

            predictions.append(state_pred.unsqueeze(1))  # [N, 1, input_dim]
            conflict_probs.append(conflict_pred.unsqueeze(1))  # [N, 1, 1]

            # 更新当前状态（用于下一步预测）
            current_encoded = self.shared_encoder(state_pred)

        # 堆叠所有时间步
        future_states = torch.cat(predictions, dim=1)  # [N, future_steps, input_dim]
        future_conflicts = torch.cat(conflict_probs, dim=1)  # [N, future_steps, 1]

        return {
            'future_states': future_states,
            'future_conflicts': future_conflicts,
            'encoded_features': encoded
        }

    def compute_uncertainty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        计算预测不确定性

        Args:
            predictions: [N, ...] 预测值
            targets: [N, ...] 真实值
            reduction: 'mean', 'sum', 或 'none'

        Returns:
            uncertainty: 不确定性得分
        """
        mse = F.mse_loss(predictions, targets, reduction='none')

        if reduction == 'mean':
            return mse.mean()
        elif reduction == 'sum':
            return mse.sum()
        else:
            return mse

    def predict_conflicts(
        self,
        gnn_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        预测冲突概率（辅助任务）

        Args:
            gnn_embedding: [N, input_dim]

        Returns:
            conflict_probs: [N, 1]
        """
        conflict_probs = self.conflict_classifier(gnn_embedding)
        return conflict_probs


class WorldModelLoss(nn.Module):
    """
    世界模型损失函数

    组合多个损失项：
    - 状态预测损失
    - 冲突分类损失
    - 速度预测损失
    - 位置预测损失
    """

    def __init__(
        self,
        state_weight: float = 1.0,
        conflict_weight: float = 0.5,
        speed_weight: float = 0.3,
        position_weight: float = 0.2,
        use_phase2: bool = False
    ):
        super().__init__()

        self.state_weight = state_weight
        self.conflict_weight = conflict_weight
        self.speed_weight = speed_weight
        self.position_weight = position_weight
        self.use_phase2 = use_phase2

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失

        Args:
            predictions: 模型预测
            targets: 真实目标

        Returns:
            total_loss, loss_dict
        """
        losses = {}
        total_loss = 0.0

        if self.use_phase2:
            # Phase 2: 多步预测损失
            pred_states = predictions['future_states']  # [N, T, D]
            target_states = targets['future_states']  # [N, T, D]

            # 状态损失（所有时间步）
            state_loss = self.mse_loss(pred_states, target_states)
            losses['state_loss'] = state_loss.item()
            total_loss += self.state_weight * state_loss

            # 冲突损失
            if 'future_conflicts' in predictions and 'future_conflicts' in targets:
                pred_conflicts = predictions['future_conflicts']  # [N, T, 1]
                target_conflicts = targets['future_conflicts']  # [N, T, 1]
                conflict_loss = self.bce_loss(pred_conflicts, target_conflicts)
                losses['conflict_loss'] = conflict_loss.item()
                total_loss += self.conflict_weight * conflict_loss

        else:
            # Phase 1: 单步预测损失
            if 'next_state' in predictions and 'next_state' in targets:
                pred_state = predictions['next_state']
                target_state = targets['next_state']
                state_loss = self.mse_loss(pred_state, target_state)
                losses['state_loss'] = state_loss.item()
                total_loss += self.state_weight * state_loss

            # 速度预测损失
            if 'speed_pred' in predictions and 'speed' in targets:
                pred_speed = predictions['speed_pred']
                target_speed = targets['speed']
                speed_loss = self.mse_loss(pred_speed, target_speed)
                losses['speed_loss'] = speed_loss.item()
                total_loss += self.speed_weight * speed_loss

            # 位置预测损失
            if 'position_pred' in predictions and 'position' in targets:
                pred_pos = predictions['position_pred']
                target_pos = targets['position']
                position_loss = self.mse_loss(pred_pos, target_pos)
                losses['position_loss'] = position_loss.item()
                total_loss += self.position_weight * position_loss

            # 冲突分类损失
            if 'conflict_prob' in predictions and 'conflict' in targets:
                pred_conflict = predictions['conflict_prob']
                target_conflict = targets['conflict']
                conflict_loss = self.bce_loss(pred_conflict, target_conflict)
                losses['conflict_loss'] = conflict_loss.item()
                total_loss += self.conflict_weight * conflict_loss

        return total_loss, losses


def create_sequence_targets(
    trajectories: Dict[str, Dict],
    future_steps: int = 5
) -> List[Dict[str, torch.Tensor]]:
    """
    从轨迹数据创建序列训练目标

    Args:
        trajectories: 轨迹字典
        future_steps: 未来预测步数

    Returns:
        样本列表
    """
    samples = []

    for veh_id, traj in trajectories.items():
        positions = np.array(traj['positions'])
        speeds = np.array(traj['speeds'])
        accelerations = np.array(traj['accelerations'])

        seq_length = len(positions)

        # 创建滑动窗口
        for i in range(seq_length - future_steps):
            sample = {
                'current_state': torch.tensor([
                    positions[i],
                    speeds[i],
                    accelerations[i]
                ], dtype=torch.float32),
                'future_states': torch.tensor(
                    np.stack([
                        positions[i+1:i+1+future_steps],
                        speeds[i+1:i+1+future_steps],
                        accelerations[i+1:i+1+future_steps]
                    ], axis=-1),
                    dtype=torch.float32
                )
            }
            samples.append(sample)

    return samples
