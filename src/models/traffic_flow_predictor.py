"""
交通流预测模块 - 用于前瞻性协同控制

根据比赛要求，实现"预测-决策"闭环框架中的预测部分。
该模块能够预测未来短时间内交通流的演化趋势。

预测目标：
- 短期速度演化（1-5秒）
- 拥堵传播趋势
- 吞吐量变化

技术方案：
- 使用时序图神经网络（Temporal GNN）
- 结合GNN空间特征和LSTM时序建模
- 轻量化设计，支持快速推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class TemporalTrafficPredictor(nn.Module):
    """
    时序交通流预测器

    结合GNN空间特征和LSTM时序建模，预测未来交通流状态。

    输入：历史交通状态序列（T步）
    输出：未来交通状态预测（F步）

    特点：
    - 轻量化：快速推理，支持在线控制
    - 多尺度：同时预测局部和全局指标
    - 不确定性：提供预测置信度
    """

    def __init__(
        self,
        node_dim: int = 9,        # 节点特征维度
        hidden_dim: int = 64,      # 隐藏层维度
        num_layers: int = 2,       # LSTM层数
        history_steps: int = 10,   # 历史步数
        future_steps: int = 5,     # 预测步数
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.device = torch.device(device)

        # 1. 编码器：处理历史节点特征
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 2. 时序建模：双向LSTM
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # LSTM输出是双向的，需要投影回单向维度
        self.lstm_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # 3. 图注意力：捕捉车辆间交互（简化版）
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 4. 预测头：输出未来状态
        self.speed_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 预测速度
        )

        self.density_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # 预测密度
        )

        # 5. 不确定性估计
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # 输出正数（标准差）
        )

        self.to(device)

    def forward(
        self,
        history_features: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            history_features: [batch_size, num_vehicles, history_steps, node_dim]
                历史节点特征序列
            history_mask: [batch_size, num_vehicles, history_steps]
                掩码（用于处理不同长度的历史）

        Returns:
            predictions: {
                'future_speeds': [batch_size, num_vehicles, future_steps],
                'future_densities': [batch_size, num_vehicles, future_steps],
                'uncertainty': [batch_size, num_vehicles, future_steps],
                'encoded_features': [batch_size, num_vehicles, hidden_dim]
            }
        """
        batch_size, num_vehicles, hist_steps, node_dim = history_features.shape

        # 1. 编码节点特征
        # [B, N, T, D] -> [B*N*T, D] -> [B*N*T, H]
        features_flat = history_features.view(-1, node_dim)
        encoded_flat = self.node_encoder(features_flat)
        encoded = encoded_flat.view(batch_size, num_vehicles, hist_steps, -1)
        # [B, N, T, H]

        # 2. 时序建模：LSTM
        # [B, N, T, H] -> [B*N, T, H]
        encoded_bn = encoded.permute(0, 1, 3, 2).contiguous().view(batch_size * num_vehicles, hist_steps, -1)

        # LSTM处理
        lstm_out, _ = self.temporal_encoder(encoded_bn)
        # [B*N, T, 2*H] (双向)

        # 投影回单向维度
        lstm_out = self.lstm_projection(lstm_out)
        # [B*N, T, H]

        # 取最后一步的隐藏状态作为当前表征
        current_repr = lstm_out[:, -1, :]  # [B*N, H]
        current_repr = current_repr.view(batch_size, num_vehicles, -1)
        # [B, N, H]

        # 3. 图注意力：捕捉车辆间交互
        attn_out, _ = self.graph_attention(
            current_repr, current_repr, current_repr
        )
        # [B, N, H]

        # 4. 预测未来状态
        predictions = []

        for t in range(self.future_steps):
            # 使用注意力后的特征进行预测
            pred_speed = self.speed_predictor(attn_out).squeeze(-1)  # [B, N]
            pred_density = self.density_predictor(attn_out).squeeze(-1)  # [B, N]
            uncertainty = self.uncertainty_head(attn_out).squeeze(-1)  # [B, N]

            predictions.append({
                'speed': pred_speed,
                'density': pred_density,
                'uncertainty': uncertainty
            })

        # 堆叠预测结果
        future_speeds = torch.stack([p['speed'] for p in predictions], dim=2)  # [B, N, F]
        future_densities = torch.stack([p['density'] for p in predictions], dim=2)
        uncertainties = torch.stack([p['uncertainty'] for p in predictions], dim=2)

        return {
            'future_speeds': future_speeds,
            'future_densities': future_densities,
            'uncertainties': uncertainties,
            'encoded_features': attn_out  # 用于决策
        }


class TrafficFlowForecaster(nn.Module):
    """
    交通流预测器 - 高层接口

    集成GNN和时序预测，提供完整的预测-决策接口。

    功能：
    1. 从历史数据提取时空特征
    2. 预测未来交通流状态
    3. 识别拥堵传播风险
    4. 评估控制策略的潜在影响
    """

    def __init__(
        self,
        gnn_model: nn.Module,  # 预训练的GNN模型
        node_dim: int = 9,
        hidden_dim: int = 64,
        history_steps: int = 10,
        future_steps: int = 5,
        device: str = 'cuda'
    ):
        super().__init__()

        self.gnn = gnn_model
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.device = torch.device(device)

        # 时序预测器
        self.temporal_predictor = TemporalTrafficPredictor(
            node_dim=gnn_model.output_dim,  # 使用GNN输出作为输入
            hidden_dim=hidden_dim,
            history_steps=history_steps,
            future_steps=future_steps,
            device=device
        )

        # 拥堵检测器
        self.congestion_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # [正常, 拥堵, 严重拥堵]
            nn.Softmax(dim=-1)
        )

        self.to(device)

    def forward(
        self,
        history_observations: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        预测未来交通流

        Args:
            history_observations: 历史观测列表 [T个观测]

        Returns:
            predictions: {
                'future_speeds': [B, N, F],
                'congestion_prob': [B, N, 3],
                'risk_vehicles': List[int],  # 高风险车辆索引
                'bottleneck_edges': List[str],  # 瓶颈路段
                'recommendations': Dict  # 控制建议
            }
        """
        if not history_observations:
            return self._empty_predictions()

        # 1. 提取历史特征
        node_features_list = []
        vehicle_ids_list = []

        # 使用最近history_steps个观测
        recent_obs = history_observations[-self.history_steps:]

        for obs in recent_obs:
            vehicle_states = obs.get('vehicle_states', {})
            vehicle_ids = obs.get('vehicle_ids', [])
            features = []

            for veh_id in vehicle_ids:
                if veh_id in vehicle_states:
                    state = vehicle_states[veh_id]
                    # 提取9维Frenet特征
                    feat = [
                        state.get('s', 0.0),
                        state.get('d', 0.0),
                        state.get('vs', 0.0),
                        state.get('vd', 0.0),
                        state.get('speed', 0.0),
                        state.get('acceleration', 0.0),
                        state.get('lane_index', 0.0),
                        state.get('angle', 0.0),
                        1.0  # is_icv placeholder
                    ]
                    features.append(feat)

            if features:
                node_features_list.append(torch.tensor(features, dtype=torch.float32))
                vehicle_ids_list.append(vehicle_ids)

        if not node_features_list:
            return self._empty_predictions()

        # 2. 堆叠历史特征 [T, N, 9] -> [1, T, N, 9]
        history_features = torch.stack(node_features_list, dim=0).unsqueeze(0)

        # 移动到正确的设备
        history_features = history_features.to(self.device)

        # 3. 使用时序预测器
        try:
            predictions = self.temporal_predictor(history_features)
        except Exception as e:
            # 预测失败时返回空预测
            return self._empty_predictions()

        # 4. 检测拥堵风险
        congestion_probs = self.congestion_detector(predictions['encoded_features'])

        # 5. 识别高风险车辆（预测速度低于阈值的车辆）
        future_speeds = predictions['future_speeds'][0]  # [N, F]
        mean_speeds = future_speeds.mean(dim=1)  # [N] - 未来F步的平均速度

        risk_threshold = 5.0  # m/s - 低于此速度视为风险车辆
        risk_indices = (mean_speeds < risk_threshold).nonzero(as_tuple=True)[0].cpu().numpy().tolist()

        return {
            'future_speeds': predictions['future_speeds'],
            'congestion_prob': congestion_probs,
            'risk_vehicles': risk_indices,
            'bottleneck_edges': [],  # TODO: 实现瓶颈检测
            'recommendations': {
                'priority_vehicles': risk_indices[:5] if risk_indices else [],  # 前5个最需要关注的车辆
                'intervention_threshold': 2.0  # 建议的干预阈值
            }
        }

    def _empty_predictions(self):
        """返回空预测（处理边界情况）"""
        return {
            'future_speeds': torch.zeros(1, 0, self.future_steps),
            'congestion_prob': torch.zeros(1, 0, 3),
            'risk_vehicles': [],
            'bottleneck_edges': [],
            'recommendations': {
                'priority_vehicles': [],
                'intervention_threshold': 2.0
            }
        }

    def predict_congestion_propagation(
        self,
        current_state: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        预测拥堵传播

        识别可能导致系统性拥堵的关键车辆和路段。

        Returns:
            {
                'congestion_probability': float,  # 未来拥塞概率
                'propagation_path': List[str],    # 拥堵传播路径
                'critical_vehicles': List[int],   # 关键车辆（需要干预）
                'time_to_congestion': int         # 预计拥塞发生时间（步数）
            }
        """
        # TODO: 实现拥堵传播预测
        pass


def create_flow_predictor(config: Dict[str, any]) -> TemporalTrafficPredictor:
    """
    创建交通流预测器的工厂函数

    Args:
        config: 配置字典

    Returns:
        predictor: 时序交通流预测器
    """
    return TemporalTrafficPredictor(
        node_dim=config.get('node_dim', 9),
        hidden_dim=config.get('predictor_hidden_dim', 64),
        num_layers=config.get('predictor_num_layers', 2),
        history_steps=config.get('history_steps', 10),
        future_steps=config.get('future_steps', 5),
        dropout=config.get('predictor_dropout', 0.1),
        device=config.get('device', 'cuda')
    )
