"""数据类定义 - 使用dataclass简化数据结构"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch_geometric.data import Data


@dataclass
class VehicleState:
    """单个车辆的状态"""
    id: str
    x: float
    y: float
    z: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0
    lane_id: str = ""
    lane_index: int = 0
    position: float = 0.0
    angle: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """转换为张量 [x, y, speed, acceleration]"""
        return torch.tensor([self.x, self.y, self.speed, self.acceleration], dtype=torch.float32)


@dataclass
class Observation:
    """环境观测"""
    vehicle_states: Dict[str, VehicleState]
    vehicle_ids: List[str]
    icv_ids: List[str]
    global_stats: torch.Tensor
    step: int

    def __len__(self) -> int:
        return len(self.vehicle_ids)


@dataclass
class Action:
    """控制动作"""
    acceleration: float  # [-3, 2] m/s²
    lane_change: float  # [0, 1] 换道概率


@dataclass
class StepResult:
    """环境步进结果"""
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class TrainingBatch:
    """训练批次数据"""
    # 当前状态 [B, T, 4]
    current_states: torch.Tensor
    # 未来状态 [B, T_future, 4]
    future_states: torch.Tensor
    # 图数据
    graph_data: Data
    # 全局统计 [B, 16]
    global_stats: torch.Tensor
    # 是否为ICV [B]
    is_icv: torch.Tensor
    # 车辆ID
    vehicle_ids: List[str]

    def to(self, device: torch.device) -> "TrainingBatch":
        """移动到指定设备"""
        return TrainingBatch(
            current_states=self.current_states.to(device),
            future_states=self.future_states.to(device),
            graph_data=self.graph_data.to(device),
            global_stats=self.global_stats.to(device),
            is_icv=self.is_icv.to(device),
            vehicle_ids=self.vehicle_ids,
        )


@dataclass
class ModelOutput:
    """模型输出"""
    # 选择的车辆ID
    selected_vehicle_ids: List[str]
    # 安全动作
    safe_actions: torch.Tensor
    # 原始动作
    raw_actions: torch.Tensor
    # 影响力分数
    influence_scores: torch.Tensor
    # 价值估计
    value_estimates: Optional[torch.Tensor] = None
    # 成本估计
    cost_estimates: Optional[torch.Tensor] = None
    # 优势估计
    advantage_estimates: Optional[torch.Tensor] = None
    # GNN嵌入
    gnn_embedding: Optional[torch.Tensor] = None
    # 世界模型预测
    world_predictions: Optional[Dict[str, torch.Tensor]] = None
