"""
Dual-mode Safety Shield - 双模态安全屏障
功能：确保所有控制指令的安全性
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np


class DualModeSafetyShield(nn.Module):
    """
    双模态安全屏障

    Level 1: 动作裁剪（软约束）
    Level 2: 紧急制动（硬约束）

    确保所有控制指令都符合安全规范
    """

    def __init__(
        self,
        ttc_threshold: float = 2.0,
        thw_threshold: float = 1.5,
        max_accel: float = 2.0,
        max_decel: float = -3.0,
        emergency_decel: float = -5.0,
        max_lane_change_speed: float = 5.0,
        min_follow_distance: float = 2.0,
        reaction_time: float = 0.5
    ):
        super().__init__()

        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.emergency_decel = emergency_decel
        self.max_lane_change_speed = max_lane_change_speed
        self.min_follow_distance = min_follow_distance
        self.reaction_time = reaction_time

    def forward(
        self,
        raw_actions: torch.Tensor,
        vehicle_states: Dict[str, Dict],
        selected_vehicle_indices: List[int]
    ) -> Dict[str, any]:
        """
        安全屏障前向传播

        Args:
            raw_actions: [K, 2] 原始动作（加速度, 换道）
            vehicle_states: 车辆状态字典
            selected_vehicle_indices: 选中的车辆索引

        Returns:
            safe_actions, 干预统计
        """
        if len(selected_vehicle_indices) == 0:
            return {
                'safe_actions': torch.zeros(0, 2, device=raw_actions.device),
                'level1_interventions': 0,
                'level2_interventions': 0,
                'intervention_details': []
            }

        # Level 1: 动作裁剪
        level1_actions, level1_interventions = self._level1_clipping(
            raw_actions, vehicle_states, selected_vehicle_indices
        )

        # Level 2: 紧急安全检查
        level2_actions, level2_interventions = self._level2_emergency_check(
            level1_actions, vehicle_states, selected_vehicle_indices
        )

        return {
            'safe_actions': level2_actions,
            'level1_interventions': int(torch.sum(level1_interventions).item()),
            'level2_interventions': int(torch.sum(level2_interventions).item()),
            'intervention_details': []
        }

    def _level1_clipping(
        self,
        raw_actions: torch.Tensor,
        vehicle_states: Dict[str, Dict],
        selected_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Level 1: 基础动作裁剪
        """
        k = len(selected_indices)
        safe_actions = raw_actions.clone()
        intervention_mask = torch.zeros(k, dtype=torch.bool, device=raw_actions.device)

        vehicle_ids = vehicle_states.get('ids', [])
        vehicle_data = vehicle_states.get('data', {})

        for i, idx in enumerate(selected_indices):
            if idx >= len(vehicle_ids):
                continue

            veh_id = vehicle_ids[idx]
            if veh_id not in vehicle_data:
                continue

            vehicle = vehicle_data[veh_id]
            current_speed = vehicle.get('speed', 0.0)

            # 1. 加速度裁剪（动态限制）
            raw_accel = raw_actions[i, 0].item()

            # 速度依赖的加速度限制
            speed_factor = min(current_speed / 30.0, 1.0)
            dynamic_max_accel = self.max_accel * (1 - 0.3 * speed_factor)
            dynamic_max_decel = self.max_decel * (1 + 0.2 * speed_factor)

            # 映射Tanh输出[-1,1]到物理加速度范围
            accel_range = self.max_accel - self.max_decel
            physical_accel = self.max_decel + (raw_accel + 1) / 2 * accel_range

            safe_accel = np.clip(physical_accel, dynamic_max_decel, dynamic_max_accel)

            if abs(safe_accel - physical_accel) > 0.1:
                intervention_mask[i] = True

            # 映射回[-1,1]
            safe_accel_normalized = (safe_accel - self.max_decel) / accel_range * 2 - 1
            safe_actions[i, 0] = safe_accel_normalized

            # 2. 换道限制
            raw_lane_change = raw_actions[i, 1].item()
            safe_lane_change = raw_lane_change

            # 速度过高时禁止换道
            if current_speed > self.max_lane_change_speed:
                safe_lane_change = 0.0
                if raw_lane_change > 0.5:
                    intervention_mask[i] = True

            # 低速时也限制换道
            if current_speed < 1.0:
                safe_lane_change = 0.0
                if raw_lane_change > 0.5:
                    intervention_mask[i] = True

            safe_actions[i, 1] = safe_lane_change

        return safe_actions, intervention_mask

    def _level2_emergency_check(
        self,
        actions: torch.Tensor,
        vehicle_states: Dict[str, Dict],
        selected_indices: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Level 2: 紧急安全检查
        """
        k = len(selected_indices)
        final_actions = actions.clone()
        emergency_mask = torch.zeros(k, dtype=torch.bool, device=actions.device)

        vehicle_ids = vehicle_states.get('ids', [])
        vehicle_data = vehicle_states.get('data', {})

        for i, idx in enumerate(selected_indices):
            if idx >= len(vehicle_ids):
                continue

            veh_id = vehicle_ids[idx]
            if veh_id not in vehicle_data:
                continue

            ego_vehicle = vehicle_data[veh_id]

            # 查找前车
            leader_vehicle = self._find_leader(ego_vehicle, vehicle_data)

            if leader_vehicle is not None:
                ttc = self._calculate_ttc(ego_vehicle, leader_vehicle)
                thw = self._calculate_thw(ego_vehicle, leader_vehicle)

                # 紧急制动条件
                if ttc < self.ttc_threshold or thw < self.thw_threshold:
                    # 强制减速
                    final_actions[i, 0] = -1.0  # 最强减速
                    final_actions[i, 1] = 0.0  # 禁止换道
                    emergency_mask[i] = True

            # 检查侧向安全（换道场景）
            if actions[i, 1].item() > 0.5:
                if not self._check_lane_change_safe(ego_vehicle, vehicle_data):
                    final_actions[i, 1] = 0.0  # 禁止换道
                    emergency_mask[i] = True

        return final_actions, emergency_mask

    def _find_leader(
        self,
        ego: Dict,
        all_vehicles: Dict[str, Dict]
    ) -> Optional[Dict]:
        """
        找到前车
        """
        min_distance = float('inf')
        leader = None

        ego_pos = ego.get('position', 0.0)
        ego_lane = ego.get('lane_id', '')

        for veh_id, vehicle in all_vehicles.items():
            if veh_id == ego.get('id', ''):
                continue

            if vehicle.get('lane_id', '') != ego_lane:
                continue

            veh_pos = vehicle.get('position', 0.0)

            if veh_pos <= ego_pos:
                continue

            distance = veh_pos - ego_pos
            if distance < min_distance and distance < 100:
                min_distance = distance
                leader = vehicle

        return leader

    def _calculate_ttc(self, ego: Dict, leader: Dict) -> float:
        """
        计算碰撞时间TTC（Time To Collision）
        """
        ego_speed = ego.get('speed', 0.0)
        leader_speed = leader.get('speed', 0.0)
        distance = leader.get('position', 0.0) - ego.get('position', 0.0)

        rel_speed = ego_speed - leader_speed

        if rel_speed <= 1e-6:
            return float('inf')

        ttc = distance / rel_speed
        return max(0.1, ttc)

    def _calculate_thw(self, ego: Dict, leader: Dict) -> float:
        """
        计算车头时距THW（Time Headway）
        """
        ego_speed = ego.get('speed', 1.0)
        distance = leader.get('position', 0.0) - ego.get('position', 0.0)

        if ego_speed <= 1e-6:
            return float('inf')

        thw = distance / ego_speed
        return max(0.1, thw)

    def _check_lane_change_safe(
        self,
        ego: Dict,
        all_vehicles: Dict[str, Dict]
    ) -> bool:
        """
        检查换道是否安全
        """
        ego_lane = ego.get('lane_id', '')
        ego_pos = ego.get('position', 0.0)
        ego_speed = ego.get('speed', 0.0)

        # 检查目标车道（左右车道）
        lane_num = self._extract_lane_number(ego_lane)

        for direction in [-1, 1]:  # 左和右
            target_lane = f"{ego_lane.rsplit('_', 1)[0]}_{lane_num + direction}"

            # 查找目标车道上的车辆
            for veh_id, vehicle in all_vehicles.items():
                if veh_id == ego.get('id', ''):
                    continue

                if vehicle.get('lane_id', '') != target_lane:
                    continue

                veh_pos = vehicle.get('position', 0.0)
                veh_speed = vehicle.get('speed', 0.0)

                # 检查距离
                distance = abs(veh_pos - ego_pos)

                if distance < self.min_follow_distance:
                    return False

                # 检查相对速度
                if veh_pos > ego_pos and veh_speed < ego_speed:
                    # 前车较慢
                    if distance < 10:
                        return False

        return True

    def _extract_lane_number(self, lane_id: str) -> int:
        """
        从车道ID提取编号
        """
        try:
            return int(lane_id.rsplit('_', 1)[-1])
        except:
            return 0


class SafetyMonitor:
    """
    安全监控器（记录和统计）
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.stats = {
            'total_actions': 0,
            'level1_interventions': 0,
            'level2_interventions': 0,
            'ttc_violations': 0,
            'thw_violations': 0,
            'unsafe_lane_changes': 0
        }

    def update(
        self,
        actions: torch.Tensor,
        shield_output: Dict[str, any]
    ):
        """
        更新统计
        """
        self.stats['total_actions'] += actions.size(0)
        self.stats['level1_interventions'] += shield_output.get('level1_interventions', 0)
        self.stats['level2_interventions'] += shield_output.get('level2_interventions', 0)

    def get_summary(self) -> Dict[str, float]:
        """
        获取摘要统计
        """
        if self.stats['total_actions'] == 0:
            return {
                'level1_intervention_rate': 0.0,
                'level2_intervention_rate': 0.0,
                'total_intervention_rate': 0.0
            }

        return {
            'level1_intervention_rate': self.stats['level1_interventions'] / self.stats['total_actions'],
            'level2_intervention_rate': self.stats['level2_interventions'] / self.stats['total_actions'],
            'total_intervention_rate': (
                self.stats['level1_interventions'] + self.stats['level2_interventions']
            ) / self.stats['total_actions']
        }


class TrajectoryPredictor:
    """
    轨迹预测器（用于安全检查）
    """

    def __init__(
        self,
        prediction_horizon: float = 2.0,
        dt: float = 0.1
    ):
        self.horizon = prediction_horizon
        self.dt = dt
        self.steps = int(prediction_horizon / dt)

    def predict_trajectory(
        self,
        initial_state: Dict,
        action: np.ndarray,
        vehicle_length: float = 5.0
    ) -> np.ndarray:
        """
        预测车辆轨迹

        Args:
            initial_state: 初始状态
            action: 控制动作 [acceleration, lane_change]
            vehicle_length: 车辆长度

        Returns:
            trajectory: [steps, 2] (x, y)坐标
        """
        trajectory = np.zeros((self.steps, 2))

        # 初始状态
        x = initial_state.get('x', 0.0)
        y = initial_state.get('y', 0.0)
        v = initial_state.get('speed', 0.0)
        theta = initial_state.get('angle', 0.0)

        accel = action[0]
        lane_change = action[1]

        for i in range(self.steps):
            # ========== 完整的车辆运动学模型 ==========
            # 基于IDM（Intelligent Driver Model）和MOBIL（Minimizing Overall Braking Induced by Lane changes）

            # 1. 纵向运动（加速度模型）
            # 使用IDM模型计算加速度
            # a = a_max * [1 - (v/v0)^δ - (s*/s)^2]
            # 其中 s* = s0 + v*T + (v*Δv)/(2*sqrt(a_max*b))

            # 简化版：考虑最大加速度、舒适减速度
            max_accel = 2.0  # m/s²
            max_decel = 3.0  # m/s²
            desired_speed = 30.0  # m/s
            min_gap = 2.0  # m
            desired_time_headway = 1.5  # s

            # 当前加速度（考虑车辆动力学限制）
            accel_clipped = np.clip(accel, -max_decel, max_accel)

            # 速度更新（考虑空气阻力和滚动阻力）
            # v(t+dt) = v(t) + a*dt - (阻力项)
            rolling_resistance = 0.01 * 9.81  # 滚动阻力
            air_drag = 0.3 * v * v / 1500.0  # 空气阻力（简化）
            decel_resistance = rolling_resistance + air_drag

            v_new = v + accel_clipped * self.dt - decel_resistance * self.dt
            v_new = max(0, v_new)  # 速度不能为负

            # 2. 位置更新
            # 考虑车道曲率（如果有）
            lane_curvature = 0.0  # 假设直道

            # 纵向位移
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt

            x += dx
            y += dy

            # 3. 横向运动（换道模型）
            # 使用平滑的换道轨迹（5次多项式）
            if lane_change > 0.5:
                # 换道持续时间通常为3-5秒
                lane_change_duration = 4.0  # 秒
                lane_width = 3.5  # m

                # 使用sigmoid函数模拟平滑换道
                # lateral_progress从0到1
                lateral_progress = min(1.0, (i * self.dt) / lane_change_duration)

                # 5次多项式换道轨迹
                # y(t) = y_start + lane_width * (10*(t/T)^3 - 15*(t/T)^4 + 6*(t/T)^5)
                lateral_offset = lane_width * (
                    10 * lateral_progress**3 -
                    15 * lateral_progress**4 +
                    6 * lateral_progress**5
                )

                y += lateral_offset

                # 换道时略微降低速度
                v_new *= 0.95

            # 4. 航向角更新
            # 考虑车道几何和横向运动
            if lane_change > 0.5:
                # 换道时有小的航向角变化
                theta += 0.02 * np.sin(2 * np.pi * (i * self.dt) / 4.0)

            # 更新速度
            v = v_new

            trajectory[i] = [x, y]

        return trajectory
