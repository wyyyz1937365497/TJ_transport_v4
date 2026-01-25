#!/usr/bin/env python3
"""
轻量级规则控制器 - 专为初赛优化

设计理念：
1. 简单但有效的控制逻辑
2. 避免MPC的复杂性和bug
3. 在保持性能的同时提供一定的优化
"""

import numpy as np
from typing import Dict, List, Set


class SimpleRuleController:
    """
    简单规则控制器

    原则：
    - 只在必要时干预
    - 使用保守的控制参数
    - 优先保证不损害性能
    """

    def __init__(self, config: Dict):
        self.config = config

        # 控制参数（保守设置）
        self.min_speed_threshold = config.get('min_speed_threshold', 5.0)  # 低于5m/s才干预
        self.accel_adjustment = config.get('accel_adjustment', 0.5)      # 轻微加速
        self.lane_change_threshold = config.get('lane_change_threshold', 3.0)  # 前车慢于3m/s才换道

    def compute_actions(
        self,
        observation: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        计算控制动作

        策略：
        1. 慢车加速：如果车辆速度<5m/s，轻微加速
        2. 简单换道：如果前车慢，尝试换道（避免复杂逻辑）
        """
        actions = {}
        vehicle_states = observation.get('vehicle_states', {})

        for veh_id in icv_ids:
            if veh_id not in vehicle_states:
                continue

            state = vehicle_states[veh_id]
            speed = state.get('speed', 0.0)
            acceleration = state.get('acceleration', 0.0)

            # ========== 策略1: 慢车加速 ==========
            if speed < self.min_speed_threshold and acceleration < 1.0:
                # 轻微加速，避免激进
                accel_action = min(self.accel_adjustment, 2.0 - acceleration)
            else:
                # 保持当前加速度
                accel_action = 0.0

            # ========== 策略2: 简单换道 ==========
            # 仅在速度很低时考虑换道（保守策略）
            lane_change_action = 0
            if speed < 3.0:  # 严重拥堵
                # 简单的轮换策略：奇数ID向左，偶数ID向右
                # 避免所有车都往一个方向换道
                lane_index = state.get('lane_index', 0)
                veh_id_num = hash(veh_id) % 100  # 简单哈希

                if lane_index < 2 and veh_id_num % 2 == 1:
                    lane_change_action = 1   # 右换道
                elif lane_index > 0 and veh_id_num % 2 == 0:
                    lane_change_action = -1  # 左换道

            actions[veh_id] = np.array([accel_action, lane_change_action])

        return actions


class AdaptiveSpeedController:
    """
    自适应速度控制器 - 更高级的规则控制

    原理：
    - 根据交通密度自适应调整速度
    - 在瓶颈前平滑减速，避免急刹
    """

    def __init__(self, config: Dict):
        self.config = config
        self.bottleneck_s_min = config.get('bottleneck_s_min', 1200.0)
        self.bottleneck_s_max = config.get('bottleneck_s_max', 2200.0)

    def compute_actions(
        self,
        observation: Dict,
        vehicle_ids: List[str],
        icv_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        自适应速度控制

        策略：
        1. 在瓶颈前平滑减速
        2. 避免急刹车
        3. 保持车距
        """
        actions = {}
        vehicle_states = observation.get('vehicle_states', {})

        for veh_id in icv_ids:
            if veh_id not in vehicle_states:
                continue

            state = vehicle_states[veh_id]
            s = state.get('s', 0.0)  # 位置
            v = state.get('speed', 0.0)  # 速度
            a = state.get('acceleration', 0.0)

            # 判断是否在瓶颈区域
            in_bottleneck = self.bottleneck_s_min < s < self.bottleneck_s_max

            accel_action = 0.0
            lane_change_action = 0

            if in_bottleneck:
                # 瓶颈区域：保持稳定速度
                target_speed = 10.0  # 10m/s = 36km/h
                if v > target_speed + 2.0:
                    accel_action = -0.5  # 轻微减速
                elif v < target_speed - 2.0:
                    accel_action = 0.5   # 轻微加速
            else:
                # 非瓶颈区域：正常行驶
                if v < 8.0 and a < 1.0:
                    accel_action = 0.5

            # 简单换道：瓶颈区域内，向外侧车道分散
            if in_bottleneck and v < 5.0:
                lane_index = state.get('lane_index', 0)
                if lane_index == 0:  # 最内侧车道
                    lane_change_action = 1  # 向右换道

            actions[veh_id] = np.array([accel_action, lane_change_action])

        return actions
