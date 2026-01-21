"""
基于规则的车辆评分器

使用交通工程理论对车辆进行重要性评分，作为神经网络评分器的备选方案。
评分综合考虑瓶颈区域、速度、车道位置、距离等因素。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class RuleBasedVehicleScorer:
    """
    基于规则的车辆评分器

    评分策略：
    1. 瓶颈区域车辆权重最高（E17/J15, E19/J17, E23/J5汇入点）
    2. 速度越慢越重要（反映拥堵）
    3. 内侧车道优先（汇入影响大）
    4. 与前车距离越近越关键（TTC计算）
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        frenet_system=None
    ):
        """
        初始化规则评分器

        Args:
            config: 配置字典，包含评分权重和阈值
            frenet_system: Frenet坐标系系统（用于位置转换）
        """
        self.config = config or {}
        self.frenet_system = frenet_system

        # 从配置中读取参数，使用默认值作为后备
        heuristic_config = self.config.get('heuristic_selector', {})

        # 瓶颈区域定义（归一化s坐标）
        self.bottleneck_s_min = heuristic_config.get('bottleneck_s_min', 1200.0)
        self.bottleneck_s_max = heuristic_config.get('bottleneck_s_max', 2200.0)

        # 关键汇入点（s坐标位置和权重）
        self.key_merge_points = [
            {'s': 1400.0, 'weight': 1.5, 'name': 'E17/J15'},  # E17汇入点（最严重瓶颈）
            {'s': 1800.0, 'weight': 1.5, 'name': 'E19/J17'},  # E19汇入点（最严重瓶颈）
            {'s': 1000.0, 'weight': 1.2, 'name': 'E23/J5'},   # E23汇入点
        ]

        # 评分权重
        self.weights = {
            'bottleneck': 0.40,   # 瓶颈区域权重（最高）
            'speed': 0.25,        # 速度评分权重
            'lane': 0.15,         # 车道位置权重
            'distance': 0.15,     # 距离评分权重
            'ttc': 0.05,          # TTC权重
        }

        # 车辆参数（用于TTC计算）
        self.max_speed = heuristic_config.get('desired_speed', 30.0)
        self.max_accel = heuristic_config.get('max_accel', 2.0)
        self.max_decel = heuristic_config.get('max_decel', -4.5)
        self.safe_time_gap = heuristic_config.get('safe_time_gap', 1.5)

        # 阈值参数
        self.speed_threshold_ratio = heuristic_config.get('speed_threshold_ratio', 0.7)
        self.ttc_threshold = self.config.get('sparse_controller', {}).get('ttc_threshold', 3.0)

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
        traci_lib = context.get('traci_lib')
        all_vehicle_ids = context.get('all_vehicle_ids', [])

        scores = {}

        for veh_id in all_vehicle_ids:
            if veh_id not in vehicle_states:
                scores[veh_id] = 0.0
                continue

            state = vehicle_states[veh_id]

            # 计算各项评分
            bottleneck_score = self._compute_bottleneck_score(state, traci_lib, veh_id)
            speed_score = self._compute_speed_score(state)
            lane_score = self._compute_lane_score(state)
            distance_score = self._compute_distance_score(state, traci_lib, veh_id)
            ttc_score = self._compute_ttc_score(state, traci_lib, veh_id)

            # 加权求和
            total_score = (
                self.weights['bottleneck'] * bottleneck_score +
                self.weights['speed'] * speed_score +
                self.weights['lane'] * lane_score +
                self.weights['distance'] * distance_score +
                self.weights['ttc'] * ttc_score
            )

            scores[veh_id] = float(np.clip(total_score, 0.0, 1.0))

        return scores

    def _compute_bottleneck_score(
        self,
        state: Dict,
        traci_lib,
        veh_id: str
    ) -> float:
        """
        计算瓶颈区域评分

        策略：
        1. 在瓶颈区域内的车辆基础分高
        2. 接近关键汇入点的车辆额外加分
        3. 使用高斯分布平滑评分（避免突变）
        """
        s = state.get('s', 0.0)

        # 基础评分：是否在瓶颈区域
        if self.bottleneck_s_min <= s <= self.bottleneck_s_max:
            base_score = 0.8
        else:
            # 在瓶颈区域附近，使用高斯衰减
            distance_to_bottleneck = min(
                abs(s - self.bottleneck_s_min),
                abs(s - self.bottleneck_s_max)
            )
            sigma = 200.0  # 衰减尺度
            base_score = 0.8 * np.exp(-(distance_to_bottleneck ** 2) / (2 * sigma ** 2))

        # 关键汇入点加分
        merge_bonus = 0.0
        for merge_point in self.key_merge_points:
            distance = abs(s - merge_point['s'])
            # 高斯分布，影响范围100米
            if distance < 100.0:
                sigma = 50.0
                bonus = merge_point['weight'] * np.exp(-(distance ** 2) / (2 * sigma ** 2))
                merge_bonus = max(merge_bonus, bonus)

        # 组合评分
        bottleneck_score = base_score + 0.2 * merge_bonus

        return float(np.clip(bottleneck_score, 0.0, 1.0))

    def _compute_speed_score(self, state: Dict) -> float:
        """
        计算速度评分

        策略：
        1. 速度越慢越重要（反映拥堵或潜在拥堵）
        2. 速度异常低的车（接近停车）最重要
        3. 使用分段函数（非线性关系）
        """
        speed = state.get('speed', 0.0)
        vs = state.get('vs', 0.0)  # 纵向速度

        # 使用纵向速度（更准确反映沿车道运动）
        effective_speed = abs(vs)

        # 速度评分：速度越慢分数越高
        # 使用分段函数：
        # - 速度 < 5 m/s: 高分（0.8-1.0）
        # - 5-15 m/s: 中等分数（0.3-0.8）
        # - > 15 m/s: 低分（0-0.3）

        if effective_speed < 5.0:
            # 速度很低，可能是拥堵或即将停车
            score = 1.0 - 0.2 * (effective_speed / 5.0)
        elif effective_speed < 15.0:
            # 中等速度
            normalized_speed = (effective_speed - 5.0) / 10.0  # [0, 1]
            score = 0.8 - 0.5 * normalized_speed
        else:
            # 高速，不太重要
            score = 0.3 * np.exp(-(effective_speed - 15.0) / 10.0)

        # 加速度修正：减速中的车辆更重要
        accel = state.get('acceleration', 0.0)
        if accel < -1.0:  # 正在减速
            score = min(score + 0.1, 1.0)
        elif accel > 1.0:  # 正在加速
            score = max(score - 0.05, 0.0)

        return float(np.clip(score, 0.0, 1.0))

    def _compute_lane_score(self, state: Dict) -> float:
        """
        计算车道位置评分

        策略：
        1. 内侧车道（汇入车道）优先
        2. 边缘车道较低
        3. 考虑车道负载（车辆密度）
        """
        lane_id = state.get('lane_id', '')
        lane_index = state.get('lane_index', 0)

        # 从lane_id提取车道编号
        # SUMO lane_id格式: "edge_id_lane_index"
        try:
            lane_index = int(lane_id.split('_')[-1]) if lane_id else lane_index
        except:
            lane_index = 0

        # 基础评分：内侧车道优先
        # 假设有4条车道（0-3），0是最内侧
        if lane_index <= 1:
            base_score = 0.8  # 内侧车道
        elif lane_index == 2:
            base_score = 0.5  # 中间车道
        else:
            base_score = 0.3  # 外侧车道

        # 边缘检测（使用d坐标）
        d = state.get('d', 0.0)
        d_abs = abs(d)

        # 车辆偏离车道中心（可能正在换道）
        if d_abs > 0.5:  # 假设车道宽度约3.5米，归一化后约0.14
            # 正在换道，重要性提高
            base_score = min(base_score + 0.2, 1.0)

        return float(np.clip(base_score, 0.0, 1.0))

    def _compute_distance_score(
        self,
        state: Dict,
        traci_lib,
        veh_id: str
    ) -> float:
        """
        计算距离评分（与前车距离）

        策略：
        1. 与前车距离越近越重要
        2. 考虑安全距离
        3. 使用TraCI获取实际距离
        """
        if traci_lib is None:
            return 0.5  # 无法获取距离信息，返回中性分数

        try:
            # 获取前车距离
            leader_info = traci_lib.vehicle.getLeader(veh_id, 100.0)  # 100米范围内

            if leader_info is None:
                # 没有前车，自由流，重要性较低
                return 0.2

            leader_id, distance = leader_info
            distance = float(distance)

            # 距离评分：距离越近分数越高
            # 使用指数衰减
            safe_distance = self.safe_time_gap * state.get('speed', 10.0)  # 安全距离

            if distance < safe_distance * 0.5:
                # 距离过近，非常关键
                score = 1.0
            elif distance < safe_distance:
                # 接近安全距离
                normalized_dist = (distance - safe_distance * 0.5) / (safe_distance * 0.5)
                score = 1.0 - 0.3 * normalized_dist
            else:
                # 安全距离之外
                score = 0.7 * np.exp(-(distance - safe_distance) / 50.0)  # 50米衰减常数

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            # TraCI调用失败，返回中性分数
            return 0.5

    def _compute_ttc_score(
        self,
        state: Dict,
        traci_lib,
        veh_id: str
    ) -> float:
        """
        计算TTC（Time To Collision）评分

        TTC = 距离 / 相对速度
        TTC越小，碰撞风险越高，车辆越重要
        """
        if traci_lib is None:
            return 0.0

        try:
            leader_info = traci_lib.vehicle.getLeader(veh_id, 100.0)

            if leader_info is None:
                return 0.0

            leader_id, distance = leader_info
            distance = float(distance)

            # 获取自身速度和前车速度
            my_speed = state.get('speed', 0.0)

            try:
                leader_speed = traci_lib.vehicle.getSpeed(leader_id)
            except:
                leader_speed = my_speed  # 假设前车速度相同

            # 相对速度（逼近速度）
            relative_speed = my_speed - leader_speed

            # 如果相对速度 <= 0，正在远离，TTC为无穷大
            if relative_speed <= 0.1:
                return 0.0

            # 计算TTC（秒）
            ttc = distance / relative_speed

            # TTC评分：TTC越小分数越高
            if ttc < self.ttc_threshold:
                # TTC小于阈值，高碰撞风险
                score = 1.0 - (ttc / self.ttc_threshold) * 0.5
            else:
                # TTC大于阈值，低风险
                score = 0.5 * np.exp(-(ttc - self.ttc_threshold) / 5.0)

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            # 计算失败
            return 0.0

    def get_score_breakdown(
        self,
        vehicle_states: Dict[str, Dict],
        context: Dict,
        veh_id: str
    ) -> Dict[str, float]:
        """
        获取单个车辆的评分详细分解（用于调试和可视化）

        Returns:
            breakdown: {
                'bottleneck': float,
                'speed': float,
                'lane': float,
                'distance': float,
                'ttc': float,
                'total': float
            }
        """
        if veh_id not in vehicle_states:
            return {
                'bottleneck': 0.0,
                'speed': 0.0,
                'lane': 0.0,
                'distance': 0.0,
                'ttc': 0.0,
                'total': 0.0
            }

        state = vehicle_states[veh_id]
        traci_lib = context.get('traci_lib')

        bottleneck_score = self._compute_bottleneck_score(state, traci_lib, veh_id)
        speed_score = self._compute_speed_score(state)
        lane_score = self._compute_lane_score(state)
        distance_score = self._compute_distance_score(state, traci_lib, veh_id)
        ttc_score = self._compute_ttc_score(state, traci_lib, veh_id)

        total_score = (
            self.weights['bottleneck'] * bottleneck_score +
            self.weights['speed'] * speed_score +
            self.weights['lane'] * lane_score +
            self.weights['distance'] * distance_score +
            self.weights['ttc'] * ttc_score
        )

        return {
            'bottleneck': float(bottleneck_score),
            'speed': float(speed_score),
            'lane': float(lane_score),
            'distance': float(distance_score),
            'ttc': float(ttc_score),
            'total': float(np.clip(total_score, 0.0, 1.0))
        }
