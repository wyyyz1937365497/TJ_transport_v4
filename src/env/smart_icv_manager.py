#!/usr/bin/env python3
"""
智能ICV管理器 - 基于Top-K机制和事件触发

核心理念：
- "按需干预"而非"均匀撒网"
- 平峰期几乎不控制（P_int ≈ 1）
- 拥堵期精准控制（关键少数车辆）

作者: TJ Transport Team v4.0
日期: 2026-01-19
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class InterventionLevel(Enum):
    """干预级别"""
    NORMAL = "normal"       # 正常：默认Top-K控制
    ELEVATED = "elevated"   # 升级：检测到拥堵风险，扩大Top-K
    EMERGENCY = "emergency" # 紧急：检测到高危事件，全面介入


@dataclass
class VehicleRiskMetrics:
    """车辆风险指标"""
    ttc: float          # Time to Collision (碰撞时间，秒)
    thw: float          # Time Headway (车头时距，秒)
    drac: float         # Deceleration Rate to Avoid Crash (m/s²)
    risk_level: float   # 综合风险级别 (0-1)


class SmartICVManager:
    """
    智能ICV管理器

    核心功能：
    1. Top-K机制：根据影响力评分只控制最关键的K辆车
    2. 动态Top-K：根据风险级别动态调整K值
    3. 事件触发：检测到高危事件立即扩大控制范围
    4. 两级安全屏障：规则卫士 + 紧急避险
    """

    def __init__(
        self,
        max_vehicles: int,
        default_top_k: int = 5,           # 默认只控制5辆（远小于25%）
        emergency_top_k: int = 15,        # 紧急情况扩大到15辆
        elevated_top_k: int = 10,         # 升级情况控制10辆
        intervention_threshold: float = 0.3,  # ICV渗透率上限（赛题要求≤30%）
        decision_interval: int = 10,      # 默认决策周期（10步）
        ttc_threshold: float = 2.0,       # TTC高危阈值（秒）
        thw_threshold: float = 1.5,       # THW高危阈值（秒）
    ):
        """
        Args:
            max_vehicles: 场景最大车辆数
            default_top_k: 默认控制的车辆数（平峰期）
            emergency_top_k: 紧急情况控制的车辆数
            elevated_top_k: 升级情况控制的车辆数
            intervention_threshold: ICV渗透率上限
            decision_interval: 决策周期（秒）
            ttc_threshold: TTC高危阈值
            thw_threshold: THW高危阈值
        """
        self.max_vehicles = max_vehicles

        # Top-K参数
        self.default_top_k = min(default_top_k, int(max_vehicles * intervention_threshold))
        self.emergency_top_k = min(emergency_top_k, int(max_vehicles * intervention_threshold))
        self.elevated_top_k = min(elevated_top_k, int(max_vehicles * intervention_threshold))

        # 确保不超过赛题限制
        self.max_icv_allowed = int(max_vehicles * intervention_threshold)

        # 风险阈值
        self.ttc_threshold = ttc_threshold
        self.thw_threshold = thw_threshold
        self.decision_interval = decision_interval

        # 当前状态
        self.current_top_k = self.default_top_k
        self.current_level = InterventionLevel.NORMAL
        self.last_decision_step = -1  # ✅ 初始值设为-1，避免与step=0冲突

        # 缓存
        self.vehicle_scores_cache: Dict[str, float] = {}
        self.vehicle_risk_cache: Dict[str, VehicleRiskMetrics] = {}

        print(f"[SmartICV] 初始化完成")
        print(f"  - Max vehicles: {max_vehicles}")
        print(f"  - Top-K range: {self.default_top_k} (default) → {self.emergency_top_k} (emergency)")
        print(f"  - Max ICV allowed: {self.max_icv_allowed} ({intervention_threshold*100}%)")
        print(f"  - TTC threshold: {self.ttc_threshold}s")
        print(f"  - Decision interval: {self.decision_interval} steps")

    def compute_risk_metrics(
        self,
        veh_id: str,
        speed: float,
        leader_speed: Optional[float],
        leader_distance: Optional[float],
        acceleration: float,
    ) -> VehicleRiskMetrics:
        """
        计算车辆风险指标

        Args:
            veh_id: 车辆ID
            speed: 当前速度 (m/s)
            leader_speed: 前车速度 (m/s)
            leader_distance: 前车距离 (m)
            acceleration: 当前加速度 (m/s²)

        Returns:
            VehicleRiskMetrics: 风险指标
        """
        # ========== 1. TTC (Time to Collision) 计算 ==========
        # TTC = distance / relative_speed
        # 如果前车比当前车快，TTC = inf（无碰撞风险）
        if leader_distance is not None and leader_speed is not None:
            relative_speed = speed - leader_speed
            if relative_speed > 0.1:  # 前车比当前车慢
                ttc = leader_distance / relative_speed
            else:
                ttc = float('inf')
        else:
            ttc = float('inf')

        # ========== 2. THW (Time Headway) 计算 ==========
        # THW = distance / speed
        if leader_distance is not None and speed > 0.1:
            thw = leader_distance / speed
        else:
            thw = float('inf')

        # ========== 3. DRAC (Deceleration Rate to Avoid Crash) 计算 ==========
        # DRAC = (relative_speed)² / (2 * distance)
        # 代表避免碰撞所需的减速度
        if leader_distance is not None and leader_distance > 0.1:
            relative_speed = max(0, speed - (leader_speed or 0))
            drac = (relative_speed ** 2) / (2 * leader_distance)
        else:
            drac = 0.0

        # ========== 4. 综合风险级别 (0-1) ==========
        risk_score = 0.0

        # TTC风险（0-40分）
        if ttc < 1.0:
            risk_score += 40.0
        elif ttc < 2.0:
            risk_score += 30.0
        elif ttc < 3.0:
            risk_score += 20.0
        elif ttc < 5.0:
            risk_score += 10.0

        # THW风险（0-30分）
        if thw < 0.5:
            risk_score += 30.0
        elif thw < 1.0:
            risk_score += 20.0
        elif thw < 1.5:
            risk_score += 10.0
        elif thw < 2.0:
            risk_score += 5.0

        # DRAC风险（0-30分）
        # DRAC > 3.0 m/s² 代表需要急刹避撞
        if drac > 5.0:
            risk_score += 30.0
        elif drac > 3.0:
            risk_score += 20.0
        elif drac > 2.0:
            risk_score += 10.0
        elif drac > 1.0:
            risk_score += 5.0

        # 归一化到0-1
        risk_level = min(risk_score / 100.0, 1.0)

        return VehicleRiskMetrics(
            ttc=ttc,
            thw=thw,
            drac=drac,
            risk_level=risk_level
        )

    def compute_influence_score(
        self,
        veh_id: str,
        position: Tuple[float, float],  # (x, y)
        speed: float,
        lane_index: int,
        edge_id: str,
        acceleration: float,
        risk_metrics: VehicleRiskMetrics,
        in_bottleneck: bool = False,
        bottleneck_proximity: float = 0.0,
    ) -> float:
        """
        计算车辆影响力评分（用于Top-K选择）

        评分公式：
        Score = α * Importance(GNN特征) + β * Risk(风险指标) + γ * Impact(预测影响)

        当前版本使用启发式评分：
        1. 位置权重（瓶颈区域）：最多30分
        2. 风险权重（TTC/THW）：最多40分
        3. 速度权重（异常车辆）：最多15分
        4. 车道权重（关键车道）：最多10分
        5. 动态权重（急加减速）：最多5分

        Args:
            veh_id: 车辆ID
            position: 车辆位置 (x, y)
            speed: 当前速度 (m/s)
            lane_index: 车道索引
            edge_id: 边缘ID
            acceleration: 加速度 (m/s²)
            risk_metrics: 风险指标
            in_bottleneck: 是否在瓶颈区域
            bottleneck_proximity: 距离瓶颈的接近度 (0-1)

        Returns:
            影响力评分 (0-100分)
        """
        score = 0.0

        # ========== 1. 风险权重（最高优先级） - 最多40分 ==========
        # ⭐ 核心改进：直接使用TTC/THW作为评分依据
        if risk_metrics.ttc < 2.0:
            score += 40.0  # 极高危：立即干预
        elif risk_metrics.ttc < 5.0:
            score += 30.0  # 高风险
        elif risk_metrics.ttc < 10.0:
            score += 20.0  # 中等风险

        # THW风险
        if risk_metrics.thw < 1.0:
            score += 10.0
        elif risk_metrics.thw < 1.5:
            score += 5.0

        # DRAC风险（需要急刹避撞）
        if risk_metrics.drac > 3.0:
            score += 15.0
        elif risk_metrics.drac > 2.0:
            score += 10.0
        elif risk_metrics.drac > 1.0:
            score += 5.0

        # ========== 2. 位置权重（瓶颈区域） - 最多30分 ==========
        if in_bottleneck:
            score += 20.0  # 瓶颈区域车辆
            # 额外奖励：距离瓶颈越近，分数越高
            score += bottleneck_proximity * 10.0  # 最多10分

        # ========== 3. 速度权重（异常车辆） - 最多15分 ==========
        if speed < 3.0:
            score += 15.0  # 极慢车（严重拥堵源）
        elif speed < 8.0:
            score += 10.0  # 慢速车
        elif speed > 25.0:
            score += 5.0   # 快速车（需要协调）

        # ========== 4. 车道权重（关键车道） - 最多10分 ==========
        if lane_index == 0:
            score += 10.0  # 最外侧车道（通常是汇流车道）
        elif lane_index == 1:
            score += 5.0   # 次外侧车道

        # ========== 5. 动态权重（急加减速） - 最多5分 ==========
        if abs(acceleration) > 2.5:
            score += 5.0  # 急加减速（不稳定因素）
        elif abs(acceleration) > 1.5:
            score += 2.5

        return score

    def detect_emergency_events(
        self,
        vehicle_risks: Dict[str, VehicleRiskMetrics]
    ) -> bool:
        """
        检测紧急事件（触发全面干预）

        触发条件：
        1. 任一车辆 TTC < 2.0s（极高危）
        2. 多辆车（≥3辆） TTC < 5.0s（高风险集群）
        3. 任一车辆 THW < 0.5s（极度接近）

        Args:
            vehicle_risks: 所有车辆的风险指标字典

        Returns:
            是否检测到紧急事件
        """
        emergency_count = 0
        high_risk_count = 0

        for veh_id, risk in vehicle_risks.items():
            # 条件1：极高危车辆
            if risk.ttc < self.ttc_threshold:
                emergency_count += 1

            # 条件2：高风险车辆
            if risk.ttc < 5.0:
                high_risk_count += 1

            # 条件3：极度接近
            if risk.thw < 0.5:
                emergency_count += 1

        # 判断：有极高危车辆 或 有多个高风险车辆
        is_emergency = (emergency_count > 0) or (high_risk_count >= 3)

        if is_emergency:
            print(f"[SmartICV] ⚠️  EMERGENCY DETECTED: {emergency_count} critical, {high_risk_count} high-risk vehicles")

        return is_emergency

    def should_update_icv_selection(self, current_step: int) -> bool:
        """
        判断是否应该更新ICV选择

        逻辑：
        1. 首次调用（从未更新过）：强制更新
        2. 定时兜底：每decision_interval步更新一次
        3. 事件触发：检测到紧急事件立即更新

        Args:
            current_step: 当前仿真步数

        Returns:
            是否应该更新ICV选择
        """
        # 首次调用（从未更新过）：强制更新
        if self.last_decision_step == -1:
            return True

        # 定时兜底
        if current_step - self.last_decision_step >= self.decision_interval:
            return True

        return False

    def select_top_k_vehicles(
        self,
        vehicle_scores: Dict[str, float],
        current_level: InterventionLevel = InterventionLevel.NORMAL
    ) -> Set[str]:
        """
        Top-K车辆选择（核心算法）

        Args:
            vehicle_scores: 所有车辆的评分字典 {veh_id: score}
            current_level: 当前干预级别

        Returns:
            选中的Top-K车辆ID集合
        """
        # 根据干预级别确定K值
        if current_level == InterventionLevel.EMERGENCY:
            k = self.emergency_top_k
        elif current_level == InterventionLevel.ELEVATED:
            k = self.elevated_top_k
        else:
            k = self.default_top_k

        # 排序并选择Top-K
        sorted_vehicles = sorted(vehicle_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_ids = set([veh_id for veh_id, score in sorted_vehicles[:k]])

        return top_k_ids

    def determine_intervention_level(
        self,
        vehicle_risks: Dict[str, VehicleRiskMetrics],
        avg_speed: float,
        congestion_detected: bool = False,
    ) -> InterventionLevel:
        """
        确定当前干预级别

        触发条件：
        - EMERGENCY: 检测到极高危事件（TTC < 2.0s）
        - ELEVATED: 检测到拥堵或多个高风险车辆
        - NORMAL: 无特殊风险

        Args:
            vehicle_risks: 所有车辆的风险指标
            avg_speed: 平均速度
            congestion_detected: 是否检测到拥堵（可选）

        Returns:
            当前干预级别
        """
        # 1. 检测紧急事件
        is_emergency = self.detect_emergency_events(vehicle_risks)
        if is_emergency:
            return InterventionLevel.EMERGENCY

        # 2. 检测升级条件
        high_risk_count = sum(1 for risk in vehicle_risks.values() if risk.ttc < 5.0)
        very_slow_speed = avg_speed < 5.0  # 平均速度低于5m/s

        if high_risk_count >= 2 or very_slow_speed or congestion_detected:
            return InterventionLevel.ELEVATED

        # 3. 默认正常级别
        return InterventionLevel.NORMAL

    def update_icv_selection(
        self,
        current_step: int,
        all_vehicles: List[Dict],  # [{veh_id, position, speed, ...}, ...]
        avg_speed: float,
        congestion_detected: bool = False,
    ) -> Tuple[Set[str], InterventionLevel]:
        """
        更新ICV选择（主入口）

        流程：
        1. 判断是否需要更新（定时兜底）
        2. 计算所有车辆的风险指标
        3. 确定当前干预级别
        4. 计算所有车辆的影响力评分
        5. 选择Top-K车辆

        Args:
            current_step: 当前仿真步数
            all_vehicles: 所有车辆信息列表
            avg_speed: 平均速度
            congestion_detected: 是否检测到拥堵

        Returns:
            (选中的ICV集合, 当前干预级别)
        """
        # ========== 1. 判断是否需要更新 ==========
        if not self.should_update_icv_selection(current_step):
            # 不需要更新，返回空集和当前级别
            return set(), self.current_level

        # ========== 2. 计算风险指标 ==========
        vehicle_risks = {}
        for veh_info in all_vehicles:
            veh_id = veh_info['veh_id']
            speed = veh_info.get('speed', 0.0)
            leader_speed = veh_info.get('leader_speed')
            leader_distance = veh_info.get('leader_distance')
            acceleration = veh_info.get('acceleration', 0.0)

            risk = self.compute_risk_metrics(
                veh_id, speed, leader_speed, leader_distance, acceleration
            )
            vehicle_risks[veh_id] = risk

        self.vehicle_risk_cache = vehicle_risks

        # ========== 3. 确定干预级别 ==========
        self.current_level = self.determine_intervention_level(
            vehicle_risks, avg_speed, congestion_detected
        )

        # 更新Top-K值
        if self.current_level == InterventionLevel.EMERGENCY:
            self.current_top_k = self.emergency_top_k
        elif self.current_level == InterventionLevel.ELEVATED:
            self.current_top_k = self.elevated_top_k
        else:
            self.current_top_k = self.default_top_k

        # ========== 4. 计算影响力评分 ==========
        vehicle_scores = {}
        for veh_info in all_vehicles:
            veh_id = veh_info['veh_id']
            risk = vehicle_risks[veh_id]

            score = self.compute_influence_score(
                veh_id=veh_id,
                position=veh_info.get('position', (0, 0)),
                speed=veh_info.get('speed', 0.0),
                lane_index=veh_info.get('lane_index', 0),
                edge_id=veh_info.get('edge_id', ''),
                acceleration=veh_info.get('acceleration', 0.0),
                risk_metrics=risk,
                in_bottleneck=veh_info.get('in_bottleneck', False),
                bottleneck_proximity=veh_info.get('bottleneck_proximity', 0.0),
            )
            vehicle_scores[veh_id] = score

        self.vehicle_scores_cache = vehicle_scores

        # ========== 5. 选择Top-K ==========
        selected_icvs = self.select_top_k_vehicles(vehicle_scores, self.current_level)

        # 更新时间戳
        self.last_decision_step = current_step

        # 打印日志
        if self.current_level != InterventionLevel.NORMAL:
            print(f"[SmartICV] Step {current_step}: Level={self.current_level.value}, "
                  f"Top-K={len(selected_icvs)}/{len(all_vehicles)} "
                  f"({len(selected_icvs)/max(len(all_vehicles),1)*100:.1f}%)")

        return selected_icvs, self.current_level

    def get_safety_barrier_action(
        self,
        veh_id: str,
        rl_action: np.ndarray,  # RL模型的输出 [acceleration, lane_change]
        risk_metrics: VehicleRiskMetrics,
    ) -> np.ndarray:
        """
        两级安全屏障

        Level 1 (规则卫士):
        - 每步检查
        - 动作范围裁剪
        - 确保v_next > 0

        Level 2 (紧急避险):
        - 触发条件：TTC < 2.0s
        - 动作：强制执行最大制动（-4.0 m/s²）
        - 训练：给予巨大负奖励

        Args:
            veh_id: 车辆ID
            rl_action: RL模型输出的原始动作
            risk_metrics: 车辆风险指标

        Returns:
            修正后的动作
        """
        # ========== Level 1: 规则卫士 ==========
        action = rl_action.copy()

        # 1. 动作范围裁剪
        # acceleration: [-4.0, 2.0] m/s²
        action[0] = np.clip(action[0], -4.0, 2.0)

        # lane_change: [0, 1] (0=保持车道, 1=变道)
        action[1] = np.clip(action[1], 0.0, 1.0)

        # ========== Level 2: 紧急避险 ==========
        if risk_metrics.ttc < self.ttc_threshold:
            # 极高危：强制最大制动
            action[0] = -4.0  # 最大制动
            action[1] = 0.0   # 禁止变道

            # 返回特殊标记，用于训练时给予负奖励
            return np.array([action[0], action[1], -1.0])  # -1.0表示紧急制动

        return action

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'current_level': self.current_level.value,
            'current_top_k': self.current_top_k,
            'last_decision_step': self.last_decision_step,
            'default_top_k': self.default_top_k,
            'emergency_top_k': self.emergency_top_k,
        }
