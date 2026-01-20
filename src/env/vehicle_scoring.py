"""
统一的车辆评分模块

提供两种评分方式：
1. 规则评分（RuleBasedScorer）：基于手工设计的规则
2. 神经网络评分（NeuralScorer）：基于GNN的评分，支持自动回退到规则评分

设计原则：
- 单一职责：每个评分器只负责一种评分方式
- 统一接口：所有评分器实现相同的接口
- 自动回退：神经网络评分失败时自动使用规则评分
"""

import numpy as np
from typing import Dict, List, Set, Optional
from abc import ABC, abstractmethod


class BaseVehicleScorer(ABC):
    """车辆评分器基类"""

    @abstractmethod
    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        计算车辆重要性评分

        Args:
            vehicle_states: 车辆状态字典 {veh_id: state_dict}
            context: 额外上下文信息（可选）

        Returns:
            评分字典 {veh_id: score}
        """
        pass


class RuleBasedScorer(BaseVehicleScorer):
    """
    基于规则的车辆评分器

    评分标准：
    1. 位置权重：瓶颈区域车辆优先（最多25分）
    2. 速度权重：速度异常车辆（最多10分）
    3. 车道权重：关键车道车辆（最多5分）
    4. 加速度权重：急加减速车辆（5分）
    5. 跟驰距离权重：跟驰风险（最多8分）
    """

    def __init__(self, frenet_system):
        """
        初始化规则评分器

        Args:
            frenet_system: Frenet坐标系系统
        """
        self.frenet_system = frenet_system

    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        计算所有车辆的规则评分

        Args:
            vehicle_states: 车辆状态字典
            context: 包含traci_lib等额外信息（可选）

        Returns:
            评分字典 {veh_id: score}
        """
        scores = {}

        for veh_id, state in vehicle_states.items():
            scores[veh_id] = self._compute_single_vehicle_score(
                veh_id, state, context
            )

        return scores

    def _compute_single_vehicle_score(
        self,
        veh_id: str,
        state: Dict,
        context: Optional[Dict] = None
    ) -> float:
        """
        计算单辆车的规则评分

        Args:
            veh_id: 车辆ID
            state: 车辆状态
            context: 上下文信息（包含traci_lib等）

        Returns:
            重要性评分（0-53分）
        """
        score = 0.0

        try:
            # ========== 1. 位置权重：优先选择瓶颈区域（最多25分） ==========
            s = state.get('s', 0.0)
            edge_id = state.get('edge_id', '')

            in_bottleneck = self.frenet_system.is_in_bottleneck(s=s, edge_id=edge_id)

            if in_bottleneck:
                score += 15.0  # 瓶颈区域车辆优先

            # 额外：距离瓶颈越近，权重越高
            if hasattr(self.frenet_system, 'bottleneck_s_range'):
                s_min, s_max = self.frenet_system.bottleneck_s_range.get(edge_id, (0, 0))
                if s_max > s_min:
                    bottleneck_center = (s_min + s_max) / 2.0
                    dist_to_bottleneck = abs(s - bottleneck_center)
                    proximity_score = max(0, 10.0 - dist_to_bottleneck / 100.0)
                    score += proximity_score

            # ========== 2. 速度权重：优先选择速度异常的车辆（最多10分） ==========
            speed = state.get('speed', 0.0)
            if speed < 5.0:
                score += 10.0  # 慢速车
            elif speed > 20.0:
                score += 5.0   # 快速车

            # ========== 3. 车道权重：优先选择关键车道（最多5分） ==========
            lane_index = state.get('lane_index', 0)
            if lane_index == 0:
                score += 5.0
            elif lane_index == 1:
                score += 3.0

            # ========== 4. 加速度权重：优先选择急加减速的车辆（5分） ==========
            acceleration = state.get('acceleration', 0.0)
            if abs(acceleration) > 2.0:
                score += 5.0

            # ========== 5. 跟驰距离权重：优先选择跟驰距离近的车辆（最多8分） ==========
            if context and 'traci_lib' in context:
                try:
                    traci_lib = context['traci_lib']
                    all_vehicle_ids = context.get('all_vehicle_ids', [])

                    leader_id = traci_lib.vehicle.getLeader(veh_id, 100.0)
                    if leader_id and leader_id in all_vehicle_ids:
                        leader_speed = traci_lib.vehicle.getSpeed(leader_id)
                        speed_diff = speed - leader_speed
                        if speed_diff < -5.0:
                            score += 8.0
                        elif speed_diff > 5.0:
                            score += 6.0
                except:
                    pass

        except Exception as e:
            score = 0.0

        return score


class NeuralScorer(BaseVehicleScorer):
    """
    基于神经网络的车辆评分器

    特性：
    - 使用GNN学习车辆交互
    - 支持自动回退到规则评分
    - 记录回退统计信息
    """

    def __init__(self, neural_scorer, rule_scorer: RuleBasedScorer, device: str = 'cuda'):
        """
        初始化神经网络评分器

        Args:
            neural_scorer: 神经网络评分器（从neural_vehicle_scorer导入）
            rule_scorer: 规则评分器（作为后备）
            device: 设备
        """
        self.neural_scorer = neural_scorer.to(device)
        self.rule_scorer = rule_scorer
        self.device = device

        # 统计信息
        self.neural_success_count = 0
        self.neural_failure_count = 0

    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        使用神经网络计算评分，失败时回退到规则评分

        Args:
            vehicle_states: 车辆状态字典
            context: 上下文信息（用于规则评分回退）

        Returns:
            评分字典 {veh_id: score}
        """
        if not vehicle_states:
            return {}

        # 尝试神经网络评分
        try:
            scores = self.neural_scorer.compute_scores(vehicle_states)
            self.neural_success_count += 1
            return scores
        except Exception as e:
            # 回退到规则评分
            self.neural_failure_count += 1
            return self.rule_scorer.compute_scores(vehicle_states, context)

    def get_stats(self) -> Dict[str, int]:
        """获取评分器统计信息"""
        return {
            'neural_success_count': self.neural_success_count,
            'neural_failure_count': self.neural_failure_count,
            'total_calls': self.neural_success_count + self.neural_failure_count,
            'success_rate': self.neural_success_count / max(1, self.neural_success_count + self.neural_failure_count)
        }


class UnifiedVehicleScorer:
    """
    统一的车辆评分器

    提供单一的接口来计算车辆评分，自动处理神经网络/规则评分的切换和回退。
    """

    def __init__(
        self,
        neural_scorer=None,
        frenet_system=None,
        use_neural: bool = True,
        device: str = 'cuda'
    ):
        """
        初始化统一评分器

        Args:
            neural_scorer: 神经网络评分器（可选）
            frenet_system: Frenet坐标系系统（规则评分需要）
            use_neural: 是否使用神经网络评分
            device: 设备
        """
        self.use_neural = use_neural and neural_scorer is not None

        # 创建规则评分器（作为后备或主要评分器）
        self.rule_scorer = RuleBasedScorer(frenet_system)

        # 创建神经网络评分器（如果启用）
        if self.use_neural:
            self.neural_scorer = NeuralScorer(
                neural_scorer=neural_scorer,
                rule_scorer=self.rule_scorer,
                device=device
            )
        else:
            self.neural_scorer = None

    def compute_scores(
        self,
        vehicle_states: Dict[str, Dict],
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        统一的评分接口

        Args:
            vehicle_states: 车辆状态字典
            context: 上下文信息（traci_lib, all_vehicle_ids等）

        Returns:
            评分字典 {veh_id: score}
        """
        if self.use_neural and self.neural_scorer is not None:
            return self.neural_scorer.compute_scores(vehicle_states, context)
        else:
            return self.rule_scorer.compute_scores(vehicle_states, context)

    def get_stats(self) -> Dict[str, int]:
        """获取评分器统计信息"""
        if self.neural_scorer:
            return self.neural_scorer.get_stats()
        else:
            return {'mode': 'rule_based_only'}


def create_vehicle_scorer_from_config(
    config: Dict,
    frenet_system,
    device: str = 'cuda'
) -> UnifiedVehicleScorer:
    """
    从配置创建车辆评分器

    Args:
        config: 配置字典
        frenet_system: Frenet坐标系系统
        device: 设备

    Returns:
        UnifiedVehicleScorer实例
    """
    # 读取神经网络ICV评分配置
    neural_icv_config = config.get('neural_icv_scoring', {})
    use_neural = neural_icv_config.get('enabled', True)

    neural_scorer_model = None
    if use_neural:
        try:
            from src.models.neural_vehicle_scorer import create_neural_icv_scorer

            # 创建NeuralICVScorer（包含模型）
            neural_icv_scorer = create_neural_icv_scorer(
                node_dim=neural_icv_config.get('node_dim', 9),
                hidden_dim=neural_icv_config.get('hidden_dim', 64),
                num_layers=neural_icv_config.get('num_layers', 3),
                num_heads=neural_icv_config.get('num_heads', 4),
                checkpoint_path=neural_icv_config.get('checkpoint_path', None),
                device=device
            )

            # 提取内部的神经网络模型
            neural_scorer_model = neural_icv_scorer.neural_scorer
            print(f"[OK] 神经网络ICV评分器已启用")
        except Exception as e:
            print(f"[WARN] 神经网络评分器初始化失败: {e}，使用规则评分")
            neural_scorer_model = None

    # 创建统一评分器
    scorer = UnifiedVehicleScorer(
        neural_scorer=neural_scorer_model,
        frenet_system=frenet_system,
        use_neural=neural_scorer_model is not None,
        device=device
    )

    return scorer
