"""
统一车辆评分模块

提供统一的车辆评分接口，支持神经网络评分和规则评分两种方式。
默认使用神经网络评分，失效时回退到规则评分。
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env.rule_based_scorer import RuleBasedVehicleScorer
from src.models.icv_gnn_scorer import create_icv_gnn_scorer


class UnifiedVehicleScorer:
    """
    统一车辆评分器

    支持两种评分方式：
    1. 神经网络评分（默认，基于GNN）
    2. 规则评分（备选，基于交通工程理论）

    优先使用神经网络评分，出错时自动回退到规则评分。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        frenet_system=None,
        device: str = 'cuda'
    ):
        """
        初始化统一评分器

        Args:
            config: 配置字典
            frenet_system: Frenet坐标系系统
            device: 设备（'cuda'或'cpu'）
        """
        self.config = config
        self.frenet_system = frenet_system
        self.device = device

        # 配置参数
        neural_config = config.get('neural_icv_scoring', {})
        self.use_neural = neural_config.get('enabled', True)  # 默认启用神经网络
        self.fallback_on_error = neural_config.get('fallback_on_error', True)

        # 初始化评分器
        self.neural_scorer = None
        self.rule_scorer = None

        if self.use_neural:
            try:
                checkpoint_path = neural_config.get('checkpoint_path')
                self.neural_scorer = create_icv_gnn_scorer(
                    config=config,
                    device=device,
                    checkpoint_path=checkpoint_path
                )
                print(f"[UnifiedVehicleScorer] 神经网络评分器已启用 (device={device})")
                if checkpoint_path:
                    print(f"[UnifiedVehicleScorer] 预训练权重: {checkpoint_path}")
                else:
                    print(f"[UnifiedVehicleScorer] 使用随机初始化模型")
            except Exception as e:
                print(f"[UnifiedVehicleScorer] 警告：神经网络评分器初始化失败: {e}")
                if self.fallback_on_error:
                    print("[UnifiedVehicleScorer] 将回退到规则评分器")
                    self.use_neural = False
                else:
                    raise

        # 始终初始化规则评分器作为备选
        try:
            self.rule_scorer = RuleBasedVehicleScorer(
                config=config,
                frenet_system=frenet_system
            )
            print("[UnifiedVehicleScorer] 规则评分器已初始化（备选）")
        except Exception as e:
            print(f"[UnifiedVehicleScorer] 警告：规则评分器初始化失败: {e}")

        # 统计信息
        self.stats = {
            'neural_calls': 0,
            'rule_calls': 0,
            'errors': 0,
        }

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
        # 优先使用神经网络评分
        if self.use_neural and self.neural_scorer is not None:
            try:
                scores = self.neural_scorer.compute_scores(vehicle_states, context)
                self.stats['neural_calls'] += 1
                return scores
            except Exception as e:
                self.stats['errors'] += 1
                print(f"[UnifiedVehicleScorer] 神经网络评分失败: {e}")

                if self.fallback_on_error and self.rule_scorer is not None:
                    print("[UnifiedVehicleScorer] 回退到规则评分器")
                    scores = self.rule_scorer.compute_scores(vehicle_states, context)
                    self.stats['rule_calls'] += 1
                    return scores
                else:
                    raise

        # 使用规则评分
        elif self.rule_scorer is not None:
            scores = self.rule_scorer.compute_scores(vehicle_states, context)
            self.stats['rule_calls'] += 1
            return scores

        else:
            raise RuntimeError("[UnifiedVehicleScorer] 没有可用的评分器")

    def get_top_k_vehicles(
        self,
        vehicle_states: Dict[str, Dict],
        context: Dict,
        k: int,
        min_score: float = 0.0
    ) -> List[str]:
        """
        获取评分最高的K个车辆

        Args:
            vehicle_states: 车辆状态字典
            context: 上下文信息
            k: 返回的车辆数量
            min_score: 最低评分阈值

        Returns:
            top_k_vehicles: 按评分排序的车辆ID列表
        """
        scores = self.compute_scores(vehicle_states, context)

        # 按评分排序
        sorted_vehicles = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 过滤低于阈值的车辆
        filtered_vehicles = [
            (veh_id, score) for veh_id, score in sorted_vehicles
            if score >= min_score
        ]

        # 取前K个
        top_k_vehicles = [
            veh_id for veh_id, score in filtered_vehicles[:k]
        ]

        return top_k_vehicles

    def get_score_breakdown(
        self,
        vehicle_states: Dict[str, Dict],
        context: Dict,
        veh_id: str
    ) -> Dict[str, Any]:
        """
        获取单个车辆的评分详细分解（用于调试和可视化）

        Args:
            vehicle_states: 车辆状态字典
            context: 上下文信息
            veh_id: 车辆ID

        Returns:
            breakdown: 评分分解字典
        """
        if self.rule_scorer is not None and veh_id in vehicle_states:
            return self.rule_scorer.get_score_breakdown(vehicle_states, context, veh_id)
        else:
            return {
                'total': 0.0,
                'error': 'Vehicle not found or rule scorer unavailable'
            }

    def get_statistics(self) -> Dict[str, int]:
        """
        获取评分器使用统计

        Returns:
            stats: 统计信息字典
        """
        return self.stats.copy()

    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'neural_calls': 0,
            'rule_calls': 0,
            'errors': 0,
        }

    def switch_to_neural(self, checkpoint_path: Optional[str] = None):
        """
        切换到神经网络评分模式

        Args:
            checkpoint_path: 预训练权重路径（可选）
        """
        if self.neural_scorer is None:
            try:
                self.neural_scorer = create_icv_gnn_scorer(
                    config=self.config,
                    device=self.device,
                    checkpoint_path=checkpoint_path
                )
                self.use_neural = True
                print("[UnifiedVehicleScorer] 已切换到神经网络评分模式")
            except Exception as e:
                print(f"[UnifiedVehicleScorer] 切换失败: {e}")
        else:
            if checkpoint_path is not None:
                self.neural_scorer.load_checkpoint(checkpoint_path)
            self.use_neural = True
            print("[UnifiedVehicleScorer] 已切换到神经网络评分模式")

    def switch_to_rule(self):
        """切换到规则评分模式"""
        self.use_neural = False
        print("[UnifiedVehicleScorer] 已切换到规则评分模式")

    def get_current_mode(self) -> str:
        """
        获取当前评分模式

        Returns:
            mode: 'neural' 或 'rule'
        """
        return 'neural' if self.use_neural else 'rule'


def create_vehicle_scorer_from_config(
    config: Dict[str, Any],
    frenet_system=None,
    device: str = 'cuda'
) -> UnifiedVehicleScorer:
    """
    从配置创建统一车辆评分器的工厂函数

    Args:
        config: 配置字典
        frenet_system: Frenet坐标系系统
        device: 设备（'cuda'或'cpu'）

    Returns:
        scorer: UnifiedVehicleScorer实例
    """
    scorer = UnifiedVehicleScorer(
        config=config,
        frenet_system=frenet_system,
        device=device
    )
    return scorer
