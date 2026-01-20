"""
智能稀疏控制器

核心功能:
1. 按需决策：并非每步都控制，而是每隔N步决策一次
2. 车辆选择：使用统一的ICV评分系统选择关键车辆
3. 干预必要性判断：只在需要时干预，避免过度控制
4. 成本追踪：实时追踪干预成本

设计原则：
- 最小化 C_int = (α×加速指令 + β×换道指令) / (T × N_ICV)
- 最大化 OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque


class SparseController:
    """
    稀疏控制器基类

    实现按需决策机制，降低干预频率
    """

    def __init__(
        self,
        decision_interval: int = 10,  # 每10步决策一次
        top_k_ratio: float = 0.05,     # 控制Top 5%的车辆
        min_k: int = 3,                # 至少控制3辆车
        max_k: int = 15                # 最多控制15辆车
    ):
        self.decision_interval = decision_interval
        self.top_k_ratio = top_k_ratio
        self.min_k = min_k
        self.max_k = max_k

        # 状态追踪
        self.current_step = 0
        self.last_decision_step = -decision_interval  # 确保第0步会决策

        # 干预成本追踪
        self.total_accel_commands = 0
        self.total_lane_changes = 0
        self.controlled_vehicles_set = set()

    def should_make_decision(self, step: int) -> bool:
        """
        判断是否应该在当前步进行决策

        Args:
            step: 当前仿真步数

        Returns:
            True if should decide, False otherwise
        """
        # 规则1：定时决策
        if (step - self.last_decision_step) >= self.decision_interval:
            return True

        # 规则2：紧急情况立即干预
        # （子类可以实现is_emergency方法）
        if hasattr(self, 'is_emergency') and self.is_emergency(step):
            return True

        return False

    def compute_k(self, num_vehicles: int) -> int:
        """
        计算应该控制的车辆数K

        Args:
            num_vehicles: 当前车辆总数

        Returns:
            k: 应该控制的车辆数
        """
        k = int(num_vehicles * self.top_k_ratio)
        k = max(self.min_k, min(k, self.max_k))
        return k

    def track_intervention(
        self,
        accel_commands: int,
        lane_changes: int,
        vehicle_ids: List[str]
    ):
        """
        追踪干预成本

        Args:
            accel_commands: 加速度指令数
            lane_changes: 换道指令数
            vehicle_ids: 被控制的车辆ID列表
        """
        self.total_accel_commands += accel_commands
        self.total_lane_changes += lane_changes
        self.controlled_vehicles_set.update(vehicle_ids)

    def get_intervention_stats(self) -> Dict[str, int]:
        """获取干预统计"""
        return {
            'total_accel_commands': self.total_accel_commands,
            'total_lane_changes': self.total_lane_changes,
            'num_controlled_vehicles': len(self.controlled_vehicles_set),
        }

    def reset(self):
        """重置控制器状态"""
        self.current_step = 0
        self.last_decision_step = -self.decision_interval
        self.total_accel_commands = 0
        self.total_lane_changes = 0
        self.controlled_vehicles_set = set()


class RuleBasedSparseController(SparseController):
    """
    基于统一ICV评分系统的稀疏控制器

    使用统一的车辆评分系统选择关键车辆（支持神经网络/规则评分）
    """

    def __init__(
        self,
        decision_interval: int = 10,
        top_k_ratio: float = 0.05,
        min_k: int = 3,
        max_k: int = 15,
        # 评分系统
        vehicle_scorer=None,
        frenet_system=None
    ):
        super().__init__(decision_interval, top_k_ratio, min_k, max_k)

        self.vehicle_scorer = vehicle_scorer
        self.frenet_system = frenet_system

    def select_critical_vehicles(
        self,
        observation: Dict,
        k: Optional[int] = None
    ) -> List[str]:
        """
        使用统一的ICV评分系统选择关键车辆

        Args:
            observation: 完整的观测字典（来自环境）
                {
                    'vehicle_states': dict {veh_id: state_dict},
                    'vehicle_ids': list,
                    'icv_ids': set,
                    'global_stats': array,
                    'step': int
                }
            k: 选择的车辆数（如果为None，则自动计算）

        Returns:
            selected_ids: 选中的车辆ID列表
        """
        if k is None:
            k = self.compute_k(len(observation.get('vehicle_ids', [])))

        # 使用统一的评分系统
        if self.vehicle_scorer is not None:
            # ✅ 使用统一的评分器（神经网络或规则）
            vehicle_states = observation.get('vehicle_states', {})
            scores = self.vehicle_scorer.compute_scores(vehicle_states)

            # Top-K选择
            sorted_vehicles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected_ids = [veh_id for veh_id, score in sorted_vehicles[:k]]
        else:
            # ⚠️ 回退到简单规则（如果没有评分器）
            vehicle_ids = observation.get('vehicle_ids', [])
            selected_ids = vehicle_ids[:k] if k < len(vehicle_ids) else vehicle_ids

        return selected_ids


class LearnedSparseController(SparseController):
    """
    基于学习的稀疏控制器

    使用训练好的GNN模型选择关键车辆
    与LightweightPolicyV5配合使用
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        decision_interval: int = 10,
        top_k_ratio: float = 0.05,
        min_k: int = 3,
        max_k: int = 15,
        device: str = 'cuda:0'
    ):
        super().__init__(decision_interval, top_k_ratio, min_k, max_k)

        self.policy_model = policy_model
        self.device = device
        self.policy_model.eval()

    def select_critical_vehicles(
        self,
        obs: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[List[str], Dict]:
        """
        使用训练好的模型选择关键车辆

        Args:
            obs: [obs_dim] 观测
            k: 选择的车辆数（如果为None，则自动计算）

        Returns:
            selected_ids: 选中的车辆ID列表
            info: 额外信息（用于调试）
        """
        with torch.no_grad():
            # 使用模型的选择方法
            selected_indices, info = self.policy_model.select_vehicles(obs)

        # 如果指定了k，截取前k个
        if k is not None and k < len(selected_indices):
            selected_indices = selected_indices[:k]
            info['k'] = k
        else:
            k = len(selected_indices)

        # 生成车辆ID（假设格式为"veh_0", "veh_1", ...）
        selected_ids = [f"veh_{idx}" for idx in selected_indices]

        return selected_ids, info

    def compute_actions(
        self,
        obs: np.ndarray,
        selected_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        为选中的车辆计算控制动作

        Args:
            obs: [obs_dim] 观测
            selected_ids: 选中的车辆ID列表

        Returns:
            actions: 动作字典
                {
                    'accel': {veh_id: accel_value},
                    'lane_change': {veh_id: lane_change_prob}
                }
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 前向传播（注意：传递tensor而非numpy array）
            actions, _, _ = self.policy_model(obs_tensor, deterministic=True)

        # 解析动作（max_vehicles * 2维）
        # 前 max_vehicles 是加速度，后 max_vehicles 是换道概率
        max_vehicles = actions.shape[1] // 2
        accel_flat = actions[0, :max_vehicles].cpu().numpy()
        lane_change_flat = actions[0, max_vehicles:].cpu().numpy()

        # 只返回选中车辆的动作
        accel_dict = {}
        lane_change_dict = {}

        for veh_id in selected_ids:
            # 从ID中提取索引（"veh_123" -> 123）
            idx = int(veh_id.split('_')[1])
            if idx < max_vehicles:
                accel_dict[veh_id] = accel_flat[idx]
                lane_change_dict[veh_id] = lane_change_flat[idx]

        return {
            'accel': accel_dict,
            'lane_change': lane_change_dict
        }


# ========== 便捷函数 ==========

def create_sparse_controller(
    controller_type: str = 'rule',
    policy_model: Optional[torch.nn.Module] = None,
    vehicle_scorer=None,
    frenet_system=None,
    **kwargs
) -> SparseController:
    """
    创建稀疏控制器

    Args:
        controller_type: 'rule' 或 'learned'
        policy_model: 策略模型（learned模式需要）
        vehicle_scorer: 统一的车辆评分器（rule模式推荐）
        frenet_system: Frenet坐标系系统（rule模式需要）
        **kwargs: 其他参数

    Returns:
        controller: 稀疏控制器实例
    """
    if controller_type == 'rule':
        return RuleBasedSparseController(
            vehicle_scorer=vehicle_scorer,
            frenet_system=frenet_system,
            **kwargs
        )
    elif controller_type == 'learned':
        if policy_model is None:
            raise ValueError("policy_model is required for learned controller")
        return LearnedSparseController(
            policy_model=policy_model,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")
