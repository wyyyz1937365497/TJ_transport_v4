"""
智能稀疏控制器

核心功能:
1. 按需决策：并非每步都控制，而是每隔N步决策一次
2. 车辆选择：只选择关键车辆进行控制
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
    基于规则的稀疏控制器

    使用交通工程规则选择关键车辆，无需训练
    适合作为baseline或快速原型
    """

    def __init__(
        self,
        decision_interval: int = 10,
        top_k_ratio: float = 0.05,
        min_k: int = 3,
        max_k: int = 15,
        # 规则参数
        bottleneck_threshold: float = 0.7,    # 瓶颈区域阈值（纵向位置）
        speed_threshold_ratio: float = 0.7,   # 速度异常阈值（相对平均速度）
        merge_zone_start: float = 0.5,        # 汇流区起始位置
        merge_zone_end: float = 0.8,          # 汇流区结束位置
        ttc_threshold: float = 3.0,           # TTC阈值（秒）
    ):
        super().__init__(decision_interval, top_k_ratio, min_k, max_k)

        self.bottleneck_threshold = bottleneck_threshold
        self.speed_threshold_ratio = speed_threshold_ratio
        self.merge_zone_start = merge_zone_start
        self.merge_zone_end = merge_zone_end
        self.ttc_threshold = ttc_threshold

    def select_critical_vehicles(
        self,
        vehicle_states: np.ndarray,  # [N, 9]
        vehicle_ids: List[str],
        k: Optional[int] = None
    ) -> List[str]:
        """
        基于规则选择关键车辆

        Args:
            vehicle_states: [N, 9] 车辆状态
                [s, d, vs, vd, speed, accel, lane, angle, is_icv]
            vehicle_ids: [N] 车辆ID列表
            k: 选择的车辆数（如果为None，则自动计算）

        Returns:
            selected_ids: 选中的车辆ID列表
        """
        if k is None:
            k = self.compute_k(len(vehicle_states))

        num_vehicles = len(vehicle_states)
        if num_vehicles == 0:
            return []

        # ========== 提取特征 ==========
        s = vehicle_states[:, 0]          # 纵向位置 [0, 1]
        lanes = vehicle_states[:, 6]      # 车道索引
        speeds = vehicle_states[:, 4]     # 速度 [0, 1] 归一化

        # ========== 规则1：瓶颈区域的车辆 ==========
        # 主线末端(s > 0.7)且非匝道(lane < 3)
        in_bottleneck = (s > self.bottleneck_threshold) & (lanes < 3)
        score_bottleneck = in_bottleneck.astype(float)

        # ========== 规则2：速度异常的车辆 ==========
        # 计算平均速度（只考虑主线车辆）
        mainline_mask = lanes < 3
        if mainline_mask.sum() > 1:
            avg_speed = speeds[mainline_mask].mean()
            speed_std = speeds[mainline_mask].std()
            # 慢车（速度 < 平均速度 - 1.5倍标准差）
            is_slow = speeds < (avg_speed - 1.5 * speed_std)
        else:
            is_slow = np.zeros(num_vehicles, dtype=bool)

        score_speed = is_slow.astype(float)

        # ========== 规则3：汇流区车辆 ==========
        # 匝道车道(lane >= 3)且在汇流区
        in_merge_zone = (
            (s > self.merge_zone_start) &
            (s < self.merge_zone_end) &
            (lanes >= 3)
        )
        score_merge = in_merge_zone.astype(float)

        # ========== 规则4：紧急车辆（TTC低）==========
        # 简化处理：使用速度和位置近似TTC
        # 实际应用中应该使用真实TTC计算
        score_emergency = np.zeros(num_vehicles)

        # ========== 组合评分 ==========
        total_score = (
            0.3 * score_bottleneck +
            0.3 * score_speed +
            0.3 * score_merge +
            0.1 * score_emergency
        )

        # ========== Top-K选择 ==========
        top_k_indices = np.argsort(total_score)[-k:]

        # 获取对应的车辆ID
        selected_ids = [vehicle_ids[i] for i in top_k_indices if i < len(vehicle_ids)]

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
    **kwargs
) -> SparseController:
    """
    创建稀疏控制器

    Args:
        controller_type: 'rule' 或 'learned'
        policy_model: 策略模型（learned模式需要）
        **kwargs: 其他参数

    Returns:
        controller: 稀疏控制器实例
    """
    if controller_type == 'rule':
        return RuleBasedSparseController(**kwargs)
    elif controller_type == 'learned':
        if policy_model is None:
            raise ValueError("policy_model is required for learned controller")
        return LearnedSparseController(
            policy_model=policy_model,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown controller_type: {controller_type}")
