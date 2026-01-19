"""
OCR直接奖励计算器

基于官方评测公式，直接计算OD完成率(OCR)增益作为奖励。

评测公式参考:
- S_efficiency = 100 * max(0, (OCR_AI - OCR_Base) / OCR_Base)
- S_stability = 100 * (0.4 * I_σv + 0.6 * I_|a|)
- P_intervention = e^(-k * C_int)
- S_total = (W_eff * S_efficiency + W_stab * S_stability) * P_int

其中:
- OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
- I_σv = -(σv_AI - σv_Base) / σv_Base
- I_|a| = -(|a|_AI - |a|_Base) / |a|_Base
- C_int = (α * Σacmd + β * Σlc) / (T_total * N_ICV)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch


@dataclass
class EpisodeStatistics:
    """单次episode的统计信息"""

    # ========== 车辆完成度 ==========
    num_arrived: int = 0          # 已到达目的地车辆数
    num_total: int = 0            # 总车辆数
    enroute_traveled: List[float] = None  # 在途车辆已行驶距离
    enroute_total: List[float] = None     # 在途车辆总距离

    # ========== 稳定性指标 ==========
    speed_samples: List[float] = None     # 所有车辆的所有时刻速度采样
    accel_samples: List[float] = None     # 所有车辆的所有时刻加速度采样

    # ========== 干预成本 ==========
    num_accel_commands: int = 0     # 加速度指令总数
    num_lane_changes: int = 0       # 换道指令总数
    num_controlled_vehicles: int = 0 # 被控制的车辆数
    total_steps: int = 0            # 总仿真步数

    def __post_init__(self):
        if self.enroute_traveled is None:
            self.enroute_traveled = []
        if self.enroute_total is None:
            self.enroute_total = []
        if self.speed_samples is None:
            self.speed_samples = []
        if self.accel_samples is None:
            self.accel_samples = []

    def update_stability(
        self,
        vehicle_speeds: np.ndarray,
        vehicle_accels: np.ndarray
    ):
        """
        更新稳定性统计

        Args:
            vehicle_speeds: [N] 所有车辆的速度
            vehicle_accels: [N] 所有车辆的加速度
        """
        self.speed_samples.extend(vehicle_speeds.tolist())
        self.accel_samples.extend(vehicle_accels.tolist())

    def update_intervention_cost(
        self,
        accel_commands: int,
        lane_changes: int,
        num_controlled: int
    ):
        """
        更新干预成本统计

        Args:
            accel_commands: 本步加速度指令数
            lane_changes: 本步换道指令数
            num_controlled: 本步被控制的车辆数
        """
        self.num_accel_commands += accel_commands
        self.num_lane_changes += lane_changes
        self.num_controlled_vehicles = max(
            self.num_controlled_vehicles,
            num_controlled
        )

    def increment_step(self):
        """步数+1"""
        self.total_steps += 1


class OCRRewardCalculator:
    """
    OCR奖励计算器

    核心功能:
    1. 追踪episode统计（完成率、稳定性、干预成本）
    2. 计算OCR及其增益
    3. 计算综合得分（直接对齐评测公式）
    """

    def __init__(
        self,
        baseline_ocr: Optional[float] = None,
        baseline_speed_std: Optional[float] = None,
        baseline_avg_accel: Optional[float] = None,
        k_penalty: float = 0.1,  # 干预成本惩罚系数
        w_efficiency: float = 0.7,  # 初赛效率权重
        w_stability: float = 0.3,   # 初赛稳定性权重
        alpha: float = 1.0,         # 加速度指令权重
        beta: float = 5.0           # 换道指令权重
    ):
        """
        Args:
            baseline_ocr: 基准OCR（需要在评测前通过baseline run获得）
            baseline_speed_std: 基准速度标准差
            baseline_avg_accel: 基准平均绝对加速度
            k_penalty: 干预成本惩罚系数（越大惩罚越重）
            w_efficiency: 效率得分权重
            w_stability: 稳定性得分权重
            alpha: 加速度指令成本权重
            beta: 换道指令成本权重
        """
        self.baseline_ocr = baseline_ocr
        self.baseline_speed_std = baseline_speed_std
        self.baseline_avg_accel = baseline_avg_accel

        self.k_penalty = k_penalty
        self.w_efficiency = w_efficiency
        self.w_stability = w_stability
        self.alpha = alpha
        self.beta = beta

        # 当前episode统计
        self.current_episode = EpisodeStatistics()

    def reset(self):
        """重置episode统计"""
        self.current_episode = EpisodeStatistics()

    def update(
        self,
        vehicle_info: List[Dict],
        accel_commands: int,
        lane_changes: int,
        num_controlled: int
    ) -> float:
        """
        更新统计并计算即时奖励

        Args:
            vehicle_info: 车辆信息列表
                [{
                    'id': str,
                    'arrived': bool,
                    'traveled': float,  # 已行驶距离
                    'total': float,     # 总距离
                    'speed': float,
                    'accel': float
                }, ...]
            accel_commands: 本步加速度指令数
            lane_changes: 本步换道指令数
            num_controlled: 本步被控制的车辆数

        Returns:
            reward: float 即时奖励
        """
        # 更新完成度统计
        for veh in vehicle_info:
            if veh['arrived']:
                self.current_episode.num_arrived += 1

            # 如果在途，记录距离
            if not veh['arrived'] and veh['total'] > 0:
                self.current_episode.enroute_traveled.append(veh['traveled'])
                self.current_episode.enroute_total.append(veh['total'])

        # 更新稳定性统计
        speeds = [v['speed'] for v in vehicle_info]
        accels = [v['accel'] for v in vehicle_info]
        self.current_episode.update_stability(
            np.array(speeds),
            np.array(accels)
        )

        # 更新干预成本统计
        self.current_episode.update_intervention_cost(
            accel_commands,
            lane_changes,
            num_controlled
        )

        # 更新步数
        self.current_episode.increment_step()

        # 计算即时奖励（简化版：OCR增益）
        # 注意：完整的S_total只在episode结束时计算
        reward = self._compute_step_reward(vehicle_info)

        return reward

    def _compute_step_reward(self, vehicle_info: List[Dict]) -> float:
        """
        计算即时奖励（简化版）

        使用速度和完成率作为即时奖励
        """
        if len(vehicle_info) == 0:
            return 0.0

        # 平均速度（归一化到[0, 1]）
        avg_speed = np.mean([v['speed'] for v in vehicle_info])
        speed_reward = avg_speed / 30.0  # 假设最大速度30m/s

        # 完成率奖励
        num_arrived = sum(1 for v in vehicle_info if v['arrived'])
        completion_rate = num_arrived / len(vehicle_info)

        # 干预惩罚（即时）
        intervention_penalty = -0.01 * (
            self.alpha * self.current_episode.num_accel_commands +
            self.beta * self.current_episode.num_lane_changes
        ) / max(self.current_episode.total_steps, 1)

        # 组合
        reward = speed_reward + completion_rate + intervention_penalty

        return reward

    def compute_episode_score(self) -> Dict[str, float]:
        """
        计算episode最终得分（完整评测公式）

        Returns:
            scores: 包含各项得分的字典
        """
        stats = self.current_episode

        # ========== 1. 计算OCR ==========
        # OCR = (N_arrived + Σ(d_traveled / d_total)) / N_total
        enroute_completion = 0.0
        if len(stats.enroute_traveled) > 0:
            enroute_completion = sum([
                t / max(tt, 1.0)
                for t, tt in zip(stats.enroute_traveled, stats.enroute_total)
            ])

        total_vehicles = stats.num_arrived + len(stats.enroute_traveled)
        ocr = (stats.num_arrived + enroute_completion) / max(total_vehicles, 1)

        # ========== 2. 计算效率得分 ==========
        if self.baseline_ocr is not None and self.baseline_ocr > 0:
            ocr_gain = (ocr - self.baseline_ocr) / self.baseline_ocr
            s_efficiency = 100.0 * max(0, ocr_gain)
        else:
            # 没有基准时，直接使用OCR作为效率得分
            s_efficiency = 100.0 * ocr

        # ========== 3. 计算稳定性得分 ==========
        # 速度标准差
        speed_std = np.std(stats.speed_samples) if len(stats.speed_samples) > 0 else 0.0

        # 平均绝对加速度
        avg_accel = np.mean(np.abs(stats.accel_samples)) if len(stats.accel_samples) > 0 else 0.0

        if self.baseline_speed_std is not None and self.baseline_speed_std > 0:
            i_speed_std = -(speed_std - self.baseline_speed_std) / self.baseline_speed_std
        else:
            i_speed_std = 0.0

        if self.baseline_avg_accel is not None and self.baseline_avg_accel > 0:
            i_accel = -(avg_accel - self.baseline_avg_accel) / self.baseline_avg_accel
        else:
            i_accel = 0.0

        s_stability = 100.0 * (
            0.4 * max(0, i_speed_std) +
            0.6 * max(0, i_accel)
        )

        # ========== 4. 计算干预成本 ==========
        if stats.total_steps > 0 and stats.num_controlled_vehicles > 0:
            c_int = (
                self.alpha * stats.num_accel_commands +
                self.beta * stats.num_lane_changes
            ) / (stats.total_steps * stats.num_controlled_vehicles)
        else:
            c_int = 0.0

        # 惩罚因子
        p_intervention = np.exp(-self.k_penalty * c_int)

        # ========== 5. 计算总分 ==========
        s_total = (
            self.w_efficiency * s_efficiency +
            self.w_stability * s_stability
        ) * p_intervention

        return {
            'ocr': ocr,
            's_efficiency': s_efficiency,
            's_stability': s_stability,
            'c_int': c_int,
            'p_intervention': p_intervention,
            's_total': s_total,
            'speed_std': speed_std,
            'avg_accel': avg_accel,
        }

    def get_statistics(self) -> Dict[str, float]:
        """获取当前统计信息（用于TensorBoard日志）"""
        stats = self.current_episode

        return {
            'num_arrived': stats.num_arrived,
            'num_total': stats.num_arrived + len(stats.enroute_traveled),
            'num_accel_commands': stats.num_accel_commands,
            'num_lane_changes': stats.num_lane_changes,
            'num_controlled_vehicles': stats.num_controlled_vehicles,
            'total_steps': stats.total_steps,
        }


class BaselineStatisticsCollector:
    """
    基准统计收集器

    用于运行baseline场景，收集基准OCR、速度标准差、加速度等
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.num_arrived = 0
        self.enroute_traveled = []
        self.enroute_total = []
        self.speed_samples = []
        self.accel_samples = []
        self.total_steps = 0

    def update(self, vehicle_info: List[Dict]):
        """
        更新统计

        Args:
            vehicle_info: 车辆信息列表
        """
        for veh in vehicle_info:
            if veh['arrived']:
                self.num_arrived += 1
            elif veh['total'] > 0:
                self.enroute_traveled.append(veh['traveled'])
                self.enroute_total.append(veh['total'])

        # 记录速度和加速度
        self.speed_samples.extend([v['speed'] for v in vehicle_info])
        self.accel_samples.extend([v['accel'] for v in vehicle_info])

        self.total_steps += 1

    def compute_baseline(self) -> Dict[str, float]:
        """计算基准统计"""
        # OCR
        enroute_completion = sum([
            t / max(tt, 1.0)
            for t, tt in zip(self.enroute_traveled, self.enroute_total)
        ])

        total_vehicles = self.num_arrived + len(self.enroute_traveled)
        ocr = (self.num_arrived + enroute_completion) / max(total_vehicles, 1)

        # 速度标准差
        speed_std = np.std(self.speed_samples) if len(self.speed_samples) > 0 else 0.0

        # 平均绝对加速度
        avg_accel = np.mean(np.abs(self.accel_samples)) if len(self.accel_samples) > 0 else 0.0

        return {
            'baseline_ocr': ocr,
            'baseline_speed_std': speed_std,
            'baseline_avg_accel': avg_accel,
            'total_steps': self.total_steps,
        }


# ========== 便捷函数 ==========

def create_ocr_reward_calculator(
    baseline_stats: Optional[Dict] = None,
    **kwargs
) -> OCRRewardCalculator:
    """
    创建OCR奖励计算器

    Args:
        baseline_stats: 基准统计字典（包含baseline_ocr等）
        **kwargs: 其他参数

    Returns:
        calculator: OCRRewardCalculator实例
    """
    if baseline_stats is None:
        baseline_stats = {}

    return OCRRewardCalculator(
        baseline_ocr=baseline_stats.get('baseline_ocr'),
        baseline_speed_std=baseline_stats.get('baseline_speed_std'),
        baseline_avg_accel=baseline_stats.get('baseline_avg_accel'),
        **kwargs
    )
