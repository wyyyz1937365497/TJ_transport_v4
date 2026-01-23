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
- C_int = (α * Σacmd + β * Σlc) / (T_total * N_ICV)  ← N_ICV是场景中ICV总数（固定值）

【修复说明】v1.1
- 修复：使用正确的N_ICV（场景中ICV总数）而非num_controlled（本步被控车辆数）
- 影响：干预成本计算从错误的高估（30倍）修复为正确的计算
- 参考：docs/交通工程赛道-评测公式.md
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
    total_steps: int = 0            # 总仿真步数
    # ✅ 移除：num_controlled_vehicles（应该使用固定的N_ICV）

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
        lane_changes: int
    ):
        """
        更新干预成本统计（修复版）

        Args:
            accel_commands: 本步加速度指令数
            lane_changes: 本步换道指令数

        注意：不再需要num_controlled参数，因为归一化应使用固定的N_ICV
        """
        self.num_accel_commands += accel_commands
        self.num_lane_changes += lane_changes
        # ✅ 移除错误的max逻辑（之前高估了干预成本）

    def increment_step(self):
        """步数+1"""
        self.total_steps += 1


class OCRRewardCalculator:
    """
    OCR奖励计算器（修复版v1.1）

    核心功能:
    1. 追踪episode统计（完成率、稳定性、干预成本）
    2. 计算OCR及其增益
    3. 计算综合得分（直接对齐评测公式）

    修复说明：
    - 使用正确的N_ICV（场景中ICV总数）计算干预成本
    - 移除了错误的num_controlled_vehicles动态归一化
    """

    def __init__(
        self,
        num_icv_total: Optional[int] = None,  # ✅ 新增：场景中ICV总数
        baseline_ocr: Optional[float] = None,
        baseline_speed_std: Optional[float] = None,
        baseline_avg_accel: Optional[float] = None,
        k_penalty: float = 0.1,  # 干预成本惩罚系数
        w_efficiency: float = 0.7,  # 初赛效率权重
        w_stability: float = 0.3,   # 初赛稳定性权重
        alpha: float = 1.0,         # 加速度指令权重
        beta: float = 5.0,          # 换道指令权重
        use_improved_reward: bool = False,  # ✅ 新增：是否使用改进的即时奖励
        use_bottleneck_rewards: bool = False,  # ✅ OCR-MAX: 是否使用Bottleneck即时奖励
        bottleneck_reward_config: Optional[Dict] = None  # ✅ OCR-MAX: Bottleneck奖励配置
    ):
        """
        Args:
            num_icv_total: 场景中ICV总数（必需）
                = max_vehicles * icv_ratio
                例如: 600 * 0.25 = 150
                如果为None，会发出警告（向后兼容旧代码）
            baseline_ocr: 基准OCR
            baseline_speed_std: 基准速度标准差
            baseline_avg_accel: 基准平均绝对加速度
            k_penalty: 干预成本惩罚系数（越大惩罚越重）
            w_efficiency: 效率得分权重
            w_stability: 稳定性得分权重
            alpha: 加速度指令成本权重
            beta: 换道指令成本权重
            use_improved_reward: 是否使用改进的即时奖励计算
        """
        # ✅ 处理num_icv_total参数
        if num_icv_total is None:
            print("⚠️  [WARNING] OCRRewardCalculator未指定num_icv_total参数！")
            print("   这会导致干预成本计算错误。建议使用create_ocr_reward_calculator()自动推断。")
            print("   临时使用默认值150（假设600辆车的25%），请尽快修复！")
            self.num_icv_total = 150  # 临时默认值
        else:
            self.num_icv_total = num_icv_total

        self.baseline_ocr = baseline_ocr
        self.baseline_speed_std = baseline_speed_std
        self.baseline_avg_accel = baseline_avg_accel

        self.k_penalty = k_penalty
        self.w_efficiency = w_efficiency
        self.w_stability = w_stability
        self.alpha = alpha
        self.beta = beta
        self.use_improved_reward = use_improved_reward
        self.use_bottleneck_rewards = use_bottleneck_rewards  # ✅ OCR-MAX

        # ✅ OCR-MAX: 初始化Bottleneck奖励计算器
        if use_bottleneck_rewards:
            from src.training.bottleneck_rewards import BottleneckRewardComputer
            self.bottleneck_computer = BottleneckRewardComputer(
                **(bottleneck_reward_config or {})
            )
            print(f"[OCRRewardCalculator] ✅ OCR-MAX: Bottleneck奖励已启用")
        else:
            self.bottleneck_computer = None

        # 当前episode统计
        self.current_episode = EpisodeStatistics()

        print(f"[OCRRewardCalculator] 初始化（v1.1修复版）")
        print(f"  - N_ICV: {self.num_icv_total}")
        print(f"  - k_penalty: {self.k_penalty}")
        print(f"  - w_efficiency: {self.w_efficiency}")
        print(f"  - w_stability: {self.w_stability}")
        print(f"  - use_improved_reward: {self.use_improved_reward}")
        print(f"  - use_bottleneck_rewards: {self.use_bottleneck_rewards}")  # ✅ OCR-MAX

    def reset(self):
        """重置episode统计"""
        self.current_episode = EpisodeStatistics()

    def update(
        self,
        vehicle_info: List[Dict],
        accel_commands: int,
        lane_changes: int,
        num_controlled: Optional[int] = None,  # 保留参数（向后兼容），但不再用于计算
        action_dict: Optional[Dict] = None  # ✅ OCR-MAX: 新增参数，用于Bottleneck奖励计算
    ) -> float:
        """
        更新统计并计算即时奖励（修复版 + OCR-MAX扩展）

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
            num_controlled: 本步被控制的车辆数（可选，向后兼容，不影响计算）
            action_dict: ✅ OCR-MAX: 执行的动作字典 {veh_id: np.array([accel, lane_change])}

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

        # 更新干预成本统计（修复版：不再传递num_controlled）
        self.current_episode.update_intervention_cost(
            accel_commands,
            lane_changes
        )

        # 更新步数
        self.current_episode.increment_step()

        # 计算即时奖励
        if self.use_improved_reward:
            reward = self._compute_step_reward_improved()
        else:
            reward = self._compute_step_reward_simple(vehicle_info)

        # ✅ OCR-MAX: 集成Bottleneck即时奖励
        if self.use_bottleneck_rewards and action_dict is not None:
            bottleneck_reward = self.bottleneck_computer.compute_step_reward(
                vehicle_info,
                action_dict
            )
            # 组合: OCR即时奖励 + Bottleneck即时奖励
            reward = reward + 0.5 * bottleneck_reward  # 权重可调

        return reward

    def _compute_step_reward_simple(self, vehicle_info: List[Dict]) -> float:
        """
        计算即时奖励（简化版 - 保持向后兼容）

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

        # 干预惩罚（简化版）
        intervention_penalty = -0.01 * (
            self.alpha * self.current_episode.num_accel_commands +
            self.beta * self.current_episode.num_lane_changes
        ) / max(self.current_episode.total_steps, 1)

        # 组合
        reward = speed_reward + completion_rate + intervention_penalty

        return reward

    def _compute_step_reward_improved(self) -> float:
        """
        计算即时奖励（改进版）

        改进点：
        1. 使用正确的干预成本估计（基于N_ICV）
        2. 使用指数衰减的惩罚因子
        3. 更好地反映最终评测公式
        """
        stats = self.current_episode
        step = stats.total_steps

        # 1. 速度奖励（从累积样本计算）
        if len(stats.speed_samples) > 0:
            # 使用最近时刻的速度样本
            recent_samples = min(len(vehicle_info), len(stats.speed_samples)) if 'vehicle_info' in locals() else len(stats.speed_samples)
            if recent_samples > 0:
                avg_speed = np.mean(stats.speed_samples[-recent_samples:])
            else:
                avg_speed = np.mean(stats.speed_samples)
            speed_reward = avg_speed / 30.0
        else:
            speed_reward = 0.0

        # 2. 完成率奖励
        total_vehicles = stats.num_arrived + len(stats.enroute_traveled)
        if total_vehicles > 0:
            completion_rate = stats.num_arrived / total_vehicles
        else:
            completion_rate = 0.0

        # 3. ✅ 改进：使用准确的干预成本估计
        if self.num_icv_total > 0 and step > 0:
            # 当前累积的平均干预成本
            estimated_c_int = (
                self.alpha * stats.num_accel_commands +
                self.beta * stats.num_lane_changes
            ) / (step * self.num_icv_total)  # ✅ 使用固定的N_ICV

            # 惩罚因子（指数衰减）
            # P_int = e^(-k * C_int)
            # 惩罚 = 1 - P_int = 1 - e^(-k * C_int)
            intervention_penalty = -1.0 * (1.0 - np.exp(-self.k_penalty * estimated_c_int))
        else:
            intervention_penalty = 0.0

        # 4. 组合（使用官方权重）
        reward = (
            self.w_efficiency * (speed_reward + completion_rate) +
            intervention_penalty
        )

        return reward

    def compute_episode_score(self) -> Dict[str, float]:
        """
        计算episode最终得分（完整评测公式 - 修复版）

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

        # ========== ✅ 4. 计算干预成本（修复版） ==========
        if stats.total_steps > 0 and self.num_icv_total > 0:
            c_int = (
                self.alpha * stats.num_accel_commands +
                self.beta * stats.num_lane_changes
            ) / (stats.total_steps * self.num_icv_total)  # ✅ 使用固定的N_ICV
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
            'num_icv_total': self.num_icv_total,  # ✅ 添加固定的N_ICV
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
    config: Optional[Dict] = None,
    baseline_stats: Optional[Dict] = None,
    **kwargs
) -> OCRRewardCalculator:
    """
    创建OCR奖励计算器（增强版 - 自动推断N_ICV）

    Args:
        config: 训练配置字典（用于推断N_ICV）
            如果提供，会自动计算: num_icv_total = max_vehicles * icv_ratio
        baseline_stats: 基准统计字典（包含baseline_ocr等）
        **kwargs: 其他参数（可以覆盖config中的值）

    Returns:
        calculator: OCRRewardCalculator实例

    使用示例:
        # 方式1：从config自动推断（推荐）
        calculator = create_ocr_reward_calculator(config=config)

        # 方式2：手动指定N_ICV
        calculator = create_ocr_reward_calculator(num_icv_total=150)

        # 方式3：覆盖config中的值
        calculator = create_ocr_reward_calculator(
            config=config,
            k_penalty=0.15,  # 覆盖config中的值
            use_improved_reward=True
        )
    """
    # 从配置推断N_ICV
    num_icv_total = kwargs.pop('num_icv_total', None)

    if num_icv_total is None and config is not None:
        # 从config自动推断
        env_config = config.get('environment', {})
        max_vehicles = env_config.get('max_vehicles', 600)
        icv_ratio = env_config.get('icv_ratio', 0.25)
        num_icv_total = int(max_vehicles * icv_ratio)

        print(f"[create_ocr_reward_calculator] 从配置推断N_ICV:")
        print(f"  - max_vehicles: {max_vehicles}")
        print(f"  - icv_ratio: {icv_ratio}")
        print(f"  - num_icv_total: {num_icv_total}")

    # OCR奖励配置（从config或kwargs）
    if config is not None:
        ocr_config = config.get('ocr_rewards', {})
        k_penalty = kwargs.pop('k_penalty', ocr_config.get('k_penalty', 0.1))
        w_efficiency = kwargs.pop('w_efficiency', ocr_config.get('w_efficiency', 0.7))
        w_stability = kwargs.pop('w_stability', ocr_config.get('w_stability', 0.3))
        alpha = kwargs.pop('alpha', ocr_config.get('alpha', 1.0))
        beta = kwargs.pop('beta', ocr_config.get('beta', 5.0))
        use_improved = kwargs.pop('use_improved_reward', ocr_config.get('use_improved_reward', False))
    else:
        # 使用默认值或kwargs中的值
        k_penalty = kwargs.pop('k_penalty', 0.1)
        w_efficiency = kwargs.pop('w_efficiency', 0.7)
        w_stability = kwargs.pop('w_stability', 0.3)
        alpha = kwargs.pop('alpha', 1.0)
        beta = kwargs.pop('beta', 5.0)
        use_improved = kwargs.pop('use_improved_reward', False)

    # 基准统计
    if baseline_stats is None:
        baseline_stats = {}

    # 创建计算器
    return OCRRewardCalculator(
        num_icv_total=num_icv_total,  # ✅ 传入推断的值
        baseline_ocr=baseline_stats.get('baseline_ocr'),
        baseline_speed_std=baseline_stats.get('baseline_speed_std'),
        baseline_avg_accel=baseline_stats.get('baseline_avg_accel'),
        k_penalty=k_penalty,
        w_efficiency=w_efficiency,
        w_stability=w_stability,
        alpha=alpha,
        beta=beta,
        use_improved_reward=use_improved
    )
