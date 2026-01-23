"""
Bottleneck-Oriented Reward Shaping for Traffic Control

核心思想：
- OCR奖励只在episode结束给出，太稀疏
- 添加即时奖励引导策略关注瓶颈区域

3个即时奖励组件：
1. Bottleneck Throughput Reward: 奖励瓶颈区域车辆快速通过
2. On-Ramp Queue Reward: 奖励减少匝道排队
3. Conflict Avoidance Reward: 奖励避免冲突（TTC低）

论文参考：
- Improved multi-agent deep RL-based VSL control for merge bottlenecks (2025)
- Physics-Informed RL for Ramp Control (2022)
"""

import numpy as np
from typing import List, Dict, Optional


class BottleneckRewardComputer:
    """
    瓶颈导向的即时奖励计算器

    Args:
        bottleneck_s_min (float): 瓶颈区域起点（米）
        bottleneck_s_max (float): 瓶颈区域终点（米）
        w_throughput (float): 吞吐量奖励权重
        w_queue (float): 排队奖励权重
        w_conflict (float): 冲突避免奖励权重
        ttc_threshold (float): TTC阈值（秒）
        min_speed (float): 最小速度（m/s）
    """

    def __init__(
        self,
        # 瓶颈区域定义
        bottleneck_s_min: float = 1200.0,  # E17, E19汇入区域
        bottleneck_s_max: float = 2200.0,

        # 奖励权重
        w_throughput: float = 0.5,
        w_queue: float = 0.3,
        w_conflict: float = 0.2,

        # 阈值参数
        ttc_threshold: float = 3.0,
        min_speed: float = 5.0
    ):
        self.bottleneck_s_min = bottleneck_s_min
        self.bottleneck_s_max = bottleneck_s_max

        self.w_throughput = w_throughput
        self.w_queue = w_queue
        self.w_conflict = w_conflict

        self.ttc_threshold = ttc_threshold
        self.min_speed = min_speed

    def compute_step_reward(
        self,
        vehicle_info: List[Dict],
        action_dict: Dict[str, np.ndarray]
    ) -> float:
        """
        计算单步即时奖励

        Args:
            vehicle_info: 车辆信息列表
                每个元素包含: {'id': str, 's': float, 'speed': float, 'ttc': float, ...}
            action_dict: 执行的动作
                {veh_id: np.array([accel, lane_change])}

        Returns:
            reward: float 即时奖励（归一化到约[-1, 1]）
        """
        # 1. 吞吐量奖励
        throughput_reward = self._compute_throughput_reward(vehicle_info)

        # 2. 排队奖励
        queue_reward = self._compute_queue_reward(vehicle_info)

        # 3. 冲突避免奖励
        conflict_reward = self._compute_conflict_reward(
            vehicle_info,
            action_dict
        )

        # 组合
        total_reward = (
            self.w_throughput * throughput_reward +
            self.w_queue * queue_reward +
            self.w_conflict * conflict_reward
        )

        return total_reward

    def _compute_throughput_reward(self, vehicle_info: List[Dict]) -> float:
        """
        吞吐量奖励：
        - 瓶颈区域内车辆速度越快，奖励越高
        - 鼓励ICV在瓶颈区域保持高速
        """
        reward = 0.0
        count = 0

        for veh in vehicle_info:
            s = veh.get('s', 0.0)
            speed = veh.get('speed', 0.0)

            # 是否在瓶颈区域
            if self.bottleneck_s_min <= s <= self.bottleneck_s_max:
                # 速度奖励（归一化到[0, 1]）
                # speed > min_speed: reward > 0
                # speed < min_speed: reward < 0
                speed_reward = (speed - self.min_speed) / self.min_speed
                speed_reward = np.clip(speed_reward, -1.0, 1.0)
                reward += speed_reward
                count += 1

        # 平均奖励
        if count > 0:
            reward = reward / count
        else:
            reward = 0.0

        return reward

    def _compute_queue_reward(self, vehicle_info: List[Dict]) -> float:
        """
        排队奖励：
        - 匝道车辆速度越快，奖励越高
        - 减少匝道排队可以改善整体流量
        """
        reward = 0.0
        count = 0

        for veh in vehicle_info:
            s = veh.get('s', 0.0)
            speed = veh.get('speed', 0.0)

            # 是否在匝道区域（假设s < 1200为匝道）
            if s < 1200.0:
                # 速度越慢，惩罚越大
                if speed < self.min_speed:
                    reward -= 1.0  # 惩罚
                else:
                    reward += 0.5  # 奖励
                count += 1

        # 平均奖励
        if count > 0:
            reward = reward / count
        else:
            reward = 0.0

        return reward

    def _compute_conflict_reward(
        self,
        vehicle_info: List[Dict],
        action_dict: Dict[str, np.ndarray]
    ) -> float:
        """
        冲突避免奖励：
        - TTC < 阈值时，减速动作奖励
        - TTC高时，加速动作奖励
        """
        reward = 0.0
        count = 0

        for veh in vehicle_info:
            veh_id = veh.get('id', '')
            ttc = veh.get('ttc', float('inf'))

            if veh_id not in action_dict:
                continue

            action = action_dict[veh_id]
            accel = action[0]  # 加速度

            # TTC低（高风险）时，奖励减速
            if ttc < self.ttc_threshold:
                if accel < 0:  # 减速
                    reward += 1.0
                else:  # 加速（危险）
                    reward -= 1.0
            # TTC高（安全）时，奖励加速
            else:
                if accel > 0:  # 加速
                    reward += 0.5
                else:  # 减速（不必要）
                    reward -= 0.2

            count += 1

        # 平均奖励
        if count > 0:
            reward = reward / count
        else:
            reward = 0.0

        return reward


# 测试代码
if __name__ == '__main__':
    # 创建奖励计算器
    computer = BottleneckRewardComputer()

    # 模拟车辆信息
    vehicle_info = [
        {'id': 'veh1', 's': 1500.0, 'speed': 10.0, 'ttc': 5.0},  # 在瓶颈区域，高速
        {'id': 'veh2', 's': 500.0, 'speed': 3.0, 'ttc': 2.0},    # 在匝道，低速，TTC低
        {'id': 'veh3', 's': 800.0, 'speed': 8.0, 'ttc': 10.0},   # 在匝道，高速，TTC高
    ]

    # 模拟动作
    action_dict = {
        'veh1': np.array([1.0, 0.0]),   # 加速（在瓶颈）
        'veh2': np.array([-1.0, 0.0]),  # 减速（在匝道，TTC低）
        'veh3': np.array([0.5, 0.0]),   # 轻微加速
    }

    # 计算奖励
    reward = computer.compute_step_reward(vehicle_info, action_dict)

    print("=== BottleneckReward测试 ===")
    print(f"总奖励: {reward:.4f}")
    print(f"吞吐量奖励权重: {computer.w_throughput}")
    print(f"排队奖励权重: {computer.w_queue}")
    print(f"冲突避免奖励权重: {computer.w_conflict}")

    # 分别测试每个组件
    throughput = computer._compute_throughput_reward(vehicle_info)
    queue = computer._compute_queue_reward(vehicle_info)
    conflict = computer._compute_conflict_reward(vehicle_info, action_dict)

    print(f"\n=== 各组件奖励 ===")
    print(f"吞吐量奖励: {throughput:.4f}")
    print(f"排队奖励: {queue:.4f}")
    print(f"冲突避免奖励: {conflict:.4f}")

    # 验证组合
    combined = (
        computer.w_throughput * throughput +
        computer.w_queue * queue +
        computer.w_conflict * conflict
    )
    print(f"\n验证组合: {combined:.4f} (应该等于总奖励: {reward:.4f})")
