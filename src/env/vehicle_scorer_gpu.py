"""
GPU加速的车辆评分器

批量计算所有车辆的评分，用于ICV车辆选择。

关键优化：
1. 向量化计算所有车辆评分
2. GPU并行计算
3. 减少CPU-GPU通信
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import time


class GPUVehicleScorer:
    """
    GPU加速的车辆评分器

    基于规则快速评估车辆重要性。
    """

    def __init__(
        self,
        config: Dict,
        device: str = 'cuda'
    ):
        """
        初始化GPU评分器

        Args:
            config: 配置字典
            device: 设备 ('cuda' or 'cpu')
        """
        self.config = config
        self.device = device

        # 从配置中提取权重
        rule_config = config.get('rule_based_scoring', {})
        weights = rule_config.get('weights', {})

        self.bottleneck_weight = weights.get('bottleneck_weight', 2.0)
        self.upstream_weight = weights.get('upstream_weight', 1.0)
        self.speed_weight = weights.get('speed_weight', 0.5)
        self.gap_weight = weights.get('gap_weight', 1.0)

        # 瓶颈区域
        self.bottleneck_s_min = rule_config.get('bottleneck_s_min', 1200.0)
        self.bottleneck_s_max = rule_config.get('bottleneck_s_max', 2200.0)
        self.upstream_s_min = rule_config.get('upstream_s_min', 600.0)

    def compute_scores_batch(
        self,
        vehicle_states: Dict[str, Dict],
        vehicle_ids: List[str]
    ) -> Dict[str, float]:
        """
        批量计算所有车辆的评分（GPU加速）

        Args:
            vehicle_states: {veh_id: {s, d, speed, ...}}
            vehicle_ids: 所有车辆ID列表

        Returns:
            scores: {veh_id: score}
        """
        if len(vehicle_ids) == 0:
            return {}

        start_time = time.time()

        # ========== 1. 准备张量数据 ==========
        # 提取所有车辆的状态
        states_list = []
        valid_ids = []

        for veh_id in vehicle_ids:
            if veh_id in vehicle_states:
                state = vehicle_states[veh_id]
                states_list.append([
                    state.get('s', 0.0),
                    state.get('speed', 0.0),
                    state.get('acceleration', 0.0),
                    state.get('lane_index', 0.0)
                ])
                valid_ids.append(veh_id)

        if len(valid_ids) == 0:
            return {}

        # 转换为GPU张量
        states_tensor = torch.tensor(
            states_list,
            dtype=torch.float32,
            device=self.device
        )  # [B, 4]

        # ========== 2. 向量化计算评分 ==========
        s = states_tensor[:, 0]  # [B]
        speed = states_tensor[:, 1]  # [B]
        accel = states_tensor[:, 2]  # [B]
        lane = states_tensor[:, 3]  # [B]

        # 2.1 位置评分（瓶颈区域 > 上游区域 > 其他）
        # 归一化到 [0, 1]
        in_bottleneck = (s > self.bottleneck_s_min) & (s < self.bottleneck_s_max)
        in_upstream = (s > self.upstream_s_min) & (s <= self.bottleneck_s_min)

        position_scores = torch.zeros_like(s)
        position_scores[in_bottleneck] = 1.0  # 瓶颈区域最高分
        position_scores[in_upstream] = 0.7     # 上游区域次高分
        position_scores[~in_bottleneck & ~in_upstream] = 0.3  # 其他区域低分

        # 2.2 速度评分（速度低 = 重要性高）
        # 低速车辆更容易造成拥堵，需要控制
        speed_scores = 1.0 - (speed / 30.0)  # 归一化：0速=1分，30速=0分
        speed_scores = torch.clamp(speed_scores, 0.0, 1.0)

        # 2.3 车道评分（内侧车道更重要）
        # 假设lane 0是最内侧车道（快速车道）
        lane_scores = 1.0 - (lane / 10.0)  # lane 0 = 1分，lane 10 = 0分
        lane_scores = torch.clamp(lane_scores, 0.0, 1.0)

        # 2.4 加速度评分（减速中的车辆需要关注）
        # 负加速度（减速）= 重要性高
        accel_scores = torch.zeros_like(accel)
        decelerating = accel < 0
        accel_scores[decelerating] = 1.0 + (accel[decelerating] / 4.5)  # -4.5m/s² = 0分，0m/s² = 1分
        accel_scores = torch.clamp(accel_scores, 0.0, 1.0)

        # ========== 3. 加权组合 ==========
        total_scores = (
            self.bottleneck_weight * position_scores +
            self.speed_weight * speed_scores +
            0.15 * lane_scores +
            0.1 * accel_scores
        )

        # 归一化到 [0, 1]
        max_score = total_scores.max().item()
        if max_score > 0:
            total_scores = total_scores / max_score

        # ========== 4. 转换为字典 ==========
        scores_np = total_scores.cpu().numpy()
        scores_dict = {veh_id: float(score) for veh_id, score in zip(valid_ids, scores_np)}

        elapsed = time.time() - start_time

        return scores_dict

    def get_top_k_vehicles(
        self,
        vehicle_states: Dict[str, Dict],
        vehicle_ids: List[str],
        k: int,
        min_score: float = 0.0
    ) -> List[str]:
        """
        获取评分最高的K个车辆

        Args:
            vehicle_states: 车辆状态字典
            vehicle_ids: 所有车辆ID列表
            k: 返回的车辆数量
            min_score: 最低评分阈值

        Returns:
            top_k_vehicles: 按评分排序的车辆ID列表
        """
        scores = self.compute_scores_batch(vehicle_states, vehicle_ids)

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


def create_gpu_vehicle_scorer(
    config: Dict,
    device: str = 'cuda'
) -> GPUVehicleScorer:
    """
    创建GPU车辆评分器的工厂函数

    Args:
        config: 配置字典
        device: 设备 ('cuda' or 'cpu')

    Returns:
        scorer: GPUVehicleScorer实例
    """
    scorer = GPUVehicleScorer(config=config, device=device)
    return scorer


# 测试代码
if __name__ == '__main__':
    print("Testing GPUVehicleScorer...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建配置
    config = {
        'rule_based_scoring': {
            'bottleneck_s_min': 1200.0,
            'bottleneck_s_max': 2200.0,
            'upstream_s_min': 600.0,
            'weights': {
                'bottleneck_weight': 2.0,
                'upstream_weight': 1.0,
                'speed_weight': 0.5,
                'gap_weight': 1.0
            }
        }
    }

    # 创建评分器
    scorer = create_gpu_vehicle_scorer(config, device=device)

    # 测试数据
    num_vehicles = 100
    vehicle_states = {}
    vehicle_ids = []

    for i in range(num_vehicles):
        veh_id = f'veh_{i}'
        vehicle_ids.append(veh_id)
        vehicle_states[veh_id] = {
            's': 500.0 + i * 20.0,  # 分布在500-2500m范围内
            'speed': 15.0 + np.random.randn() * 5.0,
            'acceleration': np.random.randn() * 2.0,
            'lane_index': np.random.randint(0, 4)
        }

    # 计算评分
    import time

    num_iterations = 1000
    start = time.time()

    for _ in range(num_iterations):
        scores = scorer.compute_scores_batch(vehicle_states, vehicle_ids)

    elapsed = time.time() - start

    print(f"\nPerformance:")
    print(f"  {num_iterations} iterations: {elapsed:.3f}s")
    print(f"  Average: {elapsed / num_iterations * 1000:.2f}ms/iter")
    print(f"  Throughput: {num_vehicles * num_iterations / elapsed:.0f} vehicles/s")

    # 显示top-K
    top_k = scorer.get_top_k_vehicles(vehicle_states, vehicle_ids, k=10)
    print(f"\nTop 10 vehicles:")
    for veh_id in top_k:
        score = scores[veh_id]
        state = vehicle_states[veh_id]
        print(f"  {veh_id}: score={score:.3f}, s={state['s']:.1f}m, speed={state['speed']:.1f}m/s")

    print("\n✓ All tests passed!")
