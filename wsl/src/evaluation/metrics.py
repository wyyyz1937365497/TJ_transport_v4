"""
评估指标计算模块 - 完整实现
"""

from typing import Any, Dict, List

import numpy as np


def compute_metrics(episode_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算单个episode的评估指标

    指标包括：
    1. OD完成率 (OCR): 车辆按时到达目的地的比例
    2. 速度标准差 (σv): 交通流稳定性指标
    3. 平均绝对加速度 (|a|avg): 舒适度指标
    4. 干预成本: 控制指令的总成本
    5. 平均速度: 交通效率指标
    6. 行程时间: 车辆平均行程时间
    7. 碰撞次数: 安全性指标
    8. 吞吐量: 单位时间通过车辆数

    Args:
        episode_data: episode数据

    Returns:
        指标字典
    """
    metrics = {}

    # 提取数据
    trajectories = episode_data.get("trajectories", {})
    rewards = episode_data.get("rewards", [])
    costs = episode_data.get("costs", [])
    global_stats = episode_data.get("global_stats", [])

    # ========== 1. OD完成率 (OCR) ==========
    # 假设成功完成行程的车辆是那些在episode结束时仍有数据的车辆
    # 这里简化处理：使用轨迹长度来判断
    completed_vehicles = 0
    total_vehicles = len(trajectories)

    for veh_id, traj in trajectories.items():
        # 如果车辆有足够的轨迹点，认为完成了行程
        if len(traj["timestamps"]) > 100:  # 至少10秒
            completed_vehicles += 1

    metrics["ocr"] = completed_vehicles / max(total_vehicles, 1)
    metrics["completed_vehicles"] = completed_vehicles
    metrics["total_vehicles"] = total_vehicles

    # ========== 2. 速度相关指标 ==========
    all_speeds = []
    all_accels = []

    for traj in trajectories.values():
        all_speeds.extend(traj["speeds"])
        all_accels.extend(traj["accelerations"])

    if all_speeds:
        metrics["avg_speed"] = float(np.mean(all_speeds))
        metrics["speed_std"] = float(np.std(all_speeds))
        metrics["max_speed"] = float(np.max(all_speeds))
        metrics["min_speed"] = float(np.min(all_speeds))
    else:
        metrics["avg_speed"] = 0.0
        metrics["speed_std"] = 0.0
        metrics["max_speed"] = 0.0
        metrics["min_speed"] = 0.0

    # ========== 3. 加速度相关指标 ==========
    if all_accels:
        metrics["avg_accel"] = float(np.mean(all_accels))
        metrics["avg_abs_accel"] = float(np.mean(np.abs(all_accels)))
        metrics["accel_std"] = float(np.std(all_accels))
        metrics["max_accel"] = float(np.max(all_accels))
        metrics["min_accel"] = float(np.min(all_accels))

        # 急刹车统计（加速度 < -3 m/s²）
        emergency_braking = sum(1 for a in all_accels if a < -3.0)
        metrics["emergency_braking_count"] = emergency_braking
        metrics["emergency_braking_ratio"] = emergency_braking / len(all_accels)
    else:
        metrics["avg_accel"] = 0.0
        metrics["avg_abs_accel"] = 0.0
        metrics["accel_std"] = 0.0
        metrics["max_accel"] = 0.0
        metrics["min_accel"] = 0.0
        metrics["emergency_braking_count"] = 0
        metrics["emergency_braking_ratio"] = 0.0

    # ========== 4. 成本指标 ==========
    if costs:
        metrics["total_cost"] = float(sum(costs))
        metrics["avg_cost"] = float(np.mean(costs))
        metrics["max_cost"] = float(np.max(costs))
        metrics["min_cost"] = float(np.min(costs))
        metrics["cost_std"] = float(np.std(costs))
    else:
        metrics["total_cost"] = 0.0
        metrics["avg_cost"] = 0.0
        metrics["max_cost"] = 0.0
        metrics["min_cost"] = 0.0
        metrics["cost_std"] = 0.0

    # ========== 5. 奖励指标 ==========
    if rewards:
        metrics["total_reward"] = float(sum(rewards))
        metrics["avg_reward"] = float(np.mean(rewards))
        metrics["max_reward"] = float(np.max(rewards))
        metrics["min_reward"] = float(np.min(rewards))
        metrics["reward_std"] = float(np.std(rewards))
    else:
        metrics["total_reward"] = 0.0
        metrics["avg_reward"] = 0.0
        metrics["max_reward"] = 0.0
        metrics["min_reward"] = 0.0
        metrics["reward_std"] = 0.0

    # ========== 6. 行程时间 ==========
    travel_times = []
    for traj in trajectories.values():
        if len(traj["timestamps"]) > 1:
            travel_time = traj["timestamps"][-1] - traj["timestamps"][0]
            travel_times.append(travel_time)

    if travel_times:
        metrics["avg_travel_time"] = float(np.mean(travel_times))
        metrics["max_travel_time"] = float(np.max(travel_times))
        metrics["min_travel_time"] = float(np.min(travel_times))
        metrics["travel_time_std"] = float(np.std(travel_times))
    else:
        metrics["avg_travel_time"] = 0.0
        metrics["max_travel_time"] = 0.0
        metrics["min_travel_time"] = 0.0
        metrics["travel_time_std"] = 0.0

    # ========== 7. 全局统计 ==========
    if global_stats:
        global_stats_array = np.array(global_stats)

        # global_stats是16维数组
        # 索引: [0]平均速度, [1]速度标准差, [6]车辆数, [9]碰撞数, [10]已到达, [11]已出发
        if global_stats_array.shape[0] > 0 and global_stats_array.shape[1] >= 12:
            metrics["collisions"] = float(np.max(global_stats_array[:, 9]))  # 最大碰撞数
            metrics["arrived_count"] = float(np.max(global_stats_array[:, 10]))  # 最终到达数
            metrics["departed_count"] = float(np.max(global_stats_array[:, 11]))  # 最终出发数
            metrics["max_vehicle_count"] = float(np.max(global_stats_array[:, 6]))  # 最大车辆数
        else:
            metrics["collisions"] = 0.0
            metrics["arrived_count"] = 0.0
            metrics["departed_count"] = 0.0
            metrics["max_vehicle_count"] = 0.0
    else:
        metrics["collisions"] = 0.0
        metrics["arrived_count"] = 0.0
        metrics["departed_count"] = 0.0
        metrics["max_vehicle_count"] = 0.0

    # ========== 8. 干预统计 ==========
    total_actions = 0
    lane_changes = 0
    accel_interventions = 0

    for traj in trajectories.values():
        actions = traj.get("actions", [])
        for action in actions:
            if len(action) >= 2:
                total_actions += 1
                if action[1] > 0.5:  # 换道
                    lane_changes += 1
                if abs(action[0]) > 0.1:  # 加速度干预
                    accel_interventions += 1

    metrics["total_actions"] = total_actions
    metrics["lane_changes"] = lane_changes
    metrics["accel_interventions"] = accel_interventions

    if total_actions > 0:
        metrics["lane_change_ratio"] = lane_changes / total_actions
        metrics["accel_intervention_ratio"] = accel_interventions / total_actions
    else:
        metrics["lane_change_ratio"] = 0.0
        metrics["accel_intervention_ratio"] = 0.0

    # ========== 9. 吞吐量 ==========
    if travel_times and len(travel_times) > 0:
        # 吞吐量 = 完成车辆数 / 总时间
        total_time = max(travel_times) if travel_times else 1.0
        metrics["throughput"] = len(travel_times) / max(total_time, 1.0)
    else:
        metrics["throughput"] = 0.0

    # ========== 10. 效率指标 ==========
    # 综合效率评分（0-100）
    # 考虑：速度(30%), 稳定性(20%), 安全性(20%), 成本(15%), 完成率(15%)
    speed_score = min(metrics["avg_speed"] / 20.0, 1.0) * 30  # 假设20m/s是优秀
    stability_score = max(1.0 - metrics["speed_std"] / 10.0, 0.0) * 20  # 速度标准差越小越好
    safety_score = max(1.0 - metrics["collisions"] / 10.0, 0.0) * 20  # 碰撞越少越好
    cost_score = max(1.0 - metrics["avg_cost"], 0.0) * 15  # 成本越低越好
    completion_score = metrics["ocr"] * 15  # 完成率

    metrics["efficiency_score"] = speed_score + stability_score + safety_score + cost_score + completion_score

    return metrics


def compute_aggregate_metrics(episode_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算多个episode的汇总指标

    Args:
        episode_metrics_list: 多个episode的指标列表

    Returns:
        汇总指标字典
    """
    if not episode_metrics_list:
        return {}

    aggregated = {}

    # 获取所有指标键
    metric_keys = episode_metrics_list[0].keys()

    # 对每个指标计算统计量
    for key in metric_keys:
        values = [m[key] for m in episode_metrics_list if key in m]

        if not values:
            continue

        if isinstance(values[0], (int, float)):
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_median"] = float(np.median(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))
        else:
            aggregated[key] = values[0]

    # 添加episode数量
    aggregated["num_episodes"] = len(episode_metrics_list)

    return aggregated


def print_metrics_report(metrics: Dict[str, Any], title: str = "评估报告"):
    """
    打印格式化的指标报告

    Args:
        metrics: 指标字典
        title: 报告标题
    """
    print("=" * 70)
    print(f"{title}")
    print("=" * 70)

    # 基本信息
    print("\n【基本信息】")
    print(f"  车辆总数: {metrics.get('total_vehicles', 'N/A')}")
    print(f"  完成车辆数: {metrics.get('completed_vehicles', 'N/A')}")
    print(f"  碰撞次数: {metrics.get('collisions', 'N/A')}")

    # 性能指标
    print("\n【性能指标】")
    print(f"  OD完成率: {metrics.get('ocr', 0):.2%}")
    print(f"  平均速度: {metrics.get('avg_speed', 0):.2f} m/s")
    print(f"  速度标准差: {metrics.get('speed_std', 0):.2f} m/s")
    print(f"  平均绝对加速度: {metrics.get('avg_abs_accel', 0):.2f} m/s²")
    print(f"  急刹车比例: {metrics.get('emergency_braking_ratio', 0):.2%}")

    # 成本指标
    print("\n【成本指标】")
    print(f"  平均成本: {metrics.get('avg_cost', 0):.4f}")
    print(f"  总成本: {metrics.get('total_cost', 0):.2f}")
    print(f"  总动作数: {metrics.get('total_actions', 0)}")
    print(f"  换道次数: {metrics.get('lane_changes', 0)}")
    print(f"  加速度干预: {metrics.get('accel_interventions', 0)}")

    # 奖励指标
    print("\n【奖励指标】")
    print(f"  总奖励: {metrics.get('total_reward', 0):.2f}")
    print(f"  平均奖励: {metrics.get('avg_reward', 0):.2f}")

    # 行程时间
    print("\n【行程时间】")
    print(f"  平均行程时间: {metrics.get('avg_travel_time', 0):.1f} s")
    print(f"  最长行程时间: {metrics.get('max_travel_time', 0):.1f} s")
    print(f"  最短行程时间: {metrics.get('min_travel_time', 0):.1f} s")

    # 综合评分
    print("\n【综合评分】")
    print(f"  效率评分: {metrics.get('efficiency_score', 0):.1f} / 100")

    print("=" * 70)
