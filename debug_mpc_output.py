#!/usr/bin/env python3
"""
调试MPC输出的lane_change值
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
import numpy as np
from src.env.competition_env import CompetitionSumoEnv
from src.mpc import MPCConfig, GPUMPCController
from src.env.vehicle_scorer_gpu import GPUVehicleScorer
import logging

logging.basicConfig(level=logging.WARNING)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 200

env = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')

# MPC controller
mpc_config_dict = config['mpc']
mpc_config = MPCConfig(
    prediction_horizon=mpc_config_dict.get('prediction_horizon', 10),
    control_horizon=mpc_config_dict.get('control_horizon', 5),
    dt=mpc_config_dict.get('dt', 0.1),
    Q_speed=mpc_config_dict.get('state_weights', {}).get('speed', 10.0),
    Q_position=mpc_config_dict.get('state_weights', {}).get('position', 1.0),
    Q_accel=mpc_config_dict.get('state_weights', {}).get('acceleration', 0.1),
    Q_gap=mpc_config_dict.get('state_weights', {}).get('gap', 50.0),
    R_accel=mpc_config_dict.get('control_weights', {}).get('acceleration', 0.5),
    R_lane=mpc_config_dict.get('control_weights', {}).get('lane_change', 0.1),
    min_accel=mpc_config_dict.get('constraints', {}).get('min_accel', -4.5),
    max_accel=mpc_config_dict.get('constraints', {}).get('max_accel', 2.0),
    min_speed=mpc_config_dict.get('constraints', {}).get('min_speed', 0.0),
    max_speed=mpc_config_dict.get('constraints', {}).get('max_speed', 30.0),
    min_gap=mpc_config_dict.get('constraints', {}).get('min_gap', 2.0),
    desired_gap=mpc_config_dict.get('constraints', {}).get('desired_gap', 5.0),
    bottleneck_s_min=mpc_config_dict.get('bottleneck_region', {}).get('s_min', 1200.0),
    bottleneck_s_max=mpc_config_dict.get('bottleneck_region', {}).get('s_max', 2200.0),
)

mpc = GPUMPCController(config=mpc_config, device='cuda')
scorer = GPUVehicleScorer(config=config, device='cuda')

# 临时修改MPC代码以输出原始值
import torch

original_solve = mpc.solve

lane_change_raw_values = []

def debug_solve(observation_dict, vehicle_ids, icv_ids, step):
    result = original_solve(observation_dict, vehicle_ids, icv_ids, step)

    # 获取原始控制序列
    if hasattr(mpc, 'last_control_sequence') and mpc.last_control_sequence is not None:
        with torch.no_grad():
            actions = mpc.last_control_sequence[:, 0, :]  # [B, 2]
            lane_changes = actions[:, 1].cpu().numpy()
            lane_change_raw_values.extend(lane_changes)

    return result

mpc.solve = debug_solve

obs = env.reset()

print("运行200步，收集MPC输出的原始lane_change值...")
for step in range(200):
    vehicle_ids = obs.get('vehicle_ids', [])
    icv_ids = obs.get('icv_ids', set())

    actions_dict = {}
    if len(icv_ids) > 0 and step >= 50:
        selected_vehicles = scorer.get_top_k_vehicles(
            obs.get('vehicle_states', {}),
            list(icv_ids),
            k=25
        )

        try:
            actions_dict = mpc.solve(
                observation_dict=obs,
                vehicle_ids=vehicle_ids,
                icv_ids=selected_vehicles,
                step=step
            )
        except Exception as e:
            pass

    obs, _, done, _ = env.step(actions_dict)

    if done:
        break

env.close()

if len(lane_change_raw_values) > 0:
    lane_array = np.array(lane_change_raw_values)
    print("\n" + "=" * 80)
    print("MPC输出的原始lane_change值分布")
    print("=" * 80)
    print(f"  样本数: {len(lane_array)}")
    print(f"\n  统计:")
    print(f"    均值: {np.mean(lane_array):.6f}")
    print(f"    标准差: {np.std(lane_array):.6f}")
    print(f"    最小值: {np.min(lane_array):.6f}")
    print(f"    最大值: {np.max(lane_array):.6f}")
    print(f"    中位数: {np.median(lane_array):.6f}")
    print(f"\n  分布:")
    print(f"    <-0.33: {np.sum(lane_array < -0.33)} ({np.sum(lane_array < -0.33)/len(lane_array)*100:.2f}%)")
    print(f"    -0.33~0.33: {np.sum((lane_array >= -0.33) & (lane_array <= 0.33))} ({np.sum((lane_array >= -0.33) & (lane_array <= 0.33))/len(lane_array)*100:.2f}%)")
    print(f"    >0.33: {np.sum(lane_array > 0.33)} ({np.sum(lane_array > 0.33)/len(lane_array)*100:.2f}%)")
    print(f"\n  P10, P50, P90:")
    print(f"    P10: {np.percentile(lane_array, 10):.6f}")
    print(f"    P50: {np.percentile(lane_array, 50):.6f}")
    print(f"    P90: {np.percentile(lane_array, 90):.6f}")

    # 检查离散化后的结果
    discretized = np.where(lane_array < -0.33, -1, np.where(lane_array > 0.33, 1, 0))
    print(f"\n  离散化后:")
    print(f"    -1 (左): {np.sum(discretized == -1)} ({np.sum(discretized == -1)/len(discretized)*100:.2f}%)")
    print(f"     0 (保持): {np.sum(discretized == 0)} ({np.sum(discretized == 0)/len(discretized)*100:.2f}%)")
    print(f"    +1 (右): {np.sum(discretized == 1)} ({np.sum(discretized == 1)/len(discretized)*100:.2f}%)")
