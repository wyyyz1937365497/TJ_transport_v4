#!/usr/bin/env python3
"""
MPC控制器诊断工具

检查项：
1. MPC生成的加速度、换道指令是否合理
2. 控制频率和影响
3. 与baseline的对比
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
import numpy as np
import torch
from src.env.competition_env import CompetitionSumoEnv
from src.mpc import MPCConfig, GPUMPCController
from src.env.vehicle_scorer_gpu import GPUVehicleScorer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 300  # 30 seconds test

print("=" * 80)
print("MPC控制器诊断")
print("=" * 80)

# Test 1: Baseline (no control)
print("\n【测试1】Baseline（无MPC控制）")
print("-" * 80)

env_baseline = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')
obs = env_baseline.reset()

baseline_stats = {
    'arrived': [],
    'departed': [],
    'vehicles': []
}

for step in range(300):
    obs, reward, done, info = env_baseline.step({})

    if step % 50 == 0:
        baseline_stats['arrived'].append(info.get('arrived_count', 0))
        baseline_stats['departed'].append(info.get('departed_count', 0))
        baseline_stats['vehicles'].append(len(obs.get('vehicle_ids', [])))
        print(f"  Step {step:3d}: Arrived={info.get('arrived_count', 0):3d}, "
              f"Departed={info.get('departed_count', 0):3d}, "
              f"Vehicles={len(obs.get('vehicle_ids', [])):3d}")

    if done:
        break

final_ocr_baseline = info.get('arrived_count', 0) / info.get('departed_count', 1)
print(f"\n  Final OCR: {final_ocr_baseline:.4f} ({final_ocr_baseline*100:.2f}%)")
env_baseline.close()

# Test 2: GPU MPC with full parameters
print("\n【测试2】GPU MPC (完整参数)")
print("-" * 80)

env_mpc = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')

# Create MPC controller
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

obs = env_mpc.reset()

mpc_stats = {
    'arrived': [],
    'departed': [],
    'vehicles': [],
    'controlled': [],
    'mpc_times': [],
    'actions': [],
    'solve_success': 0,
    'solve_failure': 0
}

print(f"  MPC Config:")
print(f"    Prediction horizon: {mpc_config.prediction_horizon}")
print(f"    Control horizon: {mpc_config.control_horizon}")
print(f"    K vehicles: {config['mpc']['vehicle_selection']['k_vehicles']}")
print(f"    State weights: speed={mpc_config.Q_speed:.1f}, gap={mpc_config.Q_gap:.1f}")
print(f"    Control weights: accel={mpc_config.R_accel:.1f}, lane={mpc_config.R_lane:.1f}")
print()

for step in range(300):
    vehicle_ids = obs.get('vehicle_ids', [])
    icv_ids = obs.get('icv_ids', set())

    # Vehicle selection
    if len(icv_ids) > 0:
        selected_vehicles = scorer.get_top_k_vehicles(
            obs.get('vehicle_states', {}),
            list(icv_ids),
            k=config['mpc']['vehicle_selection']['k_vehicles']
        )
    else:
        selected_vehicles = []

    # MPC solve
    actions_dict = {}
    try:
        actions_dict = mpc.solve(
            observation_dict=obs,
            vehicle_ids=vehicle_ids,
            icv_ids=selected_vehicles,
            step=step
        )
        mpc_stats['solve_success'] += 1
        solve_time = mpc.get_solve_time()
        mpc_stats['mpc_times'].append(solve_time)

        # Analyze actions
        if len(actions_dict) > 0:
            accels = [a[0] for a in actions_dict.values()]
            lanes = [a[1] for a in actions_dict.values()]
            mpc_stats['actions'].extend(list(zip(accels, lanes)))
            mpc_stats['controlled'].append(len(actions_dict))

    except Exception as e:
        mpc_stats['solve_failure'] += 1
        if step % 100 == 0:
            print(f"  [Step {step}] MPC failed: {e}")

    obs, reward, done, info = env_mpc.step(actions_dict)

    if step % 50 == 0:
        mpc_stats['arrived'].append(info.get('arrived_count', 0))
        mpc_stats['departed'].append(info.get('departed_count', 0))
        mpc_stats['vehicles'].append(len(obs.get('vehicle_ids', [])))

        avg_time = np.mean(mpc_stats['mpc_times']) if len(mpc_stats['mpc_times']) > 0 else 0
        print(f"  Step {step:3d}: Arrived={info.get('arrived_count', 0):3d}, "
              f"Departed={info.get('departed_count', 0):3d}, "
              f"Vehicles={len(obs.get('vehicle_ids', [])):3d}, "
              f"Controlled={len(actions_dict):2d}, "
              f"MPC time={avg_time*1000:.1f}ms")

    if done:
        break

final_ocr_mpc = info.get('arrived_count', 0) / info.get('departed_count', 1)
print(f"\n  Final OCR: {final_ocr_mpc:.4f} ({final_ocr_mpc*100:.2f}%)")
env_mpc.close()

# Analyze actions
print("\n【MPC动作分析】")
print("-" * 80)
if len(mpc_stats['actions']) > 0:
    actions_array = np.array(mpc_stats['actions'])
    accels = actions_array[:, 0]
    lanes = actions_array[:, 1]

    print(f"  Total actions: {len(actions_array)}")
    print(f"  Acceleration:")
    print(f"    Mean: {np.mean(accels):.4f}")
    print(f"    Std:  {np.std(accels):.4f}")
    print(f"    Min:  {np.min(accels):.4f}")
    print(f"    Max:  {np.max(accels):.4f}")
    print(f"    P25:  {np.percentile(accels, 25):.4f}")
    print(f"    P50:  {np.percentile(accels, 50):.4f}")
    print(f"    P75:  {np.percentile(accels, 75):.4f}")
    print(f"  Lane change:")
    print(f"    -1 (left):  {np.sum(lanes == -1)} ({np.sum(lanes == -1)/len(lanes)*100:.1f}%)")
    print(f"     0 (none): {np.sum(lanes == 0)} ({np.sum(lanes == 0)/len(lanes)*100:.1f}%)")
    print(f"    +1 (right): {np.sum(lanes == 1)} ({np.sum(lanes == 1)/len(lanes)*100:.1f}%)")

    # Check for unreasonable actions
    extreme_accel = np.sum(np.abs(accels) > 3.0)
    if extreme_accel > 0:
        print(f"\n  ⚠️  警告: {extreme_accel} 个极端加速度动作 (|a| > 3.0 m/s²)")

    # Check action distribution
    zero_accel = np.sum(np.abs(accels) < 0.01)
    print(f"\n  动作分布:")
    print(f"    接近零加速度 (|a| < 0.01): {zero_accel} ({zero_accel/len(accels)*100:.1f}%)")
    print(f"    加速 (a > 0.5): {np.sum(accels > 0.5)} ({np.sum(accels > 0.5)/len(accels)*100:.1f}%)")
    print(f"    减速 (a < -0.5): {np.sum(accels < -0.5)} ({np.sum(accels < -0.5)/len(accels)*100:.1f}%)")
else:
    print("  ❌ 没有生成任何控制动作！")

# Solve statistics
print(f"\n【MPC求解统计】")
print("-" * 80)
total_solves = mpc_stats['solve_success'] + mpc_stats['solve_failure']
print(f"  成功: {mpc_stats['solve_success']} ({mpc_stats['solve_success']/total_solves*100:.1f}%)")
print(f"  失败: {mpc_stats['solve_failure']} ({mpc_stats['solve_failure']/total_solves*100:.1f}%)")
if len(mpc_stats['mpc_times']) > 0:
    print(f"  平均求解时间: {np.mean(mpc_stats['mpc_times'])*1000:.2f} ms")
    print(f"  最大求解时间: {np.max(mpc_stats['mpc_times'])*1000:.2f} ms")

# Comparison
print("\n【性能对比】")
print("-" * 80)
print(f"  Baseline OCR:  {final_ocr_baseline:.4f} ({final_ocr_baseline*100:.2f}%)")
print(f"  MPC OCR:       {final_ocr_mpc:.4f} ({final_ocr_mpc*100:.2f}%)")
print(f"  性能下降:      {(final_ocr_baseline - final_ocr_mpc)/final_ocr_baseline*100:.1f}%")

if final_ocr_mpc < final_ocr_baseline * 0.8:
    print("\n  ❌ 严重警告: MPC导致性能下降超过20%！")
    print("\n可能原因:")
    print("  1. MPC生成的控制指令不合理（过于激进/保守）")
    print("  2. 控制频率过高导致车辆不稳定")
    print("  3. 控制车辆数量过多导致相互冲突")
    print("  4. GPU MPC实现存在bug")
elif final_ocr_mpc > final_ocr_baseline:
    print("\n  ✅ MPC改善了性能！")
else:
    print("\n  ⚠️  MPC轻微性能下降")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)
