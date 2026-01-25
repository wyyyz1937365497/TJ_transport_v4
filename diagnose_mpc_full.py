#!/usr/bin/env python3
"""
完整3600步MPC vs Baseline对比
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

logging.basicConfig(level=logging.WARNING)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=" * 80)
print("完整3600步对比测试: Baseline vs MPC")
print("=" * 80)

def run_test(use_mpc=True, k_vehicles=50):
    """运行测试"""
    env_config = config['environment'].copy()
    env_config['max_steps'] = 3600

    env = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')

    if use_mpc:
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
    else:
        mpc = None
        scorer = None

    obs = env.reset()

    actions_list = []
    mpc_times = []

    for step in range(3600):
        vehicle_ids = obs.get('vehicle_ids', [])
        icv_ids = obs.get('icv_ids', set())

        actions_dict = {}
        if use_mpc and len(icv_ids) > 0:
            # Vehicle selection
            selected_vehicles = scorer.get_top_k_vehicles(
                obs.get('vehicle_states', {}),
                list(icv_ids),
                k=k_vehicles
            )

            # MPC solve
            try:
                actions_dict = mpc.solve(
                    observation_dict=obs,
                    vehicle_ids=vehicle_ids,
                    icv_ids=selected_vehicles,
                    step=step
                )
                mpc_times.append(mpc.get_solve_time())
                if len(actions_dict) > 0:
                    actions_list.extend([a for a in actions_dict.values()])
            except Exception as e:
                pass

        obs, reward, done, info = env.step(actions_dict)

        if step % 600 == 0:
            ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0
            ctrl_str = f", Controlled={len(actions_dict):2d}" if use_mpc else ""
            print(f"  Step {step:4d}: Arrived={info.get('arrived_count', 0):3d}, "
                  f"Departed={info.get('departed_count', 0):3d}, "
                  f"OCR={ocr:.3f}{ctrl_str}")

        if done:
            break

    final_ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0

    env.close()

    return {
        'ocr': final_ocr,
        'arrived': info.get('arrived_count', 0),
        'departed': info.get('departed_count', 0),
        'actions': actions_list,
        'mpc_times': mpc_times
    }

# Test 1: Baseline
print("\n【测试1】Baseline (无MPC控制)")
print("-" * 80)
baseline_result = run_test(use_mpc=False)
print(f"\n  Final OCR: {baseline_result['ocr']:.4f} ({baseline_result['ocr']*100:.2f}%)")

# Test 2: MPC with k=50
print("\n【测试2】GPU MPC (k_vehicles=50)")
print("-" * 80)
mpc_result_50 = run_test(use_mpc=True, k_vehicles=50)
print(f"\n  Final OCR: {mpc_result_50['ocr']:.4f} ({mpc_result_50['ocr']*100:.2f}%)")
if len(mpc_result_50['mpc_times']) > 0:
    print(f"  平均MPC时间: {np.mean(mpc_result_50['mpc_times'])*1000:.1f}ms")

# Test 3: MPC with k=25 (reduced control)
print("\n【测试3】GPU MPC (k_vehicles=25 - 减少控制车辆)")
print("-" * 80)
mpc_result_25 = run_test(use_mpc=True, k_vehicles=25)
print(f"\n  Final OCR: {mpc_result_25['ocr']:.4f} ({mpc_result_25['ocr']*100:.2f}%)")
if len(mpc_result_25['mpc_times']) > 0:
    print(f"  平均MPC时间: {np.mean(mpc_result_25['mpc_times'])*1000:.1f}ms")

# Test 4: MPC with k=10 (minimal control)
print("\n【测试4】GPU MPC (k_vehicles=10 - 最小控制)")
print("-" * 80)
mpc_result_10 = run_test(use_mpc=True, k_vehicles=10)
print(f"\n  Final OCR: {mpc_result_10['ocr']:.4f} ({mpc_result_10['ocr']*100:.2f}%)")
if len(mpc_result_10['mpc_times']) > 0:
    print(f"  平均MPC时间: {np.mean(mpc_result_10['mpc_times'])*1000:.1f}ms")

# Summary
print("\n" + "=" * 80)
print("性能对比总结")
print("=" * 80)
print(f"  Baseline (无控制):     OCR={baseline_result['ocr']:.4f} ({baseline_result['ocr']*100:.2f}%)")
print(f"  MPC k=50:              OCR={mpc_result_50['ocr']:.4f} ({mpc_result_50['ocr']*100:.2f}%)")
print(f"  MPC k=25 (减少车辆):   OCR={mpc_result_25['ocr']:.4f} ({mpc_result_25['ocr']*100:.2f}%)")
print(f"  MPC k=10 (最小控制):   OCR={mpc_result_10['ocr']:.4f} ({mpc_result_10['ocr']*100:.2f}%)")
print()

# Find best
results = [
    ('Baseline', baseline_result['ocr']),
    ('MPC k=50', mpc_result_50['ocr']),
    ('MPC k=25', mpc_result_25['ocr']),
    ('MPC k=10', mpc_result_10['ocr'])
]
best = max(results, key=lambda x: x[1])
print(f"  最佳配置: {best[0]} (OCR={best[1]:.4f})")

# Analysis
if best[0] == 'Baseline':
    print("\n  ⚠️  警告: 所有MPC配置都比baseline差！")
    print("  建议使用baseline收集数据，或修复MPC实现。")
elif best[0].startswith('MPC'):
    print(f"\n  ✅ MPC配置 '{best[0]}' 改善了性能！")
    improvement = (best[1] - baseline_result['ocr']) / baseline_result['ocr'] * 100
    print(f"  相比baseline提升: {improvement:.1f}%")

# Action analysis
for name, result, k in [
    ('MPC k=50', mpc_result_50, 50),
    ('MPC k=25', mpc_result_25, 25),
    ('MPC k=10', mpc_result_10, 10)
]:
    if len(result['actions']) > 0:
        actions = np.array(result['actions'])
        print(f"\n  {name} 动作分析:")
        print(f"    平均加速度: {np.mean(actions[:, 0]):.4f}")
        print(f"    换道比例: {np.sum(actions[:, 1] != 0) / len(actions) * 100:.1f}%")

print("\n" + "=" * 80)
