#!/usr/bin/env python3
"""
快速测试lane_change离散化修复
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

print("=" * 80)
print("测试lane_change离散化修复 (600步)")
print("=" * 80)

env_config = config['environment'].copy()
env_config['max_steps'] = 600

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

obs = env.reset()

lane_changes_count = 0
actions_list = []

print("\n运行600步测试...")
for step in range(600):
    vehicle_ids = obs.get('vehicle_ids', [])
    icv_ids = obs.get('icv_ids', set())

    actions_dict = {}
    if len(icv_ids) > 0 and step >= 50:  # 等待一些车辆进入
        selected_vehicles = scorer.get_top_k_vehicles(
            obs.get('vehicle_states', {}),
            list(icv_ids),
            k=25  # 使用k=25测试
        )

        try:
            actions_dict = mpc.solve(
                observation_dict=obs,
                vehicle_ids=vehicle_ids,
                icv_ids=selected_vehicles,
                step=step
            )

            # 统计换道
            for accel, lc in actions_dict.values():
                actions_list.append((accel, lc))
                if lc != 0:
                    lane_changes_count += 1

        except Exception as e:
            pass

    obs, reward, done, info = env.step(actions_dict)

    if step % 100 == 0:
        ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0
        print(f"  Step {step:3d}: Arrived={info.get('arrived_count', 0):2d}, "
              f"Departed={info.get('departed_count', 0):3d}, "
              f"OCR={ocr:.3f}, Controlled={len(actions_dict):2d}, "
              f"Lane changes={lane_changes_count}")

    if done:
        break

final_ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0

env.close()

# 分析动作
if len(actions_list) > 0:
    actions_array = np.array(actions_list)
    accels = actions_array[:, 0]
    lanes = actions_array[:, 1]

    print("\n" + "=" * 80)
    print("MPC动作分析")
    print("=" * 80)
    print(f"  总动作数: {len(actions_array)}")
    print(f"  换道次数: {lane_changes_count} ({lane_changes_count/len(actions_array)*100:.1f}%)")
    print(f"\n  换道分布:")
    print(f"    左变道 (-1): {np.sum(lanes == -1)} ({np.sum(lanes == -1)/len(lanes)*100:.1f}%)")
    print(f"    保持 (0):    {np.sum(lanes == 0)} ({np.sum(lanes == 0)/len(lanes)*100:.1f}%)")
    print(f"    右变道 (+1): {np.sum(lanes == 1)} ({np.sum(lanes == 1)/len(lanes)*100:.1f}%)")
    print(f"\n  加速度分布:")
    print(f"    均值: {np.mean(accels):.4f}")
    print(f"    标准差: {np.std(accels):.4f}")
    print(f"    范围: [{np.min(accels):.4f}, {np.max(accels):.4f}]")
    print(f"    加速 (>0.5): {np.sum(accels > 0.5)} ({np.sum(accels > 0.5)/len(accels)*100:.1f}%)")
    print(f"    减速 (<-0.5): {np.sum(accels < -0.5)} ({np.sum(accels < -0.5)/len(accels)*100:.1f}%)")

print("\n" + "=" * 80)
print(f"最终结果: OCR={final_ocr:.4f} ({final_ocr*100:.2f}%)")
print("=" * 80)

if lane_changes_count > 0:
    print("\n✅ 修复成功！MPC现在生成了换道指令")
else:
    print("\n⚠️  仍然没有换道，可能需要调整MPC参数")
