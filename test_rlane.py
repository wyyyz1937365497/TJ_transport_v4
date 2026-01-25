#!/usr/bin/env python3
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
import numpy as np
from src.env.competition_env import CompetitionSumoEnv
from src.mpc import MPCConfig, GPUMPCController
from src.env.vehicle_scorer_gpu import GPUVehicleScorer
import logging

logging.basicConfig(level=logging.ERROR)

# Load modified config
with open('configs/mpc_lane_fix.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 200

env = CompetitionSumoEnv(config=env_config, use_gui=False, device='cpu')

mpc_config_dict = config['mpc']
mpc_config = MPCConfig(
    prediction_horizon=10,
    control_horizon=5,
    dt=0.1,
    Q_speed=10.0,
    Q_position=1.0,
    Q_accel=0.1,
    Q_gap=50.0,
    R_accel=0.5,
    R_lane=0.001,  # 降低100倍
    min_accel=-4.5,
    max_accel=2.0,
    min_speed=0.0,
    max_speed=30.0,
    min_gap=2.0,
    desired_gap=5.0,
    bottleneck_s_min=1200.0,
    bottleneck_s_max=2200.0,
)

mpc = GPUMPCController(config=mpc_config, device='cuda')
scorer = GPUVehicleScorer(config=config, device='cuda')

print('测试R_lane=0.001 (降低100倍)')
print('='*60)

obs = env.reset()
lane_changes = []
actions_all = []

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
            actions_dict = mpc.solve(obs, vehicle_ids, selected_vehicles, step)
            for accel, lc in actions_dict.values():
                actions_all.append((accel, lc))
                if lc != 0:
                    lane_changes.append(lc)
        except:
            pass

    obs, _, done, _ = env.step(actions_dict)
    if done:
        break

env.close()

if len(actions_all) > 0:
    arr = np.array(actions_all)
    print(f'总动作数: {len(arr)}')
    print(f'换道次数: {len(lane_changes)} ({len(lane_changes)/len(arr)*100:.1f}%)')
    print(f'换道分布:')
    print(f'  左(-1): {np.sum(arr[:,1] == -1)} ({np.sum(arr[:,1] == -1)/len(arr)*100:.1f}%)')
    print(f'  保持(0): {np.sum(arr[:,1] == 0)} ({np.sum(arr[:,1] == 0)/len(arr)*100:.1f}%)')
    print(f'  右(+1): {np.sum(arr[:,1] == 1)} ({np.sum(arr[:,1] == 1)/len(arr)*100:.1f}%)')
    print()
    if len(lane_changes) > 0:
        print('✅ 成功！MPC现在生成了换道指令')
    else:
        print('⚠️  仍然没有换道，可能需要进一步降低R_lane')
