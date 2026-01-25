#!/usr/bin/env python3
"""
测试环境中的traci连接和arrived/departed统计
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
from src.env.competition_env import CompetitionSumoEnv
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 500  # 50 seconds

env = CompetitionSumoEnv(
    config=env_config,
    use_gui=False,
    device='cpu'
)

print("=== Testing Environment ===")
obs = env.reset()

for step in range(500):
    # Step with no action
    obs, reward, done, info = env.step({})

    if step % 100 == 0:
        print(f"[Step {step}]")
        print(f"  Arrived: {info.get('arrived_count', 0)}")
        print(f"  Departed: {info.get('departed_count', 0)}")
        print(f"  Vehicles: {len(obs.get('vehicle_ids', []))}")

    if done:
        print(f"\n[Episode done at step {step}]")
        print(f"  Final arrived: {info.get('arrived_count', 0)}")
        print(f"  Final departed: {info.get('departed_count', 0)}")
        print(f"  OCR: {info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0.0}")
        break

env.close()
print("\n=== Test Complete ===")
