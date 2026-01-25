#!/usr/bin/env python3
"""
测试baseline性能（无MPC控制）
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
from src.env.competition_env import CompetitionSumoEnv
import logging

logging.basicConfig(level=logging.WARNING)

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

env_config = config['environment'].copy()
env_config['max_steps'] = 3600  # Full 6 minutes

env = CompetitionSumoEnv(
    config=env_config,
    use_gui=False,
    device='cpu'
)

print("=== Baseline Test (No Control) ===")
obs = env.reset()

for step in range(3600):
    # No control action
    obs, reward, done, info = env.step({})

    if step % 600 == 0:
        print(f"[Step {step}] Arrived: {info.get('arrived_count', 0)}, Departed: {info.get('departed_count', 0)}, OCR: {info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0.0:.3f}")

    if done:
        print(f"\n[Episode done at step {step}]")
        print(f"  Final arrived: {info.get('arrived_count', 0)}")
        print(f"  Final departed: {info.get('departed_count', 0)}")
        departed = info.get('departed_count', 0)
        arrived = info.get('arrived_count', 0)
        ocr = arrived / departed if departed > 0 else 0.0
        print(f"  Final OCR: {ocr:.4f} ({ocr*100:.2f}%)")
        break

env.close()
print("\n=== Baseline Test Complete ===")
