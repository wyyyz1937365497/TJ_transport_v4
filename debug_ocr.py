#!/usr/bin/env python3
"""
诊断OCR=0问题
"""
import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import yaml
import numpy as np
from src.env.competition_env import CompetitionSumoEnv

# Load config
with open('configs/mpc.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create environment (env_config needs to have sumo_config at top level)
env_config = config['environment'].copy()
env_config['max_steps'] = 1000  # Longer test (100 seconds)

env = CompetitionSumoEnv(
    config=env_config,
    use_gui=False,
    device='cpu'
)

print("=== Starting Environment ===")
obs = env.reset()

# Import traci directly (environment already has connection)
import traci

# Tracking
departed_vehicles = set()
arrived_vehicles = set()
previous_vehicles = set()

for step in range(1000):
    vehicle_ids = obs.get('vehicle_ids', [])
    current_vehicles = set(vehicle_ids)

    # Track departed
    newly_departed = current_vehicles - previous_vehicles
    departed_vehicles.update(newly_departed)

    # Track arrived
    try:
        arrived = traci.simulation.getArrivedIDList()
        if arrived:
            arrived_vehicles.update(arrived)
            print(f"[Step {step}] {len(arrived)} vehicles arrived: {arrived[:5]}")
    except Exception as e:
        print(f"[Step {step}] Error getting arrived: {e}")

    # Track departed via traci
    try:
        departed = traci.simulation.getDepartedIDList()
        if departed:
            print(f"[Step {step}] {len(departed)} vehicles departed: {departed[:5]}")
    except Exception as e:
        print(f"[Step {step}] Error getting departed: {e}")

    previous_vehicles = current_vehicles

    # Random action (no MPC)
    action_dict = {}
    next_obs, reward, done, info = env.step(action_dict)

    if done:
        print(f"[Step {step}] Episode done")
        break

    obs = next_obs

    if step % 20 == 0:
        print(f"[Step {step}] Current: {len(current_vehicles)} vehicles, "
              f"Total departed: {len(departed_vehicles)}, "
              f"Total arrived: {len(arrived_vehicles)}")

env.close()

print("\n=== Final Statistics ===")
print(f"Total departed: {len(departed_vehicles)}")
print(f"Total arrived: {len(arrived_vehicles)}")
print(f"OCR: {len(arrived_vehicles) / len(departed_vehicles) if len(departed_vehicles) > 0 else 0.0}")
