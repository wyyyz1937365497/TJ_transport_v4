#!/usr/bin/env python3
"""
GPU加速的MPC专家演示收集脚本

使用GPU加速的MPC控制器生成专家演示数据。

关键优化：
1. GPU MPC控制器（基于梯度，不用CVXPY）
2. GPU车辆评分器（向量化计算）
3. 减少CPU-GPU通信
4. LibSUMO加速仿真

Usage:
    python scripts/collect_mpc_demonstrations_gpu.py \
        --config configs/mpc.yaml \
        --num_episodes 50 \
        --output_dir data/demonstrations/mpc_gpu
"""

import os
import sys
import argparse
import yaml
import pickle
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, List, Tuple

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.env.competition_env import CompetitionSumoEnv
from src.mpc import MPCConfig, GPUMPCController
from src.env.vehicle_scorer_gpu import GPUVehicleScorer


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    Convert observation dictionary to flattened tensor

    Args:
        obs_dict: Environment observation dictionary
        max_vehicles: Maximum number of vehicles (for padding)

    Returns:
        flattened_obs: [obs_dim] flattened observation
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])
    icv_ids = obs_dict.get('icv_ids', set())
    global_stats = obs_dict.get('global_stats', np.zeros(32))

    num_vehicles = len(vehicle_ids)

    # Extract 9D vehicle features (normalized)
    vehicle_features = np.zeros((max_vehicles, 9), dtype=np.float32)

    for i, veh_id in enumerate(vehicle_ids[:max_vehicles]):
        if veh_id in vehicle_states:
            state = vehicle_states[veh_id]
            vehicle_features[i, 0] = state.get('s', 0.0) / 1000.0
            vehicle_features[i, 1] = state.get('d', 0.0) / 10.0
            vehicle_features[i, 2] = state.get('vs', 0.0) / 30.0
            vehicle_features[i, 3] = state.get('vd', 0.0) / 10.0
            vehicle_features[i, 4] = state.get('speed', 0.0) / 30.0
            vehicle_features[i, 5] = state.get('acceleration', 0.0) / 3.0
            vehicle_features[i, 6] = state.get('lane_index', 0.0) / 10.0
            vehicle_features[i, 7] = state.get('angle', 0.0) / 360.0
            vehicle_features[i, 8] = 1.0 if veh_id in icv_ids else 0.0

    # Ensure global_stats is 32D
    global_stats_flat = global_stats.flatten()
    if len(global_stats_flat) < 32:
        global_stats_flat = np.concatenate([
            global_stats_flat,
            np.zeros(32 - len(global_stats_flat), dtype=np.float32)
        ])
    elif len(global_stats_flat) > 32:
        global_stats_flat = global_stats_flat[:32]

    # Flatten and concatenate
    vehicle_features_flat = vehicle_features.flatten()
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs.astype(np.float32)


def collect_single_episode_gpu(args: Tuple) -> Dict:
    """
    Collect a single episode using GPU MPC

    Args:
        args: (episode_id, config_dict, seed, device)

    Returns:
        episode_data: Episode dictionary
    """
    episode_id, config_dict, seed, device = args

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env_config = config_dict['environment'].copy()
    env_config['max_steps'] = config_dict['data_collection'].get('timeout', 400) * 10

    # Ensure scoring configuration
    if 'neural_icv_scoring' not in env_config:
        env_config['neural_icv_scoring'] = {'enabled': False}
    else:
        env_config['neural_icv_scoring']['enabled'] = False

    if 'rule_based_scoring' not in env_config:
        env_config['rule_based_scoring'] = {'enabled': True}
    else:
        env_config['rule_based_scoring']['enabled'] = True

    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device='cpu'  # SUMO在CPU上运行
    )

    # Create GPU MPC controller
    mpc_config_dict = config_dict['mpc']

    # Convert to MPCConfig
    mpc_config = MPCConfig(
        prediction_horizon=mpc_config_dict.get('prediction_horizon', 10),
        control_horizon=mpc_config_dict.get('control_horizon', 5),
        dt=mpc_config_dict.get('dt', 0.1),

        # State weights
        Q_speed=mpc_config_dict.get('state_weights', {}).get('speed', 10.0),
        Q_position=mpc_config_dict.get('state_weights', {}).get('position', 1.0),
        Q_accel=mpc_config_dict.get('state_weights', {}).get('acceleration', 0.1),
        Q_gap=mpc_config_dict.get('state_weights', {}).get('gap', 50.0),

        # Control weights
        R_accel=mpc_config_dict.get('control_weights', {}).get('acceleration', 0.5),
        R_lane=mpc_config_dict.get('control_weights', {}).get('lane_change', 0.1),

        # Constraints
        min_accel=mpc_config_dict.get('constraints', {}).get('min_accel', -4.5),
        max_accel=mpc_config_dict.get('constraints', {}).get('max_accel', 2.0),
        min_speed=mpc_config_dict.get('constraints', {}).get('min_speed', 0.0),
        max_speed=mpc_config_dict.get('constraints', {}).get('max_speed', 30.0),
        min_gap=mpc_config_dict.get('constraints', {}).get('min_gap', 2.0),
        desired_gap=mpc_config_dict.get('constraints', {}).get('desired_gap', 5.0),

        # Bottleneck
        bottleneck_s_min=mpc_config_dict.get('bottleneck_region', {}).get('s_min', 1200.0),
        bottleneck_s_max=mpc_config_dict.get('bottleneck_region', {}).get('s_max', 2200.0),

        # Solver (not used for GPU MPC)
        solver=mpc_config_dict.get('solver', {}).get('name', 'OSQP'),
        verbose=mpc_config_dict.get('solver', {}).get('verbose', False),
        max_iters=mpc_config_dict.get('solver', {}).get('max_iters', 1000),
        tolerance=mpc_config_dict.get('solver', {}).get('tolerance', 1e-4),

        # Incremental optimization
        update_interval=mpc_config_dict.get('incremental_optimization', {}).get('update_interval', 3),
        warm_start=mpc_config_dict.get('incremental_optimization', {}).get('warm_start', True)
    )

    mpc_controller = GPUMPCController(config=mpc_config, device=device)

    # Create GPU vehicle scorer
    scorer = GPUVehicleScorer(config=config_dict, device=device)

    # Episode data
    transitions = []

    # Reset environment
    obs_dict = env.reset()

    # Tracking variables
    departed_vehicles = set()
    arrived_vehicles = set()
    previous_vehicles = set()

    total_reward = 0.0
    step = 0
    max_steps = 3600

    # Import traci for tracking
    try:
        import libsumo as traci_lib
    except ImportError:
        import traci as traci_lib

    # Performance tracking
    mpc_times = []
    scorer_times = []

    try:
        for step in range(max_steps):
            # Parse observation
            vehicle_states = obs_dict.get('vehicle_states', {})
            vehicle_ids = obs_dict.get('vehicle_ids', [])
            icv_ids = obs_dict.get('icv_ids', set())

            # Track departed vehicles
            current_vehicles = set(vehicle_ids)
            newly_departed = current_vehicles - previous_vehicles
            departed_vehicles.update(newly_departed)
            previous_vehicles = current_vehicles

            # ========== GPU车辆评分 ==========
            scorer_start = time.time()
            if len(icv_ids) > 0:
                selected_vehicles = scorer.get_top_k_vehicles(
                    vehicle_states,
                    list(icv_ids),
                    k=config_dict['mpc'].get('vehicle_selection', {}).get('k_vehicles', 25)
                )
            else:
                selected_vehicles = []
            scorer_times.append(time.time() - scorer_start)

            # ========== GPU MPC求解 ==========
            try:
                actions_dict = mpc_controller.solve(
                    observation_dict=obs_dict,
                    vehicle_ids=vehicle_ids,
                    icv_ids=selected_vehicles,
                    step=step
                )
                mpc_times.append(mpc_controller.get_solve_time())
            except Exception as e:
                # MPC failed, use fallback
                print(f"[Episode {episode_id}] GPU MPC failed at step {step}: {e}")
                actions_dict = {}

            # Store transition
            obs_flat = flatten_observation(obs_dict)

            # Flatten actions
            actions_flat = np.zeros(32 * 2, dtype=np.float32)
            mask = np.zeros(32, dtype=np.float32)

            for i, veh_id in enumerate(vehicle_ids[:32]):
                if veh_id in actions_dict:
                    action = actions_dict[veh_id]
                    actions_flat[i * 2] = float(action[0])     # acceleration
                    actions_flat[i * 2 + 1] = float(action[1]) # lane_change
                    mask[i] = 1.0  # 标记为受控车辆

            transition = {
                'obs': obs_flat,
                'actions': actions_flat,
                'mask': mask,
                'vehicle_ids': vehicle_ids,
                'icv_ids': list(icv_ids),
                'selected_vehicles': selected_vehicles
            }

            transitions.append(transition)

            # Execute actions
            next_obs_dict, reward, done, info = env.step(actions_dict)
            total_reward += reward

            # ✅ 修复：使用环境info获取arrived/departed统计（避免traci连接问题）
            arrived_count = info.get('arrived_count', 0)
            if arrived_count > len(arrived_vehicles):
                arrived_vehicles = set(range(arrived_count))

            obs_dict = next_obs_dict

            if done:
                break

            # 每100步输出一次性能统计
            if (step + 1) % 100 == 0:
                avg_mpc_time = np.mean(mpc_times[-100:]) if len(mpc_times) > 0 else 0
                avg_scorer_time = np.mean(scorer_times[-100:]) if len(scorer_times) > 0 else 0
                print(f"  [Episode {episode_id}] Step {step+1}/{max_steps} | "
                      f"MPC: {avg_mpc_time*1000:.1f}ms | Scorer: {avg_scorer_time*1000:.1f}ms")

    except Exception as e:
        print(f"[Episode {episode_id}] Error at step {step}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        env.close()

    # Calculate OCR
    ocr = len(arrived_vehicles) / len(departed_vehicles) if len(departed_vehicles) > 0 else 0.0

    episode_data = {
        'episode_id': episode_id,
        'transitions': transitions,
        'ocr': ocr,
        'reward': total_reward,
        'num_steps': step + 1,
        'departed': len(departed_vehicles),
        'arrived': len(arrived_vehicles),
        'performance': {
            'avg_mpc_time_ms': np.mean(mpc_times) * 1000 if len(mpc_times) > 0 else 0,
            'avg_scorer_time_ms': np.mean(scorer_times) * 1000 if len(scorer_times) > 0 else 0
        }
    }

    return episode_data


def main():
    parser = argparse.ArgumentParser(description='Collect GPU MPC expert demonstrations')
    parser.add_argument('--config', type=str, default='configs/mpc.yaml',
                        help='Path to config YAML')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/mpc_gpu',
                        help='Output directory')
    parser.add_argument('--min_ocr', type=float, default=0.5469,
                        help='Minimum OCR for filtering')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"GPU MPC演示收集")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes
    all_transitions = []
    episode_ocrs = []
    episode_rewards = []

    start_time = time.time()

    for episode_id in range(args.num_episodes):
        print(f"Episode {episode_id + 1}/{args.num_episodes}")

        episode_args = (episode_id, config_dict, 42 + episode_id, device)
        episode_data = collect_single_episode_gpu(episode_args)

        # Filter by OCR
        if episode_data['ocr'] >= args.min_ocr:
            all_transitions.extend(episode_data['transitions'])
            episode_ocrs.append(episode_data['ocr'])
            episode_rewards.append(episode_data['reward'])

            print(f"  OCR: {episode_data['ocr']:.4f} ✓ (accepted)")
        else:
            print(f"  OCR: {episode_data['ocr']:.4f} ✗ (rejected, below {args.min_ocr})")

        # Print performance
        perf = episode_data['performance']
        print(f"  Performance: MPC={perf['avg_mpc_time_ms']:.1f}ms, "
              f"Scorer={perf['avg_scorer_time_ms']:.1f}ms\n")

    # Save
    output_file = output_dir / 'demonstrations.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_transitions, f)

    # Save metadata
    metadata = {
        'num_episodes': args.num_episodes,
        'num_filtered': len(episode_ocrs),
        'ocr_mean': np.mean(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_std': np.std(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_min': np.min(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_max': np.max(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'reward_mean': np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
        'reward_std': np.std(episode_rewards) if len(episode_rewards) > 0 else 0.0,
        'total_transitions': len(all_transitions),
        'collection_time_minutes': (time.time() - start_time) / 60.0,
        'min_ocr_threshold': args.min_ocr,
        'device': device
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Collection Statistics")
    print(f"{'='*60}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Filtered episodes (OCR >= {args.min_ocr}): {len(episode_ocrs)}")
    print(f"Collection time: {metadata['collection_time_minutes']:.1f} min")

    if len(episode_ocrs) > 0:
        print(f"\nOCR Statistics:")
        print(f"  Mean: {metadata['ocr_mean']:.4f}")
        print(f"  Std:  {metadata['ocr_std']:.4f}")
        print(f"  Min:  {metadata['ocr_min']:.4f}")
        print(f"  Max:  {metadata['ocr_max']:.4f}")

        print(f"\nReward Statistics:")
        print(f"  Mean: {metadata['reward_mean']:.2f}")
        print(f"  Std:  {metadata['reward_std']:.2f}")

    print(f"\nTotal transitions: {len(all_transitions)}")
    print(f"{'='*60}\n")

    print(f"✓ Demonstrations saved to {output_file}")
    print(f"✓ Metadata saved to {metadata_file}")

    if len(episode_ocrs) == 0:
        print(f"\n⚠️  WARNING: No episodes passed OCR filter!")
        print(f"   Consider checking MPC configuration or lowering min_ocr threshold")


if __name__ == '__main__':
    main()
