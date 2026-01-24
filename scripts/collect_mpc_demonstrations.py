#!/usr/bin/env python3
"""
MPC专家演示收集脚本

使用MPC控制器生成专家演示数据，用于训练模仿学习模型。

关键特性：
1. 使用MPC最优控制策略
2. 多车辆协同优化
3. 瓶颈区域优化
4. 并行收集（多worker）
5. 数据质量检查

Usage:
    # 基本使用
    python scripts/collect_mpc_demonstrations.py \
        --config configs/mpc.yaml \
        --num_episodes 50 \
        --num_workers 4

    # 自定义输出
    python scripts/collect_mpc_demonstrations.py \
        --config configs/mpc.yaml \
        --num_episodes 50 \
        --output_dir data/demonstrations/mpc_custom

Output:
    - data/demonstrations/mpc/demonstrations.pkl: 演示数据
    - data/demonstrations/mpc/metadata.json: 元数据
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
from multiprocessing import Pool, Manager
from typing import Dict, List, Tuple

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.competition_env import CompetitionSumoEnv
from src.env.rule_based_scorer import RuleBasedVehicleScorer
from src.mpc import MPCController, MPCConfig


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


def select_top_k_vehicles(
    scorer: RuleBasedVehicleScorer,
    vehicle_states: dict,
    context: dict,
    k: int = 25
) -> list:
    """Select top-k vehicles using rule-based scorer"""
    scores = scorer.compute_scores(vehicle_states, context)

    # Filter to ICV vehicles
    icv_ids = context.get('icv_ids', set())
    icv_scores = {veh_id: scores.get(veh_id, 0.0) for veh_id in icv_ids}

    # Sort by score
    sorted_vehicles = sorted(icv_scores.items(), key=lambda x: x[1], reverse=True)
    selected_vehicle_ids = [veh_id for veh_id, score in sorted_vehicles[:k]]

    return selected_vehicle_ids


def collect_single_episode(args: Tuple) -> Dict:
    """
    Collect a single episode using MPC

    Args:
        args: (episode_id, config_dict, seed)

    Returns:
        episode_data: Episode dictionary
    """
    episode_id, config_dict, seed = args

    # Set random seed
    np.random.seed(seed)

    # Create environment
    env_config = config_dict['environment'].copy()
    env_config['max_steps'] = config_dict['data_collection'].get('timeout', 400) * 10  # ~400s

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
        device='cpu'
    )

    # Create MPC controller
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

        # Solver
        solver=mpc_config_dict.get('solver', {}).get('name', 'OSQP'),
        verbose=mpc_config_dict.get('solver', {}).get('verbose', False),
        max_iters=mpc_config_dict.get('solver', {}).get('max_iters', 1000),
        tolerance=mpc_config_dict.get('solver', {}).get('tolerance', 1e-4),

        # Incremental optimization
        update_interval=mpc_config_dict.get('incremental_optimization', {}).get('update_interval', 3),
        warm_start=mpc_config_dict.get('incremental_optimization', {}).get('warm_start', True)
    )

    mpc_controller = MPCController(config=mpc_config)

    # Create scorer for vehicle selection
    scorer = RuleBasedVehicleScorer(config=config_dict)

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

            # Select top-k vehicles
            if len(icv_ids) > 0:
                context = {
                    'traci_lib': None,
                    'all_vehicle_ids': vehicle_ids,
                    'icv_ids': icv_ids
                }
                selected_vehicles = select_top_k_vehicles(
                    scorer, vehicle_states, context,
                    k=config_dict['mpc'].get('vehicle_selection', {}).get('k_vehicles', 25)
                )
            else:
                selected_vehicles = []

            # MPC solve
            try:
                actions_dict = mpc_controller.solve(
                    observation_dict=obs_dict,
                    vehicle_ids=vehicle_ids,
                    icv_ids=selected_vehicles,
                    step=step
                )
            except Exception as e:
                # MPC failed, use fallback
                print(f"[Episode {episode_id}] MPC failed at step {step}: {e}")
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

            # Track arrived vehicles
            try:
                arrived = traci_lib.simulation.getArrivedIDList()
                if arrived:
                    arrived_vehicles.update(arrived)
            except:
                pass

            obs_dict = next_obs_dict

            if done:
                break

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
        'episode_length': step + 1,
        'departed_count': len(departed_vehicles),
        'arrived_count': len(arrived_vehicles)
    }

    return episode_data


def main():
    parser = argparse.ArgumentParser(description='Collect MPC expert demonstrations')

    parser.add_argument('--config', type=str, default='configs/mpc.yaml',
                        help='Path to config YAML')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/mpc',
                        help='Output directory')
    parser.add_argument('--min_ocr', type=float, default=0.5469,
                        help='Minimum OCR for filtering')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print("MPC Expert Demonstration Collection")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print(f"Min OCR: {args.min_ocr}")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare episode arguments
    episode_args = []
    for i in range(args.num_episodes):
        seed = config.get('global', {}).get('seed', 42) + i
        episode_args.append((i, config, seed))

    # Collect episodes
    print(f"\nCollecting {args.num_episodes} episodes with {args.num_workers} workers...")
    print("-"*70)

    start_time = time.time()

    if args.num_workers > 1:
        # Parallel collection
        with Pool(args.num_workers) as pool:
            episodes = list(tqdm(
                pool.imap(collect_single_episode, episode_args),
                total=args.num_episodes,
                desc="Collecting"
            ))
    else:
        # Sequential collection
        episodes = []
        for episode_arg in tqdm(episode_args, desc="Collecting"):
            episode = collect_single_episode(episode_arg)
            episodes.append(episode)

    collect_time = time.time() - start_time

    # Filter episodes by OCR
    filtered_episodes = [ep for ep in episodes if ep['ocr'] >= args.min_ocr]

    print("\n" + "="*70)
    print("Collection Statistics")
    print("="*70)
    print(f"Total episodes: {len(episodes)}")
    print(f"Filtered episodes (OCR >= {args.min_ocr}): {len(filtered_episodes)}")
    print(f"Collection time: {collect_time/60:.1f} min")

    if len(episodes) > 0:
        ocr_list = [ep['ocr'] for ep in episodes]
        reward_list = [ep['reward'] for ep in episodes]

        print(f"\nOCR Statistics:")
        print(f"  Mean: {np.mean(ocr_list):.4f}")
        print(f"  Std:  {np.std(ocr_list):.4f}")
        print(f"  Min:  {np.min(ocr_list):.4f}")
        print(f"  Max:  {np.max(ocr_list):.4f}")

        print(f"\nReward Statistics:")
        print(f"  Mean: {np.mean(reward_list):.2f}")
        print(f"  Std:  {np.std(reward_list):.2f}")

        # Count total transitions
        total_transitions = sum(len(ep['transitions']) for ep in filtered_episodes)
        print(f"\nTotal transitions: {total_transitions}")

    print("="*70)

    # Save demonstrations
    demonstrations = {
        'episodes': filtered_episodes,
        'metadata': {
            'num_episodes': len(filtered_episodes),
            'total_transitions': sum(len(ep['transitions']) for ep in filtered_episodes),
            'collection_time': collect_time,
            'config': config
        }
    }

    demo_path = output_dir / 'demonstrations.pkl'
    with open(demo_path, 'wb') as f:
        pickle.dump(demonstrations, f)

    print(f"\n✓ Demonstrations saved to {demo_path}")

    # Save metadata
    if len(episodes) > 0:
        metadata = {
            'num_episodes': len(episodes),
            'num_filtered': len(filtered_episodes),
            'ocr_mean': float(np.mean(ocr_list)),
            'ocr_std': float(np.std(ocr_list)),
            'ocr_min': float(np.min(ocr_list)),
            'ocr_max': float(np.max(ocr_list)),
            'reward_mean': float(np.mean(reward_list)),
            'reward_std': float(np.std(reward_list)),
            'total_transitions': total_transitions,
            'collection_time_minutes': collect_time / 60,
            'min_ocr_threshold': args.min_ocr
        }

        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Metadata saved to {metadata_path}")

    # Check if successful
    if len(filtered_episodes) == 0:
        print("\n⚠️  WARNING: No episodes passed OCR filter!")
        print("   Consider checking MPC configuration or lowering min_ocr threshold")
    else:
        mean_ocr = np.mean([ep['ocr'] for ep in filtered_episodes])
        if mean_ocr > args.min_ocr:
            print(f"\n✅ SUCCESS: Mean OCR {mean_ocr:.4f} > baseline {args.min_ocr:.4f}")
        else:
            print(f"\n⚠️  WARNING: Mean OCR {mean_ocr:.4f} < baseline {args.min_ocr:.4f}")


if __name__ == '__main__':
    main()
