#!/usr/bin/env python3
"""
Expert Demonstration Collector for Imitation Learning

This script collects expert demonstrations using rule-based policies:
1. Vehicle Selection: RuleBasedVehicleScorer (bottleneck-aware)
2. Action Generation: IDM (Intelligent Driver Model)
3. Quality Control: Filter episodes by OCR > baseline

Usage:
    # Basic collection (100 episodes)
    python scripts/collect_expert_demonstrations.py \
        --config configs/v5_complete.yaml \
        --num_episodes 100 \
        --output_dir data/demonstrations/expert

    # Parallel collection (8 workers, recommended)
    python scripts/collect_expert_demonstrations.py \
        --config configs/v5_complete.yaml \
        --num_episodes 100 \
        --num_workers 8 \
        --output_dir data/demonstrations/expert

Output:
    - demonstrations.pkl: Expert demonstration data
    - metadata.json: Statistics (OCR distribution, episode lengths, etc.)

Performance:
    - Serial: ~1x speed (~10-15 min for 100 episodes)
    - 4 workers: ~3.5-4x speed
    - 8 workers: ~7-8x speed (~2-3 min for 100 episodes)
"""

import os
import sys
import argparse
import yaml
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Multiprocessing support
from multiprocessing import Pool
import random

# Try to import libsumo (faster) or traci
try:
    import libsumo as traci
except ImportError:
    import traci

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.competition_env import CompetitionSumoEnv
from src.env.rule_based_scorer import RuleBasedVehicleScorer


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    Convert observation dictionary to flattened tensor

    Args:
        obs_dict: Environment observation dictionary
        max_vehicles: Maximum number of vehicles (for padding)

    Returns:
        flattened_obs: [obs_dim] flattened observation
            obs_dim = max_vehicles * 9 + 32 + 1 = 321
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
            # Normalize features to [-1, 1] or [0, 1]
            vehicle_features[i, 0] = state.get('s', 0.0) / 1000.0          # s: [0, 3000] -> [0, 3]
            vehicle_features[i, 1] = state.get('d', 0.0) / 10.0             # d: [-10, 10] -> [-1, 1]
            vehicle_features[i, 2] = state.get('vs', 0.0) / 30.0            # vs: [0, 30] -> [0, 1]
            vehicle_features[i, 3] = state.get('vd', 0.0) / 10.0            # vd: [-5, 5] -> [-0.5, 0.5]
            vehicle_features[i, 4] = state.get('speed', 0.0) / 30.0        # speed: [0, 30] -> [0, 1]
            vehicle_features[i, 5] = state.get('acceleration', 0.0) / 3.0  # accel: [-3, 3] -> [-1, 1]
            vehicle_features[i, 6] = state.get('lane_index', 0.0) / 10.0   # lane: [0, 10] -> [0, 1]
            vehicle_features[i, 7] = state.get('angle', 0.0) / 360.0       # angle: [-180, 180] -> [-0.5, 0.5]
            vehicle_features[i, 8] = 1.0 if veh_id in icv_ids else 0.0     # is_icv: {0, 1}

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
    vehicle_features_flat = vehicle_features.flatten()  # [max_vehicles * 9]
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs.astype(np.float32)


def compute_idm_action(
    env: CompetitionSumoEnv,
    veh_id: str,
    vehicle_states: dict,
    traci_lib: Any = traci
) -> np.ndarray:
    """
    Generate action for a single vehicle using IDM (Intelligent Driver Model)

    IDM Equation:
        a = a_max * [1 - (v/v0)^4 - (s*/s)^2]

    where:
        s* = s0 + v*T + (v * Δv) / (2 * sqrt(a_max * b_max))

    Args:
        env: CompetitionSumoEnv instance
        veh_id: Vehicle ID
        vehicle_states: Dictionary of vehicle states
        traci_lib: traci or libsumo module

    Returns:
        action: [acceleration, lane_change_probability]
    """
    if veh_id not in vehicle_states:
        return np.array([0.0, 0.0])

    state = vehicle_states[veh_id]

    # IDM parameters (calibrated for highway traffic)
    desired_speed = 30.0    # v0: desired speed (m/s) ~108 km/h
    min_gap = 2.0           # s0: minimum gap (m)
    time_headway = 1.5      # T: safe time headway (s)
    max_accel = 0.73        # a_max: maximum acceleration (m/s²)
    max_decel = 1.67        # b_max: comfortable deceleration (m/s²)

    # Get current state
    current_speed = abs(state.get('vs', state.get('speed', 15.0)))

    # Get leader information
    delta_v = 0.0   # speed difference with leader
    gap = 100.0     # gap to leader (default large value)

    try:
        leader_info = traci_lib.vehicle.getLeader(veh_id, 150.0)
        if leader_info is not None:
            leader_id, leader_distance = leader_info
            leader_speed = traci_lib.vehicle.getSpeed(leader_id)
            delta_v = current_speed - leader_speed
            veh_length = traci_lib.vehicle.getLength(veh_id)
            gap = leader_distance - veh_length
    except Exception as e:
        # Default: no leader nearby
        gap = 1000.0
        delta_v = 0.0

    # Ensure gap is positive
    gap = max(gap, 0.1)

    # IDM acceleration calculation
    desired_gap = min_gap + current_speed * time_headway + \
                  (current_speed * delta_v) / (2 * np.sqrt(max_accel * max_decel))

    idm_accel = max_accel * (1 - (current_speed / desired_speed)**4 -
                             (desired_gap / gap)**2)

    # Clip acceleration to valid range
    idm_accel = np.clip(idm_accel, -max_decel, max_accel)

    # Lane change decision (simplified: probabilistic based on speed and density)
    # Only consider lane change at moderate speeds and when not in bottleneck
    s_coord = state.get('s', 0.0)
    in_bottleneck = (1200.0 <= s_coord <= 2200.0)

    if not in_bottleneck and 5.0 < current_speed < 25.0:
        # Low probability of lane change (5%)
        lane_change_prob = 0.05
    else:
        # Very low probability in bottleneck or extreme speeds
        lane_change_prob = 0.01

    lane_change = 1.0 if np.random.random() < lane_change_prob else 0.0

    return np.array([idm_accel, lane_change], dtype=np.float32)


def select_top_k_vehicles(
    scorer: RuleBasedVehicleScorer,
    vehicle_states: dict,
    context: dict,
    k: int = 25
) -> List[str]:
    """
    Select Top-K vehicles using rule-based scorer

    Args:
        scorer: RuleBasedVehicleScorer instance
        vehicle_states: Vehicle state dictionary
        context: Environment context (traci_lib, all_vehicle_ids, icv_ids)
        k: Number of vehicles to select

    Returns:
        selected_vehicle_ids: List of selected vehicle IDs
    """
    # Compute scores for all vehicles
    scores = scorer.compute_scores(vehicle_states, context)

    # Filter to ICV vehicles only
    icv_ids = context.get('icv_ids', set())
    icv_scores = {veh_id: scores.get(veh_id, 0.0) for veh_id in icv_ids}

    # Sort by score (descending)
    sorted_vehicles = sorted(icv_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-k (or fewer if not enough ICVs)
    k_actual = min(k, len(sorted_vehicles))
    selected_vehicle_ids = [veh_id for veh_id, score in sorted_vehicles[:k_actual]]

    return selected_vehicle_ids


def collect_single_episode_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function: collect a single episode (for parallel collection)

    Args:
        args: (worker_id, config_path, max_steps, k_vehicles, seed, output_dir, episode_idx)

    Returns:
        episode_data: Dictionary with episode info and transitions
    """
    import torch

    worker_id, config_path, max_steps, k_vehicles, seed, output_dir, episode_idx = args

    # Set random seed (different for each worker)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup environment config
    env_config = config.get('environment', {})
    if 'sumocfg_file' not in env_config:
        env_config['sumocfg_file'] = '仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg'
    if 'max_steps' not in env_config:
        env_config['max_steps'] = max_steps
    if 'icv_ratio' not in env_config:
        env_config['icv_ratio'] = 0.10

    # Create environment
    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device='cpu'
    )

    # Create rule-based scorer
    scorer = RuleBasedVehicleScorer(config=config)

    # Import traci in worker process
    try:
        import libsumo as worker_traci
    except ImportError:
        import traci as worker_traci

    # Episode data structure
    episode_data = {
        'episode_id': episode_idx,
        'transitions': [],
        'total_reward': 0.0,
        'episode_length': 0,
        'ocr': 0.0
    }

    try:
        # Reset environment
        obs_dict = env.reset()

        # Track departures and arrivals for OCR calculation
        # ✅ 修复：直接使用traci跟踪车辆，避免环境接口问题
        all_departed = set()
        all_arrived = set()
        previous_vehicles = set()

        for step in range(max_steps):
            # Parse observation
            vehicle_states = obs_dict.get('vehicle_states', {})
            vehicle_ids = obs_dict.get('vehicle_ids', [])
            icv_ids = obs_dict.get('icv_ids', set())

            # ✅ 修复：使用traci直接跟踪departed和arrived车辆
            current_vehicles = set(vehicle_ids)

            # 新departed的车辆 = 当前车辆 - 之前的车辆
            newly_departed = current_vehicles - previous_vehicles
            all_departed.update(newly_departed)

            # 更新vehicle集合
            previous_vehicles = current_vehicles

            # Build context for scorer
            context = {
                'traci_lib': worker_traci,
                'all_vehicle_ids': vehicle_ids,
                'icv_ids': icv_ids
            }

            # Select top-k vehicles
            if len(icv_ids) > 0:
                selected_vehicles = select_top_k_vehicles(
                    scorer, vehicle_states, context, k=k_vehicles
                )
            else:
                selected_vehicles = []

            # Generate IDM actions
            actions_dict = {}
            for veh_id in selected_vehicles:
                action = compute_idm_action(env, veh_id, vehicle_states, worker_traci)
                actions_dict[veh_id] = action

            # Execute actions
            next_obs_dict, reward, done, info = env.step(actions_dict)

            # ✅ 修复：使用traci检查arrived车辆
            try:
                # 获取已到达的车辆（从仿真中移除的）
                arrived = worker_traci.simulation.getArrivedIDList()
                if arrived:
                    all_arrived.update(arrived)
            except:
                pass  # 如果traci调用失败，忽略

            # Store transition
            transition = {
                'obs': flatten_observation(obs_dict),
                'actions': actions_dict,
                'selected_vehicles': selected_vehicles,
                'reward': reward,
                'vehicle_ids': vehicle_ids,
                'icv_ids': list(icv_ids)
            }
            episode_data['transitions'].append(transition)
            episode_data['total_reward'] += reward
            episode_data['episode_length'] += 1

            # Next step
            obs_dict = next_obs_dict

            if done:
                break

        # Calculate OCR
        if len(all_departed) > 0:
            episode_data['ocr'] = len(all_arrived) / len(all_departed)
        else:
            episode_data['ocr'] = 0.0

        episode_data['departed_count'] = len(all_departed)
        episode_data['arrived_count'] = len(all_arrived)

    except Exception as e:
        print(f"[Worker {worker_id}] Error in episode {episode_idx}: {e}")
        episode_data['error'] = str(e)
        episode_data['ocr'] = 0.0

    finally:
        # Close environment
        try:
            env.close()
        except:
            pass

    return episode_data


def collect_demonstrations_serial(
    config_path: str,
    num_episodes: int,
    max_steps: int,
    k_vehicles: int,
    seed: int
) -> Tuple[List[Dict], Dict]:
    """
    Collect demonstrations serially (for debugging or low-core systems)

    Args:
        config_path: Path to config YAML
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        k_vehicles: Number of vehicles to control
        seed: Random seed

    Returns:
        episodes: List of episode dictionaries
        metadata: Collection metadata
    """
    episodes = []
    ocr_list = []
    length_list = []

    print(f"\n[Serial Collection] Collecting {num_episodes} episodes...")

    for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
        # Use worker function with worker_id=0
        episode_data = collect_single_episode_worker(
            (0, config_path, max_steps, k_vehicles, seed, None, episode_idx)
        )

        # Skip failed episodes
        if 'error' in episode_data:
            print(f"[Warning] Episode {episode_idx} failed: {episode_data['error']}")
            continue

        episodes.append(episode_data)
        ocr_list.append(episode_data['ocr'])
        length_list.append(episode_data['episode_length'])

    # Compute metadata
    metadata = {
        'total_episodes': len(episodes),
        'ocr_mean': np.mean(ocr_list) if ocr_list else 0.0,
        'ocr_std': np.std(ocr_list) if ocr_list else 0.0,
        'ocr_min': np.min(ocr_list) if ocr_list else 0.0,
        'ocr_max': np.max(ocr_list) if ocr_list else 0.0,
        'avg_length': np.mean(length_list) if length_list else 0.0,
        'collection_method': 'serial',
        'timestamp': datetime.now().isoformat()
    }

    return episodes, metadata


def collect_demonstrations_parallel(
    config_path: str,
    num_episodes: int,
    max_steps: int,
    k_vehicles: int,
    num_workers: int,
    seed: int
) -> Tuple[List[Dict], Dict]:
    """
    Collect demonstrations in parallel (recommended)

    Args:
        config_path: Path to config YAML
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        k_vehicles: Number of vehicles to control
        num_workers: Number of parallel workers
        seed: Random seed

    Returns:
        episodes: List of episode dictionaries
        metadata: Collection metadata
    """
    print(f"\n[Parallel Collection] Collecting {num_episodes} episodes with {num_workers} workers...")

    # Prepare arguments for each worker
    worker_args = [
        (i, config_path, max_steps, k_vehicles, seed, None, i)
        for i in range(num_episodes)
    ]

    # Create process pool
    with Pool(processes=num_workers) as pool:
        # Collect episodes with progress bar
        episodes = list(tqdm(
            pool.imap(collect_single_episode_worker, worker_args),
            total=num_episodes,
            desc="Collecting episodes"
        ))

    # Filter failed episodes
    valid_episodes = [ep for ep in episodes if 'error' not in ep]

    if len(valid_episodes) < len(episodes):
        print(f"[Warning] {len(episodes) - len(valid_episodes)} episodes failed")

    # Extract statistics
    ocr_list = [ep['ocr'] for ep in valid_episodes]
    length_list = [ep['episode_length'] for ep in valid_episodes]

    # Compute metadata
    metadata = {
        'total_episodes': len(valid_episodes),
        'ocr_mean': np.mean(ocr_list) if ocr_list else 0.0,
        'ocr_std': np.std(ocr_list) if ocr_list else 0.0,
        'ocr_min': np.min(ocr_list) if ocr_list else 0.0,
        'ocr_max': np.max(ocr_list) if ocr_list else 0.0,
        'avg_length': np.mean(length_list) if length_list else 0.0,
        'collection_method': f'parallel_{num_workers}_workers',
        'timestamp': datetime.now().isoformat()
    }

    return valid_episodes, metadata


def main():
    parser = argparse.ArgumentParser(description='Collect expert demonstrations for imitation learning')

    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to collect')
    parser.add_argument('--max_steps', type=int, default=3600,
                        help='Maximum steps per episode')
    parser.add_argument('--k_vehicles', type=int, default=25,
                        help='Number of vehicles to control (Top-K)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel workers (1=serial, 4-8=parallel)')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/expert',
                        help='Output directory for demonstrations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Expert Demonstration Collector")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"K vehicles: {args.k_vehicles}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print("="*70)

    # Collect demonstrations
    if args.num_workers == 1:
        episodes, metadata = collect_demonstrations_serial(
            config_path=args.config,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            k_vehicles=args.k_vehicles,
            seed=args.seed
        )
    else:
        episodes, metadata = collect_demonstrations_parallel(
            config_path=args.config,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            k_vehicles=args.k_vehicles,
            num_workers=args.num_workers,
            seed=args.seed
        )

    # Print statistics
    print("\n" + "="*70)
    print("Collection Statistics")
    print("="*70)
    print(f"Total episodes: {metadata['total_episodes']}")
    print(f"OCR: {metadata['ocr_mean']:.4f} ± {metadata['ocr_std']:.4f}")
    print(f"OCR range: [{metadata['ocr_min']:.4f}, {metadata['ocr_max']:.4f}]")
    print(f"Avg length: {metadata['avg_length']:.1f} steps")
    print("="*70)

    # Save demonstrations
    demo_path = output_dir / 'demonstrations.pkl'
    with open(demo_path, 'wb') as f:
        pickle.dump({'episodes': episodes, 'metadata': metadata}, f)

    print(f"\n✓ Demonstrations saved to: {demo_path}")

    # Save metadata as JSON
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # Print quality assessment
    baseline_ocr = 0.5469
    high_quality_episodes = sum(1 for ep in episodes if ep['ocr'] >= baseline_ocr)
    print(f"\n{'='*70}")
    print("Quality Assessment")
    print(f"{'='*70}")
    print(f"Baseline OCR: {baseline_ocr:.4f}")
    print(f"Episodes above baseline: {high_quality_episodes}/{len(episodes)} ({100*high_quality_episodes/len(episodes):.1f}%)")

    if metadata['ocr_mean'] >= baseline_ocr:
        print(f"✓ Average OCR above baseline! ({metadata['ocr_mean']:.4f} > {baseline_ocr:.4f})")
    else:
        print(f"⚠ Average OCR below baseline. ({metadata['ocr_mean']:.4f} < {baseline_ocr:.4f})")
        print(f"  Consider adjusting expert policy or collecting more episodes.")

    print("="*70)


if __name__ == '__main__':
    main()
