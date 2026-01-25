#!/usr/bin/env python3
"""
并行MPC演示收集脚本（基于spawn + LibSUMO）

关键特性：
1. 使用multiprocessing.spawn避免fork问题
2. LibSUMO支持多进程并行（官方文档）
3. LibSUMO自动管理端口（不需要手动设置）
4. GPU加速的MPC控制器和车辆评分器

基于官方文档说明：
> running parallel instances of libsumo requires the multiprocessing module (in python)

关键点：
- ✅ LibSUMO支持多进程并行（使用multiprocessing）
- ✅ 不设置端口（LibSUMO自动管理）
- ✅ 使用libsumo.start()而不是traci.start()

基于之前的并行数据收集实现:
- commit b965d85: feat(stage0): 添加SUMO并行数据收集能力
- commit f74f031: feat(env): 修复并行数据收集中的SUMO连接状态共享问题

Usage:
    # 使用4个worker并行收集（推荐）
    python scripts/collect_mpc_parallel.py \
        --config configs/mpc.yaml \
        --num_episodes 50 \
        --num_workers 4 \
        --output_dir data/demonstrations/mpc_parallel

    # 使用8个worker高性能收集
    python scripts/collect_mpc_parallel.py \
        --config configs/mpc.yaml \
        --num_episodes 100 \
        --num_workers 8 \
        --output_dir data/demonstrations/mpc_parallel
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
from multiprocessing import Pool
import multiprocessing

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ========== 关键：使用LibSUMO支持多进程并行 ==========
# 根据官方文档：libsumo支持多进程，只需要使用multiprocessing模块
# 不设置端口（LibSUMO自动管理）
import torch
from src.env.competition_env import CompetitionSumoEnv
from src.mpc import MPCConfig, GPUMPCController
from src.env.vehicle_scorer_gpu import GPUVehicleScorer

# ========== 关键：设置spawn启动方法 ==========
# 避免fork导致的TraCI状态共享问题
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # 已经设置过，忽略
    pass


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """Convert observation dictionary to flattened tensor"""
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


def collect_single_episode_worker(args: Tuple) -> Dict:
    """
    Worker函数：收集单个episode（在独立进程中运行）

    Args:
        args: (episode_id, config_dict, seed, device, worker_id, port)

    Returns:
        episode_data: Episode dictionary
    """
    episode_id, config_dict, seed, device, worker_id, port = args

    # ========== 关键：LibSUMO多进程并行设置 ==========
    # 根据官方文档：
    # - LibSUMO支持多进程并行，使用multiprocessing模块
    # - 不需要也不支持设置端口（LibSUMO自动管理）
    # - 不使用traci.init或traci.connect，直接使用libsumo.start()

    # 1. 清理任何现有的连接（确保状态干净）
    try:
        import libsumo
        libsumo.close()
    except:
        pass

    # 2. 不设置端口参数，让LibSUMO自动管理
    # （不设置 SUMO_PORT 环境变量）

    # 3. Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ========== 创建环境和控制器 ==========
    env_config = config_dict['environment'].copy()
    env_config['max_steps'] = config_dict['data_collection'].get('timeout', 400) * 10

    # 不设置port参数（LibSUMO自动管理）
    # env_config['port'] = port  # 注释掉！

    # Ensure scoring configuration
    if 'neural_icv_scoring' not in env_config:
        env_config['neural_icv_scoring'] = {'enabled': False}
    else:
        env_config['neural_icv_scoring']['enabled'] = False

    if 'rule_based_scoring' not in env_config:
        env_config['rule_based_scoring'] = {'enabled': True}
    else:
        env_config['rule_based_scoring']['enabled'] = True

    try:
        env = CompetitionSumoEnv(
            config=env_config,
            use_gui=False,
            device='cpu'
        )
    except Exception as e:
        return {
            'episode_id': episode_id,
            'worker_id': worker_id,
            'ocr': 0.0,
            'error': str(e),
            'transitions': [],
            'num_steps': 0,
            'departed': 0,
            'arrived': 0,
            'performance': {'avg_mpc_time_ms': 0, 'avg_scorer_time_ms': 0}
        }

    # Create GPU MPC controller
    mpc_config_dict = config_dict['mpc']

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
        import traci as traci_lib
    except ImportError:
        import libsumo as traci_lib

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

            # GPU车辆评分
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

            # GPU MPC求解
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
                if step % 500 == 0:  # 减少错误打印
                    print(f"[Worker {worker_id}] GPU MPC failed at step {step}: {e}")
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
                    mask[i] = 1.0

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
        print(f"[Worker {worker_id}] Error at step {step}: {e}")

    finally:
        # ========== 关键：清理环境 ==========
        try:
            env.close()
        except:
            pass

        try:
            import traci
            traci.close()
        except:
            pass

    # Calculate OCR
    ocr = len(arrived_vehicles) / len(departed_vehicles) if len(departed_vehicles) > 0 else 0.0

    episode_data = {
        'episode_id': episode_id,
        'worker_id': worker_id,
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
    parser = argparse.ArgumentParser(description='Collect GPU MPC demonstrations (parallel with spawn)')
    parser.add_argument('--config', type=str, default='configs/mpc.yaml',
                        help='Path to config YAML')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to collect')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of parallel workers (recommended: 4-8)')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/mpc_parallel',
                        help='Output directory')
    parser.add_argument('--min_ocr', type=float, default=0.5469,
                        help='Minimum OCR for filtering')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"并行GPU MPC演示收集（基于spawn + LibSUMO）")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print(f"Start method: spawn (避免fork问题)")
    print(f"SUMO Mode: LibSUMO (多进程并行，自动端口管理)")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for each worker
    # LibSUMO会自动分配端口，不需要手动指定
    worker_args = []

    current_episode = 0
    episodes_per_worker = args.num_episodes // args.num_workers
    remainder = args.num_episodes % args.num_workers

    for worker_id in range(args.num_workers):
        num_episodes_worker = episodes_per_worker + (1 if worker_id < remainder else 0)

        for episode_in_worker in range(num_episodes_worker):
            episode_id = current_episode
            worker_args.append((
                episode_id,
                config_dict,
                42 + episode_id,  # seed
                device,
                worker_id,
                None  # port=None，让LibSUMO自动管理
            ))
            current_episode += 1

    print(f"Work distribution:")
    for worker_id in range(args.num_workers):
        num_episodes_worker = episodes_per_worker + (1 if worker_id < remainder else 0)
        print(f"  Worker {worker_id}: {num_episodes_worker} episodes (auto-port)")
    print()

    # Collect episodes
    start_time = time.time()

    # Use multiprocessing pool with spawn
    with Pool(processes=args.num_workers) as pool:
        # tqdm progress bar
        results = list(tqdm(
            pool.imap_unordered(collect_single_episode_worker, worker_args),
            total=len(worker_args),
            desc="Collecting episodes"
        ))

    collection_time = time.time() - start_time

    # Sort results by episode_id
    results.sort(key=lambda x: x['episode_id'])

    # Filter and aggregate results
    all_transitions = []
    episode_ocrs = []
    episode_rewards = []

    for episode_data in results:
        # Filter by OCR
        if episode_data['ocr'] >= args.min_ocr:
            all_transitions.extend(episode_data['transitions'])
            episode_ocrs.append(episode_data['ocr'])
            episode_rewards.append(episode_data['reward'])

            print(f"Episode {episode_data['episode_id']} (Worker {episode_data['worker_id']}): "
                  f"OCR={episode_data['ocr']:.4f} ✓")
        else:
            print(f"Episode {episode_data['episode_id']} (Worker {episode_data['worker_id']}): "
                  f"OCR={episode_data['ocr']:.4f} ✗ (below {args.min_ocr})")

    # Save
    output_file = output_dir / 'demonstrations.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(all_transitions, f)

    # Save metadata
    metadata = {
        'num_episodes': args.num_episodes,
        'num_workers': args.num_workers,
        'num_filtered': len(episode_ocrs),
        'ocr_mean': np.mean(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_std': np.std(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_min': np.min(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'ocr_max': np.max(episode_ocrs) if len(episode_ocrs) > 0 else 0.0,
        'reward_mean': np.mean(episode_rewards) if len(episode_rewards) > 0 else 0.0,
        'reward_std': np.std(episode_rewards) if len(episode_rewards) > 0 else 0.0,
        'total_transitions': len(all_transitions),
        'collection_time_minutes': collection_time / 60.0,
        'min_ocr_threshold': args.min_ocr,
        'device': device,
        'start_method': 'spawn',
        'mode': 'LibSUMO parallel (auto-port management)'
    }

    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Collection Statistics")
    print(f"{'='*70}")
    print(f"Total episodes: {args.num_episodes}")
    print(f"Workers: {args.num_workers}")
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
    print(f"Speedup: ~{args.num_workers}x (vs single worker)")
    print(f"{'='*70}\n")

    print(f"✓ Demonstrations saved to {output_file}")
    print(f"✓ Metadata saved to {metadata_file}")

    if len(episode_ocrs) == 0:
        print(f"\n⚠️  WARNING: No episodes passed OCR filter!")
        print(f"   Consider checking MPC configuration or lowering min_ocr threshold")


if __name__ == '__main__':
    main()
