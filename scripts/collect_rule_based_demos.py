#!/usr/bin/env python3
"""
使用简单规则控制器收集演示数据

比MPC更简单、更可靠、性能更好
"""

import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import argparse
import yaml
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import time

from src.env.competition_env import CompetitionSumoEnv
from src.env.vehicle_scorer_gpu import GPUVehicleScorer
sys.path.append('/home/wyyyz/TJ_transport_v4/src')
from controllers.simple_rule_controller import AdaptiveSpeedController


def flatten_observation(obs_dict: Dict) -> np.ndarray:
    """将观测字典展平为向量"""
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])

    obs_flat = np.zeros(32 * 2, dtype=np.float32)

    for i, veh_id in enumerate(vehicle_ids[:32]):
        if veh_id in vehicle_states:
            state = vehicle_states[veh_id]
            obs_flat[i * 2] = state.get('speed', 0.0)
            obs_flat[i * 2 + 1] = state.get('acceleration', 0.0)

    return obs_flat


def collect_episode(args):
    """Collect one episode with rule-based controller"""
    episode_id, config_dict, seed, worker_id = args

    np.random.seed(seed)

    env_config = config_dict['environment'].copy()
    env_config['max_steps'] = 3600

    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device='cpu'
    )

    # Create rule-based controller
    controller_config = {
        'bottleneck_s_min': 1200.0,
        'bottleneck_s_max': 2200.0
    }
    controller = AdaptiveSpeedController(controller_config)

    obs = env.reset()
    transitions = []

    total_reward = 0.0
    controller_times = []

    for step in range(3600):
        vehicle_ids = obs.get('vehicle_ids', [])
        icv_ids = list(obs.get('icv_ids', set()))

        # Rule-based control
        start_time = time.time()
        actions_dict = controller.compute_actions(obs, vehicle_ids, icv_ids)
        controller_times.append(time.time() - start_time)

        # Store transition
        obs_flat = flatten_observation(obs)

        actions_flat = np.zeros(32 * 2, dtype=np.float32)
        mask = np.zeros(32, dtype=np.float32)

        for i, veh_id in enumerate(vehicle_ids[:32]):
            if veh_id in actions_dict:
                action = actions_dict[veh_id]
                actions_flat[i * 2] = float(action[0])
                actions_flat[i * 2 + 1] = float(action[1])
                mask[i] = 1.0

        transition = {
            'obs': obs_flat,
            'actions': actions_flat,
            'mask': mask,
            'vehicle_ids': vehicle_ids,
            'icv_ids': icv_ids,
            'selected_vehicles': icv_ids[:25]  # 最多25辆
        }
        transitions.append(transition)

        # Execute actions
        next_obs, reward, done, info = env.step(actions_dict)
        total_reward += reward

        obs = next_obs

        if done:
            break

    env.close()

    ocr = info.get('arrived_count', 0) / info.get('departed_count', 1) if info.get('departed_count', 0) > 0 else 0.0

    return {
        'episode_id': episode_id,
        'worker_id': worker_id,
        'transitions': transitions,
        'ocr': ocr,
        'reward': total_reward,
        'num_steps': step + 1,
        'departed': info.get('departed_count', 0),
        'arrived': info.get('arrived_count', 0),
        'performance': {
            'avg_controller_time_ms': np.mean(controller_times) * 1000 if len(controller_times) > 0 else 0
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Collect rule-based demonstrations')
    parser.add_argument('--config', type=str, default='configs/mpc.yaml')
    parser.add_argument('--num_episodes', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/rule_based')
    parser.add_argument('--min_ocr', type=float, default=0.50)

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    multiprocessing.set_start_method('spawn', force=True)

    episode_args = [
        (i, config, 42 + i, i % args.num_workers)
        for i in range(args.num_episodes)
    ]

    print("=" * 80)
    print("规则控制器数据收集")
    print("=" * 80)
    print(f"Episodes: {args.num_episodes}")
    print(f"Workers: {args.num_workers}")
    print(f"Output: {args.output_dir}")
    print(f"Min OCR: {args.min_ocr}")
    print("=" * 80)

    all_episodes = []
    ocr_list = []

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(collect_episode, episode_args),
            total=args.num_episodes,
            desc="Collecting"
        ))

    for result in results:
        print(f"Episode {result['episode_id']} (Worker {result['worker_id']}): "
              f"OCR={result['ocr']:.4f} ({'✓' if result['ocr'] >= args.min_ocr else '✗'})")

        if result['ocr'] >= args.min_ocr:
            all_episodes.append(result)
            ocr_list.append(result['ocr'])

    # Save
    import pickle
    with open(output_dir / 'demonstrations.pkl', 'wb') as f:
        pickle.dump(all_episodes, f)

    metadata = {
        'num_episodes': len(all_episodes),
        'ocr_mean': float(np.mean(ocr_list)) if len(ocr_list) > 0 else 0.0,
        'ocr_std': float(np.std(ocr_list)) if len(ocr_list) > 0 else 0.0,
        'ocr_min': float(np.min(ocr_list)) if len(ocr_list) > 0 else 0.0,
        'ocr_max': float(np.max(ocr_list)) if len(ocr_list) > 0 else 0.0,
        'min_ocr_threshold': args.min_ocr,
        'controller_type': 'adaptive_speed_rule',
        'device': 'cpu'
    }

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("收集完成")
    print("=" * 80)
    print(f"总episodes: {args.num_episodes}")
    print(f"合格episodes: {len(all_episodes)}")
    print(f"OCR均值: {metadata['ocr_mean']:.4f} ({metadata['ocr_mean']*100:.2f}%)")
    print(f"OCR标准差: {metadata['ocr_std']:.4f}")
    print(f"OCR范围: [{metadata['ocr_min']:.4f}, {metadata['ocr_max']:.4f}]")
    print(f"\n✓ 数据已保存到 {output_dir}")


if __name__ == '__main__':
    main()
