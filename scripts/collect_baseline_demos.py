#!/usr/bin/env python3
"""
Baseline数据收集 - 无MPC控制

优点：
- 性能稳定（OCR ~54.69%）
- 无MPC bug
- 可作为模仿学习的基础数据
"""

import sys
sys.path.append('/home/wyyyz/TJ_transport_v4')

import argparse
import yaml
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm import tqdm

from src.env.competition_env import CompetitionSumoEnv


def collect_episode(args):
    """Collect one baseline episode"""
    episode_id, config_dict, seed, worker_id = args

    np.random.seed(seed)

    env_config = config_dict['environment'].copy()
    env_config['max_steps'] = 3600

    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device='cpu'
    )

    obs = env.reset()
    transitions = []

    total_reward = 0.0
    for step in range(3600):
        # Flatten observation
        vehicle_ids = obs.get('vehicle_ids', [])
        obs_flat = np.zeros(32 * 2, dtype=np.float32)

        for i, veh_id in enumerate(vehicle_ids[:32]):
            veh_state = obs.get('vehicle_states', {}).get(veh_id, {})
            obs_flat[i * 2] = veh_state.get('speed', 0.0)
            obs_flat[i * 2 + 1] = veh_state.get('acceleration', 0.0)

        # Dummy actions (no control)
        actions_flat = np.zeros(32 * 2, dtype=np.float32)
        mask = np.zeros(32, dtype=np.float32)

        transition = {
            'obs': obs_flat,
            'actions': actions_flat,
            'mask': mask,
            'vehicle_ids': vehicle_ids,
            'icv_ids': list(obs.get('icv_ids', set())),
            'selected_vehicles': []
        }
        transitions.append(transition)

        # No control actions
        next_obs, reward, done, info = env.step({})
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
        'performance': {'avg_mpc_time_ms': 0, 'avg_scorer_time_ms': 0}
    }


def main():
    parser = argparse.ArgumentParser(description='Collect baseline demonstrations (no MPC control)')
    parser.add_argument('--config', type=str, default='configs/mpc.yaml', help='Config path')
    parser.add_argument('--num_episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations/baseline')
    parser.add_argument('--min_ocr', type=float, default=0.50, help='Minimum OCR')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Prepare arguments
    episode_args = [
        (i, config, 42 + i, i % args.num_workers)
        for i in range(args.num_episodes)
    ]

    # Collect
    print(f"Collecting {args.num_episodes} baseline episodes with {args.num_workers} workers...")

    all_episodes = []
    ocr_list = []

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(collect_episode, episode_args),
            total=args.num_episodes,
            desc="Episodes"
        ))

    for result in results:
        if result['ocr'] >= args.min_ocr:
            all_episodes.append(result)
            ocr_list.append(result['ocr'])

    # Save
    import pickle
    with open(output_dir / 'demonstrations.pkl', 'wb') as f:
        pickle.dump(all_episodes, f)

    metadata = {
        'num_episodes': len(all_episodes),
        'ocr_mean': np.mean(ocr_list) if len(ocr_list) > 0 else 0.0,
        'ocr_std': np.std(ocr_list) if len(ocr_list) > 0 else 0.0,
        'min_ocr_threshold': args.min_ocr,
        'baseline_mode': True
    }

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Saved {len(all_episodes)} episodes to {output_dir}")
    print(f"  OCR mean: {metadata['ocr_mean']:.4f}")
    print(f"  OCR std: {metadata['ocr_std']:.4f}")


if __name__ == '__main__':
    main()
