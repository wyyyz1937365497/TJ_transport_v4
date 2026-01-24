#!/usr/bin/env python3
"""
Imitation Learning Evaluation Script

This script evaluates a trained imitation learning model by:
1. Running the policy in the SUMO environment
2. Computing OCR (OD Completion Rate)
3. Comparing with baseline
4. Calculating competition score

Usage:
    # Basic evaluation
    python scripts/evaluate_imitation.py \
        --checkpoint checkpoints/imitation_learning/best.pth \
        --config configs/imitation_learning.yaml \
        --num_episodes 5

    # Evaluation with visualization
    python scripts/evaluate_imitation.py \
        --checkpoint checkpoints/imitation_learning/best.pth \
        --config configs/imitation_learning.yaml \
        --num_episodes 5 \
        --use_gui

Output:
    - OCR statistics
    - Comparison with baseline
    - Competition score
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

import torch

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.competition_env import CompetitionSumoEnv
from src.models.simplified_icv_policy import SimplifiedICVPolicy
from src.env.rule_based_scorer import RuleBasedVehicleScorer


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


def convert_actions_to_dict(
    actions: np.ndarray,
    vehicle_ids: list,
    selected_vehicles: list
) -> dict:
    """
    Convert flattened action array to action dictionary

    Args:
        actions: [64] flattened action array (32 vehicles * 2)
        vehicle_ids: List of vehicle IDs (in observation order)
        selected_vehicles: List of vehicles to control

    Returns:
        actions_dict: {veh_id: [acceleration, lane_change]}
    """
    actions_dict = {}

    # Reshape actions to [32, 2]
    actions_reshaped = actions.reshape(32, 2)

    # ✅ 修复：动作与观测中的车辆位置一一对应
    # 观测中actions[i]对应vehicle_ids[i]
    # 只有selected_vehicles中的车辆才应用控制
    for i, veh_id in enumerate(vehicle_ids[:32]):  # 按vehicle_ids顺序
        if veh_id in selected_vehicles:  # 只有被选中的车辆才应用动作
            action = actions_reshaped[i]
            actions_dict[veh_id] = action

    return actions_dict


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


def evaluate_episode(
    env: CompetitionSumoEnv,
    policy: SimplifiedICVPolicy,
    scorer: RuleBasedVehicleScorer,
    device: str,
    max_steps: int = 3600,
    k_vehicles: int = 25
) -> dict:
    """
    Evaluate a single episode

    Args:
        env: CompetitionSumoEnv instance
        policy: Trained policy network
        scorer: Rule-based vehicle scorer
        device: Device to run inference on
        max_steps: Maximum steps per episode
        k_vehicles: Number of vehicles to control

    Returns:
        episode_stats: Dictionary with episode statistics
    """
    obs_dict = env.reset()

    # ✅ 修复：使用traci直接跟踪车辆，不依赖环境接口
    departed_vehicles = set()
    arrived_vehicles = set()
    previous_vehicles = set()

    total_reward = 0.0
    step = 0

    # Import traci for direct tracking
    try:
        import libsumo as traci_lib
    except ImportError:
        import traci as traci_lib

    for step in range(max_steps):
        # Parse observation
        vehicle_states = obs_dict.get('vehicle_states', {})
        vehicle_ids = obs_dict.get('vehicle_ids', [])
        icv_ids = obs_dict.get('icv_ids', set())

        # ✅ 修复：使用traci直接跟踪departed和arrived车辆
        current_vehicles = set(vehicle_ids)

        # 新departed的车辆 = 当前车辆 - 之前的车辆
        newly_departed = current_vehicles - previous_vehicles
        departed_vehicles.update(newly_departed)

        # 更新vehicle集合
        previous_vehicles = current_vehicles

        # Select top-k vehicles
        if len(icv_ids) > 0:
            context = {
                'traci_lib': None,  # Not needed for scorer
                'all_vehicle_ids': vehicle_ids,
                'icv_ids': icv_ids
            }
            selected_vehicles = select_top_k_vehicles(
                scorer, vehicle_states, context, k=k_vehicles
            )
        else:
            selected_vehicles = []

        # Prepare observation
        obs_flat = flatten_observation(obs_dict)
        obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).float().to(device)

        # Policy inference (deterministic)
        with torch.no_grad():
            outputs = policy(obs_tensor, deterministic=True)
            action = outputs['actions'][0].cpu().numpy()

        # Convert to action dictionary
        actions_dict = convert_actions_to_dict(action, vehicle_ids, selected_vehicles)

        # Execute actions
        next_obs_dict, reward, done, info = env.step(actions_dict)

        # ✅ 修复：使用traci检查arrived车辆
        try:
            arrived = traci_lib.simulation.getArrivedIDList()
            if arrived:
                arrived_vehicles.update(arrived)
        except:
            pass  # 如果traci调用失败，忽略

        # Accumulate reward
        total_reward += reward

        # Next step
        obs_dict = next_obs_dict

        if done:
            break

    # Calculate OCR
    ocr = len(arrived_vehicles) / len(departed_vehicles) if len(departed_vehicles) > 0 else 0.0

    episode_stats = {
        'ocr': ocr,
        'departed_count': len(departed_vehicles),
        'arrived_count': len(arrived_vehicles),
        'total_reward': total_reward,
        'episode_length': step + 1
    }

    return episode_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate imitation learning model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/imitation_learning.yaml',
                        help='Path to config YAML')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='Number of episodes to evaluate')
    parser.add_argument('--max_steps', type=int, default=3600,
                        help='Maximum steps per episode')
    parser.add_argument('--k_vehicles', type=int, default=25,
                        help='Number of vehicles to control')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--use_gui', action='store_true',
                        help='Use SUMO GUI for visualization')
    parser.add_argument('--baseline_ocr', type=float, default=0.5469,
                        help='Baseline OCR for comparison')
    parser.add_argument('--output_dir', type=str, default='competition_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("\n" + "="*70)
    print("Imitation Learning Model Evaluation")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print(f"Baseline OCR: {args.baseline_ocr:.4f}")
    print("="*70)

    # Create environment
    env_config = config.get('environment', {})

    # ✅ 修复：环境期望的键是sumo_config或sumocfg，而不是sumocfg_file
    if 'sumo_config' not in env_config or not env_config['sumo_config']:
        if 'sumocfg' not in env_config or not env_config['sumocfg']:
            env_config['sumo_config'] = '仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg'
        else:
            env_config['sumo_config'] = env_config['sumocfg']

    env_config['max_steps'] = args.max_steps
    env_config['icv_ratio'] = env_config.get('icv_ratio', 0.10)

    # ✅ 修复：禁用神经网络评分器，使用规则评分器（与训练时一致）
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
        use_gui=args.use_gui,
        device='cpu'  # Environment runs on CPU
    )

    # Create policy network
    policy_config = config.get('policy', {})
    policy = SimplifiedICVPolicy(
        obs_dim=policy_config.get('obs_dim', 321),
        node_dim=policy_config.get('node_dim', 9),
        hidden_dim=policy_config.get('hidden_dim', 128),
        num_layers=policy_config.get('num_layers', 3),
        num_vehicles=policy_config.get('num_vehicles', 32),
        device=args.device,
        use_safety_shield=False
    )

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    print("✓ Checkpoint loaded successfully")

    if 'epoch' in checkpoint:
        print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    if 'best_val_loss' in checkpoint:
        print(f"  Best validation loss: {checkpoint['best_val_loss']:.6f}")

    # Create scorer for vehicle selection
    scorer = RuleBasedVehicleScorer(config=config)

    # Evaluate
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    print("-"*70)

    episode_results = []
    ocr_list = []

    start_time = time.time()

    for episode_idx in tqdm(range(args.num_episodes), desc="Evaluating"):
        try:
            episode_stats = evaluate_episode(
                env=env,
                policy=policy,
                scorer=scorer,
                device=args.device,
                max_steps=args.max_steps,
                k_vehicles=args.k_vehicles
            )

            episode_results.append(episode_stats)
            ocr_list.append(episode_stats['ocr'])

            print(f"\nEpisode {episode_idx + 1}:")
            print(f"  OCR: {episode_stats['ocr']:.4f} ({episode_stats['arrived_count']}/{episode_stats['departed_count']})")
            print(f"  Reward: {episode_stats['total_reward']:.2f}")
            print(f"  Length: {episode_stats['episode_length']} steps")

        except Exception as e:
            print(f"\n[Error] Episode {episode_idx + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    eval_time = time.time() - start_time

    # Compute statistics
    ocr_mean = np.mean(ocr_list) if ocr_list else 0.0
    ocr_std = np.std(ocr_list) if ocr_list else 0.0
    ocr_min = np.min(ocr_list) if ocr_list else 0.0
    ocr_max = np.max(ocr_list) if ocr_list else 0.0

    # Competition score
    competition_score = max(0, 100 * (ocr_mean - args.baseline_ocr) / args.baseline_ocr)

    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Total episodes: {len(episode_results)}")
    print(f"Evaluation time: {eval_time/60:.1f} min")
    print(f"\nOCR Statistics:")
    print(f"  Mean: {ocr_mean:.4f} ± {ocr_std:.4f}")
    print(f"  Min:  {ocr_min:.4f}")
    print(f"  Max:  {ocr_max:.4f}")
    print(f"\nBaseline Comparison:")
    print(f"  Baseline OCR: {args.baseline_ocr:.4f}")
    print(f"  Model OCR:    {ocr_mean:.4f}")
    print(f"  Improvement:  {(ocr_mean - args.baseline_ocr)*100:+.2f}%")
    print(f"\nCompetition Score:")
    print(f"  Score: {competition_score:.2f}")
    print("="*70)

    # Determine success
    if ocr_mean > args.baseline_ocr:
        print("\n✅ SUCCESS: Model OCR above baseline!")
        if ocr_mean >= 0.55:
            print("🎉 EXCELLENT: OCR >= 55% (初赛通过标准)")
    else:
        print("\n⚠️  WARNING: Model OCR below baseline")
        print("   Consider retraining with more data or different hyperparameters")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'checkpoint': str(args.checkpoint),
        'config': str(args.config),
        'num_episodes': len(episode_results),
        'evaluation_time_minutes': eval_time / 60,
        'ocr_mean': float(ocr_mean),
        'ocr_std': float(ocr_std),
        'ocr_min': float(ocr_min),
        'ocr_max': float(ocr_max),
        'baseline_ocr': float(args.baseline_ocr),
        'improvement_pct': float((ocr_mean - args.baseline_ocr) * 100),
        'competition_score': float(competition_score),
        'episodes': episode_results
    }

    results_path = output_dir / 'imitation_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")

    # Close environment
    env.close()


if __name__ == '__main__':
    main()
