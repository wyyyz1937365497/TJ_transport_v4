#!/usr/bin/env python3
"""
OCR-MAX架构评估脚本

功能：
1. 评估SimplifiedICVPolicy模型的性能
2. 计算OCR（OD完成率）- 比赛核心指标
3. 分析Mean Return、效率、稳定性
4. 生成详细评估报告
5. 可选：与基线模型对比

使用方法：
    # 基础评估
    python scripts/evaluate_ocr_max.py \
        --config configs/ocr_max.yaml \
        --checkpoint checkpoints/ocr_max/stage2_best.pth \
        --num_eval_episodes 20 \
        --device cuda

    # 与基线对比
    python scripts/evaluate_ocr_max.py \
        --config configs/ocr_max.yaml \
        --checkpoint checkpoints/ocr_max/stage2_best.pth \
        --baseline_checkpoint checkpoints/ocr_max/stage1_best.pth \
        --num_eval_episodes 20 \
        --device cuda

输出：
    - logs/ocr_max/evaluation/results.json
    - logs/ocr_max/evaluation/summary.txt
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.simplified_icv_policy import SimplifiedICVPolicy
from src.env.competition_env import CompetitionSumoEnv


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    将观测字典转换为扁平化张量

    Args:
        obs_dict: 环境返回的观测字典
        max_vehicles: 最大车辆数（用于padding）

    Returns:
        flattened_obs: [obs_dim] 扁平化观测
            obs_dim = max_vehicles * 9 + 32 + 1 = 321
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])
    icv_ids = obs_dict.get('icv_ids', set())
    global_stats = obs_dict.get('global_stats', np.zeros(32))

    num_vehicles = len(vehicle_ids)

    # 提取9维车辆特征（归一化）
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

    # 确保global_stats是32维
    global_stats_flat = global_stats.flatten()
    if len(global_stats_flat) < 32:
        global_stats_flat = np.concatenate([
            global_stats_flat,
            np.zeros(32 - len(global_stats_flat), dtype=np.float32)
        ])
    elif len(global_stats_flat) > 32:
        global_stats_flat = global_stats_flat[:32]

    # 扁平化并拼接
    vehicle_features_flat = vehicle_features.flatten()
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs


def convert_actions_to_dict(action_array: np.ndarray, vehicle_ids: list, icv_ids: set) -> dict:
    """
    将扁平动作数组转换为字典格式（只控制ICV）

    Args:
        action_array: [max_vehicles * 2] 扁平动作数组
        vehicle_ids: 车辆ID列表
        icv_ids: ICV ID集合

    Returns:
        actions_dict: {vehicle_id: [acceleration, lane_change]}
    """
    actions_dict = {}

    # 重塑为 [max_vehicles, 2]
    actions_reshaped = action_array.reshape(32, 2)

    for i, veh_id in enumerate(vehicle_ids):
        if veh_id in icv_ids and i < 32:
            accel = actions_reshaped[i, 0].item()
            lane_change = actions_reshaped[i, 1].item()

            # 反归一化加速度
            accel = accel * 3.0  # [-3, 3] m/s²

            actions_dict[veh_id] = np.array([accel, lane_change])

    return actions_dict


def evaluate_model(
    policy: SimplifiedICVPolicy,
    env: CompetitionSumoEnv,
    num_episodes: int = 20,
    max_steps: int = 3600,
    device: str = 'cuda',
    desc: str = "Evaluating"
) -> Dict:
    """
    评估模型性能

    Args:
        policy: SimplifiedICVPolicy
        env: CompetitionSumoEnv
        num_episodes: 评估episode数
        max_steps: 每个episode最大步数
        device: 设备
        desc: 进度条描述

    Returns:
        results: 评估结果字典
    """
    policy.eval()

    episode_returns = []
    episode_lengths = []
    episode_ocrs = []  # OCR指标

    all_speeds = []
    all_accelerations = []

    print(f"\n{'='*80}")
    print(f"评估模型: {desc}")
    print(f"{'='*80}\n")

    for episode_idx in tqdm(range(num_episodes), desc=desc):
        obs_dict = env.reset()
        episode_reward = 0.0
        episode_length = 0

        # 记录初始车辆
        initial_vehicles = set(obs_dict.get('vehicle_ids', []))

        for step in range(max_steps):
            # 扁平化观测
            obs_flat = flatten_observation(obs_dict)
            obs_tensor = torch.from_numpy(obs_flat).unsqueeze(0).float()

            # 确保输入在正确的设备上
            obs_tensor = obs_tensor.to(device)

            # 策略推理（确定性）
            with torch.no_grad():
                outputs = policy(obs_tensor, deterministic=True)
                action = outputs['actions'][0].cpu().numpy()

            # 转换为动作字典
            vehicle_ids = obs_dict.get('vehicle_ids', [])
            icv_ids = obs_dict.get('icv_ids', set())
            actions_dict = convert_actions_to_dict(action, vehicle_ids, icv_ids)

            # 记录速度和加速度
            for veh_id in vehicle_ids:
                if veh_id in obs_dict.get('vehicle_states', {}):
                    state = obs_dict['vehicle_states'][veh_id]
                    all_speeds.append(state.get('speed', 0.0))
                    all_accelerations.append(state.get('acceleration', 0.0))

            # 执行动作
            next_obs_dict, reward, done, info = env.step(actions_dict)

            episode_reward += reward
            episode_length += 1

            obs_dict = next_obs_dict

            if done:
                break

        # 计算OCR
        final_vehicles = set(info.get('arrived_vehicles', set()))
        ocr = len(final_vehicles) / max(len(initial_vehicles), 1)

        episode_returns.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_ocrs.append(ocr)

        print(f"Episode {episode_idx + 1}/{num_episodes}: "
              f"Return={episode_reward:.2f}, "
              f"OCR={ocr:.4f}, "
              f"Length={episode_length}")

    # 统计
    results = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_ocr': np.mean(episode_ocrs),
        'std_ocr': np.std(episode_ocrs),
        'mean_speed': np.mean(all_speeds),
        'std_speed': np.std(all_speeds),
        'mean_acceleration': np.mean(all_accelerations),
        'std_acceleration': np.std(all_accelerations),
        'episode_returns': episode_returns,
        'episode_ocrs': episode_ocrs,
        'episode_lengths': episode_lengths
    }

    return results


def print_results(results: Dict, model_name: str = "Model"):
    """打印评估结果"""
    print(f"\n{'='*80}")
    print(f"{model_name} 评估结果")
    print(f"{'='*80}\n")

    print(f"📊 核心指标:")
    print(f"  Mean Return:  {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean OCR:     {results['mean_ocr']:.4f} ± {results['std_ocr']:.4f} ⭐")
    print(f"  Mean Length:  {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    print(f"\n🚗 车辆性能:")
    print(f"  Mean Speed:        {results['mean_speed']:.2f} ± {results['std_speed']:.2f} m/s")
    print(f"  Mean Acceleration: {results['mean_acceleration']:.3f} ± {results['std_acceleration']:.3f} m/s²")

    print(f"\n📈 Episode详情:")
    for i, (ret, ocr) in enumerate(zip(results['episode_returns'], results['episode_ocrs'])):
        print(f"  Episode {i+1:2d}: Return={ret:7.2f}, OCR={ocr:.4f}")

    print(f"\n{'='*80}\n")


def save_results(results: Dict, output_path: str, metadata: Dict):
    """保存评估结果到JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 只保存标量结果，不保存数组
    save_results = {
        'metadata': metadata,
        'metrics': {
            'mean_return': float(results['mean_return']),
            'std_return': float(results['std_return']),
            'mean_length': float(results['mean_length']),
            'std_length': float(results['std_length']),
            'mean_ocr': float(results['mean_ocr']),
            'std_ocr': float(results['std_ocr']),
            'mean_speed': float(results['mean_speed']),
            'std_speed': float(results['std_speed']),
            'mean_acceleration': float(results['mean_acceleration']),
            'std_acceleration': float(results['std_acceleration']),
        },
        'episodes': [
            {
                'episode_id': i,
                'return': float(ret),
                'ocr': float(ocr),
                'length': int(length)
            }
            for i, (ret, ocr, length) in enumerate(
                zip(results['episode_returns'], results['episode_ocrs'], results['episode_lengths'])
            )
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"✅ 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='评估OCR-MAX模型')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                        help='基线模型检查点路径（用于对比）')
    parser.add_argument('--num_eval_episodes', type=int, default=20,
                        help='评估episode数')
    parser.add_argument('--max_steps', type=int, default=3600,
                        help='每个episode最大步数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='logs/ocr_max/evaluation',
                        help='输出目录')

    args = parser.parse_args()

    # 加载配置
    print(f"加载配置: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env_config = config.get('environment', {})
    if 'sumocfg_file' not in env_config:
        env_config['sumocfg_file'] = '仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg'
    if 'max_steps' not in env_config:
        env_config['max_steps'] = 3600
    if 'icv_ratio' not in env_config:
        env_config['icv_ratio'] = 0.10

    # 创建环境
    print("创建环境...")
    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device=args.device
    )

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    policy_config = config.get('policy', {})
    policy = SimplifiedICVPolicy(
        obs_dim=policy_config.get('obs_dim', 321),
        node_dim=policy_config.get('node_dim', 9),
        hidden_dim=policy_config.get('hidden_dim', 128),
        num_layers=policy_config.get('num_layers', 3),
        num_vehicles=policy_config.get('num_vehicles', 32),
        device=args.device,
        use_safety_shield=True  # 评估时启用SafetyShield
    )

    policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"✅ 模型加载成功")

    # 评估主模型
    results = evaluate_model(
        policy=policy,
        env=env,
        num_episodes=args.num_eval_episodes,
        max_steps=args.max_steps,
        device=args.device,
        desc=f"Evaluating ({Path(args.checkpoint).stem})"
    )

    # 打印结果
    model_name = Path(args.checkpoint).stem
    print_results(results, model_name)

    # 保存结果
    metadata = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'num_eval_episodes': args.num_eval_episodes,
        'max_steps': args.max_steps,
        'device': args.device,
        'evaluation_time': datetime.now().isoformat()
    }
    save_results(results, f"{args.output_dir}/{model_name}_results.json", metadata)

    # 对比基线（如果提供）
    if args.baseline_checkpoint is not None:
        print(f"\n{'='*80}")
        print(f"与基线模型对比")
        print(f"{'='*80}\n")

        print(f"加载基线模型: {args.baseline_checkpoint}")
        baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=args.device, weights_only=False)

        baseline_policy = SimplifiedICVPolicy(
            obs_dim=policy_config.get('obs_dim', 321),
            node_dim=policy_config.get('node_dim', 9),
            hidden_dim=policy_config.get('hidden_dim', 128),
            num_layers=policy_config.get('num_layers', 3),
            num_vehicles=policy_config.get('num_vehicles', 32),
            device=args.device,
            use_safety_shield=True
        )
        baseline_policy.load_state_dict(baseline_checkpoint['policy_state_dict'])

        baseline_results = evaluate_model(
            policy=baseline_policy,
            env=env,
            num_episodes=args.num_eval_episodes,
            max_steps=args.max_steps,
            device=args.device,
            desc=f"Evaluating Baseline ({Path(args.baseline_checkpoint).stem})"
        )

        baseline_name = Path(args.baseline_checkpoint).stem
        print_results(baseline_results, baseline_name)

        # 对比分析
        print(f"\n{'='*80}")
        print(f"对比分析")
        print(f"{'='*80}\n")

        print(f"📊 核心指标对比:")
        print(f"  Mean Return:")
        print(f"    {model_name}:  {results['mean_return']:.2f}")
        print(f"    {baseline_name}: {baseline_results['mean_return']:.2f}")
        diff = results['mean_return'] - baseline_results['mean_return']
        print(f"    差异: {diff:+.2f} ({diff/abs(baseline_results['mean_return'])*100:+.1f}%)")

        print(f"\n  Mean OCR ⭐:")
        print(f"    {model_name}:  {results['mean_ocr']:.4f}")
        print(f"    {baseline_name}: {baseline_results['mean_ocr']:.4f}")
        diff_ocr = results['mean_ocr'] - baseline_results['mean_ocr']
        print(f"    差异: {diff_ocr:+.4f} ({diff_ocr/abs(baseline_results['mean_ocr'])*100:+.1f}%)")

        # 保存对比结果
        comparison_results = {
            'metadata': {
                **metadata,
                'baseline_checkpoint': args.baseline_checkpoint,
                'comparison_time': datetime.now().isoformat()
            },
            'model': {
                'name': model_name,
                'checkpoint': args.checkpoint,
                'mean_return': float(results['mean_return']),
                'mean_ocr': float(results['mean_ocr'])
            },
            'baseline': {
                'name': baseline_name,
                'checkpoint': args.baseline_checkpoint,
                'mean_return': float(baseline_results['mean_return']),
                'mean_ocr': float(baseline_results['mean_ocr'])
            },
            'comparison': {
                'return_diff': float(diff),
                'return_diff_pct': float(diff/abs(baseline_results['mean_return'])*100),
                'ocr_diff': float(diff_ocr),
                'ocr_diff_pct': float(diff_ocr/abs(baseline_results['mean_ocr'])*100)
            }
        }

        comparison_path = f"{args.output_dir}/comparison_{model_name}_vs_{baseline_name}.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n✅ 对比结果已保存到: {comparison_path}")

    env.close()

    print(f"\n{'='*80}")
    print(f"评估完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
