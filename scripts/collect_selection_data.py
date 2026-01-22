#!/usr/bin/env python3
"""
收集ICV选择训练数据（并行版本）

使用规则选择器收集专家演示数据，用于预训练ImportancePredictor。

支持多进程并行收集数据，显著加快速度。

输出：
- data/icv_selection/observations.npy: 观测数据 [N, max_vehicles, 9]
- data/icv_selection/selections.npy: 选择标签 [N, max_vehicles] (0或1)
- data/icv_selection/metadata.json: 元数据

使用方法：
    # 单进程（默认）
    python scripts/collect_selection_data.py --config configs/v5_complete.yaml --num_episodes 100

    # 多进程并行（推荐）
    python scripts/collect_selection_data.py --config configs/v5_complete.yaml --num_episodes 100 --num_workers 4
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, Manager
import itertools

import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.competition_env import CompetitionSumoEnv
from src.env.rule_based_scorer import RuleBasedVehicleScorer


def flatten_observation_to_tensor(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    将观测字典转换为车辆特征张量

    Args:
        obs_dict: 观测字典
        max_vehicles: 最大车辆数

    Returns:
        vehicle_features: [max_vehicles, 9] 车辆特征矩阵
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])
    icv_ids = obs_dict.get('icv_ids', set())

    num_vehicles = len(vehicle_ids)
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

    return vehicle_features


def collect_single_episode(args):
    """
    Worker函数：收集单个episode的数据（带增量保存）

    Args:
        args: (worker_id, config_path, max_steps, max_vehicles, k_ratio, seed, output_dir, episode_idx)

    Returns:
        (num_samples, episode_idx)
    """
    worker_id, config_path, max_steps, max_vehicles, k_ratio, seed, output_dir, episode_idx = args

    # 设置随机种子（确保每个worker有不同的种子）
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 创建环境（每个worker独立的SUMO实例）
    env = CompetitionSumoEnv(
        config=config,
        use_gui=False,
        device='cpu'
    )

    # 创建规则选择器
    rule_scorer = RuleBasedVehicleScorer(config=config)

    # 增量保存：每个episode的数据立即保存
    episode_observations = []
    episode_selections = []

    try:
        obs = env.reset()

        # 预热
        if len(obs.get('vehicle_ids', [])) == 0:
            obs, _, done, _ = env.step({})

        for step in range(max_steps):
            # 提取车辆特征
            vehicle_features = flatten_observation_to_tensor(obs, max_vehicles)

            # 使用规则选择器选择ICV
            vehicle_states = obs.get('vehicle_states', {})
            all_vehicle_ids = obs.get('vehicle_ids', [])

            if all_vehicle_ids:
                # 计算规则评分
                context = {
                    'traci_lib': None,
                    'all_vehicle_ids': all_vehicle_ids
                }
                scores = rule_scorer.compute_scores(vehicle_states, context)

                # 选择Top-K
                k = max(5, int(len(all_vehicle_ids) * k_ratio))
                k = min(k, len(all_vehicle_ids))

                sorted_vehicles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected_ids = set([veh_id for veh_id, _ in sorted_vehicles[:k]])

                # 创建选择标签
                selection = np.zeros(max_vehicles, dtype=np.float32)
                for i, veh_id in enumerate(all_vehicle_ids[:max_vehicles]):
                    if veh_id in selected_ids:
                        selection[i] = 1.0
            else:
                selection = np.zeros(max_vehicles, dtype=np.float32)

            # 存储数据
            episode_observations.append(vehicle_features.copy())
            episode_selections.append(selection.copy())

            # 执行随机动作（只为继续仿真）
            action_dict = {}
            next_obs, _, done, _ = env.step(action_dict)

            obs = next_obs

            if done:
                break

    finally:
        # 确保环境被正确关闭
        env.close()

    # 增量保存：立即保存这个episode的数据
    episode_observations = np.array(episode_observations)
    episode_selections = np.array(episode_selections)

    # 保存到临时文件
    temp_file = Path(output_dir) / f'temp_episode_{episode_idx}.npz'
    np.savez_compressed(temp_file, observations=episode_observations, selections=episode_selections)

    return (len(episode_observations), episode_idx)


def collect_selection_data_parallel(
    config_path: str,
    num_episodes: int = 100,
    max_steps: int = 3600,
    max_vehicles: int = 32,
    k_ratio: float = 0.10,
    num_workers: int = 4,
    output_dir: Path = None
):
    """
    并行收集ICV选择数据（带增量保存）

    Args:
        config_path: 配置文件路径
        num_episodes: 收集的episode数
        max_steps: 每个episode的最大步数
        max_vehicles: 最大车辆数
        k_ratio: ICV比例
        num_workers: 并行worker数
        output_dir: 输出目录

    Returns:
        data: Dict包含observations和selections
    """
    print(f"\n开始并行收集数据（增量保存版本）...")
    print(f"  总Episodes: {num_episodes}")
    print(f"  并行Workers: {num_workers}")
    print(f"  每Worker Episodes: {num_episodes // num_workers}")
    print(f"  每Episode步数: {max_steps}")
    print(f"  ICV比例: {k_ratio}")
    print(f"  💾 增量保存模式：每个episode立即保存到磁盘")

    # 准备worker参数
    base_seed = 42
    worker_args = []
    episodes_per_worker = num_episodes // num_workers

    for worker_id in range(num_workers):
        # 每个worker处理一部分episodes
        start_epi = worker_id * episodes_per_worker
        end_epi = start_epi + episodes_per_worker if worker_id < num_workers - 1 else num_episodes

        for epi in range(start_epi, end_epi):
            episode_idx = epi
            seed = base_seed + worker_id * 1000 + epi
            worker_args.append((worker_id, config_path, max_steps, max_vehicles, k_ratio, seed, str(output_dir), episode_idx))

    # 并行收集（增量保存）
    results = []
    total_samples = 0

    with Pool(processes=num_workers) as pool:
        # 使用imap_unordered以便实时显示进度
        for result in tqdm(
            pool.imap_unordered(collect_single_episode, worker_args),
            total=len(worker_args),
            desc="收集数据（并行，增量保存）"
        ):
            results.append(result)
            total_samples += result[0]

    print(f"\n合并数据...")
    # 合并所有临时文件
    all_observations = []
    all_selections = []

    for i in range(num_episodes):
        temp_file = output_dir / f'temp_episode_{i}.npz'
        if temp_file.exists():
            data = np.load(temp_file)
            all_observations.append(data['observations'])
            all_selections.append(data['selections'])
            # 删除临时文件
            temp_file.unlink()

    print(f"  合并完成：{len(all_observations)} episodes")

    return {
        'observations': np.concatenate(all_observations, axis=0),  # [N, max_vehicles, 9]
        'selections': np.concatenate(all_selections, axis=0)        # [N, max_vehicles]
    }


def collect_selection_data_serial(
    env,
    rule_scorer,
    num_episodes: int = 100,
    max_steps: int = 3600,
    max_vehicles: int = 32,
    k_ratio: float = 0.10
):
    """
    串行收集ICV选择数据（单进程版本，保留兼容性）

    Args:
        env: SUMO环境
        rule_scorer: 规则选择器
        num_episodes: 收集的episode数
        max_steps: 每个episode的最大步数
        max_vehicles: 最大车辆数
        k_ratio: ICV比例

    Returns:
        data: Dict包含observations和selections
    """
    observations = []
    selections = []

    print(f"\n开始收集数据（单进程）...")
    print(f"  Episodes: {num_episodes}")
    print(f"  每Episode步数: {max_steps}")
    print(f"  ICV比例: {k_ratio}")

    for episode in tqdm(range(num_episodes), desc="收集数据"):
        obs = env.reset()

        # 预热
        if len(obs.get('vehicle_ids', [])) == 0:
            obs, _, done, _ = env.step({})

        for step in range(max_steps):
            # 提取车辆特征
            vehicle_features = flatten_observation_to_tensor(obs, max_vehicles)

            # 使用规则选择器选择ICV
            vehicle_states = obs.get('vehicle_states', {})
            all_vehicle_ids = obs.get('vehicle_ids', [])

            if all_vehicle_ids:
                # 计算规则评分
                context = {
                    'traci_lib': None,
                    'all_vehicle_ids': all_vehicle_ids
                }
                scores = rule_scorer.compute_scores(vehicle_states, context)

                # 选择Top-K
                k = max(5, int(len(all_vehicle_ids) * k_ratio))
                k = min(k, len(all_vehicle_ids))

                sorted_vehicles = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                selected_ids = set([veh_id for veh_id, _ in sorted_vehicles[:k]])

                # 创建选择标签
                selection = np.zeros(max_vehicles, dtype=np.float32)
                for i, veh_id in enumerate(all_vehicle_ids[:max_vehicles]):
                    if veh_id in selected_ids:
                        selection[i] = 1.0
            else:
                selection = np.zeros(max_vehicles, dtype=np.float32)

            # 存储数据
            observations.append(vehicle_features.copy())
            selections.append(selection.copy())

            # 执行随机动作（只为继续仿真）
            action_dict = {}
            next_obs, _, done, _ = env.step(action_dict)

            obs = next_obs

            if done:
                break

    return {
        'observations': np.array(observations),  # [N, max_vehicles, 9]
        'selections': np.array(selections)        # [N, max_vehicles]
    }


def main():
    parser = argparse.ArgumentParser(description="收集ICV选择训练数据（并行版本）")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='收集的episode数')
    parser.add_argument('--output_dir', type=str, default='data/icv_selection',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备（数据收集用CPU即可）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='并行worker数（默认4，设为1则为单进程）')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("ICV选择数据收集（并行版本）")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"Episodes: {args.num_episodes}")
    print(f"并行Workers: {args.num_workers}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集数据参数
    max_vehicles = config['environment']['icv_config']['max_vehicles']
    k_ratio = config['environment']['icv_config']['penetration_rate']
    max_steps = config['environment']['max_steps'] // 10  # 36000步 -> 3600步（缩短收集时间）

    # 根据worker数选择并行或串行
    if args.num_workers > 1:
        # 并行收集（增量保存）
        print(f"\n🚀 使用{args.num_workers}个进程并行收集（增量保存）...")
        data = collect_selection_data_parallel(
            config_path=args.config,
            num_episodes=args.num_episodes,
            max_steps=max_steps,
            max_vehicles=max_vehicles,
            k_ratio=k_ratio,
            num_workers=args.num_workers,
            output_dir=output_dir
        )
    else:
        # 串行收集（单进程）
        print(f"\n📝 使用单进程收集...")

        # 创建环境
        print("创建环境...")
        env = CompetitionSumoEnv(
            config=config,
            use_gui=False,
            device=args.device
        )

        # 创建规则选择器
        print("初始化规则选择器...")
        rule_scorer = RuleBasedVehicleScorer(config=config)

        data = collect_selection_data_serial(
            env=env,
            rule_scorer=rule_scorer,
            num_episodes=args.num_episodes,
            max_steps=max_steps,
            max_vehicles=max_vehicles,
            k_ratio=k_ratio
        )

        env.close()

    # 保存数据
    print(f"\n保存数据...")
    np.save(output_dir / 'observations.npy', data['observations'])
    np.save(output_dir / 'selections.npy', data['selections'])

    # 保存元数据
    metadata = {
        'num_samples': len(data['observations']),
        'max_vehicles': max_vehicles,
        'node_dim': 9,
        'k_ratio': k_ratio,
        'num_episodes': args.num_episodes,
        'collection_date': datetime.now().isoformat(),
        'config_file': args.config
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ 数据收集完成！")
    print(f"  样本数: {metadata['num_samples']}")
    print(f"  观测形状: {data['observations'].shape}")
    print(f"  选择标签形状: {data['selections'].shape}")
    print(f"  正样本比例: {data['selections'].mean():.4f}")
    print(f"\n数据已保存到: {output_dir}")


if __name__ == '__main__':
    main()
