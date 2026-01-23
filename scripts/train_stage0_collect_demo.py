#!/usr/bin/env python3
"""
Stage 0: 收集规则基线演示数据

功能：
1. 使用RuleBasedVehicleScorer选择Top-K车辆
2. 使用IDM模型生成演示动作
3. 收集完整episode数据用于Stage 1模仿学习

使用方法：
    python scripts/train_stage0_collect_demo.py --config configs/v5_complete.yaml --num_episodes 50

输出：
    - data/demonstrations/demonstrations.pkl
    - data/demonstrations/info.json
"""

import os
import sys
import argparse
import yaml
import json
import pickle
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.competition_env import CompetitionSumoEnv
from src.env.rule_based_scorer import RuleBasedVehicleScorer


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    将观测字典转换为扁平化张量

    Args:
        obs_dict: 环境返回的观测字典
        max_vehicles: 最大车辆数（用于padding）

    Returns:
        flattened_obs: [obs_dim] 扁平化观测
            obs_dim = max_vehicles * 9 + 32 + 1
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
    vehicle_features_flat = vehicle_features.flatten()  # [max_vehicles * 9]
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs


def collect_idm_action(env, veh_id: str, vehicle_states: dict) -> np.ndarray:
    """
    使用IDM模型生成单辆车的动作

    Args:
        env: CompetitionSumoEnv
        veh_id: 车辆ID
        vehicle_states: 车辆状态字典

    Returns:
        action: [acceleration, lane_change]
    """
    if veh_id not in vehicle_states:
        return np.array([0.0, 0.0])

    state = vehicle_states[veh_id]

    # IDM参数
    desired_speed = 30.0  # m/s
    min_gap = 2.0  # 最小车间距
    time_headway = 1.5  # 车头时距
    accel = 0.73  # 最大加速度
    decel = 1.67  # 最大减速度

    # 获取当前速度和前车信息
    current_speed = state.get('speed', 15.0)

    # 简化版IDM：计算期望加速度
    delta_v = 0.0  # 与前车的速度差
    gap = 100.0  # 与前车的距离（默认值）

    try:
        # 尝试获取前车信息
        leader_info = env.traci_lib.vehicle.getLeader(veh_id, 100.0)
        if leader_info is not None:
            leader_id, leader_distance = leader_info
            leader_speed = env.traci_lib.vehicle.getSpeed(leader_id)
            delta_v = current_speed - leader_speed
            gap = leader_distance - env.traci_lib.vehicle.getLength(veh_id)
    except:
        gap = 100.0
        delta_v = 0.0

    # IDM加速度公式
    desired_gap = min_gap + current_speed * time_headway + \
                  (current_speed * delta_v) / (2 * np.sqrt(accel * decel))

    idm_accel = accel * (1 - (current_speed / desired_speed)**4 -
                         (desired_gap / (gap + 1e-6))**2)

    # 限制加速度范围
    idm_accel = np.clip(idm_accel, -decel, accel)

    # 换道决策（简化：随机换道，概率较低）
    lane_change = np.random.choice([0.0, 1.0], p=[0.95, 0.05])

    return np.array([idm_accel, lane_change])


def select_top_k_vehicles(scorer, vehicle_states: dict, context: dict, k: int = 25) -> list:
    """
    选择Top-K车辆

    Args:
        scorer: RuleBasedVehicleScorer
        vehicle_states: 车辆状态
        context: 环境上下文
        k: 选择数量

    Returns:
        selected_vehicle_ids: 选中的车辆ID列表
    """
    # 计算评分
    scores = scorer.compute_scores(vehicle_states, context)

    # 过滤ICV
    icv_ids = context.get('icv_ids', set())
    icv_scores = {veh_id: scores.get(veh_id, 0.0) for veh_id in icv_ids}

    # 选择Top-K
    sorted_vehicles = sorted(icv_scores.items(), key=lambda x: x[1], reverse=True)
    selected_vehicle_ids = [veh_id for veh_id, score in sorted_vehicles[:k]]

    return selected_vehicle_ids


def collect_demonstrations(
    env,
    scorer,
    num_episodes: int = 50,
    max_steps: int = 3600,
    k_vehicles: int = 25,
    output_dir: str = 'data/demonstrations'
) -> dict:
    """
    收集演示数据

    Args:
        env: CompetitionSumoEnv
        scorer: RuleBasedVehicleScorer
        num_episodes: 收集episode数量
        max_steps: 每个episode最大步数
        k_vehicles: 选择的车辆数量
        output_dir: 输出目录

    Returns:
        demonstrations: {
            'episodes': [...],
            'metadata': {...}
        }
    """
    demonstrations = {
        'episodes': [],
        'metadata': {
            'num_episodes': 0,
            'collection_time': datetime.now().isoformat(),
            'k_vehicles': k_vehicles,
            'max_steps': max_steps
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    print(f"开始收集演示数据...")
    print(f"  Episodes: {num_episodes}")
    print(f"  每episode最大步数: {max_steps}")
    print(f"  选择车辆数: {k_vehicles}")

    for episode_idx in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs_dict = env.reset()
        episode_data = {
            'episode_id': episode_idx,
            'transitions': [],
            'total_reward': 0.0,
            'episode_length': 0
        }

        for step in range(max_steps):
            # 解析观测
            vehicle_states = obs_dict.get('vehicle_states', {})
            vehicle_ids = obs_dict.get('vehicle_ids', [])
            icv_ids = obs_dict.get('icv_ids', set())

            # 构建context
            context = {
                'traci_lib': env.traci_lib,
                'all_vehicle_ids': vehicle_ids,
                'icv_ids': icv_ids
            }

            # 选择Top-K车辆
            if len(icv_ids) > 0:
                selected_vehicles = select_top_k_vehicles(
                    scorer, vehicle_states, context, k=k_vehicles
                )
            else:
                selected_vehicles = []

            # 生成动作
            actions_dict = {}
            for veh_id in selected_vehicles:
                action = collect_idm_action(env, veh_id, vehicle_states)
                actions_dict[veh_id] = action

            # 执行动作
            next_obs_dict, reward, done, info = env.step(actions_dict)

            # 存储转换
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

            # 更新观测
            obs_dict = next_obs_dict

            if done:
                break

        episode_data['episode_length'] = len(episode_data['transitions'])
        demonstrations['episodes'].append(episode_data)

        print(f"\nEpisode {episode_idx + 1}/{num_episodes}: "
              f"length={episode_data['episode_length']}, "
              f"reward={episode_data['total_reward']:.2f}")

    demonstrations['metadata']['num_episodes'] = num_episodes

    # 计算统计信息
    all_lengths = [ep['episode_length'] for ep in demonstrations['episodes']]
    all_rewards = [ep['total_reward'] for ep in demonstrations['episodes']]

    demonstrations['metadata']['avg_length'] = np.mean(all_lengths)
    demonstrations['metadata']['std_length'] = np.std(all_lengths)
    demonstrations['metadata']['avg_reward'] = np.mean(all_rewards)
    demonstrations['metadata']['std_reward'] = np.std(all_rewards)

    print(f"\n演示数据收集完成！")
    print(f"  平均episode长度: {demonstrations['metadata']['avg_length']:.1f} ± "
          f"{demonstrations['metadata']['std_length']:.1f}")
    print(f"  平均episode奖励: {demonstrations['metadata']['avg_reward']:.2f} ± "
          f"{demonstrations['metadata']['std_reward']:.2f}")

    return demonstrations


def main():
    parser = argparse.ArgumentParser(description='Stage 0: 收集演示数据')
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='收集episode数量')
    parser.add_argument('--k_vehicles', type=int, default=25,
                        help='选择的车辆数量')
    parser.add_argument('--output_dir', type=str, default='data/demonstrations',
                        help='输出目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 创建环境
    print("创建环境...")
    env_config = config.get('environment', {})

    # 设置默认值
    if 'sumocfg_file' not in env_config:
        env_config['sumocfg_file'] = '仿真环境_初赛_1.0/仿真环境-初赛/sumo_train.sumocfg'
    if 'max_steps' not in env_config:
        env_config['max_steps'] = 3600
    if 'icv_ratio' not in env_config:
        env_config['icv_ratio'] = 0.10

    # CompetitionSumoEnv接受完整的config字典
    env = CompetitionSumoEnv(
        config=env_config,
        use_gui=False,
        device='cpu'  # Stage 0使用CPU即可
    )

    # 创建规则评分器
    print("创建规则评分器...")
    scorer = RuleBasedVehicleScorer(config=config)

    # 收集演示数据
    demonstrations = collect_demonstrations(
        env=env,
        scorer=scorer,
        num_episodes=args.num_episodes,
        max_steps=env_config.get('max_steps', 3600),
        k_vehicles=args.k_vehicles,
        output_dir=args.output_dir
    )

    # 保存演示数据
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, 'demonstrations.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(demonstrations, f)

    # 保存元信息
    info_file = os.path.join(args.output_dir, 'info.json')
    info = {
        'config_file': args.config,
        'num_episodes': args.num_episodes,
        'k_vehicles': args.k_vehicles,
        'seed': args.seed,
        'output_file': output_file,
        'metadata': demonstrations['metadata']
    }
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n演示数据已保存到: {output_file}")
    print(f"元信息已保存到: {info_file}")


if __name__ == '__main__':
    main()
