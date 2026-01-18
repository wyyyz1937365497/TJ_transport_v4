"""
初赛模型评估脚本

评估训练好的模型在初赛场景下的性能。

使用方法：
    # 评估Level 5模型
    python evaluate_preliminary.py --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip

    # 评估并显示详细信息
    python evaluate_preliminary.py --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip --verbose

    # 自定义评估episodes数
    python evaluate_preliminary.py --checkpoint checkpoints/competition/preliminary/level5/custom_ppo.zip --episodes 50

初赛核心指标：
    - 平均速度（核心）
    - 吞吐量（核心）
    - 干预率（次要）
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.ideal_policy_v4 import create_ideal_traffic_policy_v4
from src.env.competition_env import CompetitionSumoEnv


def evaluate_model(
    checkpoint_path: str,
    config_path: str = 'configs/competition_preliminary.yaml',
    num_episodes: int = 20,
    verbose: bool = True,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    评估模型性能

    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
        num_episodes: 评估episodes数
        verbose: 是否显示详细信息
        device: 计算设备

    Returns:
        评估结果字典
    """
    print("\n" + "="*80)
    print("初赛模型评估")
    print("="*80)
    print(f"模型: {checkpoint_path}")
    print(f"Episodes: {num_episodes}")
    print("="*80 + "\n")

    # 加载配置
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置设备
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"[DEVICE] 使用设备: {device}\n")

    # 创建环境（使用初赛官方场景参数）
    env_config = config['environment'].copy()
    env_config.update({
        'max_vehicles': 32,
        'inflow_rate': 2000,
        'icv_ratio': 0.25,  # 25%智能车
        'disturbance_level': 0.5,
    })

    print("[ENV] 创建评估环境...")
    print(f"  - max_vehicles: {env_config['max_vehicles']}")
    print(f"  - inflow_rate: {env_config['inflow_rate']}")
    print(f"  - icv_ratio: {env_config['icv_ratio']*100}%")
    print(f"  - disturbance_level: {env_config['disturbance_level']}\n")

    env = CompetitionSumoEnv(config=env_config)

    # 创建模型
    print("[MODEL] 加载模型...")
    policy_cls = create_ideal_traffic_policy_v4(config)

    # 加载检查点
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 检查点不存在: {checkpoint_path}")
        return None

    print(f"[CHECKPOINT] 加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        # CustomPPOTrainer保存的格式
        policy = policy_cls.to(device)
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Stable-Baselines3格式（如果使用）
        print("[WARNING] 未找到model_state_dict，尝试其他格式...")
        policy.load(checkpoint_path)

    policy.eval()
    print("[OK] 模型加载完成\n")

    # 评估循环
    print("[EVAL] 开始评估...\n")

    all_rewards = []
    all_lengths = []
    all_avg_speeds = []
    all_throughputs = []
    all_intervention_rates = []

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=config['seed'] + episode)
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (done or truncated):
            # 预处理观测
            observation = env._format_observation(obs)

            # 模型预测
            with torch.no_grad():
                action_dict = policy(observation, deterministic=True)

            # 提取动作
            if isinstance(action_dict, dict):
                actions = action_dict.get('actions', action_dict)
            else:
                actions = action_dict

            # 环境步进
            obs, reward, done, truncated, info = env.step(actions)
            episode_reward += reward
            episode_length += 1

        # 记录指标
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)

        # 从info中提取指标
        if 'avg_speed' in info:
            all_avg_speeds.append(info['avg_speed'])
        if 'throughput' in info:
            all_throughputs.append(info['throughput'])
        if 'intervention_rate' in info:
            all_intervention_rates.append(info['intervention_rate'])

        if verbose and (episode + 1) % 5 == 0:
            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"Reward={episode_reward:.1f}, "
                  f"Length={episode_length}, "
                  f"Speed={info.get('avg_speed', 0):.2f} m/s")

    env.close()

    # 计算统计数据
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_length': np.mean(all_lengths),
        'mean_avg_speed': np.mean(all_avg_speeds) if all_avg_speeds else 0,
        'mean_throughput': np.mean(all_throughputs) if all_throughputs else 0,
        'mean_intervention_rate': np.mean(all_intervention_rates) if all_intervention_rates else 0,
        'all_rewards': all_rewards,
        'all_lengths': all_lengths,
    }

    # 打印结果
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"Episodes数: {results['num_episodes']}")
    print()
    print("⭐ 初赛核心指标：")
    print(f"  平均速度:     {results['mean_avg_speed']:.3f} ± {np.std(all_avg_speeds) if all_avg_speeds else 0:.3f} m/s")
    print(f"  吞吐量:       {results['mean_throughput']:.3f} ± {np.std(all_throughputs) if all_throughputs else 0:.3f} veh/step")
    print()
    print("次要指标：")
    print(f"  平均奖励:     {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  平均长度:     {results['mean_length']:.1f} steps")
    print(f"  干预率:       {results['mean_intervention_rate']*100:.2f}%")
    print("="*80 + "\n")

    # 性能评估
    print("[PERFORMANCE] 性能评估:")
    if results['mean_avg_speed'] > 15:
        print("  ⭐⭐⭐ 优秀！平均速度 > 15 m/s")
    elif results['mean_avg_speed'] > 12:
        print("  ⭐⭐ 良好！平均速度 > 12 m/s")
    elif results['mean_avg_speed'] > 10:
        print("  ⭐ 一般，平均速度 > 10 m/s")
    else:
        print("  ⚠️  需要改进，平均速度 < 10 m/s")

    if results['mean_intervention_rate'] < 0.3:
        print("  ✅ 干预率合理 (< 30%)")
    elif results['mean_intervention_rate'] < 0.5:
        print("  ⚠️  干预率偏高 (30-50%)")
    else:
        print("  ❌ 干预率过高 (> 50%)，可能影响评分")

    print()

    return results


def main():
    parser = argparse.ArgumentParser(description='初赛模型评估脚本')

    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/competition/preliminary/level5/custom_ppo.zip',
        help='模型检查点路径'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/competition_preliminary.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=20,
        help='评估episodes数'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细进度信息'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备（cuda:0, cpu等）'
    )

    args = parser.parse_args()

    # 设置设备
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 评估模型
    results = evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.episodes,
        verbose=args.verbose,
        device=device
    )

    if results is None:
        print("[ERROR] 评估失败")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
