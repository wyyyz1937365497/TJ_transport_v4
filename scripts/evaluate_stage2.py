#!/usr/bin/env python3
"""
Stage 2: 引导探索评估脚本（Guided Exploration Evaluation）

功能：
1. 评估PPO训练的策略性能
2. 计算OCR（Output Competition Ratio）- 比赛关键指标
3. 分析效率、稳定性和成本指标
4. 生成可视化报告

使用方法：
    python scripts/evaluate_stage2.py \
        --config configs/v5_complete.yaml \
        --checkpoint checkpoints/v5_complete/stage2_best.pth \
        --num_eval_episodes 20 \
        --visualize \
        --device cuda

输出：
    - logs/v5_complete/eval_stage2/evaluation_results.json
    - logs/v5_complete/eval_stage2/plots/
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.env.competition_env import CompetitionSumoEnv
from src.training.ocr_rewards import OCRRewardComputer


class Stage2Evaluator:
    """
    Stage 2评估器

    评估PPO训练的策略在交通仿真环境中的表现
    """

    def __init__(self, policy, env, config, device='cuda'):
        """
        Args:
            policy: JointICVPolicy
            env: CompetitionSumoEnv
            config: 配置字典
            device: 设备
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.device = device

        # 奖励计算器
        self.reward_computer = OCRRewardComputer()

        # 评估结果存储
        self.episode_results = []

    def evaluate_episode(self, render=False):
        """
        评估单个episode

        Args:
            render: 是否渲染

        Returns:
            episode_metrics: Dict
        """
        obs, _ = self.env.reset()
        done = False
        truncated = False

        episode_data = {
            'steps': 0,
            'total_reward': 0.0,
            'ocr_rewards': [],
            'efficiency_rewards': [],
            'stability_rewards': [],
            'cost_rewards': [],
            'interventions': [],  # 每步控制的车辆数
            'speeds': [],  # 所有车辆的平均速度
            'flows': [],  # 每个车道的流量
            'collisions': 0,
            'emergency_braking': 0,
            'icv_count': 0,
            'vehicle_count': 0
        }

        step_count = 0
        max_steps = 36000  # 1小时仿真

        while not (done or truncated) and step_count < max_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.policy(obs_tensor, deterministic=True)

            action = outputs['actions'][0].cpu().numpy()

            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)

            # 统计
            episode_data['steps'] += 1
            episode_data['total_reward'] += reward

            # 分解奖励
            if 'reward_components' in info:
                rewards = info['reward_components']
                episode_data['ocr_rewards'].append(rewards.get('efficiency', 0) +
                                                   rewards.get('stability', 0) +
                                                   rewards.get('cost', 0))
                episode_data['efficiency_rewards'].append(rewards.get('efficiency', 0))
                episode_data['stability_rewards'].append(rewards.get('stability', 0))
                episode_data['cost_rewards'].append(rewards.get('cost', 0))

            # 统计干预车辆数
            obs_dim = obs.shape[0]
            vehicle_dim = (obs_dim - 32 - 1)
            num_vehicles = vehicle_dim // 9
            actions_reshaped = action.reshape(num_vehicles, 2)
            num_controlled = (np.abs(actions_reshaped[:, 0]) > 0.01).sum()
            episode_data['interventions'].append(num_controlled)

            # 统计速度和流量
            if 'average_speed' in info:
                episode_data['speeds'].append(info['average_speed'])

            if 'lane_flows' in info:
                episode_data['flows'].append(info['lane_flows'])

            # 安全事件
            if 'collision' in info and info['collision']:
                episode_data['collisions'] += 1

            if 'emergency_braking' in info and info['emergency_braking']:
                episode_data['emergency_braking'] += 1

            # ICV和车辆统计
            if 'num_icv' in info:
                episode_data['icv_count'] = max(episode_data['icv_count'], info['num_icv'])

            if 'num_vehicles' in info:
                episode_data['vehicle_count'] = max(episode_data['vehicle_count'], info['num_vehicles'])

            obs = next_obs
            step_count += 1

        # 计算聚合指标
        episode_data['mean_reward'] = episode_data['total_reward'] / episode_data['steps']
        episode_data['mean_ocr'] = np.mean(episode_data['ocr_rewards']) if episode_data['ocr_rewards'] else 0
        episode_data['mean_efficiency'] = np.mean(episode_data['efficiency_rewards']) if episode_data['efficiency_rewards'] else 0
        episode_data['mean_stability'] = np.mean(episode_data['stability_rewards']) if episode_data['stability_rewards'] else 0
        episode_data['mean_cost'] = np.mean(episode_data['cost_rewards']) if episode_data['cost_rewards'] else 0
        episode_data['mean_interventions'] = np.mean(episode_data['interventions']) if episode_data['interventions'] else 0
        episode_data['mean_speed'] = np.mean(episode_data['speeds']) if episode_data['speeds'] else 0
        episode_data['intervention_rate'] = episode_data['mean_interventions'] / episode_data['icv_count'] if episode_data['icv_count'] > 0 else 0

        return episode_data

    def evaluate(self, num_episodes=20):
        """
        评估多个episode

        Args:
            num_episodes: 评估episode数

        Returns:
            results: Dict
        """
        print("\n" + "=" * 80)
        print(f"Stage 2 策略评估（{num_episodes} episodes）")
        print("=" * 80)

        all_results = []

        for episode in tqdm(range(num_episodes), desc="评估episodes"):
            episode_metrics = self.evaluate_episode()
            all_results.append(episode_metrics)

            if (episode + 1) % 5 == 0:
                print(f"\nEpisode {episode + 1}:")
                print(f"  总Reward: {episode_metrics['total_reward']:.2f}")
                print(f"  平均OCR: {episode_metrics['mean_ocr']:.4f}")
                print(f"  平均速度: {episode_metrics['mean_speed']:.2f} m/s")
                print(f"  干预率: {episode_metrics['intervention_rate']:.2%}")
                print(f"  碰撞次数: {episode_metrics['collisions']}")

        # 聚合结果
        results = {
            'num_episodes': num_episodes,
            'total_reward': {
                'mean': np.mean([r['total_reward'] for r in all_results]),
                'std': np.std([r['total_reward'] for r in all_results]),
                'min': np.min([r['total_reward'] for r in all_results]),
                'max': np.max([r['total_reward'] for r in all_results])
            },
            'ocr': {
                'mean': np.mean([r['mean_ocr'] for r in all_results]),
                'std': np.std([r['mean_ocr'] for r in all_results]),
                'min': np.min([r['mean_ocr'] for r in all_results]),
                'max': np.max([r['mean_ocr'] for r in all_results])
            },
            'efficiency': {
                'mean': np.mean([r['mean_efficiency'] for r in all_results]),
                'std': np.std([r['mean_efficiency'] for r in all_results])
            },
            'stability': {
                'mean': np.mean([r['mean_stability'] for r in all_results]),
                'std': np.std([r['mean_stability'] for r in all_results])
            },
            'cost': {
                'mean': np.mean([r['mean_cost'] for r in all_results]),
                'std': np.std([r['mean_cost'] for r in all_results])
            },
            'average_speed': {
                'mean': np.mean([r['mean_speed'] for r in all_results]),
                'std': np.std([r['mean_speed'] for r in all_results])
            },
            'intervention_rate': {
                'mean': np.mean([r['intervention_rate'] for r in all_results]),
                'std': np.std([r['intervention_rate'] for r in all_results])
            },
            'total_collisions': np.sum([r['collisions'] for r in all_results]),
            'total_emergency_braking': np.sum([r['emergency_braking'] for r in all_results]),
            'episodes': all_results
        }

        return results

    def print_summary(self, results):
        """打印评估摘要"""
        print("\n" + "=" * 80)
        print("Stage 2 评估结果摘要")
        print("=" * 80)

        print(f"\n📊 奖励指标:")
        print(f"  总Reward: {results['total_reward']['mean']:.2f} ± {results['total_reward']['std']:.2f}")
        print(f"  范围: [{results['total_reward']['min']:.2f}, {results['total_reward']['max']:.2f}]")

        print(f"\n🎯 OCR指标（比赛关键指标）:")
        print(f"  平均OCR: {results['ocr']['mean']:.4f} ± {results['ocr']['std']:.4f}")
        print(f"  范围: [{results['ocr']['min']:.4f}, {results['ocr']['max']:.4f}]")

        print(f"\n⚡ 效率指标:")
        print(f"  平均速度: {results['average_speed']['mean']:.2f} ± {results['average_speed']['std']:.2f} m/s")
        print(f"  效率奖励: {results['efficiency']['mean']:.4f} ± {results['efficiency']['std']:.4f}")

        print(f"\n🛡️ 安全指标:")
        print(f"  总碰撞数: {results['total_collisions']}")
        print(f"  总急刹车: {results['total_emergency_braking']}")
        print(f"  稳定性奖励: {results['stability']['mean']:.4f} ± {results['stability']['std']:.4f}")

        print(f"\n💰 成本指标:")
        print(f"  干预率: {results['intervention_rate']['mean']:.2%} ± {results['intervention_rate']['std']:.2%}")
        print(f"  成本奖励: {results['cost']['mean']:.4f} ± {results['cost']['std']:.4f}")

        # 评级
        print(f"\n📈 性能评级:")

        ocr = results['ocr']['mean']
        if ocr > 0.9:
            ocr_grade = "优秀 🌟"
        elif ocr > 0.8:
            ocr_grade = "良好 ✅"
        elif ocr > 0.7:
            ocr_grade = "及格 ⚠️"
        else:
            ocr_grade = "需要改进 ❌"

        print(f"  OCR性能: {ocr_grade}")

        speed = results['average_speed']['mean']
        if speed > 20:
            speed_grade = "优秀 🌟"
        elif speed > 15:
            speed_grade = "良好 ✅"
        elif speed > 10:
            speed_grade = "及格 ⚠️"
        else:
            speed_grade = "需要改进 ❌"

        print(f"  速度性能: {speed_grade}")

        collision_rate = results['total_collisions'] / results['num_episodes']
        if collision_rate == 0:
            safety_grade = "优秀 🌟"
        elif collision_rate < 1:
            safety_grade = "良好 ✅"
        elif collision_rate < 5:
            safety_grade = "及格 ⚠️"
        else:
            safety_grade = "需要改进 ❌"

        print(f"  安全性能: {safety_grade} ({collision_rate:.2f} collisions/episode)")

        intervention = results['intervention_rate']['mean']
        if intervention < 0.1:
            intervention_grade = "优秀 🌟"
        elif intervention < 0.3:
            intervention_grade = "良好 ✅"
        elif intervention < 0.5:
            intervention_grade = "及格 ⚠️"
        else:
            intervention_grade = "需要改进 ❌"

        print(f"  干预效率: {intervention_grade}")

        # 建议
        print(f"\n💡 建议:")

        can_proceed = True
        issues = []

        if ocr < 0.7:
            can_proceed = False
            issues.append("OCR过低，建议继续训练Stage 2或调整超参数")

        if speed < 15:
            can_proceed = False
            issues.append("平均速度偏低，建议检查效率奖励权重")

        if collision_rate > 2:
            can_proceed = False
            issues.append("碰撞率过高，建议增加稳定性奖励权重或启用SafetyShield")

        if intervention > 0.5:
            issues.append("干预率较高，但可以进入Stage 3学习约束优化")

        if can_proceed:
            print("  ✅ 模型性能良好，可以进入Stage 3约束优化训练")
        else:
            print("  ⚠️ 建议改进以下问题后再进入Stage 3:")
            for issue in issues:
                print(f"     - {issue}")

    def visualize_results(self, results, save_dir):
        """
        生成可视化图表

        Args:
            results: 评估结果
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        episodes = list(range(1, len(results['episodes']) + 1))

        # 图1: Reward分布
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1.1 总Reward
        axes[0, 0].bar(episodes, [r['total_reward'] for r in results['episodes']])
        axes[0, 0].axhline(results['total_reward']['mean'], color='red', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Total Reward per Episode')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 1.2 OCR
        axes[0, 1].bar(episodes, [r['mean_ocr'] for r in results['episodes']], color='green')
        axes[0, 1].axhline(results['ocr']['mean'], color='red', linestyle='--', label='Mean')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('OCR')
        axes[0, 1].set_title('OCR per Episode')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1])

        # 1.3 平均速度
        axes[1, 0].bar(episodes, [r['mean_speed'] for r in results['episodes']], color='orange')
        axes[1, 0].axhline(results['average_speed']['mean'], color='red', linestyle='--', label='Mean')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Speed (m/s)')
        axes[1, 0].set_title('Average Speed per Episode')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 1.4 干预率
        axes[1, 1].bar(episodes, [r['intervention_rate'] for r in results['episodes']], color='purple')
        axes[1, 1].axhline(results['intervention_rate']['mean'], color='red', linestyle='--', label='Mean')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Intervention Rate')
        axes[1, 1].set_title('Intervention Rate per Episode')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'reward_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图表已保存: {save_dir / 'reward_metrics.png'}")

        # 图2: 奖励分解
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        reward_types = ['efficiency', 'stability', 'cost']
        colors = ['blue', 'green', 'red']
        titles = ['Efficiency Reward', 'Stability Reward', 'Cost Reward']

        for idx, (rtype, color, title) in enumerate(zip(reward_types, colors, titles)):
            means = [r[f'mean_{rtype}'] for r in results['episodes']]
            axes[idx].bar(episodes, means, color=color)
            axes[idx].axhline(results[rtype]['mean'], color='red', linestyle='--', label='Mean')
            axes[idx].set_xlabel('Episode')
            axes[idx].set_ylabel('Reward')
            axes[idx].set_title(title)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'reward_components.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图表已保存: {save_dir / 'reward_components.png'}")

        # 图3: 安全统计
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 3.1 碰撞数
        axes[0].bar(episodes, [r['collisions'] for r in results['episodes']], color='red')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Collisions')
        axes[0].set_title('Collisions per Episode')
        axes[0].grid(True, alpha=0.3)

        # 3.2 急刹车
        axes[1].bar(episodes, [r['emergency_braking'] for r in results['episodes']], color='orange')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Emergency Braking')
        axes[1].set_title('Emergency Braking per Episode')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'safety_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 图表已保存: {save_dir / 'safety_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Guided Exploration Evaluation")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--num_eval_episodes', type=int, default=20,
                        help='评估episode数')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化图表')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Stage 2: 引导探索评估")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"评估Episodes: {args.num_eval_episodes}")
    print(f"设备: {args.device}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        config=config,
        use_gui=False,
        device=args.device
    )

    # 创建策略
    print("\n创建策略...")
    policy = create_joint_icv_policy(
        obs_dim=config['policy']['obs_dim'],
        node_dim=config['policy']['node_dim'],
        hidden_dim=config['policy']['hidden_dim'],
        num_layers=config['policy']['num_layers'],
        initial_k_ratio=config['policy']['sparse_gate']['initial_k_ratio'],
        device=args.device
    )

    # 加载权重
    print(f"\n加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint)
    print("  ✅ 权重已加载")

    # 设置为评估模式
    policy.eval()

    # 创建评估器
    evaluator = Stage2Evaluator(policy, env, config, args.device)

    # 评估
    results = evaluator.evaluate(args.num_eval_episodes)

    # 打印摘要
    evaluator.print_summary(results)

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['global']['log_dir']) / 'eval_stage2'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f'evaluation_results_{timestamp}.json'

    # 转换numpy类型为Python类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    results_serializable = convert_to_serializable(results)

    with open(result_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n✅ 评估结果已保存: {result_file}")

    # 可视化
    if args.visualize:
        print("\n生成可视化图表...")
        plot_dir = output_dir / f'plots_{timestamp}'
        evaluator.visualize_results(results, plot_dir)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
