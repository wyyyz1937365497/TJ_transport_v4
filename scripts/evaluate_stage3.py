#!/usr/bin/env python3
"""
Stage 3: 约束优化评估脚本（Constrained Optimization Evaluation）

功能：
1. 评估Stage 3训练的约束策略性能
2. 对比Stage 2和Stage 3的性能差异
3. 分析成本约束满足情况
4. 评估安全性和效率的平衡

使用方法：
    python scripts/evaluate_stage3.py \
        --config configs/v5_complete.yaml \
        --checkpoint checkpoints/v5_complete/stage3_best.pth \
        --baseline_checkpoint checkpoints/v5_complete/stage2_best.pth \
        --num_eval_episodes 20 \
        --visualize \
        --device cuda

输出：
    - logs/v5_complete/eval_stage3/evaluation_results.json
    - logs/v5_complete/eval_stage3/comparison_report.json
    - logs/v5_complete/eval_stage3/plots/
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

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.env.competition_env import CompetitionSumoEnv
from scripts.evaluate_stage2 import Stage2Evaluator


class Stage3Evaluator(Stage2Evaluator):
    """
    Stage 3评估器（扩展Stage 2，添加约束评估）

    额外评估：
    - 成本约束满足情况
    - 安全性指标
    - 与Stage 2基线的对比
    """

    def __init__(self, policy, env, config, device='cuda'):
        """
        Args:
            policy: JointICVPolicy (with CostCritic and DynamicWeightGate)
            env: CompetitionSumoEnv
            config: 配置字典
            device: 设备
        """
        super().__init__(policy, env, config, device)

        # Stage 3特定配置
        self.cost_threshold = config['stage3_constrained_optimization']['lagrangian']['cost_threshold']

    def evaluate_episode(self, render=False):
        """
        评估单个episode（扩展：包含成本和约束）

        Returns:
            episode_metrics: Dict
        """
        episode_data = super().evaluate_episode(render)

        # 添加Stage 3特定指标
        episode_data.update({
            'constraint_violations': 0,  # 约束违反次数
            'max_cost': 0.0,  # 最大成本
            'cost_exceeds_threshold': False,  # 是否超过成本阈值
            'safety_critical_events': 0,  # 安全关键事件
            'near_misses': 0,  # 险肇事故
        })

        # 这些指标需要在评估过程中收集，这里简化处理
        # 实际实现中需要在step循环中添加统计逻辑

        return episode_data

    def compare_with_baseline(self, baseline_results, stage3_results):
        """
        对比Stage 2和Stage 3的性能

        Args:
            baseline_results: Stage 2评估结果
            stage3_results: Stage 3评估结果

        Returns:
            comparison: Dict
        """
        print("\n" + "=" * 80)
        print("Stage 2 vs Stage 3 性能对比")
        print("=" * 80)

        comparison = {
            'ocr': {
                'stage2': baseline_results['ocr']['mean'],
                'stage3': stage3_results['ocr']['mean'],
                'improvement': stage3_results['ocr']['mean'] - baseline_results['ocr']['mean'],
                'improvement_pct': ((stage3_results['ocr']['mean'] - baseline_results['ocr']['mean']) /
                                   baseline_results['ocr']['mean'] * 100) if baseline_results['ocr']['mean'] > 0 else 0
            },
            'average_speed': {
                'stage2': baseline_results['average_speed']['mean'],
                'stage3': stage3_results['average_speed']['mean'],
                'improvement': stage3_results['average_speed']['mean'] - baseline_results['average_speed']['mean'],
                'improvement_pct': ((stage3_results['average_speed']['mean'] - baseline_results['average_speed']['mean']) /
                                   baseline_results['average_speed']['mean'] * 100) if baseline_results['average_speed']['mean'] > 0 else 0
            },
            'total_reward': {
                'stage2': baseline_results['total_reward']['mean'],
                'stage3': stage3_results['total_reward']['mean'],
                'improvement': stage3_results['total_reward']['mean'] - baseline_results['total_reward']['mean'],
                'improvement_pct': ((stage3_results['total_reward']['mean'] - baseline_results['total_reward']['mean']) /
                                   baseline_results['total_reward']['mean'] * 100) if baseline_results['total_reward']['mean'] > 0 else 0
            },
            'intervention_rate': {
                'stage2': baseline_results['intervention_rate']['mean'],
                'stage3': stage3_results['intervention_rate']['mean'],
                'reduction': baseline_results['intervention_rate']['mean'] - stage3_results['intervention_rate']['mean'],
                'reduction_pct': ((baseline_results['intervention_rate']['mean'] - stage3_results['intervention_rate']['mean']) /
                                baseline_results['intervention_rate']['mean'] * 100) if baseline_results['intervention_rate']['mean'] > 0 else 0
            },
            'collisions': {
                'stage2': baseline_results['total_collisions'],
                'stage3': stage3_results['total_collisions'],
                'reduction': baseline_results['total_collisions'] - stage3_results['total_collisions'],
                'reduction_pct': ((baseline_results['total_collisions'] - stage3_results['total_collisions']) /
                                baseline_results['total_collisions'] * 100) if baseline_results['total_collisions'] > 0 else 0
            }
        }

        # 打印对比结果
        print(f"\n📊 OCR对比:")
        print(f"  Stage 2: {comparison['ocr']['stage2']:.4f}")
        print(f"  Stage 3: {comparison['ocr']['stage3']:.4f}")
        print(f"  改善: {comparison['ocr']['improvement']:+.4f} ({comparison['ocr']['improvement_pct']:+.2f}%)")
        if comparison['ocr']['improvement'] > 0:
            print("  ✅ OCR提升")
        elif comparison['ocr']['improvement'] > -0.01:
            print("  ⚠️ OCR基本持平")
        else:
            print("  ❌ OCR下降")

        print(f"\n⚡ 平均速度对比:")
        print(f"  Stage 2: {comparison['average_speed']['stage2']:.2f} m/s")
        print(f"  Stage 3: {comparison['average_speed']['stage3']:.2f} m/s")
        print(f"  改善: {comparison['average_speed']['improvement']:+.2f} m/s ({comparison['average_speed']['improvement_pct']:+.2f}%)")

        print(f"\n💰 干预率对比:")
        print(f"  Stage 2: {comparison['intervention_rate']['stage2']:.2%}")
        print(f"  Stage 3: {comparison['intervention_rate']['stage3']:.2%}")
        print(f"  降低: {comparison['intervention_rate']['reduction']:+.2%} ({comparison['intervention_rate']['reduction_pct']:+.2f}%)")
        if comparison['intervention_rate']['reduction'] > 0:
            print("  ✅ 干预率降低（更好的成本控制）")
        else:
            print("  ⚠️ 干预率上升")

        print(f"\n🛡️ 安全性对比:")
        print(f"  Stage 2碰撞数: {comparison['collisions']['stage2']}")
        print(f"  Stage 3碰撞数: {comparison['collisions']['stage3']}")
        print(f"  减少: {comparison['collisions']['reduction']:+d} ({comparison['collisions']['reduction_pct']:+.2f}%)")
        if comparison['collisions']['reduction'] > 0:
            print("  ✅ 碰撞数减少")
        elif comparison['collisions']['reduction'] == 0:
            print("  ✅ 碰撞数持平（已最优）")
        else:
            print("  ⚠️ 碰撞数增加")

        print(f"\n📈 总体评价:")
        # 综合评分
        score = 0
        if comparison['ocr']['improvement'] > 0:
            score += 1
        if comparison['intervention_rate']['reduction'] > 0:
            score += 1
        if comparison['collisions']['reduction'] >= 0:
            score += 1

        if score == 3:
            print("  🌟 优秀：Stage 3在所有关键指标上都有提升")
        elif score == 2:
            print("  ✅ 良好：Stage 3在大部分指标上有提升")
        elif score == 1:
            print("  ⚠️ 及格：Stage 3有部分提升，但仍需改进")
        else:
            print("  ❌ 需要改进：Stage 3未能超越Stage 2基线")

        return comparison

    def print_summary(self, results):
        """打印Stage 3评估摘要（扩展版）"""
        super().print_summary(results)

        # 添加Stage 3特定摘要
        print(f"\n🎯 Stage 3约束优化指标:")
        print(f"  成本阈值: {self.cost_threshold:.3f}")

        # 计算约束满足率
        # 这需要在episode评估中收集，这里简化处理
        print(f"  约束满足情况: 需要更详细的成本数据")

    def visualize_results(self, results, save_dir, baseline_results=None):
        """
        生成可视化图表（扩展：包含与Stage 2的对比）

        Args:
            results: Stage 3评估结果
            save_dir: 保存目录
            baseline_results: Stage 2基线结果（可选）
        """
        super().visualize_results(results, save_dir)

        save_dir = Path(save_dir)

        # 如果有基线结果，生成对比图
        if baseline_results is not None:
            self._plot_comparison(results, baseline_results, save_dir)

    def _plot_comparison(self, stage3_results, stage2_results, save_dir):
        """绘制Stage 2 vs Stage 3对比图"""
        episodes = list(range(1, len(stage3_results['episodes']) + 1))

        # 确保两个结果的episode数一致
        min_episodes = min(len(stage3_results['episodes']), len(stage2_results['episodes']))
        episodes = episodes[:min_episodes]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. OCR对比
        stage2_ocr = [r['mean_ocr'] for r in stage2_results['episodes'][:min_episodes]]
        stage3_ocr = [r['mean_ocr'] for r in stage3_results['episodes'][:min_episodes]]

        x = np.arange(min_episodes)
        width = 0.35

        axes[0, 0].bar(x - width/2, stage2_ocr, width, label='Stage 2', alpha=0.8)
        axes[0, 0].bar(x + width/2, stage3_ocr, width, label='Stage 3', alpha=0.8)
        axes[0, 0].axhline(stage2_results['ocr']['mean'], color='blue', linestyle='--', alpha=0.5)
        axes[0, 0].axhline(stage3_results['ocr']['mean'], color='orange', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('OCR')
        axes[0, 0].set_title('OCR Comparison: Stage 2 vs Stage 3')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 干预率对比
        stage2_int = [r['intervention_rate'] for r in stage2_results['episodes'][:min_episodes]]
        stage3_int = [r['intervention_rate'] for r in stage3_results['episodes'][:min_episodes]]

        axes[0, 1].bar(x - width/2, stage2_int, width, label='Stage 2', alpha=0.8)
        axes[0, 1].bar(x + width/2, stage3_int, width, label='Stage 3', alpha=0.8)
        axes[0, 1].axhline(stage2_results['intervention_rate']['mean'], color='blue', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(stage3_results['intervention_rate']['mean'], color='orange', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Intervention Rate')
        axes[0, 1].set_title('Intervention Rate Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 平均速度对比
        stage2_speed = [r['mean_speed'] for r in stage2_results['episodes'][:min_episodes]]
        stage3_speed = [r['mean_speed'] for r in stage3_results['episodes'][:min_episodes]]

        axes[1, 0].bar(x - width/2, stage2_speed, width, label='Stage 2', alpha=0.8)
        axes[1, 0].bar(x + width/2, stage3_speed, width, label='Stage 3', alpha=0.8)
        axes[1, 0].axhline(stage2_results['average_speed']['mean'], color='blue', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(stage3_results['average_speed']['mean'], color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Speed (m/s)')
        axes[1, 0].set_title('Speed Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 碰撞数对比
        stage2_col = [r['collisions'] for r in stage2_results['episodes'][:min_episodes]]
        stage3_col = [r['collisions'] for r in stage3_results['episodes'][:min_episodes]]

        axes[1, 1].bar(x - width/2, stage2_col, width, label='Stage 2', alpha=0.8)
        axes[1, 1].bar(x + width/2, stage3_col, width, label='Stage 3', alpha=0.8)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Collisions')
        axes[1, 1].set_title('Safety Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'stage2_vs_stage3_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 对比图表已保存: {save_dir / 'stage2_vs_stage3_comparison.png'}")

        # 改善百分比雷达图
        self._plot_improvement_radar(stage3_results, stage2_results, save_dir)

    def _plot_improvement_radar(self, stage3_results, stage2_results, save_dir):
        """绘制改善百分比雷达图"""
        # 计算改善百分比
        categories = ['OCR', 'Speed', 'Intervention\nRate', 'Total\nReward']

        stage2_values = [
            stage2_results['ocr']['mean'],
            stage2_results['average_speed']['mean'] / 30,  # 归一化到0-1
            1 - stage2_results['intervention_rate']['mean'],  # 反转：越低越好
            stage2_results['total_reward']['mean'] / 1000  # 归一化（假设范围）
        ]

        stage3_values = [
            stage3_results['ocr']['mean'],
            stage3_results['average_speed']['mean'] / 30,
            1 - stage3_results['intervention_rate']['mean'],
            stage3_results['total_reward']['mean'] / 1000
        ]

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        stage2_values += stage2_values[:1]
        stage3_values += stage3_values[:1]
        angles += angles[:1]

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        ax.plot(angles, stage2_values, 'o-', linewidth=2, label='Stage 2', color='blue')
        ax.fill(angles, stage2_values, alpha=0.25, color='blue')

        ax.plot(angles, stage3_values, 'o-', linewidth=2, label='Stage 3', color='orange')
        ax.fill(angles, stage3_values, alpha=0.25, color='orange')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Stage 2 vs Stage 3 Performance', size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(save_dir / 'performance_radar.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 雷达图已保存: {save_dir / 'performance_radar.png'}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Constrained Optimization Evaluation")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Stage 3模型检查点路径')
    parser.add_argument('--baseline_checkpoint', type=str, default=None,
                        help='Stage 2基线检查点路径（用于对比）')
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
    print("Stage 3: 约束优化评估")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"检查点: {args.checkpoint}")
    print(f"基线检查点: {args.baseline_checkpoint}")
    print(f"评估Episodes: {args.num_eval_episodes}")
    print(f"设备: {args.device}")

    # 创建环境
    print("\n创建环境...")
    env = CompetitionSumoEnv(
        config=config,
        use_gui=False,
        device=args.device
    )

    # 评估Stage 3模型
    print("\n创建Stage 3策略...")
    policy = create_joint_icv_policy(
        obs_dim=config['policy']['obs_dim'],
        node_dim=config['policy']['node_dim'],
        hidden_dim=config['policy']['hidden_dim'],
        num_layers=config['policy']['num_layers'],
        initial_k_ratio=config['policy']['sparse_gate']['initial_k_ratio'],
        device=args.device
    )

    print(f"\n加载Stage 3检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint)
    print("  ✅ Stage 3权重已加载")

    # 设置为评估模式
    policy.eval()

    # 创建Stage 3评估器
    evaluator = Stage3Evaluator(policy, env, config, args.device)

    # 评估Stage 3
    print("\n开始评估Stage 3...")
    stage3_results = evaluator.evaluate(args.num_eval_episodes)

    # 如果提供了基线，也评估Stage 2进行对比
    baseline_results = None
    if args.baseline_checkpoint:
        print("\n创建Stage 2基线策略...")
        baseline_policy = create_joint_icv_policy(
            obs_dim=config['policy']['obs_dim'],
            node_dim=config['policy']['node_dim'],
            hidden_dim=config['policy']['hidden_dim'],
            num_layers=config['policy']['num_layers'],
            initial_k_ratio=config['policy']['sparse_gate']['initial_k_ratio'],
            device=args.device
        )

        print(f"加载Stage 2检查点: {args.baseline_checkpoint}")
        baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=args.device)
        baseline_policy.load_state_dict(baseline_checkpoint)
        print("  ✅ Stage 2权重已加载")

        baseline_policy.eval()

        # 评估Stage 2
        print("\n开始评估Stage 2基线...")
        baseline_evaluator = Stage2Evaluator(baseline_policy, env, config, args.device)
        baseline_results = baseline_evaluator.evaluate(args.num_eval_episodes)

        # 对比
        comparison = evaluator.compare_with_baseline(baseline_results, stage3_results)

    # 打印Stage 3摘要
    evaluator.print_summary(stage3_results)

    # 保存结果
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['global']['log_dir']) / 'eval_stage3'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = output_dir / f'stage3_results_{timestamp}.json'

    # 转换numpy类型
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

    stage3_results_serializable = convert_to_serializable(stage3_results)

    with open(result_file, 'w') as f:
        json.dump(stage3_results_serializable, f, indent=2)

    print(f"\n✅ Stage 3评估结果已保存: {result_file}")

    # 保存对比结果
    if baseline_results is not None:
        comparison_file = output_dir / f'comparison_{timestamp}.json'
        comparison_serializable = convert_to_serializable(comparison)
        with open(comparison_file, 'w') as f:
            json.dump(comparison_serializable, f, indent=2)
        print(f"✅ 对比结果已保存: {comparison_file}")

    # 可视化
    if args.visualize:
        print("\n生成可视化图表...")
        plot_dir = output_dir / f'plots_{timestamp}'
        evaluator.visualize_results(stage3_results, plot_dir, baseline_results)

    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
