#!/usr/bin/env python3
"""
Baseline评估脚本 - 无任何ICV控制

用途：测量没有AI干预时的OCR，作为比赛baseline

运行方式：
    python scripts/evaluate_baseline.py --num_episodes 5
"""

import argparse
import json
from pathlib import Path
import sys
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import traci

from src.env.competition_env import CompetitionSumoEnv
from src.utils.config import load_config


def evaluate_baseline(
    config_path: str = "configs/v5_complete.yaml",
    num_episodes: int = 5,
    max_steps: int = 3600,
    use_gui: bool = False,
    output_dir: str = "logs/baseline"
) -> dict:
    """
    评估无ICV控制的baseline性能

    Args:
        config_path: 配置文件路径
        num_episodes: 评估episode数
        max_steps: 每个episode最大步数
        use_gui: 是否使用GUI
        output_dir: 输出目录

    Returns:
        评估结果字典
    """
    print("=" * 80)
    print("Baseline评估 - 无ICV控制")
    print("=" * 80)

    # 加载配置
    config = load_config(config_path)
    env_config = config.get('environment', {})

    # 修改配置：禁用ICV控制
    env_config['icv_ratio'] = 0.0  # 关键：设置ICV比例为0
    env_config['use_gui'] = use_gui
    env_config['max_steps'] = max_steps

    # 创建环境
    print("\n创建环境...")
    print(f"  SUMO配置: {env_config.get('sumo_cfg', 'N/A')}")
    print(f"  ICV比例: 0% (无控制)")
    print(f"  最大步数: {max_steps}")

    env = CompetitionSumoEnv(config=env_config)

    # 存储结果
    all_returns = []
    all_ocrs = []
    all_lengths = []
    all_speeds = []
    all_accelerations = []

    # 运行评估
    print(f"\n开始评估 ({num_episodes} episodes)...")
    print("-" * 80)

    for episode in range(num_episodes):
        episode_start = time.time()
        obs, info = env.reset()
        done = False
        step = 0
        episode_return = 0.0

        # 收集速度和加速度
        episode_speeds = []
        episode_accelerations = []

        print(f"\n[Episode {episode + 1}/{num_episodes}]")

        while not done and step < max_steps:
            # 不应用任何控制（使用SUMO默认驾驶行为）
            # 空 action_dict 表示所有车辆使用SUMO的IDM模型
            action_dict = {}

            # 执行一步
            obs, reward, done, truncated, info = env.step(action_dict)
            episode_return += reward
            step += 1

            # 收集统计信息
            if 'vehicle_stats' in info:
                for veh_id, stats in info['vehicle_stats'].items():
                    if 'speed' in stats:
                        episode_speeds.append(stats['speed'])
                    if 'acceleration' in stats:
                        episode_accelerations.append(stats['acceleration'])

            # 每1000步输出一次进度
            if step % 1000 == 0:
                elapsed = time.time() - episode_start
                print(f"  Step {step}/{max_steps} | Return: {episode_return:.2f} | Elapsed: {elapsed:.1f}s")

        episode_length = step
        episode_time = time.time() - episode_start

        # 计算OCR
        arrived_vehicles = env.stats.get('arrived_vehicles', [])
        departed_vehicles = env.stats.get('departed_vehicles', [])

        if len(departed_vehicles) > 0:
            ocr = len(arrived_vehicles) / len(departed_vehicles)
        else:
            ocr = 0.0

        # 计算平均速度和加速度
        mean_speed = np.mean(episode_speeds) if episode_speeds else 0.0
        std_speed = np.std(episode_speeds) if episode_speeds else 0.0
        mean_accel = np.mean(episode_accelerations) if episode_accelerations else 0.0
        abs_accel = np.mean([abs(a) for a in episode_accelerations]) if episode_accelerations else 0.0

        all_returns.append(episode_return)
        all_ocrs.append(ocr)
        all_lengths.append(episode_length)
        all_speeds.extend(episode_speeds)
        all_accelerations.extend(episode_accelerations)

        print(f"\n  Episode {episode + 1} 完成:")
        print(f"    Return: {episode_return:.2f}")
        print(f"    OCR: {ocr:.4f} ({len(arrived_vehicles)}/{len(departed_vehicles)})")
        print(f"    Length: {episode_length} 步")
        print(f"    Time: {episode_time:.1f}s")
        print(f"    Mean Speed: {mean_speed:.2f} ± {std_speed:.2f} m/s")
        print(f"    Mean Accel: {mean_accel:.3f} m/s²")

        # 关闭SUMO
        env.close()

    # 计算总体统计
    results = {
        'mean_return': np.mean(all_returns),
        'std_return': np.std(all_returns),
        'mean_ocr': np.mean(all_ocrs),
        'std_ocr': np.std(all_ocrs),
        'mean_length': np.mean(all_lengths),
        'std_length': np.std(all_lengths),
        'mean_speed': np.mean(all_speeds) if all_speeds else 0.0,
        'std_speed': np.std(all_speeds) if all_speeds else 0.0,
        'mean_acceleration': np.mean(all_accelerations) if all_accelerations else 0.0,
        'std_acceleration': np.std(all_accelerations) if all_accelerations else 0.0,
        'abs_acceleration': np.mean([abs(a) for a in all_accelerations]) if all_accelerations else 0.0,
        'episode_returns': all_returns,
        'episode_ocrs': all_ocrs,
        'episode_lengths': all_lengths,
        'num_episodes': num_episodes,
        'config': {
            'icv_ratio': 0.0,
            'max_steps': max_steps,
            'use_gui': use_gui
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline评估 - 无ICV控制")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='评估episode数')
    parser.add_argument('--max_steps', type=int, default=3600,
                        help='每个episode最大步数')
    parser.add_argument('--use_gui', action='store_true',
                        help='是否使用GUI')
    parser.add_argument('--output_dir', type=str, default='logs/baseline',
                        help='输出目录')

    args = parser.parse_args()

    # 运行评估
    results = evaluate_baseline(
        config_path=args.config,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        use_gui=args.use_gui,
        output_dir=args.output_dir
    )

    # 打印结果
    print("\n" + "=" * 80)
    print("Baseline评估结果（无ICV控制）")
    print("=" * 80 + "\n")

    print("📊 核心指标:")
    print(f"  Mean Return:  {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean OCR:     {results['mean_ocr']:.4f} ± {results['std_ocr']:.4f} ⭐")
    print(f"  Mean Length:  {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    print("\n🚗 车辆性能:")
    print(f"  Mean Speed:        {results['mean_speed']:.2f} ± {results['std_speed']:.2f} m/s")
    print(f"  Mean Acceleration: {results['mean_acceleration']:.3f} ± {results['std_acceleration']:.3f} m/s²")
    print(f"  Abs Acceleration:  {results['abs_acceleration']:.3f} m/s²")

    print("\n📈 Episode详情:")
    for i, (ret, ocr) in enumerate(zip(results['episode_returns'], results['episode_ocrs'])):
        print(f"  Episode {i+1:2d}: Return={ret:7.2f}, OCR={ocr:.4f}")

    print("\n" + "=" * 80)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存到: {output_path}")
    print("\n" + "=" * 80 + "\n")

    # 输出baseline OCR（方便复制到评估脚本）
    print(f"\n📌 Baseline OCR: {results['mean_ocr']:.4f}")
    print(f"   用法: python scripts/evaluate_ocr_max.py --baseline_ocr {results['mean_ocr']:.4f}\n")


if __name__ == '__main__':
    main()
