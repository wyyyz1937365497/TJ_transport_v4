"""
主训练脚本
功能：执行完整的三阶段训练流程
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import create_model_from_config
from src.algorithms import Trainer
from src.evaluation import XLSXResultGenerator, generate_evaluation_report


def load_config(config_path: str) -> dict:
    """加载配置文件（支持JSON和YAML格式）"""
    import os

    with open(config_path, 'r', encoding='utf-8') as f:
        # 根据文件扩展名选择加载方式
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # 自动检测设备
    if config.get('device', 'cuda') == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        config['device'] = 'cpu'

    # 从YAML配置中提取路径配置
    if 'paths' in config:
        paths = config['paths']
        if 'checkpoint_dir' in paths and 'checkpoint_dir' not in config:
            config['checkpoint_dir'] = paths['checkpoint_dir']
        if 'log_dir' in paths and 'log_dir' not in config:
            config['log_dir'] = paths['log_dir']

    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能交通控制系统训练（完整4阶段）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练阶段说明:
  Phase 1 - 世界模型预训练：学习基础交通动力学
  Phase 2 - PPO训练：带安全屏障的强化学习训练
  Phase 3 - 端到端微调：联合优化所有组件
  Phase 4 - 约束优化：使用拉格朗日乘子法平衡性能与成本

配置文件选项:
  --config configs/training_config.json (JSON格式)
  --config configs/windows_base.yaml (YAML格式，基础配置)
  --config configs/windows_high_performance.yaml (YAML格式，高性能配置)
  --config configs/windows_extreme_performance.yaml (YAML格式，极限性能配置)

示例:
  # 完整4阶段训练
  python train.py --config configs/windows_base.yaml --phase all

  # 只运行阶段1（世界模型预训练）
  python train.py --config configs/windows_base.yaml --phase 1

  # 跳过数据收集，使用已有数据
  python train.py --config configs/windows_base.yaml --phase 1 --skip-data-collection
        """
    )
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='配置文件路径（支持JSON和YAML格式）')
    parser.add_argument('--phase', type=str, default='all', choices=['1', '2', '3', '4', 'all'],
                       help='训练阶段: 1, 2, 3, 4, 或 all')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--generate-xlsx', action='store_true',
                       help='生成XLSX结果文件')
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='跳过SUMO数据收集，使用已生成的数据')

    args = parser.parse_args()

    # 加载配置
    print("="*70)
    print("🎯 智能交通协同控制系统 - 完整4阶段训练")
    print("="*70)

    config = load_config(args.config)
    print(f"\n📋 配置文件: {args.config}")
    print(f"🔧 设备: {config.get('device', 'cuda')}")
    print(f"🌱 种子: {config.get('seed', 42)}")

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 创建模型
    print(f"\n🏗️  创建模型...")
    model = create_model_from_config(config)

    # 创建训练器
    trainer = Trainer(config)

    # 仅评估模式
    if args.eval_only:
        print("\n📊 仅评估模式")

        # 加载模型
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
        else:
            print(f"⚠️  检查点不存在: {checkpoint_path}")
            print("   将使用随机初始化的模型")

        # 评估
        metrics = generate_evaluation_report(model, config)

        # 生成XLSX
        if args.generate_xlsx:
            generator = XLSXResultGenerator(config['evaluation']['output_dir'])
            filepath = generator.generate_results(
                model=model,
                config=config,
                num_episodes=config['evaluation']['num_episodes']
            )
            print(f"\n✅ XLSX文件已生成: {filepath}")

        return

    # 训练模式
    start_time = time.time()

    if args.phase == 'all' or args.phase == '1':
        print("\n" + "="*70)
        print("🔄 阶段1：世界模型预训练")
        print("="*70)

        model = trainer.train_phase1(
            model=model,
            num_episodes=config.get('phase1', {}).get('num_episodes', 20),
            epochs=config.get('phase1', {}).get('epochs', 20),
            batch_size=config.get('phase1', {}).get('batch_size', 128),
            learning_rate=config.get('phase1', {}).get('lr', 1e-4),
            skip_data_collection=args.skip_data_collection
        )

    if args.phase == 'all' or args.phase == '2':
        print("\n" + "="*70)
        print("🔄 阶段2：带安全屏障的RL训练")
        print("="*70)

        model = trainer.train_phase2(
            model=model,
            num_envs=config.get('phase2', {}).get('num_envs', 2),
            total_timesteps=config.get('phase2', {}).get('total_timesteps', 10000),
            learning_rate=config.get('phase2', {}).get('lr', 3e-4)
        )

    if args.phase == 'all' or args.phase == '3':
        print("\n" + "="*70)
        print("🔄 阶段3：端到端微调（所有组件联合优化）")
        print("="*70)

        model = trainer.train_phase3(
            model=model,
            total_timesteps=config.get('phase3', {}).get('total_timesteps', 5000),
            learning_rate=config.get('phase3', {}).get('lr', 1e-5),
            freeze_bn=config.get('phase3', {}).get('freeze_bn', True)
        )

    if args.phase == 'all' or args.phase == '4':
        print("\n" + "="*70)
        print("🔄 阶段4：约束优化训练（拉格朗日乘子法）")
        print("="*70)

        model = trainer.train_phase4(
            model=model,
            total_timesteps=config.get('phase4', {}).get('total_timesteps', 5000),
            cost_limit=config.get('phase4', {}).get('cost_limit', 0.1),
            learning_rate=config.get('phase4', {}).get('lr', 1e-4)
        )

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("🎉 训练完成!（4阶段训练）")
    print("="*70)
    print(f"   总耗时: {total_time/3600:.2f} 小时")

    # 保存训练历史
    history_path = os.path.join(config['log_dir'], 'training_history.json')
    trainer.save_history(history_path)

    # 最终评估
    print("\n📊 最终评估...")
    metrics = generate_evaluation_report(model, config)

    # 生成XLSX结果文件
    if args.generate_xlsx:
        print("\n📊 生成XLSX结果文件...")
        generator = XLSXResultGenerator(config['evaluation']['output_dir'])
        filepath = generator.generate_results(
            model=model,
            config=config,
            num_episodes=config['evaluation']['num_episodes']
        )
        print(f"✅ XLSX文件已生成: {filepath}")

    print("\n✅ 所有任务完成!")
    print(f"   - 模型检查点: {config['checkpoint_dir']}")
    print(f"   - 训练日志: {config['log_dir']}")
    print(f"   - 结果文件: {config['evaluation']['output_dir']}")


if __name__ == "__main__":
    main()
