"""
统一训练脚本 - 融合 train.py 和 train_sb3.py

支持完整的 4 阶段训练流程：
- Phase 1: 世界模型预训练（监督学习）
- Phase 2: SB3 PPO 训练
- Phase 3: SB3 端到端微调
- Phase 4: CPO 约束优化

特性：
1. 默认使用 SB3 优化路径（Phase 2-4）
2. 保持与 train.py 完全相同的命令行接口
3. 支持所有原有配置文件
4. 提供传统实现作为备用选项

使用示例：
    # 完整 4 阶段训练
    python train_unified.py --config configs/windows_base.yaml

    # 只运行特定阶段
    python train_unified.py --config configs/windows_base.yaml --phase 2

    # 覆盖参数
    python train_unified.py --config configs/windows_base.yaml --envs 8 --timesteps 50000

    # 使用传统实现（向后兼容）
    python train_unified.py --config configs/windows_base.yaml --legacy
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入模型创建函数（可能失败）
from src.models import create_model_from_config
# 导入 SB3 训练管道
from src.algorithms.training_sb3 import TrainingPipelineSB3

# 导入评估模块
from src.evaluation import XLSXResultGenerator, generate_evaluation_report

def load_config(config_path: str) -> dict:
    """
    加载配置文件（支持 JSON 和 YAML 格式）

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        # 根据文件扩展名选择加载方式
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # 从 YAML 配置中提取路径配置
    if 'paths' in config:
        paths = config['paths']
        if 'checkpoint_dir' in paths and 'checkpoint_dir' not in config:
            config['checkpoint_dir'] = paths['checkpoint_dir']
        if 'log_dir' in paths and 'log_dir' not in config:
            config['log_dir'] = paths['log_dir']
        if 'data_dir' in paths and 'data_dir' not in config:
            config['data_dir'] = paths['data_dir']
        if 'result_dir' in paths and 'result_dir' not in config:
            config['result_dir'] = paths['result_dir']

    return config


def apply_command_line_overrides(config: dict, args: argparse.Namespace):
    """
    应用命令行参数覆盖配置

    Args:
        config: 配置字典
        args: 命令行参数
    """
    overrides_applied = []

    # 覆盖并行环境数
    if args.envs is not None:
        config.setdefault('training', {}).setdefault('phase2', {})['num_envs'] = args.envs
        overrides_applied.append(f"num_envs = {args.envs}")

    # 覆盖训练步数
    if args.timesteps is not None:
        config.setdefault('training', {}).setdefault('phase2', {})['total_timesteps'] = args.timesteps
        config.setdefault('training', {}).setdefault('phase3', {})['total_timesteps'] = args.timesteps
        config.setdefault('training', {}).setdefault('phase4', {})['total_timesteps'] = args.timesteps
        overrides_applied.append(f"total_timesteps = {args.timesteps}")

    # 覆盖学习率
    if args.learning_rate is not None:
        for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
            config.setdefault('training', {}).setdefault(phase, {})['learning_rate'] = args.learning_rate
        overrides_applied.append(f"learning_rate = {args.learning_rate}")

    # 覆盖批次大小
    if args.batch_size is not None:
        for phase in ['phase1', 'phase2', 'phase3']:
            config.setdefault('training', {}).setdefault(phase, {})['batch_size'] = args.batch_size
        overrides_applied.append(f"batch_size = {args.batch_size}")

    # 覆盖 n_steps
    if args.n_steps is not None:
        for phase in ['phase2', 'phase3']:
            config.setdefault('training', {}).setdefault(phase, {})['n_steps'] = args.n_steps
        overrides_applied.append(f"n_steps = {args.n_steps}")

    # 覆盖设备
    if args.device is not None:
        config['device'] = args.device
        overrides_applied.append(f"device = {args.device}")

    # 覆盖随机种子
    if args.seed is not None:
        config['seed'] = args.seed
        overrides_applied.append(f"seed = {args.seed}")

    # 打印覆盖信息
    if overrides_applied:
        print("\n📝 命令行参数覆盖:")
        for override in overrides_applied:
            print(f"   - {override}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能交通控制系统统一训练脚本（支持 SB3 优化）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
训练模式:
  --use-sb3     使用 SB3 优化实现（Phase 2-4，默认）
  --legacy      使用传统自定义实现（向后兼容）

阶段选择:
  --phase 1     只运行 Phase 1（世界模型预训练）
  --phase 2     只运行 Phase 2（PPO 训练）
  --phase 3     只运行 Phase 3（端到端微调）
  --phase 4     只运行 Phase 4（约束优化）
  --phase all   运行完整 4 阶段训练（默认）

配置文件选项:
  configs/training_config.json           (JSON 格式)
  configs/windows_base.yaml              (YAML 格式，标准配置)
  configs/windows_high_performance.yaml  (YAML 格式，高性能配置)
  configs/windows_extreme_performance.yaml (YAML 格式，极限性能配置)
  configs/quick_test.yaml                (YAML 格式，快速测试)

示例:
  # 完整 4 阶段训练（SB3 优化）
  python train_unified.py --config configs/windows_base.yaml

  # 快速测试
  python train_unified.py --config configs/quick_test.yaml --phase 2

  # 覆盖并行环境数和训练步数
  python train_unified.py --config configs/windows_base.yaml --envs 8 --timesteps 50000

  # 使用传统实现
  python train_unified.py --config configs/windows_base.yaml --legacy

  # 只运行特定阶段
  python train_unified.py --config configs/windows_base.yaml --phase 2
        """
    )

    # ========== 配置文件参数 ==========
    parser.add_argument('--config', type=str, default='configs/training_config.json',
                       help='配置文件路径（支持 JSON 和 YAML 格式）')

    # ========== 训练模式选择 ==========
    parser.add_argument('--use-sb3', action='store_true', default=True,
                       help='使用 SB3 优化实现（Phase 2-4，默认）')
    parser.add_argument('--legacy', action='store_true',
                       help='使用传统自定义实现（向后兼容）')

    # ========== 阶段选择 ==========
    parser.add_argument('--phase', type=str, default='all',
                       choices=['1', '2', '3', '4', 'all'],
                       help='训练阶段: 1, 2, 3, 4, 或 all（默认: all）')

    # ========== 评估模式 ==========
    parser.add_argument('--eval-only', action='store_true',
                       help='仅评估模式')
    parser.add_argument('--generate-xlsx', action='store_true',
                       help='生成 XLSX 结果文件')

    # ========== 数据收集 ==========
    parser.add_argument('--skip-data-collection', action='store_true',
                       help='跳过 SUMO 数据收集，使用已生成的数据')

    # ========== 参数覆盖（与 train_sb3.py 兼容） ==========
    parser.add_argument('--envs', type=int, default=None,
                       help='并行环境数量（覆盖配置文件）')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='总训练步数（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--n-steps', type=int, default=None,
                       help='每次 rollout 的步数（覆盖配置文件）')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'], default=None,
                       help='训练设备（覆盖配置文件）')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（覆盖配置文件）')

    args = parser.parse_args()

    # ========== 加载配置 ==========
    print("="*70)
    print("🎯 智能交通协同控制系统 - 统一训练脚本")
    print("="*70)

    config = load_config(args.config)
    print(f"\n📋 配置文件: {args.config}")

    # 应用命令行参数覆盖
    apply_command_line_overrides(config, args)

    print(f"\n🔧 设备: {config.get('device', 'cuda')}")
    print(f"🌱 种子: {config.get('seed', 42)}")

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # ========== 选择训练管道 ==========
    print("\n✨ 使用 SB3 优化训练管道")
    trainer = TrainingPipelineSB3(config)
    # ========== 仅评估模式 ==========
    if args.eval_only:
        print("\n📊 仅评估模式")

        # 加载模型
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
        if os.path.exists(checkpoint_path):
            # 加载 SB3 模型
            from stable_baselines3 import PPO
            model = PPO.load(checkpoint_path.replace('.pth', '_sb3.zip'))
        else:
            print(f"⚠️  检查点不存在: {checkpoint_path}")
            print("   将使用随机初始化的模型")

        # 评估
        metrics = generate_evaluation_report(model, config)

        # 生成 XLSX
        if args.generate_xlsx:
            generator = XLSXResultGenerator(config.get('result_dir', 'results'))
            filepath = generator.generate_results(
                model=model,
                config=config,
                num_episodes=config.get('evaluation', {}).get('num_episodes', 5)
            )
            print(f"\n✅ XLSX 文件已生成: {filepath}")

        return

    # ========== 训练模式 ==========
    start_time = time.time()

    # Phase 1: 世界模型预训练
    if args.phase in ['1', 'all']:
        print("\n" + "="*70)
        print("🔄 阶段 1：世界模型预训练")
        print("="*70)

        phase1_config = config.get('training', {}).get('phase1', {})

        # 使用 SB3 管道（复用传统 Phase 1）
        model = trainer.train_phase1(
            model=None,
            num_episodes=phase1_config.get('num_episodes', 20),
            epochs=phase1_config.get('epochs', 20),
            batch_size=phase1_config.get('batch_size', 128),
            learning_rate=phase1_config.get('learning_rate', 1e-4),
            skip_data_collection=args.skip_data_collection
        )

    # Phase 2: PPO 训练
    if args.phase in ['2', 'all']:
        print("\n" + "="*70)
        print("🔄 阶段 2：PPO 训练")
        print("="*70)

        phase2_config = config.get('training', {}).get('phase2', {})

        model = trainer.train_phase2_sb3(
            total_timesteps=phase2_config.get('total_timesteps', 100000),
            num_envs=phase2_config.get('num_envs', 4),
            learning_rate=phase2_config.get('learning_rate', 3e-4)
        )

    # Phase 3: 端到端微调
    if args.phase in ['3', 'all']:
        print("\n" + "="*70)
        print("🔄 阶段 3：端到端微调")
        print("="*70)

        phase3_config = config.get('training', {}).get('phase3', {})

        # 检查是否启用
        if not phase3_config.get('enabled', True):
            print("⏭️  Phase 3 已禁用 (phase3.enabled: false)")
            print("   跳过端到端微调")
        else:
            # SB3 端到端微调
            model = trainer.train_phase3_sb3(
                total_timesteps=phase3_config.get('total_timesteps', 50000),
                learning_rate=phase3_config.get('learning_rate', 1e-5),
                freeze_bn=phase3_config.get('freeze_bn', True)
            )

    # Phase 4: 约束优化训练
    if args.phase in ['4', 'all']:
        print("\n" + "="*70)
        print("🔄 阶段 4：约束优化训练")
        print("="*70)

        phase4_config = config.get('training', {}).get('phase4', {})

        # 检查是否启用
        if not phase4_config.get('enabled', True):
            print("⏭️  Phase 4 已禁用 (phase4.enabled: false)")
            print("   跳过约束优化训练")
        else:
            # CPO 约束优化（拉格朗日 PPO）
            model = trainer.train_phase4_sb3(
                total_timesteps=phase4_config.get('total_timesteps', 50000),
                cost_limit=phase4_config.get('cost_limit', 0.1),
                learning_rate=phase4_config.get('learning_rate', 1e-4)
            )
    total_time = time.time() - start_time

    # ========== 训练完成 ==========
    print("\n" + "="*70)
    print("🎉 训练完成！（4 阶段训练）")
    print("="*70)
    print(f"   总耗时: {total_time/3600:.2f} 小时")

    # 保存训练历史
    history_path = os.path.join(config['log_dir'], 'training_history.json')
    trainer.save_history(history_path)

    # 最终评估
    print("\n📊 最终评估...")
    metrics = generate_evaluation_report(model, config)

    # 生成 XLSX 结果文件
    if args.generate_xlsx:
        print("\n📊 生成 XLSX 结果文件...")
        generator = XLSXResultGenerator(config.get('result_dir', 'results'))
        filepath = generator.generate_results(
            model=model,
            config=config,
            num_episodes=config.get('evaluation', {}).get('num_episodes', 5)
        )
        print(f"✅ XLSX 文件已生成: {filepath}")

    print("\n✅ 所有任务完成！")
    print(f"   - 模型检查点: {config['checkpoint_dir']}")
    print(f"   - 训练日志: {config['log_dir']}")
    print(f"   - 结果文件: {config.get('result_dir', 'results')}")


if __name__ == "__main__":
    main()
