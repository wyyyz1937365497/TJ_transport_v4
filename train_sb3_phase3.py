"""
Phase 3: 端到端微调 - 使用Stable-Baselines3加速

这个脚本使用Stable-Baselines3的PPO进行Phase 3的端到端微调，
联合优化GNN + World Model + Controller所有组件。

相比原生PyTorch实现，SB3版本可以提供40-60%的性能提升。
"""

import os
import sys

# Windows UTF-8编码修复
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold
)
from stable_baselines3.common.utils import set_random_seed

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.env.vec_env import create_parallel_envs
from src.models.sb3_full_policy import create_full_traffic_policy


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件（支持YAML和JSON格式）"""
    with open(config_path, 'r', encoding='utf-8') as f:
        # 根据文件扩展名选择加载方式
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            import json
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


def train_phase3_sb3(
    config: Dict[str, Any],
    phase2_checkpoint: str = None,
    total_timesteps: int = 50000,
    num_envs: int = 4,
    checkpoint_dir: str = "checkpoints_sb3_phase3",
    freeze_bn: bool = True
):
    """
    使用Stable-Baselines3 PPO进行Phase 3端到端微调

    Args:
        config: 配置字典
        phase2_checkpoint: Phase 2检查点路径
        total_timesteps: 总训练步数
        num_envs: 并行环境数量
        checkpoint_dir: 检查点目录
        freeze_bn: 是否冻结BatchNorm层
    """
    print("="*70)
    print("🚀 Phase 3: 端到端微调 (Stable-Baselines3加速)")
    print("="*70)
    print(f"   - 总步数: {total_timesteps:,}")
    print(f"   - 并行环境: {num_envs}")
    print(f"   - 设备: {config.get('device', 'cuda')}")
    print(f"   - 检查点: {checkpoint_dir}")
    print(f"   - 冻结BN: {freeze_bn}")
    print("="*70)

    # 设置随机种子
    seed = config.get('seed', 42)
    set_random_seed(seed)

    # 创建并行环境
    print("\n📊 创建并行环境...")
    env_config = config.get('environment', {})

    # 修改配置以适应Gymnasium环境
    env_config['max_steps'] = env_config.get('max_steps', 3600)

    # 转换相对路径为绝对路径
    for key in ['sumo_cfg', 'net_file', 'route_file']:
        if key in env_config:
            path = env_config[key]
            if not os.path.isabs(path):
                abs_path = str((Path.cwd() / path).resolve())
                if not os.path.exists(abs_path):
                    print(f"⚠️  警告: 配置文件不存在: {abs_path}")
                env_config[key] = abs_path

    # 创建训练环境
    vec_env = create_parallel_envs(
        config=env_config,
        num_envs=num_envs,
        base_port=8813,
        seed=seed
    )

    print(f"✅ 环境创建成功")
    print(f"   - 观测空间: {vec_env.observation_space}")
    print(f"   - 动作空间: {vec_env.action_space}")

    # 使用底层的VecEnv
    vec_env = vec_env.vec_env

    # 创建自定义策略
    print("\n🧠 创建完整交通策略（GNN + World Model + Controller）...")

    # 合并配置
    full_config = {
        'device': config.get('device', 'cuda'),
        **config.get('model', {}),
        **config.get('gnn', {}),
        **config.get('world_model', {}),
        **config.get('controller', {}),
        **config.get('safety', {}),
        **env_config
    }

    # 提取模型配置
    model_config = {
        'node_dim': full_config.get('node_dim', 9),
        'edge_dim': full_config.get('edge_dim', 4),
        'gnn_hidden_dim': full_config.get('hidden_dim', 64),
        'gnn_output_dim': full_config.get('output_dim', 256),
        'gnn_layers': full_config.get('num_layers', 3),
        'gnn_heads': full_config.get('heads', 4),
        'gnn_dropout': full_config.get('dropout', 0.1),
        'world_hidden_dim': full_config.get('world_hidden_dim', 128),
        'future_steps': full_config.get('future_steps', 5),
        'world_dropout': full_config.get('world_dropout', 0.1),
        'world_num_layers': full_config.get('world_num_layers', 2),
        'world_bidirectional': full_config.get('world_bidirectional', False),
        'global_dim': full_config.get('global_dim', 16),
        'controller_hidden_dim': full_config.get('controller_hidden_dim', 128),
        'action_dim': full_config.get('action_dim', 2),
        'top_k': full_config.get('top_k', 5),
        'controller_dropout': full_config.get('controller_dropout', 0.2),
        'ttc_threshold': full_config.get('ttc_threshold', 2.0),
        'thw_threshold': full_config.get('thw_threshold', 1.5),
        'max_accel': full_config.get('max_accel', 2.0),
        'max_decel': full_config.get('max_decel', -3.0),
        'emergency_decel': full_config.get('emergency_decel', -5.0),
        'max_lane_change_speed': full_config.get('max_lane_change_speed', 5.0),
        'interaction_radius': full_config.get('interaction_radius', 100.0),
        'max_neighbors': full_config.get('max_neighbors', 8),
        'lane_change_distance': full_config.get('lane_change_distance', 50.0)
    }

    policy_class = create_full_traffic_policy(model_config)

    # 从配置文件读取Phase 3参数
    training_config = config.get('training', {}).get('phase3', {})

    # PPO超参数（端到端微调使用较小的学习率）
    learning_rate = training_config.get('learning_rate', 1e-5)
    n_steps = training_config.get('n_steps', 2048)
    batch_size = training_config.get('batch_size', 256)
    n_epochs = training_config.get('update_epochs', 10)
    gamma = training_config.get('gamma', 0.99)
    gae_lambda = training_config.get('gae_lambda', 0.95)
    clip_range = training_config.get('clip_epsilon', 0.2)
    ent_coef = training_config.get('entropy_coef', 0.01)
    vf_coef = training_config.get('value_loss_coef', 0.5)

    # 创建PPO模型
    model = PPO(
        policy=policy_class,
        env=vec_env,
        verbose=1,
        tensorboard_log="./logs_sb3_phase3/tensorboard/",
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        target_kl=0.03,
        stats_window_size=100,
        seed=seed,
        device=config.get('device', 'cuda'),
        _init_setup_model=True
    )

    # 加载Phase 2权重（如果提供）
    if phase2_checkpoint is None:
        # 尝试从配置中获取
        phase2_checkpoint = training_config.get('phase2_model_path')
        if phase2_checkpoint is None:
            phase2_checkpoint = os.path.join(
                config.get('checkpoint_dir', 'checkpoints'),
                'ppo_phase2.pth'
            )

    if phase2_checkpoint and os.path.exists(phase2_checkpoint):
        print(f"\n📂 加载Phase 2权重: {phase2_checkpoint}")
        model.policy.load_phase2_weights(phase2_checkpoint)

        # 解冻所有组件进行端到端训练
        model.policy.unfreeze_all()

        # 冻结BatchNorm（如果需要）
        if freeze_bn:
            model.policy.freeze_batch_norm()

    print("\n🏋️ 开始端到端微调...")
    print("="*70)

    # 设置回调
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 定期保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="ppo_e2e_phase3",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # 评估环境
    eval_env_wrapper = create_parallel_envs(
        config=env_config,
        num_envs=1,
        base_port=8900,
        seed=seed
    )

    eval_callback = EvalCallback(
        eval_env_wrapper.vec_env,
        best_model_save_path=checkpoint_dir,
        log_path="./logs_sb3_phase3/eval/",
        eval_freq=5000,
        deterministic=False,
        render=False
    )

    start_time = time.time()

    # 训练
    print("\n🏋️ 开始端到端微调...")
    print("="*70)
    print(f"📊 训练统计:")
    print(f"   - 总步数: {total_timesteps:,}")
    print(f"   - 学习率: {learning_rate:.6f}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - PPO n_steps: {n_steps}")
    print(f"   - PPO epochs: {n_epochs}")
    print(f"   - 冻结BN: {freeze_bn}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=100  # 每100步记录一次
    )

    print("\n" + "="*70)
    print("✅ Phase 3端到端微调完成！")
    print("="*70)
    print(f"📊 训练统计:")
    print(f"   - 总步数: {total_timesteps:,}")
    print(f"   - 训练时长: {format(time.time() - start_time, '.2f')}s")
    print(f"   - 吞吐量: {total_timesteps/(time.time() - start_time):.1f} steps/s")

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "e2e_phase3_sb3")
    model.save(final_model_path)
    print(f"📦 最终模型已保存: {final_model_path}")

    # 关闭环境
    vec_env.close()
    eval_env_wrapper.close()

    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Phase 3: 端到端微调（Stable-Baselines3加速）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 3端到端微调说明:
  联合优化所有组件（GNN + World Model + Controller）
  使用较小的学习率进行微调，防止破坏预训练权重

性能优化:
  --envs 4           标准配置（4个并行环境）
  --envs 16          高性能配置（16个并行环境）
  --envs 32          极限性能配置（32个并行环境）

配置文件选项:
  --config configs/windows_base.yaml
  --config configs/windows_high_performance.yaml
  --config configs/windows_extreme_performance.yaml

示例:
  # 标准训练
  python train_sb3_phase3.py --config configs/windows_base.yaml

  # 指定Phase 2检查点
  python train_sb3_phase3.py --config configs/windows_base.yaml --phase2-checkpoint checkpoints/ppo_phase2.pth

  # 高性能训练
  python train_sb3_phase3.py --config configs/windows_high_performance.yaml --envs 16 --timesteps 100000
        """
    )

    # 配置文件
    parser.add_argument("--config", type=str, default="configs/windows_base.yaml",
                       help="配置文件路径 (默认: configs/windows_base.yaml)")

    # Phase 2检查点
    parser.add_argument("--phase2-checkpoint", type=str, default=None,
                       help="Phase 2检查点路径 (默认从配置读取)")

    # 性能参数
    parser.add_argument("--envs", type=int, default=None,
                       help="并行环境数量 (覆盖配置文件)")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="总训练步数 (覆盖配置文件)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="PPO批次大小 (覆盖配置文件)")
    parser.add_argument("--n-steps", type=int, default=None,
                       help="每次rollout的步数 (覆盖配置文件)")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="学习率 (覆盖配置文件)")

    # 其他参数
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_sb3_phase3",
                       help="检查点目录 (默认: checkpoints_sb3_phase3)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu", "auto"], default=None,
                       help="训练设备 (覆盖配置文件)")
    parser.add_argument("--seed", type=int, default=None,
                       help="随机种子 (覆盖配置文件)")
    parser.add_argument("--freeze-bn", action='store_true', default=True,
                       help="冻结BatchNorm层 (默认: True)")
    parser.add_argument("--no-freeze-bn", dest='freeze_bn', action='store_false',
                       help="不冻结BatchNorm层")

    args = parser.parse_args()

    # 加载配置
    print(f"📋 加载配置: {args.config}")
    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.envs is not None:
        config.setdefault('training', {}).setdefault('phase3', {})['num_envs'] = args.envs
        print(f"   - 命令行覆盖: num_envs = {args.envs}")

    if args.timesteps is not None:
        config.setdefault('training', {}).setdefault('phase3', {})['total_timesteps'] = args.timesteps
        print(f"   - 命令行覆盖: total_timesteps = {args.timesteps}")

    if args.batch_size is not None:
        config.setdefault('training', {}).setdefault('phase3', {})['batch_size'] = args.batch_size
        print(f"   - 命令行覆盖: batch_size = {args.batch_size}")

    if args.n_steps is not None:
        config.setdefault('training', {}).setdefault('phase3', {})['n_steps'] = args.n_steps
        print(f"   - 命令行覆盖: n_steps = {args.n_steps}")

    if args.learning_rate is not None:
        config.setdefault('training', {}).setdefault('phase3', {})['learning_rate'] = args.learning_rate
        print(f"   - 命令行覆盖: learning_rate = {args.learning_rate}")

    if args.device is not None:
        config['device'] = args.device
        print(f"   - 命令行覆盖: device = {args.device}")

    if args.seed is not None:
        config['seed'] = args.seed
        print(f"   - 命令行覆盖: seed = {args.seed}")

    print()

    # 从配置文件获取默认值
    training_config = config.get('training', {}).get('phase3', {})

    # 确定最终使用的参数值
    final_timesteps = args.timesteps if args.timesteps is not None else training_config.get('total_timesteps', 50000)
    final_num_envs = args.envs if args.envs is not None else training_config.get('num_envs', 4)

    # 训练
    train_phase3_sb3(
        config=config,
        phase2_checkpoint=args.phase2_checkpoint,
        total_timesteps=final_timesteps,
        num_envs=final_num_envs,
        checkpoint_dir=args.checkpoint_dir,
        freeze_bn=args.freeze_bn
    )

    print("\n🎉 Phase 3训练完成！")
    print(f"📊 TensorBoard日志: ./logs_sb3_phase3/tensorboard/")
    print(f"💾 模型检查点: {args.checkpoint_dir}")
    print("\n使用以下命令查看训练曲线:")
    print(f"   tensorboard --logdir=./logs_sb3_phase3/tensorboard/")


if __name__ == "__main__":
    main()
