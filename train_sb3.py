"""
基于Stable-Baselines3的训练脚本
使用SubprocVecEnv解决SUMO多进程并行问题
"""

import os
import sys
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

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.env.vec_env import create_parallel_envs
from src.models.sb3_policy import create_custom_policy


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def train_sb3(
    config: Dict[str, Any],
    total_timesteps: int = 100000,
    num_envs: int = 4,
    checkpoint_dir: str = "checkpoints_sb3"
):
    """
    使用Stable-Baselines3的PPO训练

    Args:
        config: 配置字典
        total_timesteps: 总训练步数
        num_envs: 并行环境数量
        checkpoint_dir: 检查点目录
    """
    print("="*70)
    print("🚀 Stable-Baselines3 PPO训练")
    print("="*70)
    print(f"   - 总步数: {total_timesteps:,}")
    print(f"   - 并行环境: {num_envs}")
    print(f"   - 设备: {config.get('device', 'cuda')}")
    print(f"   - 检查点: {checkpoint_dir}")
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
    import os
    from pathlib import Path

    # 尝试找到可用的SUMO配置文件
    # 如果原配置文件不存在，使用SUMO测试网络
    needs_fallback = False
    for key in ['sumo_cfg', 'net_file', 'route_file']:
        if key in env_config:
            path = env_config[key]
            if not os.path.isabs(path):
                # 转换为绝对路径（解析..）
                abs_path = str((Path.cwd() / path).resolve())
                if not os.path.exists(abs_path):
                    needs_fallback = True
                    break
                env_config[key] = abs_path

    # 如果原配置不存在，使用SUMO测试网格
    if needs_fallback:
        print(f"\n⚠️  原始配置文件不存在，使用SUMO测试网络")
        print(f"   这是演示模式，实际训练时请提供正确的配置文件\n")

        # 创建临时配置使用SUMO自带的网格网络
        import tempfile
        import subprocess

        # 使用SUMO的网格生成工具
        tmpdir = tempfile.mkdtemp(prefix="sumo_test_")

        # 生成网格网络
        net_file = os.path.join(tmpdir, "grid.net.xml")
        subprocess.run([
            "netconvert",
            "--grid", "--grid.number=3",
            "--grid.length=500",
            "-o", net_file
        ], check=True, capture_output=True)

        # 生成简单的交通流
        route_file = os.path.join(tmpdir, "grid.rou.xml")
        with open(route_file, 'w') as f:
            f.write('''<routes>
    <vType id="car" accel="2.6" decel="4.5" length="5" minGap="2.5" maxSpeed="30" sigma="0.5"/>
    <flow id="flow1" type="car" begin="0" end="3600" vehsPerHour="500" from="0/0to" to="1/1to"/>
</routes>''')

        # 生成sumocfg
        sumo_cfg = os.path.join(tmpdir, "grid.sumocfg")
        with open(sumo_cfg, 'w') as f:
            f.write(f'''<configuration>
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(route_file)}"/>
    </input>
    <time>
        <step-length value="0.1"/>
    </time>
</configuration>''')

        env_config['sumo_cfg'] = sumo_cfg
        env_config['net_file'] = net_file
        env_config['route_file'] = route_file

        print(f"   - 使用测试网络: {tmpdir}")

    vec_env = create_parallel_envs(
        config=env_config,
        num_envs=num_envs,
        base_port=8813,
        seed=seed
    )

    print(f"✅ 环境创建成功")
    print(f"   - 观测空间: {vec_env.observation_space}")
    print(f"   - 动作空间: {vec_env.action_space}")

    # 直接使用底层的VecEnv
    vec_env = vec_env.vec_env

    # 创建自定义策略
    print("\n🧠 创建自定义PPO策略...")
    model_config = config.get('model', {})

    # 合并配置
    full_config = {
        'device': config.get('device', 'cuda'),
        **model_config,
        **env_config
    }

    # 创建模型
    policy_class = create_custom_policy(full_config)

    model = PPO(
        policy=policy_class,
        env=vec_env,
        verbose=1,
        tensorboard_log="./logs_sb3/tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        stats_window_size=100,
        seed=seed,
        device=config.get('device', 'cuda'),
        _init_setup_model=True
    )

    print("✅ 模型创建成功")
    print(f"   - 策略: {policy_class.__name__}")
    print(f"   - 学习率: 3e-4")
    print(f"   - 批次大小: 64")
    print(f"   - PPO epochs: 10")

    # 设置回调
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # 定期保存检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="ppo_sumo",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    # 早停回调（可选）
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=100.0,  # 根据实际情况调整
        verbose=1
    )

    # 评估回调
    eval_env_wrapper = create_parallel_envs(
        config=env_config,
        num_envs=1,  # 评估只用1个环境
        base_port=8900,
        seed=seed
    )

    eval_callback = EvalCallback(
        eval_env_wrapper.vec_env,  # 使用底层的VecEnv
        best_model_save_path=checkpoint_dir,
        log_path="./logs_sb3/eval/",
        eval_freq=5000,
        deterministic=False,
        render=False
    )

    print("\n🏋️ 开始训练...")
    print("="*70)

    # 训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "ppo_sumo_final")
    model.save(final_model_path)
    print(f"📦 最终模型已保存: {final_model_path}")

    # 关闭环境
    vec_env.close()
    eval_env.close()

    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用Stable-Baselines3训练SUMO交通控制")
    parser.add_argument("--config", type=str, default="wsl/config/base.yaml",
                       help="配置文件路径")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="总训练步数")
    parser.add_argument("--envs", type=int, default=4,
                       help="并行环境数量")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_sb3",
                       help="检查点目录")
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2", "phase3", "all"],
                       default="all", help="训练阶段")

    args = parser.parse_args()

    # 加载配置
    print(f"📋 加载配置: {args.config}")
    config = load_config(args.config)

    # 根据阶段训练
    if args.phase in ["phase1", "all"]:
        print("\n" + "="*70)
        print("🎯 阶段1: 世界模型预训练（通过PPO隐式学习）")
        print("="*70)
        # 注意：在这个简化版本中，我们通过端到端训练隐式地学习世界模型
        # 如果需要显式的世界模型预训练，可以使用原来的代码

    if args.phase in ["phase2", "all"]:
        print("\n" + "="*70)
        print("🎯 阶段2: PPO训练（主要训练阶段）")
        print("="*70)

        train_sb3(
            config=config,
            total_timesteps=args.timesteps,
            num_envs=args.envs,
            checkpoint_dir=args.checkpoint_dir
        )

    print("\n🎉 所有训练完成！")
    print(f"📊 TensorBoard日志: ./logs_sb3/tensorboard/")
    print(f"💾 模型检查点: {args.checkpoint_dir}")
    print("\n使用以下命令查看训练曲线:")
    print(f"   tensorboard --logdir=./logs_sb3/tensorboard/")


if __name__ == "__main__":
    main()
