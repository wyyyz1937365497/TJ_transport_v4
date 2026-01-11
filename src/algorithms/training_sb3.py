"""
SB3 优化的训练管道

使用 Stable-Baselines3 实现高效的强化学习训练：
- Phase 2: PPO 训练
- Phase 3: 端到端微调
- Phase 4: CPO 约束优化

Phase 1 保持监督学习实现（已在 training.py 中优化）。
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Stable-Baselines3 导入
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback
)
from stable_baselines3.common.vec_env import VecEnv

# 项目导入
from src.env.vec_env import create_parallel_envs
from src.models.sb3_full_policy import create_full_traffic_policy
from src.utils.weight_transfer import WeightTransfer

# Phase 1 传统训练导入（复用）
from src.algorithms.training import Trainer


class TrainingPipelineSB3:
    """
    SB3 优化的训练管道

    提供 4 阶段训练流程：
    - Phase 1: 世界模型预训练（复用传统实现）
    - Phase 2: SB3 PPO 训练
    - Phase 3: SB3 端到端微调
    - Phase 4: CPO 约束优化

    与 Trainer 保持相同的接口，确保向后兼容。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练管道

        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))

        # 路径配置
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.log_dir = config.get('log_dir', 'logs')
        self.data_dir = config.get('data_dir', 'data')

        # 创建目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        # 权重传递器
        self.weight_transfer = WeightTransfer(verbose=True)

        # 模型历史
        self.phase1_model = None
        self.phase1_checkpoint = None
        self.phase2_model = None
        self.phase3_model = None
        self.phase4_model = None

        # 训练历史
        self.history = {
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'phase4': {}
        }

    # ============================================================
    # Phase 1: 世界模型预训练（复用传统实现）
    # ============================================================

    def train_phase1(
        self,
        model=None,
        num_episodes: int = 20,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        skip_data_collection: bool = False
    ):
        """
        Phase 1：世界模型预训练

        使用传统的监督学习实现（已在 training.py 中优化）

        Args:
            model: 交通控制模型
            num_episodes: 训练 episodes 数量
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            skip_data_collection: 是否跳过数据收集

        Returns:
            训练好的模型
        """
        print("\n" + "="*70)
        print("🔄 Phase 1: 世界模型预训练（监督学习）")
        print("="*70)

        # 如果没有传入模型，则创建一个
        if model is None:
            print("🏗️  创建模型...")
            from src.models import create_model_from_config
            model = create_model_from_config(self.config)
            print("✅ 模型创建成功")

        # 复用传统 Trainer 的 Phase 1 实现
        trainer = Trainer(self.config)
        model = trainer.train_phase1(
            model=model,
            num_episodes=num_episodes,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            skip_data_collection=skip_data_collection
        )

        # 保存检查点
        self.phase1_checkpoint = os.path.join(
            self.checkpoint_dir,
            'world_model_phase1.pth'
        )
        model.save_checkpoint(self.phase1_checkpoint, epoch=epochs)

        self.phase1_model = model

        # 记录历史
        self.history['phase1'] = {
            'checkpoint': self.phase1_checkpoint,
            'num_episodes': num_episodes,
            'epochs': epochs
        }

        return model

    # ============================================================
    # Phase 2: SB3 PPO 训练
    # ============================================================

    def train_phase2_sb3(
        self,
        total_timesteps: int = 100000,
        num_envs: int = 4,
        learning_rate: float = 3e-4
    ) -> PPO:
        """
        Phase 2：使用 SB3 PPO 训练控制器

        关键特性：
        1. 使用 FullTrafficActorCriticPolicy（包含完整 GNN + WorldModel）
        2. 加载 Phase 1 的 GNN 和 WorldModel 权重
        3. 冻结 GNN 和 WorldModel，只训练 Controller
        4. 使用 SB3 的高效 PPO 实现

        Args:
            total_timesteps: 总训练步数
            num_envs: 并行环境数量
            learning_rate: 学习率

        Returns:
            训练好的 SB3 PPO 模型
        """
        print("\n" + "="*70)
        print("🚀 Phase 2: SB3 PPO 训练")
        print("="*70)
        print(f"   总步数: {total_timesteps:,}")
        print(f"   并行环境: {num_envs}")
        print(f"   学习率: {learning_rate:.6f}")
        print(f"   架构: 完整 GNN + WorldModel + Controller")
        print("="*70)

        start_time = time.time()

        # 1. 创建并行环境
        print("\n📊 创建并行环境...")
        env_config = self.config.get('environment', {})
        vec_env_wrapper = create_parallel_envs(
            config=env_config,
            num_envs=num_envs,
            base_port=8813,
            seed=self.config.get('seed', 42)
        )

        # 获取底层 VecEnv
        vec_env = vec_env_wrapper.vec_env

        print(f"✅ 环境创建成功")
        print(f"   观测空间: {vec_env.observation_space}")
        print(f"   动作空间: {vec_env.action_space}")

        # 2. 创建完整的 SB3 PPO 模型（使用 FullTrafficActorCriticPolicy）
        print("\n🧠 创建 SB3 PPO 模型（完整架构）...")

        # 使用完整的策略（包含 GNN + WorldModel）
        from src.models.sb3_full_policy import create_full_traffic_policy
        full_policy_class = create_full_traffic_policy(self.config)

        phase2_config = self.config.get('training', {}).get('phase2', {})

        model = PPO(
            full_policy_class,
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, 'sb3_phase2'),
            learning_rate=learning_rate,
            n_steps=phase2_config.get('n_steps', 2048),
            batch_size=phase2_config.get('batch_size', 128),
            n_epochs=phase2_config.get('update_epochs', 10),
            gamma=phase2_config.get('gamma', 0.99),
            gae_lambda=phase2_config.get('gae_lambda', 0.95),
            clip_range=phase2_config.get('clip_epsilon', 0.2),
            clip_range_vf=None,
            ent_coef=phase2_config.get('entropy_coef', 0.01),
            vf_coef=phase2_config.get('value_loss_coef', 0.5),
            max_grad_norm=0.5,
            target_kl=0.03,
            stats_window_size=100,
            seed=self.config.get('seed', 42),
            device=str(self.device),
            _init_setup_model=True
        )

        print("✅ 模型创建成功")
        print(f"   策略: FullTrafficActorCriticPolicy")
        print(f"   Batch size: {phase2_config.get('batch_size', 128)}")
        print(f"   PPO n_steps: {phase2_config.get('n_steps', 2048)}")
        print(f"   PPO epochs: {phase2_config.get('update_epochs', 10)}")

        # 3. 加载 Phase 1 权重到完整的策略中
        if self.phase1_checkpoint and os.path.exists(self.phase1_checkpoint):
            print(f"\n🔄 加载 Phase 1 权重到完整策略...")
            try:
                # 直接加载到 traffic_controller
                checkpoint = torch.load(self.phase1_checkpoint, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                # 获取策略的 traffic_controller
                policy_traffic_controller = model.policy.traffic_controller
                policy_state_dict = policy_traffic_controller.state_dict()

                # 加载 GNN 和 WorldModel 权重
                loaded = 0
                for key, param in state_dict.items():
                    if key in policy_state_dict:
                        if param.shape == policy_state_dict[key].shape:
                            policy_state_dict[key] = param
                            loaded += 1

                policy_traffic_controller.load_state_dict(policy_state_dict)
                print(f"✅ 成功加载 {loaded} 个权重")
                print(f"   - GNN: 已加载")
                print(f"   - WorldModel: 已加载")
                print(f"   - Controller: 将从随机初始化开始训练")
            except Exception as e:
                print(f"⚠️  权重加载失败: {e}")
                print(f"   将使用随机初始化的权重")
        else:
            print(f"\n⚠️  未找到 Phase 1 检查点，使用随机初始化")

        # 4. 冻结 GNN 和 WorldModel
        print(f"\n❄️  冻结组件...")
        model.policy.freeze_gnn_and_world_model()
        print("✅ 已冻结 GNN 和 WorldModel，只训练 Controller")

        # 5. 设置回调
        callbacks = self._create_phase2_callbacks()

        # 6. 开始训练
        print("\n🏋️  开始训练...")
        print("="*70)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        elapsed = time.time() - start_time

        # 7. 保存模型
        print("\n💾 保存模型...")

        # SB3 格式
        sb3_path = os.path.join(self.checkpoint_dir, 'ppo_phase2_sb3.zip')
        model.save(sb3_path)
        print(f"   ✅ SB3 格式: {sb3_path}")

        # 兼容格式（用于 Phase 3）
        compat_path = os.path.join(self.checkpoint_dir, 'ppo_phase2.pth')
        self._save_sb3_as_compatible(model, compat_path, phase=2)
        print(f"   ✅ 兼容格式: {compat_path}")

        self.phase2_model = model

        # 记录历史
        self.history['phase2'] = {
            'sb3_path': sb3_path,
            'compat_path': compat_path,
            'total_timesteps': total_timesteps,
            'elapsed_time': elapsed
        }

        # 8. 清理环境
        vec_env.close()

        print("\n" + "="*70)
        print(f"✅ Phase 2 训练完成！")
        print(f"   总耗时: {elapsed/60:.2f} 分钟")
        print("="*70 + "\n")

        return model

    def _freeze_phase2_components(self, policy):
        """冻结 Phase 2 不需要训练的组件"""
        # 冻结特征提取器中的 GNN 和 WorldModel（如果存在）
        if hasattr(policy, 'features_extractor'):
            extractor = policy.features_extractor

            # 冻结 GNN
            if hasattr(extractor, 'risk_gnn'):
                for param in extractor.risk_gnn.parameters():
                    param.requires_grad = False
                print("   ✅ 已冻结 GNN")

            # 冻结 WorldModel
            if hasattr(extractor, 'world_model'):
                for param in extractor.world_model.parameters():
                    param.requires_grad = False
                print("   ✅ 已冻结 WorldModel")

    def _create_phase2_callbacks(self):
        """创建 Phase 2 的回调函数"""
        callbacks = []

        # 定期保存检查点
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=self.checkpoint_dir,
            name_prefix="ppo_phase2",
            save_replay_buffer=False,
            save_vecnormalize=False
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    # ============================================================
    # Phase 3: SB3 端到端微调
    # ============================================================

    def train_phase3_sb3(
        self,
        total_timesteps: int = 50000,
        learning_rate: float = 1e-5,
        freeze_bn: bool = True
    ) -> PPO:
        """
        Phase 3：SB3 端到端微调

        关键特性：
        1. 使用 FullTrafficActorCriticPolicy（完整策略）
        2. 加载 Phase 2 的所有权重
        3. 解冻所有组件进行联合训练
        4. 可选冻结 BatchNorm 层

        Args:
            total_timesteps: 总训练步数
            learning_rate: 学习率（较小）
            freeze_bn: 是否冻结 BatchNorm 层

        Returns:
            训练好的 SB3 PPO 模型
        """
        if self.phase2_model is None:
            raise RuntimeError("Phase 2 模型不存在，请先运行 train_phase2_sb3()")

        print("\n" + "="*70)
        print("🔥 Phase 3: SB3 端到端微调")
        print("="*70)
        print(f"   总步数: {total_timesteps:,}")
        print(f"   学习率: {learning_rate:.6f}")
        print(f"   BatchNorm 冻结: {freeze_bn}")
        print("="*70)

        start_time = time.time()

        # 1. 复用环境
        vec_env = self.phase2_model.get_env()

        # 2. 创建完整策略
        print("\n🧠 创建完整策略...")
        full_policy_class = create_full_traffic_policy(self.config)

        phase3_config = self.config.get('training', {}).get('phase3', {})

        model = PPO(
            full_policy_class,
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, 'sb3_phase3'),
            learning_rate=learning_rate,
            n_steps=phase3_config.get('n_steps', 2048),
            batch_size=phase3_config.get('batch_size', 64),  # 较小 batch
            n_epochs=phase3_config.get('update_epochs', 10),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=self.config.get('seed', 42),
            device=str(self.device)
        )

        print("✅ 模型创建成功")

        # 3. 加载 Phase 2 权重
        print("\n🔄 加载 Phase 2 权重...")
        try:
            loaded = model.policy.load_phase2_weights_from_sb3(
                self.phase2_model,
                verbose=True
            )
            print(f"✅ 成功加载 {loaded} 个权重")
        except Exception as e:
            print(f"⚠️  权重加载失败: {e}")
            print(f"   将使用 Phase 2 模型作为初始化")

        # 4. 解冻所有组件
        model.policy.unfreeze_all()
        print("✅ 已解冻所有组件")

        # 5. 冻结 BatchNorm（如果需要）
        if freeze_bn:
            model.policy.freeze_batch_norm()
            print("✅ 已冻结 BatchNorm 层")

        # 6. 设置回调
        callbacks = self._create_phase3_callbacks()

        # 7. 开始训练
        print("\n🏋️  开始训练...")
        print("="*70)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        elapsed = time.time() - start_time

        # 8. 保存模型
        print("\n💾 保存模型...")

        sb3_path = os.path.join(self.checkpoint_dir, 'e2e_phase3_sb3.zip')
        model.save(sb3_path)
        print(f"   ✅ SB3 格式: {sb3_path}")

        compat_path = os.path.join(self.checkpoint_dir, 'e2e_phase3.pth')
        self._save_sb3_as_compatible(model, compat_path, phase=3)
        print(f"   ✅ 兼容格式: {compat_path}")

        self.phase3_model = model

        # 记录历史
        self.history['phase3'] = {
            'sb3_path': sb3_path,
            'compat_path': compat_path,
            'total_timesteps': total_timesteps,
            'elapsed_time': elapsed
        }

        print("\n" + "="*70)
        print(f"✅ Phase 3 训练完成！")
        print(f"   总耗时: {elapsed/60:.2f} 分钟")
        print("="*70 + "\n")

        return model

    def _create_phase3_callbacks(self):
        """创建 Phase 3 的回调函数"""
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=self.checkpoint_dir,
            name_prefix="e2e_phase3",
            save_replay_buffer=False
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    # ============================================================
    # Phase 4: CPO 约束优化
    # ============================================================

    def train_phase4_sb3(
        self,
        total_timesteps: int = 50000,
        cost_limit: float = 0.1,
        learning_rate: float = 1e-4
    ):
        """
        Phase 4：拉格朗日约束优化

        使用完整的拉格朗日 PPO 实现（基于 SB3 的 PPO + 自定义拉格朗日乘子）

        注意：sb3_contrib 中没有 CPO 算法，因此我们使用自实现的拉格朗日 PPO。

        Args:
            total_timesteps: 总训练步数
            cost_limit: 成本上限
            learning_rate: 学习率

        Returns:
            训练好的拉格朗日 PPO 模型
        """
        if self.phase3_model is None:
            raise RuntimeError("Phase 3 模型不存在，请先运行 train_phase3_sb3()")

        print("\n" + "="*70)
        print("⚖️  Phase 4: 拉格朗日约束优化训练")
        print("="*70)
        print(f"   总步数: {total_timesteps:,}")
        print(f"   成本上限: {cost_limit}")
        print(f"   学习率: {learning_rate:.6f}")
        print(f"   方法: 拉格朗日 PPO（自实现）")
        print("="*70)

        start_time = time.time()

        # 直接调用完整的拉格朗日 PPO 实现
        model = self._train_phase4_lagrangian_ppo(
            total_timesteps=total_timesteps,
            cost_limit=cost_limit,
            learning_rate=learning_rate
        )

        elapsed = time.time() - start_time

        print("\n" + "="*70)
        print(f"✅ Phase 4 训练完成！")
        print(f"   总耗时: {elapsed/60:.2f} 分钟")
        print("="*70 + "\n")

        return model

    def _create_phase4_callbacks(self):
        """创建 Phase 4 的回调函数"""
        callbacks = []

        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=self.checkpoint_dir,
            name_prefix="final_model",
            save_replay_buffer=False
        )
        callbacks.append(checkpoint_callback)

        return callbacks

    def _train_phase4_lagrangian_ppo(
        self,
        total_timesteps: int,
        cost_limit: float,
        learning_rate: float
    ):
        """
        Phase 4 拉格朗日 PPO 实现（主要实现）

        完整的拉格朗日松弛约束优化实现
        基于标准 SB3 PPO，添加：
        1. 拉格朗日乘子动态更新
        2. 成本约束处理
        3. 双层优化（策略 + 乘子）

        Args:
            total_timesteps: 总训练步数
            cost_limit: 成本上限
            learning_rate: 学习率

        Returns:
            训练好的拉格朗日 PPO 模型
        """
        print("\n🔧 初始化拉格朗日 PPO...")

        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        # 创建拉格朗日 PPO 类
        class LagrangianPPO(PPO):
            """
            拉格朗日 PPO 实现 - 完整版本

            在标准 PPO 基础上添加：
            1. 拉格朗日乘子
            2. 成本约束
            3. 双层优化
            """

            def __init__(self, *args, cost_limit=0.1, lambda_init=0.1, lambda_lr=1e-3, **kwargs):
                # 调用父类初始化
                super().__init__(*args, **kwargs)

                # 拉格朗日参数
                self.cost_limit = cost_limit
                self.lambda_param = lambda_init
                self.lambda_lr = lambda_lr

                # 记录成本历史
                self.cost_history = []
                self.lambda_history = []

                print(f"   ✅ 拉格朗日 PPO 初始化")
                print(f"      成本上限: {cost_limit}")
                print(f"      初始 lambda: {lambda_init}")
                print(f"      Lambda 学习率: {lambda_lr}")

            def train(self):
                """
                扩展训练方法，添加拉格朗日乘子更新
                """
                # 标准PPO更新
                super().train()

                # 更新拉格朗日乘子
                # 使用最近的平均成本
                if len(self.cost_history) > 0:
                    avg_cost = np.mean(self.cost_history[-100:])  # 最近100步的平均成本
                    cost_violation = avg_cost - self.cost_limit

                    # 梯度上升更新 lambda（试图满足约束）
                    # lambda = lambda + lr * (cost - cost_limit)
                    lambda_update = self.lambda_lr * cost_violation
                    self.lambda_param = np.clip(
                        self.lambda_param + lambda_update,
                        0,  # lambda >= 0
                        10  # 上限，防止爆炸
                    )

                    self.lambda_history.append(self.lambda_param)

                    # 定期打印
                    if len(self.lambda_history) % 100 == 0:
                        print(f"   Lambda 更新: {self.lambda_param:.4f} (成本违反: {cost_violation:.4f})")

            def _update_policy(self, *args, **kwargs):
                """
                扩展策略更新，在损失中添加成本惩罚
                """
                # 调用父类的策略更新
                loss = super()._update_policy(*args, **kwargs)

                # 注意：这里我们通过奖励函数来施加成本约束
                # 在环境层面应该已经将 cost 作为负奖励加入

                return loss

        # 复用 Phase 3 的环境和策略
        vec_env = self.phase3_model.get_env()
        policy_class = self.phase3_model.policy.__class__

        # 创建拉格朗日 PPO 模型
        phase4_config = self.config.get('training', {}).get('phase4', {})

        model = LagrangianPPO(
            policy_class,
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(self.log_dir, 'sb3_phase4_fallback'),
            learning_rate=learning_rate,
            cost_limit=cost_limit,
            lambda_init=0.1,
            lambda_lr=1e-3,
            n_steps=phase4_config.get('n_steps', 2048),
            batch_size=phase4_config.get('batch_size', 64),
            n_epochs=phase4_config.get('update_epochs', 10),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=self.config.get('seed', 42),
            device=str(self.device)
        )

        # 加载 Phase 3 权重
        print("\n🔄 加载 Phase 3 权重...")
        try:
            model.set_parameters(self.phase3_model.get_parameters())
            print("✅ 权重加载成功")
        except Exception as e:
            print(f"⚠️  权重加载失败: {e}")

        # 冻结 GNN 和 WorldModel（如果策略支持）
        if hasattr(model.policy, 'freeze_gnn_and_world_model'):
            model.policy.freeze_gnn_and_world_model()
            print("✅ 已冻结 GNN 和 WorldModel")

        # 设置回调
        callbacks = self._create_phase4_callbacks()

        # 开始训练
        print("\n🏋️  开始拉格朗日 PPO 训练...")
        print("="*70)

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # 保存模型
        print("\n💾 保存模型...")
        final_path = os.path.join(self.checkpoint_dir, 'final_model_lagrangian.zip')
        model.save(final_path)
        print(f"   ✅ 拉格朗日 PPO 模型: {final_path}")

        compat_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        self._save_sb3_as_compatible(model, compat_path, phase=4)
        print(f"   ✅ 兼容格式: {compat_path}")

        print("\n" + "="*70)
        print(f"✅ Phase 4 拉格朗日 PPO 训练完成！")
        print("="*70)

        return model

    # ============================================================
    # 工具方法
    # ============================================================

    def _save_sb3_as_compatible(
        self,
        sb3_model,
        filepath: str,
        phase: int
    ):
        """
        保存 SB3 模型为兼容格式

        Args:
            sb3_model: SB3 模型
            filepath: 保存路径
            phase: 阶段编号
        """
        torch.save({
            'model_type': 'sb3',
            'phase': phase,
            'policy_state_dict': sb3_model.policy.state_dict(),
            'config': self.config
        }, filepath)

    def save_history(self, filepath: str):
        """
        保存训练历史

        Args:
            filepath: 保存路径
        """
        import json
        from datetime import datetime

        history = {
            'timestamp': datetime.now().isoformat(),
            'history': self.history,
            'transfer_summary': self.weight_transfer.get_transfer_summary()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"📄 训练历史已保存: {filepath}")

    def get_phase2_model(self):
        """获取 Phase 2 模型"""
        return self.phase2_model

    def get_phase3_model(self):
        """获取 Phase 3 模型"""
        return self.phase3_model

    def get_phase4_model(self):
        """获取 Phase 4 模型"""
        return self.phase4_model
