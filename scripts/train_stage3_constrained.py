#!/usr/bin/env python3
"""
Stage 3: 约束优化训练（Constrained Optimization Training）

功能：
1. 加载Stage 2训练好的策略
2. 引入CostCritic进行成本预测
3. 激活DynamicWeightGate进行动态权重调整
4. 拉格朗日松弛优化

使用方法：
    python scripts/train_stage3_constrained.py --config configs/v5_complete.yaml --resume checkpoints/v5_complete/stage2_best.pth

输出：
    - checkpoints/v5_complete/stage3_constrained.pth
    - logs/v5_complete/stage3/
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.models.cost_critic import CostCritic, create_cost_critic, LagrangianOptimizer
from src.models.dynamic_weight_gate import DynamicWeightGate, create_dynamic_weight_gate
from src.env.competition_env import CompetitionSumoEnv
from scripts.train_stage2_guided_exploration import PPOTrainer


class ConstrainedPPOTrainer(PPOTrainer):
    """
    约束PPO训练器（Stage 3）

    继承自Stage 2的PPOTrainer，添加成本约束优化
    """

    def __init__(self, policy, env, config, device='cuda',
                 use_cache=True, force_refresh=False, cache_dir=None):
        """
        Args:
            policy: JointICVPolicy
            env: CompetitionSumoEnv
            config: 配置字典
            device: 设备
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新缓存
            cache_dir: 缓存目录
        """
        super().__init__(policy, env, config, device,
                        use_cache=use_cache,
                        force_refresh=force_refresh,
                        cache_dir=cache_dir)

        # Stage 3特定配置
        self.use_cost_critic = config['stage3_constrained_optimization']['ppo']['use_cost_critic']
        self.use_dynamic_gate = config['stage3_constrained_optimization']['ppo']['use_dynamic_gate']

        # CostCritic
        if self.use_cost_critic:
            self.cost_critic = create_cost_critic(
                hidden_dim=config['policy']['hidden_dim'],
                global_dim=64,
                device=device
            )
            print(f"CostCritic参数量: {sum(p.numel() for p in self.cost_critic.parameters()):,}")

            # 优化器
            self.cost_optimizer = optim.Adam(
                self.cost_critic.parameters(),
                lr=3e-4,
                weight_decay=1e-5
            )

        # DynamicWeightGate
        if self.use_dynamic_gate:
            self.dynamic_gate = create_dynamic_weight_gate(
                global_dim=64,
                hidden_dim=32,
                device=device
            )
            print(f"DynamicWeightGate参数量: {sum(p.numel() for p in self.dynamic_gate.parameters()):,}")

        # 拉格朗日优化器
        lagrangian_config = config['stage3_constrained_optimization']['lagrangian']
        self.lagrangian = LagrangianOptimizer(
            initial_lambda=lagrangian_config['initial_lambda'],
            lambda_lr=lagrangian_config['lambda_lr'],
            cost_threshold=lagrangian_config['cost_threshold']
        )

        # 损失权重
        self.loss_weights = config['stage3_constrained_optimization']['ppo'].get('loss_weights', {})
        self.cost_loss_weight = self.loss_weights.get('cost', 0.5)

    def collect_rollouts(self, num_steps, cache_dir=None, use_cache=True, force_refresh=False):
        """
        收集rollout数据（扩展：包含成本，支持缓存）

        注意: 对于PPO训练，缓存效果有限，因为策略会不断更新。
        缓存主要用于调试和重复实验。

        Args:
            num_steps: 收集的步数
            cache_dir: 缓存目录路径
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新缓存

        Returns:
            rollouts: Dict
        """
        import hashlib
        import pickle
        from pathlib import Path

        # 生成缓存key
        if cache_dir is None:
            cache_dir = Path("cache/stage3_rollouts")
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        # 创建缓存key（基于参数）
        cache_params = {
            'num_steps': num_steps,
            'stage': 'stage3',
        }
        cache_key = hashlib.md5(str(cache_params).encode()).hexdigest()[:12]
        cache_file = cache_dir / f"rollouts_{cache_key}.pkl"

        # 尝试从缓存加载
        if use_cache and not force_refresh and cache_file.exists():
            print(f"[缓存] 发现rollout缓存: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_rollouts = pickle.load(f)

                # 验证缓存数据
                if len(cached_rollouts['observations']) == num_steps:
                    print(f"[缓存] 成功加载rollout（{num_steps}步）")
                    return cached_rollouts
                else:
                    print(f"[缓存] rollout长度不匹配，将重新收集")
            except Exception as e:
                print(f"[缓存] 加载失败: {e}，将重新收集")

        # 缓存未命中，进行rollout收集
        self.policy.eval()
        rollouts = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'costs': [],  # Stage 3新增：成本
            'dones': []
        }

        obs, _ = self.env.reset()
        episode_rewards = []
        episode_costs = []

        for step in tqdm(range(num_steps), desc="收集rollouts（含成本）"):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.policy(obs_tensor, deterministic=False)

                # 计算成本（如果启用CostCritic）
                cost_value = torch.zeros(1, device=self.device)
                if self.use_cost_critic and 'cost_value' in outputs:
                    cost_value = outputs['cost_value']

            action = outputs['actions'][0].cpu().numpy()
            value = outputs['value'][0, 0].cpu().item()
            log_prob = outputs['log_prob'][0].cpu().item()

            # 执行动作
            next_obs, reward, done, truncated, info = self.env.step(action)

            # 计算干预成本
            cost = self.compute_intervention_cost(action, obs, next_obs)

            # 存储到buffer
            rollouts['observations'].append(obs.copy())
            rollouts['actions'].append(action.copy())
            rollouts['log_probs'].append(log_prob)
            rollouts['rewards'].append(reward)
            rollouts['values'].append(value)
            rollouts['costs'].append(cost + cost_value.cpu().item())  # 基础成本 + 预测成本
            rollouts['dones'].append(done or truncated)

            obs = next_obs

            if done or truncated:
                obs, _ = self.env.reset()

        # 转换为numpy数组
        for key in rollouts.keys():
            rollouts[key] = np.array(rollouts[key])

        # 保存到缓存
        if use_cache or force_refresh:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(rollouts, f)
                cache_size_kb = cache_file.stat().st_size / 1024
                print(f"[缓存] Rollout已缓存: {cache_file} ({cache_size_kb:.1f} KB)")
            except Exception as e:
                print(f"[缓存] 保存失败: {e}")

        return rollouts

    def compute_intervention_cost(self, actions, obs, next_obs):
        """
        计算干预成本

        Args:
            actions: [N*2]
            obs: [obs_dim]
            next_obs: [obs_dim]

        Returns:
            cost: float
        """
        # 简化：每控制一辆车成本为0.01
        obs_dim = obs.shape[0]
        vehicle_dim = (obs_dim - 32 - 1)
        num_vehicles = vehicle_dim // 9

        # 计算控制的车辆数（动作不为0）
        actions_reshaped = actions.reshape(num_vehicles, 2)
        num_controlled = (np.abs(actions_reshaped[:, 0]) > 0.01).sum()

        # 成本 = 每车成本 * 控制车辆数
        cost = 0.01 * num_controlled

        return cost

    def update_policy(self, rollouts):
        """
        更新策略（约束优化版本）

        Args:
            rollouts: rollout数据

        Returns:
            metrics: Dict
        """
        observations = rollouts['observations']
        actions = rollouts['actions']
        old_log_probs = rollouts['log_probs']
        rewards = rollouts['rewards']
        values = rollouts['values']
        costs = rollouts['costs']
        dones = rollouts['dones']

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones)

        # 转换为tensor
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32).to(self.device)
        costs_tensor = torch.as_tensor(costs, dtype=torch.float32).to(self.device)

        # Minibatch更新
        batch_size = observations.shape[0]
        minibatch_size = batch_size // self.num_minibatches

        total_policy_loss = 0
        total_value_loss = 0
        total_cost_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(self.update_epochs):
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                # 获取minibatch数据
                mb_obs = obs_tensor[mb_indices]
                mb_actions = actions_tensor[mb_indices]
                mb_old_log_probs = old_log_probs_tensor[mb_indices]
                mb_advantages = advantages_tensor[mb_indices]
                mb_returns = returns_tensor[mb_indices]
                mb_costs = costs_tensor[mb_indices]

                # 重新计算log_prob和entropy
                self.policy.train()
                outputs = self.policy(mb_obs, deterministic=False)

                new_log_probs = outputs['log_prob']
                entropy = outputs['entropy']  # 使用正确的熵（高斯分布的熵）
                value = outputs['value']

                # 获取成本预测（如果有）
                if self.use_cost_critic and 'cost_value' in outputs:
                    pred_cost = outputs['cost_value'].squeeze(1)
                else:
                    pred_cost = mb_costs

                # 计算ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(value.squeeze(1), mb_returns)

                # Cost loss（Stage 3新增）
                cost_loss = nn.functional.mse_loss(pred_cost, mb_costs)

                # 拉格朗日损失
                lagrangian_loss = self.lagrangian.compute_lagrangian_loss(
                    torch.from_numpy(mb_returns).float(),
                    torch.from_numpy(mb_costs).float()
                )

                # 总损失
                loss = (policy_loss +
                       self.vf_coef * value_loss +
                       self.cost_loss_weight * cost_loss -
                       self.entropy_coef * entropy.mean())

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                                   self.config['stage3_constrained_optimization']['ppo']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 更新CostCritic（如果有）
                if self.use_cost_critic:
                    self.cost_optimizer.zero_grad()
                    cost_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), 1.0)
                    self.cost_optimizer.step()

                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_cost_loss += cost_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # 更新学习率
        self.scheduler.step()

        # 更新λ（拉格朗日乘数）
        mean_cost = costs.mean()
        lambda_update = self.lagrangian.update_lambda(mean_cost)

        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'cost_loss': total_cost_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_reward': rewards.mean(),
            'mean_cost': mean_cost,
            'lambda': lambda_update['new_lambda'],
            'constraint_violation': lambda_update['constraint_violation']
        }

        return metrics

    def train(self):
        """训练循环（Stage 3）"""
        print("\n" + "=" * 80)
        print("开始Stage 3 约束优化训练")
        print("=" * 80)

        best_reward = float('-inf')
        checkpoint_dir = Path(self.config['global']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(self.num_iterations):
            print(f"\n迭代 {iteration + 1}/{self.num_iterations}")

            # 收集rollouts
            rollouts = self.collect_rollouts(self.num_steps_per_iteration)

            # 更新策略
            metrics = self.update_policy(rollouts)

            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Cost Loss: {metrics['cost_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
            print(f"  Mean Cost: {metrics['mean_cost']:.4f}")
            print(f"  Lambda: {metrics['lambda']:.4f}")
            print(f"  Constraint Violation: {metrics['constraint_violation']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存检查点
            if (iteration + 1) % self.config['stage3_constrained_optimization']['checkpoint']['save_interval'] == 0:
                checkpoint_path = checkpoint_dir / f'stage3_iter_{iteration+1}.pth'
                torch.save(self.policy.state_dict(), checkpoint_path)
                print(f"  检查点已保存: {checkpoint_path}")

            # 保存最佳模型
            if metrics['mean_reward'] > best_reward:
                best_reward = metrics['mean_reward']
                best_checkpoint = checkpoint_dir / 'stage3_best.pth'
                torch.save(self.policy.state_dict(), best_checkpoint)
                print(f"  ✅ 最佳模型已保存: {best_checkpoint}")

        print("\n" + "=" * 80)
        print("🎉 Stage 3 训练完成！")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Constrained Optimization Training")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复检查点路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    # 缓存相关参数
    parser.add_argument('--use_cache', type=lambda x: x.lower() == 'true', default=True,
                        help='是否使用缓存 (True/False, 默认: True)')
    parser.add_argument('--force_refresh', action='store_true',
                        help='强制刷新缓存，重新收集数据')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='自定义缓存目录路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Stage 3: 约束优化训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"设备: {args.device}")

    if args.resume:
        print(f"恢复检查点: {args.resume}")

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

    # 加载Stage 2权重
    if args.resume:
        print(f"\n加载Stage 2策略权重...")
        checkpoint = torch.load(args.resume, map_location=args.device)
        policy.load_state_dict(checkpoint)
        print(f"  ✅ 策略权重已加载")

    # 启用Stage 3组件
    print("\n启用Stage 3组件...")

    # CostCritic
    if config['stage3_constrained_optimization']['cost_critic']['enabled']:
        policy.enable_cost_critic(global_dim=64)
        print(f"  ✅ CostCritic已启用")

    # DynamicWeightGate
    if config['stage3_constrained_optimization']['dynamic_gate']['enabled']:
        policy.enable_dynamic_gate(global_dim=64)
        print(f"  ✅ DynamicWeightGate已启用")

    print(f"\n策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 创建训练器
    trainer = ConstrainedPPOTrainer(
        policy, env, config, args.device,
        use_cache=args.use_cache,
        force_refresh=args.force_refresh,
        cache_dir=args.cache_dir
    )

    # 训练
    start_time = time.time()
    trainer.train()
    elapsed_time = time.time() - start_time

    print(f"\n训练完成！")
    print(f"  训练时长: {elapsed_time/3600:.2f} 小时")

    # 保存最终模型
    final_checkpoint = Path(config['global']['checkpoint_dir']) / 'stage3_final.pth'
    torch.save(policy.state_dict(), final_checkpoint)
    print(f"  最终模型已保存: {final_checkpoint}")


if __name__ == '__main__':
    main()
