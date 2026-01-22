#!/usr/bin/env python3
"""
Stage 2: 基于规则的ICV选择训练（Rule-Based ICV Selection Training）

功能：
1. 使用基于规则的ICV选择器（不训练选择器）
2. PPO训练只优化策略网络（加速度和换道）
3. 固定奖励权重

与标准Stage 2的区别：
- ICV选择使用规则而不是神经网络
- 策略网络不学习选择，只学习控制
- 用于验证规则选择器是否优于神经网络选择器

使用方法：
    python scripts/train_stage2_rule_based.py --config configs/v5_complete.yaml

输出：
    - checkpoints/v5_complete/stage2_rule_based.pth
    - logs/v5_complete/stage2_rule_based/
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.joint_icv_policy import create_joint_icv_policy
from src.models.world_model import WorldModel, create_world_model
from src.env.competition_env import CompetitionSumoEnv
from src.training.ocr_rewards import OCRRewardCalculator as OCRRewardComputer
from src.env.rule_based_scorer import RuleBasedVehicleScorer


def flatten_observation(obs_dict: dict, max_vehicles: int = 32) -> np.ndarray:
    """
    将观测字典转换为扁平化张量

    Args:
        obs_dict: 环境返回的观测字典
        max_vehicles: 最大车辆数（用于padding）

    Returns:
        flattened_obs: [obs_dim] 扁平化观测
            obs_dim = max_vehicles * 9 + 32 + 1
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    vehicle_ids = obs_dict.get('vehicle_ids', [])
    icv_ids = obs_dict.get('icv_ids', set())
    global_stats = obs_dict.get('global_stats', np.zeros(32))

    num_vehicles = len(vehicle_ids)

    # 提取9维车辆特征（归一化）
    vehicle_features = np.zeros((max_vehicles, 9), dtype=np.float32)

    for i, veh_id in enumerate(vehicle_ids[:max_vehicles]):
        if veh_id in vehicle_states:
            state = vehicle_states[veh_id]
            vehicle_features[i, 0] = state.get('s', 0.0) / 1000.0
            vehicle_features[i, 1] = state.get('d', 0.0) / 10.0
            vehicle_features[i, 2] = state.get('vs', 0.0) / 30.0
            vehicle_features[i, 3] = state.get('vd', 0.0) / 10.0
            vehicle_features[i, 4] = state.get('speed', 0.0) / 30.0
            vehicle_features[i, 5] = state.get('acceleration', 0.0) / 3.0
            vehicle_features[i, 6] = state.get('lane_index', 0.0) / 10.0
            vehicle_features[i, 7] = state.get('angle', 0.0) / 360.0
            vehicle_features[i, 8] = 1.0 if veh_id in icv_ids else 0.0

    # 确保global_stats是32维
    global_stats_flat = global_stats.flatten()
    if len(global_stats_flat) < 32:
        global_stats_flat = np.concatenate([
            global_stats_flat,
            np.zeros(32 - len(global_stats_flat), dtype=np.float32)
        ])
    elif len(global_stats_flat) > 32:
        global_stats_flat = global_stats_flat[:32]

    # 扁平化并拼接
    vehicle_features_flat = vehicle_features.flatten()  # [max_vehicles * 9]
    num_vehicles_array = np.array([num_vehicles], dtype=np.float32)

    flattened_obs = np.concatenate([
        vehicle_features_flat,
        global_stats_flat,
        num_vehicles_array
    ])

    return flattened_obs


def convert_actions_to_dict(action_array: np.ndarray, vehicle_ids: list) -> dict:
    """
    将扁平动作数组转换为字典格式

    Args:
        action_array: [max_vehicles * 2] 扁平动作数组
        vehicle_ids: 车辆ID列表

    Returns:
        actions_dict: {vehicle_id: [acceleration, lane_change]}
    """
    max_vehicles = action_array.shape[0] // 2
    actions_reshaped = action_array.reshape(max_vehicles, 2)

    actions_dict = {}
    for i, veh_id in enumerate(vehicle_ids[:max_vehicles]):
        if i < max_vehicles:
            actions_dict[veh_id] = actions_reshaped[i]

    return actions_dict


def select_icv_by_rules(obs_dict: dict, rule_scorer: RuleBasedVehicleScorer,
                        k_ratio: float = 0.10) -> set:
    """
    使用规则选择器选择ICV

    Args:
        obs_dict: 观测字典
        rule_scorer: 规则评分器
        k_ratio: K值比例

    Returns:
        icv_ids: 选择的ICV车辆ID集合
    """
    vehicle_states = obs_dict.get('vehicle_states', {})
    all_vehicle_ids = obs_dict.get('vehicle_ids', [])

    if not all_vehicle_ids:
        return set()

    # 使用规则评分器计算所有车辆的评分
    context = {
        'traci_lib': None,  # 规则评分器可以在没有TraCI的情况下工作
        'all_vehicle_ids': all_vehicle_ids
    }

    scores = rule_scorer.compute_scores(vehicle_states, context)

    # 选择Top-K车辆
    k = max(5, int(len(all_vehicle_ids) * k_ratio))
    k = min(k, len(all_vehicle_ids))

    # 按分数排序
    sorted_vehicles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 选择Top-K
    icv_ids = set([veh_id for veh_id, _ in sorted_vehicles[:k]])

    return icv_ids


class PPOTrainer:
    """
    PPO训练器（Stage 2 - 基于规则的ICV选择）
    """

    def __init__(self, policy, env, config, device='cuda',
                 use_cache=False, force_refresh=False, cache_dir=None):
        """
        Args:
            policy: JointICVPolicy
            env: CompetitionSumoEnv
            config: 配置字典
            device: 设备
            use_cache: 是否使用缓存（规则模式默认关闭）
            force_refresh: 是否强制刷新缓存
            cache_dir: 缓存目录
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.device = device
        self.use_cache = use_cache
        self.force_refresh = force_refresh
        self.cache_dir = cache_dir

        # 🔥 新增：初始化规则选择器
        self.rule_scorer = RuleBasedVehicleScorer(config=config)
        print("[PPOTrainer] 规则ICV选择器已初始化")

        # PPO配置
        ppo_config = config['stage2_guided_exploration']['ppo']

        self.num_iterations = ppo_config['num_iterations']
        self.num_steps_per_iteration = ppo_config['num_steps_per_iteration']
        self.batch_size = ppo_config['batch_size']
        self.num_minibatches = ppo_config['num_minibatches']

        self.gamma = ppo_config['gamma']
        self.gae_lambda = ppo_config['gae_lambda']
        self.clip_param = ppo_config['clip_param']

        self.update_epochs = ppo_config['update_epochs']
        self.entropy_coef = ppo_config['entropy_coef']
        self.vf_coef = ppo_config['vf_coef']

        # 优化器
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=ppo_config['lr'],  # 直接从 ppo_config 读取
            weight_decay=ppo_config['weight_decay']  # 直接从 ppo_config 读取
        )

        # 学习率调度器
        lr_schedule_config = ppo_config.get('lr_schedule', {})
        if lr_schedule_config.get('enabled', False):
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=lr_schedule_config.get('start_factor', 0.1),
                total_iters=lr_schedule_config.get('total_iters', 100),
                end_factor=lr_schedule_config.get('end_factor', 1.0)
            )
        else:
            self.scheduler = None

        # 混合精度训练
        self.scaler = GradScaler()

        # 奖励计算（初赛配置）
        # 根据初赛规则：Pint=1, Wstability=0，只优化效率
        max_vehicles = config['environment']['icv_config']['max_vehicles']
        penetration_rate = config['environment']['icv_config']['penetration_rate']
        num_icv_total = int(max_vehicles * penetration_rate)

        self.reward_computer = OCRRewardComputer(
            num_icv_total=num_icv_total,  # 600 * 0.10 = 60
            w_efficiency=1.0,              # 🔥 初赛：只关心效率
            w_stability=0.0,                # 🔥 初赛：不关心稳定性
            use_improved_reward=True        # 使用改进的即时奖励
        )

        # Rollout buffer
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

    def collect_rollouts(self, num_steps, cache_dir=None, use_cache=True, force_refresh=False):
        """
        收集rollout数据（支持缓存）

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
            cache_dir = Path("cache/stage2_rollouts")
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)

        # 创建缓存key（基于参数）
        cache_params = {
            'num_steps': num_steps,
            'stage': 'stage2',
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

        # 重置buffer为空列表（每次收集rollouts时）
        self.buffer = {
            'observations': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': []
        }

        obs = self.env.reset()  # 只返回一个值

        # 🔥 修复：预热环境，让车辆出发
        # reset后vehicle_ids可能是空的，需要step一次让车辆开始仿真
        if len(obs.get('vehicle_ids', [])) == 0:
            obs, _, done, _ = self.env.step({})
            # 不记录这一步的数据

        episode_rewards = []

        for step in tqdm(range(num_steps), desc="收集rollouts"):
            # 将观测字典转换为扁平化张量
            obs_flat = flatten_observation(obs, max_vehicles=self.policy.max_vehicles)
            obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 前向传播
            with torch.no_grad():
                outputs = self.policy(obs_tensor, deterministic=False)

            action = outputs['actions'][0].cpu().numpy()
            value = outputs['value'][0, 0].cpu().item()
            log_prob = outputs['log_prob'][0].cpu().item()

            # 将动作转换为字典格式
            # 🔥 基于规则的ICV选择：使用规则选择器而不是环境返回的icv_ids
            rule_selected_icv_ids = select_icv_by_rules(obs, self.rule_scorer, k_ratio=0.10)
            vehicle_ids = list(rule_selected_icv_ids) if rule_selected_icv_ids else obs.get('vehicle_ids', [])
            action_dict = convert_actions_to_dict(action, vehicle_ids)

            # 执行动作
            next_obs, reward, done, info = self.env.step(action_dict)
            truncated = False  # 环境不返回truncated，统一使用done

            # 存储到buffer
            self.buffer['observations'].append(obs_flat.copy())
            self.buffer['actions'].append(action.copy())
            self.buffer['log_probs'].append(log_prob)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['dones'].append(done or truncated)

            obs = next_obs

            if done or truncated:
                obs = self.env.reset()  # 只返回一个值
                # 预热环境，让车辆出发
                if len(obs.get('vehicle_ids', [])) == 0:
                    obs, _, done, _ = self.env.step({})

        # 转换为tensor
        for key in self.buffer.keys():
            self.buffer[key] = np.array(self.buffer[key])

        # 保存到缓存
        if use_cache or force_refresh:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.buffer, f)
                cache_size_kb = cache_file.stat().st_size / 1024
                print(f"[缓存] Rollout已缓存: {cache_file} ({cache_size_kb:.1f} KB)")
            except Exception as e:
                print(f"[缓存] 保存失败: {e}")

        return self.buffer

    def compute_gae(self, rewards, values, dones):
        """
        计算GAE优势

        Args:
            rewards: [T]
            values: [T]
            dones: [T]

        Returns:
            advantages: [T]
            returns: [T]
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        returns = np.zeros_like(rewards)
        last_return = 0

        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            returns[t] = last_return = rewards[t] + self.gamma * (1 - dones[t]) * last_return

        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update_policy(self, rollouts):
        """
        更新策略

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
        dones = rollouts['dones']

        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones)

        # 🔥 保存原始returns用于显示Mean Return
        original_returns = returns.copy()

        # 🔥 方案1: 归一化returns（降低Value Loss，帮助Value网络学习）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 转换为tensor
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32).to(self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32).to(self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32).to(self.device)

        # Minibatch更新
        batch_size = observations.shape[0]
        minibatch_size = batch_size // self.num_minibatches

        total_policy_loss = 0
        total_value_loss = 0
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

                # 重新计算log_prob和entropy
                self.policy.train()
                outputs = self.policy(mb_obs, deterministic=False)

                new_log_probs = outputs['log_prob']
                entropy = outputs['entropy']  # 使用正确的熵（高斯分布的熵）
                value = outputs['value']

                # 计算ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # PPO clip loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                # 🔥 方案2: 使用Huber Loss (smooth_l1_loss)替代MSE，对异常值更鲁棒
                value_loss = nn.functional.smooth_l1_loss(value.squeeze(1), mb_returns)

                # 总损失
                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config['stage2_guided_exploration']['ppo']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 统计
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # 更新学习率
        self.scheduler.step()

        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_reward': rewards.mean(),
            'mean_return': original_returns.mean()  # 🔥 使用原始returns（未归一化）
        }

        return metrics

    def train(self):
        """训练循环"""
        print("\n" + "=" * 80)
        print("开始Stage 2 PPO训练")
        print("=" * 80)

        best_reward = float('-inf')
        checkpoint_dir = Path(self.config['global']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(self.num_iterations):
            print(f"\n迭代 {iteration + 1}/{self.num_iterations}")

            # 收集rollouts
            rollouts = self.collect_rollouts(
                self.num_steps_per_iteration,
                cache_dir=self.cache_dir,
                use_cache=self.use_cache,
                force_refresh=self.force_refresh
            )

            # 更新策略
            metrics = self.update_policy(rollouts)

            print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
            print(f"  Value Loss: {metrics['value_loss']:.4f}")
            print(f"  Entropy: {metrics['entropy']:.4f}")
            print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
            print(f"  Mean Return: {metrics['mean_return']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存检查点
            if (iteration + 1) % self.config['stage2_guided_exploration']['checkpoint']['save_interval'] == 0:
                checkpoint_path = checkpoint_dir / f'stage2_rule_iter_{iteration+1}.pth'
                torch.save(self.policy.state_dict(), checkpoint_path)
                print(f"  检查点已保存: {checkpoint_path}")

            # 保存最佳模型
            if metrics['mean_return'] > best_reward:
                best_reward = metrics['mean_return']
                best_checkpoint = checkpoint_dir / 'stage2_rule_best.pth'
                torch.save(self.policy.state_dict(), best_checkpoint)
                print(f"  ✅ 最佳模型已保存: {best_checkpoint}")

        print("\n" + "=" * 80)
        print("🎉 Stage 2 规则选择器训练完成！")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Rule-Based ICV Selection Training")
    parser.add_argument('--config', type=str, default='configs/v5_complete.yaml',
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复检查点路径（Stage 1的WorldModel）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')

    # 缓存相关参数
    parser.add_argument('--use_cache', type=lambda x: x.lower() == 'true', default=False,
                        help='是否使用缓存 (True/False, 默认: False) - PPO训练应禁用缓存')
    parser.add_argument('--force_refresh', action='store_true',
                        help='强制刷新缓存，重新收集数据')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='自定义缓存目录路径')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("Stage 2: 基于规则的ICV选择训练")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"设备: {args.device}")
    print(f"\n📌 特点：使用规则选择器选择ICV，策略只学习控制")

    if args.resume:
        print(f"\n恢复Stage 1检查点: {args.resume}")

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

    print(f"策略参数量: {sum(p.numel() for p in policy.parameters()):,}")

    # 加载Stage 1权重（如果有）
    if args.resume and config['stage2_guided_exploration']['resume_from_stage1']:
        print(f"\n加载Stage 1 WorldModel权重...")
        # 注意：这里需要从Stage 1检查点加载WorldModel权重
        # 简化：创建一个新的WorldModel并加载权重
        world_model = create_world_model(
            hidden_dim=config['policy']['hidden_dim'],
            latent_dim=config['stage1_world_observer']['world_model']['latent_dim'],
            num_vehicles=config['environment']['icv_config']['max_vehicles'],
            device=args.device
        )

        stage1_checkpoint = torch.load(args.resume, map_location=args.device)
        world_model.load_state_dict(stage1_checkpoint['world_model'])
        print(f"  ✅ WorldModel权重已加载")

        # 启用WorldModel
        policy.use_world_model = True
        policy.world_model = world_model
        print(f"  ✅ WorldModel已启用")

    # 创建训练器
    trainer = PPOTrainer(
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


if __name__ == '__main__':
    main()
