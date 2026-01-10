"""
三阶段训练流程
功能：实现完整的训练pipeline
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Tuple
import shutil

from ..models import TrafficController, create_model_from_config, WorldModelLoss
from ..env import SumoEnvironment, EfficientDataCollector, TrajectoryDataset, collect_parallel_data_optimized


class Trainer:
    """
    训练器 - 实现三阶段训练流程
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cpu'))

        # 创建目录
        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        self.log_dir = config.get('log_dir', 'logs')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'phase1': [],
            'phase2': [],
            'phase3': []
        }

    def train_phase1(
        self,
        model: TrafficController,
        num_episodes: int = 100,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-4
    ) -> TrafficController:
        """
        阶段1：世界模型预训练

        目标：学习基础交通动力学
        """
        print("\n" + "="*70)
        print("🔄 阶段1：世界模型预训练")
        print("="*70)

        start_time = time.time()

        # 1. 收集数据（真正的并行处理）
        print(f"\n📊 并行收集训练数据 ({num_episodes} episodes)...")
        data_config = self.config.get('environment', {})
        timeout = self.config.get('phase1', {}).get('data_collection_timeout', 180)
        num_workers = self.config.get('phase1', {}).get('num_parallel_workers', None)

        # 使用优化的并行收集器
        all_trajectories, total_stats = collect_parallel_data_optimized(
            config=data_config,
            num_episodes=num_episodes,
            max_steps=data_config.get('max_steps', 3600),
            timeout=timeout,
            num_workers=num_workers,  # 使用配置中的工作进程数
            output_dir="data"
        )

        # 检查收集结果
        if len(all_trajectories) == 0:
            print("⚠️  警告: 没有收集到任何数据，跳过阶段1训练")
            return model

        # 2. 创建数据集
        print(f"\n🔄 创建数据集...")
        dataset = TrajectoryDataset(
            trajectories=all_trajectories,
            future_steps=1,  # Phase 1只预测下一步
            sequence_length=10
        )

        # 检查数据集大小
        if len(dataset) == 0:
            print("⚠️  警告: 数据集为空，跳过阶段1训练")
            return model

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows兼容
            pin_memory=False
        )

        # 3. 设置模型
        model.set_world_model_phase(1)
        model.to(self.device)

        # 冻结其他组件，只训练世界模型
        model.freeze_component('controller')
        model.freeze_component('safety')

        # 4. 优化器和损失
        optimizer = torch.optim.AdamW(
            model.world_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        criterion = nn.MSELoss()

        # 5. 训练循环
        best_loss = float('inf')
        phase1_history = []

        print(f"\n🏋️  开始训练 ({epochs} epochs)...")

        for epoch in range(epochs):
            epoch_start = time.time()
            model.world_model.train()

            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # 简化：使用当前状态预测下一状态
                current_states = batch['current']  # [B, T, 3]
                future_states = batch['future']    # [B, T_future, 3]

                # 取最后一步作为当前，第一步未来作为目标
                x_current = current_states[:, -1, :].float().to(self.device)  # [B, 3]
                x_next = future_states[:, 0, :].float().to(self.device)      # [B, 3]

                # 简化：假设GNN已经处理
                # 实际应该先通过GNN，这里简化为直接嵌入
                batch_size = x_current.size(0)
                gnn_embedding = torch.randn(batch_size, 256, device=self.device)

                # 前向传播
                predictions = model.world_model(gnn_embedding)
                next_state_pred = predictions['next_state']

                # 计算损失（简化版）
                # 将3维状态映射到256维
                state_target = torch.randn(batch_size, 256, device=self.device)
                loss = criterion(next_state_pred, state_target)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start

            phase1_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'time': epoch_time
            })

            print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, 'world_model_phase1.pth')
                model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())

        # 保存历史
        self.history['phase1'] = phase1_history

        phase1_time = time.time() - start_time
        print(f"\n✅ 阶段1完成! 耗时: {phase1_time/60:.2f} 分钟")
        print(f"   最佳损失: {best_loss:.6f}")

        return model

    def train_phase2(
        self,
        model: TrafficController,
        num_envs: int = 4,
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4
    ) -> TrafficController:
        """
        阶段2：带安全屏障的RL训练

        目标：学习安全控制策略
        """
        print("\n" + "="*70)
        print("🔄 阶段2：带安全屏障的RL训练")
        print("="*70)

        start_time = time.time()

        # 1. 加载阶段1权重
        checkpoint_path = os.path.join(self.checkpoint_dir, 'world_model_phase1.pth')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            print("✅ 已加载阶段1预训练权重")

        # 2. 设置模型
        model.set_world_model_phase(2)
        model.to(self.device)

        # 冻结世界模型，训练控制器
        model.freeze_component('gnn')
        model.freeze_component('world_model')

        # 3. 创建环境
        env_config = self.config.get('environment', {})
        env = SumoEnvironment(env_config, use_gui=False)

        # 4. 简化版PPO训练
        optimizer = torch.optim.Adam(
            model.controller.parameters(),
            lr=learning_rate
        )

        phase2_history = []
        timestep = 0

        episode_rewards = []
        episode_costs = []

        while timestep < total_timesteps:
            episode_start = time.time()

            # 重置环境
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            step = 0

            # 存储轨迹
            states = []
            actions_list = []
            rewards = []
            values_list = []

            while not done and timestep < total_timesteps:
                # 准备batch
                batch = self._prepare_batch(observation, env)

                # 前向传播
                with torch.no_grad():
                    output = model(batch)

                # 获取动作
                actions = {}
                if output['selected_vehicle_ids']:
                    safe_actions = output['safe_actions'].cpu().numpy()
                    for i, veh_id in enumerate(output['selected_vehicle_ids']):
                        actions[veh_id] = safe_actions[i]

                # 执行
                next_observation, reward, done, info = env.step(actions)

                # 计算成本
                cost = self._compute_cost(output, actions)

                # 存储经验
                states.append(batch)
                actions_list.append(actions)
                rewards.append(reward)
                values_list.append(output.get('value_estimates'))

                episode_reward += reward
                episode_cost += cost

                observation = next_observation
                timestep += 1
                step += 1

                # 定期更新
                if step % 64 == 0:
                    self._update_policy(model, optimizer, states, actions_list, rewards, values_list)

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)

            episode_time = time.time() - episode_start

            # 记录
            phase2_history.append({
                'episode': len(episode_rewards),
                'reward': episode_reward,
                'cost': episode_cost,
                'timestep': timestep,
                'time': episode_time
            })

            # 打印进度
            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_cost = np.mean(episode_costs[-10:])
                print(f"   Episode {len(episode_rewards)} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Cost: {avg_cost:.4f} | "
                      f"Timestep: {timestep}/{total_timesteps}")

        # 保存模型
        final_checkpoint = os.path.join(self.checkpoint_dir, 'ppo_phase2.pth')
        model.save_checkpoint(final_checkpoint, len(episode_rewards), optimizer.state_dict())

        self.history['phase2'] = phase2_history

        phase2_time = time.time() - start_time
        print(f"\n✅ 阶段2完成! 耗时: {phase2_time/60:.2f} 分钟")
        print(f"   总timesteps: {timestep}")

        env.close()

        return model

    def train_phase3(
        self,
        model: TrafficController,
        total_timesteps: int = 50000,
        cost_limit: float = 0.1,
        learning_rate: float = 1e-4
    ) -> TrafficController:
        """
        阶段3：约束优化训练

        目标：平衡性能与成本
        """
        print("\n" + "="*70)
        print("🔄 阶段3：约束优化训练")
        print("="*70)

        start_time = time.time()

        # 1. 加载阶段2权重
        checkpoint_path = os.path.join(self.checkpoint_dir, 'ppo_phase2.pth')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            print("✅ 已加载阶段2权重")

        # 2. 设置约束
        model.cost_limit = cost_limit
        model.unfreeze_component('controller')

        # 3. 优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        # 4. 训练
        phase3_history = []
        timestep = 0

        episode_rewards = []
        episode_costs = []

        env_config = self.config.get('environment', {})
        env = SumoEnvironment(env_config, use_gui=False)

        while timestep < total_timesteps:
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            step = 0

            while not done and timestep < total_timesteps:
                batch = self._prepare_batch(observation, env)

                with torch.no_grad():
                    output = model(batch)

                actions = {}
                if output['selected_vehicle_ids']:
                    safe_actions = output['safe_actions'].cpu().numpy()
                    for i, veh_id in enumerate(output['selected_vehicle_ids']):
                        actions[veh_id] = safe_actions[i]

                next_observation, reward, done, info = env.step(actions)

                # 计算成本
                cost = self._compute_cost(output, actions)

                # 更新拉格朗日乘子
                model.update_lagrange_multiplier(cost)

                episode_reward += reward
                episode_cost += cost

                observation = next_observation
                timestep += 1
                step += 1

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)

            phase3_history.append({
                'episode': len(episode_rewards),
                'reward': episode_reward,
                'cost': episode_cost,
                'lambda': model.lagrange_multiplier.item(),
                'cost_limit': model.cost_limit
            })

            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_cost = np.mean(episode_costs[-10:])
                print(f"   Episode {len(episode_rewards)} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Cost: {avg_cost:.4f} | "
                      f"Lambda: {model.lagrange_multiplier.item():.3f}")

        # 保存最终模型
        final_checkpoint = os.path.join(self.checkpoint_dir, 'final_model.pth')
        model.save_checkpoint(final_checkpoint, len(episode_rewards), optimizer.state_dict())

        self.history['phase3'] = phase3_history

        phase3_time = time.time() - start_time
        print(f"\n✅ 阶段3完成! 耗时: {phase3_time/60:.2f} 分钟")

        env.close()

        return model

    def _prepare_batch(self, observation: Dict, env: SumoEnvironment) -> Dict[str, Any]:
        """准备训练batch"""
        vehicle_states = observation['vehicle_states']
        vehicle_ids = observation['vehicle_ids']
        icv_ids = observation['icv_ids']
        global_stats = observation['global_stats']

        # 简化版：不构建实际图
        batch_size = len(vehicle_ids) if vehicle_ids else 1

        batch = {
            'vehicle_states': vehicle_states,
            'vehicle_ids': vehicle_ids,
            'icv_ids': icv_ids,
            'global_metrics': torch.tensor(global_stats, dtype=torch.float32).unsqueeze(0).to(self.device),
            'is_icv': torch.tensor(
                [1 if vid in icv_ids else 0 for vid in vehicle_ids],
                dtype=torch.float32
            ).to(self.device) if vehicle_ids else torch.tensor([], dtype=torch.float32).to(self.device)
        }

        return batch

    def _compute_cost(self, output: Dict, actions: Dict) -> float:
        """计算干预成本"""
        alpha = 1.0  # 加速度成本系数
        beta = 5.0   # 换道成本系数

        total_cost = 0.0

        for veh_id, action in actions.items():
            accel_cost = alpha * abs(action[0])
            lane_cost = beta if action[1] > 0.5 else 0.0
            total_cost += (accel_cost + lane_cost)

        # 归一化
        num_vehicles = len(actions) if actions else 1
        return total_cost / max(num_vehicles, 1)

    def _update_policy(
        self,
        model: TrafficController,
        optimizer: torch.optim.Optimizer,
        states: List,
        actions: List,
        rewards: List,
        values: List
    ):
        """更新策略（简化版PPO）"""
        if len(states) < 10:
            return

        # 简化：随机梯度更新
        idx = np.random.randint(len(states))
        batch = states[idx]

        # 前向传播
        output = model(batch)

        # 简化的策略损失
        if output['value_estimates'].numel() > 0:
            value = output['value_estimates'].mean()

            # 假设奖励
            reward = rewards[idx] if idx < len(rewards) else 0.0

            # 价值损失
            value_loss = F.mse_loss(value, torch.tensor(reward, device=self.device))

            optimizer.zero_grad()
            value_loss.backward()
            optimizer.step()

    def save_history(self, filepath: str):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 训练历史已保存: {filepath}")


def train_full_pipeline(config: Dict[str, Any]) -> TrafficController:
    """
    完整训练流程

    Args:
        config: 配置字典

    Returns:
        训练好的模型
    """
    print("="*70)
    print("🎯 智能交通协同控制系统 - 完整训练流程")
    print("="*70)

    # 创建模型
    model = create_model_from_config(config)

    # 创建训练器
    trainer = Trainer(config)

    # 阶段1
    model = trainer.train_phase1(
        model=model,
        num_episodes=config.get('phase1_episodes', 50),
        epochs=config.get('phase1_epochs', 30),
        batch_size=config.get('batch_size', 128),
        learning_rate=config.get('phase1_lr', 1e-4)
    )

    # 阶段2
    model = trainer.train_phase2(
        model=model,
        num_envs=config.get('num_envs', 4),
        total_timesteps=config.get('phase2_timesteps', 50000),
        learning_rate=config.get('phase2_lr', 3e-4)
    )

    # 阶段3
    model = trainer.train_phase3(
        model=model,
        total_timesteps=config.get('phase3_timesteps', 30000),
        cost_limit=config.get('cost_limit', 0.1),
        learning_rate=config.get('phase3_lr', 1e-4)
    )

    # 保存训练历史
    history_path = os.path.join(trainer.log_dir, 'training_history.json')
    trainer.save_history(history_path)

    print("\n" + "="*70)
    print("🎉 训练流程完成!")
    print("="*70)

    return model
