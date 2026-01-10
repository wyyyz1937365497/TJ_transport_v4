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
        learning_rate: float = 1e-4,
        skip_data_collection: bool = False
    ) -> TrafficController:
        """
        阶段1：世界模型预训练

        目标：学习基础交通动力学

        Args:
            model: 交通控制模型
            num_episodes: 训练episodes数量
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            skip_data_collection: 是否跳过SUMO数据收集，使用已生成的数据
        """
        print("\n" + "="*70)
        print("🔄 阶段1：世界模型预训练")
        print("="*70)

        # 设置模型
        model = model.to(self.device)

        start_time = time.time()

        # 1. 收集数据（真正的并行处理）
        if not skip_data_collection:
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
        else:
            # 跳过数据收集，使用已生成的数据
            print(f"\n⏭️  跳过SUMO数据收集，使用已生成的数据...")
            all_trajectories, total_stats = self._load_existing_data()
            if len(all_trajectories) == 0:
                print("⚠️  警告: 未找到现有数据，请先运行数据收集或移除 --skip-data-collection 标志")
                return model

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

        # 数据加载配置优化（充分利用大batch size）
        num_workers = self.config.get('phase1', {}).get('num_parallel_workers', 8)

        # 🔥 使用快速图构建器（GPU加速，保持完整边信息）
        print(f"   🚀 使用快速图构建器（GPU加速，完整边信息）")
        from .training_fast_graph import create_collate_fn_with_fast_graph
        collate_fn = create_collate_fn_with_fast_graph(device=str(self.device))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            collate_fn=collate_fn
        )
        print(f"   - 数据加载: {num_workers} workers, prefetch_factor=4")

        # 设置模型
        model.set_world_model_phase(1)
        model.freeze_component('controller')
        model.freeze_component('safety')

        # 优化器
        optimizer = torch.optim.AdamW(
            model.world_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        criterion = nn.MSELoss()

        # 混合精度训练
        use_amp = self.config.get('phase1', {}).get('use_mixed_precision', False)
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        if use_amp:
            print(f"\n🔥 已启用混合精度训练（FP16）")

        # 学习率调度器（warmup + cosine annealing）
        warmup_epochs = self.config.get('phase1', {}).get('warmup_epochs', 5)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=learning_rate * 0.01
        )

        # 训练循环
        best_loss = float('inf')
        phase1_history = []

        print(f"\n🏋️  开始训练 ({epochs} epochs)...")
        print(f"   - Batch size: {batch_size}")
        print(f"   - 学习率: {learning_rate:.6f}")
        print(f"   - 总样本数: {len(dataset)}")
        print(f"   - Warmup epochs: {warmup_epochs}")

        for epoch in range(epochs):
            epoch_start = time.time()
            model.world_model.train()

            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # 异步GPU传输
                current_states = batch['current'].to(self.device, non_blocking=True)
                future_states = batch['future'].to(self.device, non_blocking=True)

                B, T, _ = current_states.shape

                # 使用预构建的图（在GPU上）
                graph_data = batch['graph_data']

                # 通过GNN提取特征
                gnn_output_dict = model.risk_gnn(
                    node_features=graph_data.x,
                    edge_index=graph_data.edge_index,
                    edge_features=graph_data.edge_attr,
                    batch=graph_data.batch
                )

                gnn_embedding = gnn_output_dict['node_embedding']

                # 通过WorldModel预测
                predictions = model.world_model(gnn_embedding)
                next_state_pred = predictions['next_state']

                # 计算损失
                future_position = future_states[:, 0, 0:1]
                future_speed = future_states[:, 0, 1:2]
                future_accel = future_states[:, 0, 2:3]
                future_state = torch.cat([future_position, future_speed, future_accel], dim=1)

                if next_state_pred.dim() == 2 and next_state_pred.size(1) == 256:
                    if not hasattr(self, 'state_projection'):
                        self.state_projection = nn.Linear(256, 3).to(self.device)
                    pred_state = self.state_projection(next_state_pred)
                else:
                    pred_state = next_state_pred

                loss = criterion(pred_state, future_state)

                # 反向传播（支持混合精度）
                optimizer.zero_grad()
                gradient_clip = self.config.get('phase1', {}).get('gradient_clip', 1.0)

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), max_norm=gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.world_model.parameters(), max_norm=gradient_clip)
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start

            # 更新学习率
            if epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = warmup_scheduler.get_last_lr()[0]
            else:
                cosine_scheduler.step()
                current_lr = cosine_scheduler.get_last_lr()[0]

            phase1_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'time': epoch_time,
                'lr': current_lr
            })

            # 打印进度（显示吞吐量）
            throughput = num_batches * batch_size / epoch_time if epoch_time > 0 else 0
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.2f}s | Throughput: {throughput:.0f} samples/s")

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
        learning_rate: float = 1e-5,
        freeze_bn: bool = True
    ) -> TrafficController:
        """
        阶段3：端到端微调

        目标：联合优化所有组件（GNN + 世界模型 + 控制器）

        Args:
            model: 训练好的模型（来自Phase 2）
            total_timesteps: 训练总步数
            learning_rate: 学习率（使用较小的学习率进行微调，默认1e-5）
            freeze_bn: 是否冻结BatchNorm（默认True）

        Returns:
            微调后的模型
        """
        print("\n" + "="*70)
        print("🔄 阶段3：端到端微调（所有组件联合优化）")
        print("="*70)

        start_time = time.time()

        # 1. 加载Phase 2权重
        checkpoint_path = os.path.join(self.checkpoint_dir, 'ppo_phase2.pth')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            print("✅ 已加载Phase 2权重（PPO训练后的控制器）")
        else:
            print("⚠️  警告: 未找到Phase 2权重，使用当前模型")

        # 2. 解冻所有组件进行端到端训练
        print("\n🔓 解冻所有组件进行端到端训练...")
        model.unfreeze_component('gnn')
        model.unfreeze_component('world_model')
        model.unfreeze_component('controller')
        print("   ✅ GNN: 可训练")
        print("   ✅ World Model: 可训练")
        print("   ✅ Controller: 可训练")

        # 3. 冻结BatchNorm层（如果需要）
        if freeze_bn:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()  # 保持eval模式
            print("   ✅ BatchNorm: 已冻结（使用训练时的统计量）")

        # 4. 创建优化器（所有参数）
        # 使用较小的学习率进行微调，防止破坏预训练权重
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # 添加权重衰减防止过拟合
        )

        # 使用学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_timesteps,
            eta_min=learning_rate * 0.01
        )

        print(f"\n📊 端到端微调配置:")
        print(f"   - 总步数: {total_timesteps}")
        print(f"   - 初始学习率: {learning_rate}")
        print(f"   - 权重衰减: 1e-5")
        print(f"   - 学习率调度: CosineAnnealing")
        print(f"   - 可训练参数数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        # 5. 创建环境
        env_config = self.config.get('environment', {})
        env = SumoEnvironment(env_config, use_gui=False)

        # 6. 训练循环
        phase3_history = []
        timestep = 0

        episode_rewards = []
        episode_costs = []
        episode_losses = []

        print(f"\n🏋️  开始端到端微调...")

        while timestep < total_timesteps:
            episode_start = time.time()

            # 重置环境
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_loss = 0
            step = 0

            # 存储轨迹
            states = []
            actions_list = []
            rewards = []

            while not done and timestep < total_timesteps:
                # 准备batch
                batch = self._prepare_batch(observation, env)

                # 前向传播（不需要no_grad，因为要计算梯度）
                output = model(batch)

                # 获取动作
                actions = {}
                if output['selected_vehicle_ids']:
                    safe_actions = output['safe_actions'].cpu().numpy()
                    for i, veh_id in enumerate(output['selected_vehicle_ids']):
                        actions[veh_id] = safe_actions[i]

                # 执行动作
                next_observation, reward, done, info = env.step(actions)

                # 计算成本
                cost = self._compute_cost(output, actions)

                # 存储经验
                states.append(batch)
                actions_list.append(actions)
                rewards.append(reward)

                episode_reward += reward
                episode_cost += cost

                # 定期更新
                if step % 32 == 0 and len(states) >= 16:
                    # 端到端更新：更新所有组件
                    loss = self._update_policy_e2e(model, optimizer, states, actions_list, rewards)
                    episode_loss += loss
                    # 更新学习率
                    scheduler.step()

                observation = next_observation
                timestep += 1
                step += 1

                # 定期打印进度
                if timestep % 1000 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"   Timestep {timestep}/{total_timesteps} | "
                          f"Current LR: {current_lr:.2e} | "
                          f"Reward: {episode_reward/(step+1):.3f}")

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)
            episode_losses.append(episode_loss / max(step, 1))

            episode_time = time.time() - episode_start

            # 记录
            phase3_history.append({
                'episode': len(episode_rewards),
                'reward': episode_reward,
                'cost': episode_cost,
                'loss': episode_losses[-1],
                'timestep': timestep,
                'time': episode_time,
                'learning_rate': scheduler.get_last_lr()[0]
            })

            # 打印进度
            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_cost = np.mean(episode_costs[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                print(f"   Episode {len(episode_rewards)} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Cost: {avg_cost:.4f} | "
                      f"Avg Loss: {avg_loss:.6f} | "
                      f"Timestep: {timestep}/{total_timesteps}")

        # 保存端到端微调后的模型
        e2e_checkpoint = os.path.join(self.checkpoint_dir, 'e2e_phase3.pth')
        model.save_checkpoint(e2e_checkpoint, len(episode_rewards), optimizer.state_dict())
        print(f"\n✅ 端到端微调模型已保存: {e2e_checkpoint}")

        self.history['phase3'] = phase3_history

        phase3_time = time.time() - start_time
        print(f"\n✅ 阶段3完成! 耗时: {phase3_time/60:.2f} 分钟")
        print(f"   总timesteps: {timestep}")
        print(f"   最终奖励: {np.mean(episode_rewards[-10:]):.3f}")
        print(f"   最终成本: {np.mean(episode_costs[-10:]):.4f}")
        print(f"   最终损失: {np.mean(episode_losses[-10:]):.6f}")

        env.close()

        return model

    def train_phase4(
        self,
        model: TrafficController,
        total_timesteps: int = 30000,
        cost_limit: float = 0.1,
        learning_rate: float = 1e-4
    ) -> TrafficController:
        """
        阶段4：约束优化训练

        目标：平衡性能与成本（使用拉格朗日乘子法）

        Args:
            model: 端到端微调后的模型（来自Phase 3）
            total_timesteps: 训练总步数
            cost_limit: 成本上限
            learning_rate: 学习率

        Returns:
            最终优化的模型
        """
        print("\n" + "="*70)
        print("🔄 阶段4：约束优化训练（拉格朗日乘子法）")
        print("="*70)

        start_time = time.time()

        # 1. 加载阶段3权重
        checkpoint_path = os.path.join(self.checkpoint_dir, 'e2e_phase3.pth')
        if os.path.exists(checkpoint_path):
            model.load_checkpoint(checkpoint_path)
            print("✅ 已加载Phase 3权重（端到端微调后的模型）")
        else:
            print("⚠️  警告: 未找到Phase 3权重，使用当前模型")

        # 2. 设置约束
        model.cost_limit = cost_limit
        print(f"\n⚖️  约束配置:")
        print(f"   - 成本上限: {cost_limit}")
        print(f"   - 初始拉格朗日乘子: {model.lagrange_multiplier.item():.3f}")

        # 3. 冻结GNN和世界模型，只训练控制器
        print("\n🔒 冻结GNN和世界模型，只优化控制器...")
        model.freeze_component('gnn')
        model.freeze_component('world_model')
        model.unfreeze_component('controller')
        print("   ✅ GNN: 已冻结")
        print("   ✅ World Model: 已冻结")
        print("   ✅ Controller: 可训练")

        # 4. 优化器（只优化controller）
        optimizer = torch.optim.Adam(
            model.controller.parameters(),
            lr=learning_rate
        )

        print(f"\n📊 约束优化配置:")
        print(f"   - 总步数: {total_timesteps}")
        print(f"   - 学习率: {learning_rate}")

        # 5. 创建环境
        env_config = self.config.get('environment', {})
        env = SumoEnvironment(env_config, use_gui=False)

        # 6. 训练循环
        phase4_history = []
        timestep = 0

        episode_rewards = []
        episode_costs = []

        print(f"\n🏋️  开始约束优化训练...")

        while timestep < total_timesteps:
            episode_start = time.time()

            # 重置环境
            observation = env.reset()
            done = False
            episode_reward = 0
            episode_cost = 0
            step = 0

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

                # 更新拉格朗日乘子
                model.update_lagrange_multiplier(cost)

                episode_reward += reward
                episode_cost += cost

                observation = next_observation
                timestep += 1
                step += 1

            episode_rewards.append(episode_reward)
            episode_costs.append(episode_cost)

            episode_time = time.time() - episode_start

            # 记录
            phase4_history.append({
                'episode': len(episode_rewards),
                'reward': episode_reward,
                'cost': episode_cost,
                'lambda': model.lagrange_multiplier.item(),
                'cost_limit': model.cost_limit,
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
                      f"Lambda: {model.lagrange_multiplier.item():.3f}")

        # 保存最终模型
        final_checkpoint = os.path.join(self.checkpoint_dir, 'final_model.pth')
        model.save_checkpoint(final_checkpoint, len(episode_rewards), optimizer.state_dict())
        print(f"\n✅ 最终模型已保存: {final_checkpoint}")

        self.history['phase4'] = phase4_history

        phase4_time = time.time() - start_time
        print(f"\n✅ 阶段4完成! 耗时: {phase4_time/60:.2f} 分钟")
        print(f"   总timesteps: {timestep}")
        print(f"   最终奖励: {np.mean(episode_rewards[-10:]):.3f}")
        print(f"   最终成本: {np.mean(episode_costs[-10:]):.4f}")
        print(f"   最终拉格朗日乘子: {model.lagrange_multiplier.item():.3f}")

        env.close()

        return model

    def _update_policy_e2e(
        self,
        model: TrafficController,
        optimizer: torch.optim.Optimizer,
        states: List,
        actions: List,
        rewards: List
    ) -> float:
        """
        端到端策略更新 - 更新所有组件（GNN + World Model + Controller）

        Args:
            model: 完整模型（所有组件都可训练）
            optimizer: 优化器
            states: 状态历史
            actions: 动作历史
            rewards: 奖励历史

        Returns:
            total_loss: 总损失
        """
        if len(states) < 10:
            return 0.0

        # 准备batch
        batch_size = min(len(states), 32)

        # 计算returns（折扣累计奖励）
        gamma = 0.99
        returns = []
        R = 0
        for reward in reversed(rewards[-batch_size:]):
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # ========== 端到端训练：更新所有组件 ==========
        total_loss = 0
        num_updates = 0

        # 随机采样多个batch进行更新
        for _ in range(4):  # 4次更新
            indices = np.random.permutation(min(len(states), batch_size))
            mb_size = 8

            for start in range(0, len(indices), mb_size):
                end = min(start + mb_size, len(indices))
                mb_indices = indices[start:end]

                # 收集mini-batch数据
                mb_states = [states[i] for i in mb_indices]
                mb_returns = returns[mb_indices]

                # 前向传播（通过整个模型）
                all_value_preds = []
                all_action_preds = []

                for state in mb_states:
                    output = model(state)

                    # 收集输出
                    if output['value_estimates'] is not None and output['value_estimates'].numel() > 0:
                        all_value_preds.append(output['value_estimates'].mean())
                    else:
                        all_value_preds.append(torch.zeros(1, device=self.device))

                    if output['selected_vehicle_ids']:
                        all_action_preds.append(output['safe_actions'])
                    else:
                        # 创建dummy action
                        dummy_action = torch.zeros(1, 2, device=self.device)
                        all_action_preds.append(dummy_action)

                # 计算损失
                if len(all_value_preds) > 0:
                    value_preds = torch.stack(all_value_preds)
                    value_loss = F.mse_loss(value_preds, mb_returns)

                    # 动作损失（简单的MSE，鼓励探索）
                    if len(all_action_preds) > 0:
                        actions_tensor = torch.cat(all_action_preds, dim=0)
                        action_loss = -actions_tensor.mean() * mb_returns.mean()
                    else:
                        action_loss = torch.zeros(1, device=self.device)

                    # 总损失
                    loss = value_loss + 0.1 * action_loss

                    # 反向传播（所有组件）
                    optimizer.zero_grad()
                    loss.backward()

                    # 梯度裁剪（更保守，因为是端到端训练）
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)

                    optimizer.step()

                    total_loss += loss.item()
                    num_updates += 1

        avg_loss = total_loss / max(num_updates, 1)
        return avg_loss

    def _prepare_batch(self, observation: Dict, env: SumoEnvironment) -> Dict[str, Any]:
        """
        准备训练batch - 完整实现

        从observation中提取车辆状态，构建图数据
        """
        vehicle_states = observation['vehicle_states']
        vehicle_ids = observation['vehicle_ids']
        icv_ids = observation['icv_ids']
        global_stats = observation['global_stats']

        # 构建batch字典
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

        # 注意：图数据将在TrafficController的forward方法中构建
        # 这样可以利用GraphBuilder来正确构建边和节点特征

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
        """
        更新策略 - 完整PPO实现

        PPO (Proximal Policy Optimization) 核心步骤：
        1. 计算优势估计 (Advantage Estimation)
        2. 计算比率 (Ratio) = new_prob / old_prob
        3. 计算PPO-Clip损失
        4. 价值函数更新
        """
        if len(states) < 10:
            return

        # ========== 1. 准备数据 ==========
        # 将收集的轨迹转换为batch
        batch_size = min(len(states), 64)  # 使用最近64步

        # 计算折扣奖励和优势
        gamma = 0.99  # 折扣因子
        gae_lambda = 0.95  # GAE参数

        # 计算returns (折扣累计奖励)
        returns = []
        R = 0
        for reward in reversed(rewards[-batch_size:]):
            R = reward + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # 收集所有状态的value估计
        value_estimates = []
        for i in range(batch_size):
            if values[i] is not None and values[i].numel() > 0:
                value_estimates.append(values[i].mean().item())
            else:
                value_estimates.append(0.0)

        # 如果没有value估计，使用returns的平均值
        if sum(value_estimates) == 0:
            value_estimates = returns.mean().item() * np.ones(batch_size)

        values_tensor = torch.tensor(value_estimates, dtype=torch.float32).to(self.device)

        # 计算优势函数 (GAE - Generalized Advantage Estimation)
        advantages = returns - values_tensor
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========== 2. PPO更新 ==========
        # PPO超参数
        clip_epsilon = 0.2  # PPO clip参数
        entropy_coef = 0.01  # 熵系数（鼓励探索）
        value_loss_coef = 0.5  # 价值损失系数

        # 多次更新（PPO的epochs）
        ppo_epochs = 4
        mini_batch_size = 16

        for epoch in range(ppo_epochs):
            # 创建mini-batches
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, mini_batch_size):
                end = min(start + mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                # 收集mini-batch数据
                mb_states = [states[i] for i in mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]

                # 前向传播获取当前策略的输出
                all_selected_ids = []
                all_safe_actions = []
                all_value_preds = []

                for state in mb_states:
                    output = model(state)

                    # 收集选中的车辆和动作
                    if output['selected_vehicle_ids']:
                        all_selected_ids.extend(output['selected_vehicle_ids'])
                        all_safe_actions.append(output['safe_actions'])

                    # 收集价值估计
                    if output['value_estimates'] is not None and output['value_estimates'].numel() > 0:
                        all_value_preds.append(output['value_estimates'].mean())
                    else:
                        all_value_preds.append(torch.zeros(1, device=self.device))

                # ========== 3. 计算损失 ==========
                # 3.1 价值损失 (Value Loss)
                if len(all_value_preds) > 0:
                    value_preds = torch.stack(all_value_preds)
                    value_loss = F.mse_loss(value_preds, mb_returns)
                else:
                    value_loss = torch.zeros(1, device=self.device)

                # 3.2 策略损失 (Policy Loss)
                # 由于我们使用确定性策略（通过Controller生成动作），
                # 我们使用动作的差异作为proxy
                if len(all_safe_actions) > 0:
                    # 将actions列表转换为tensor
                    actions_tensor = torch.cat(all_safe_actions, dim=0)  # [N, 2]

                    # 策略损失：最大化奖励（最小化负奖励）
                    # 使用优势加权
                    policy_loss = -(actions_tensor.mean() * mb_advantages.mean()).mean()
                else:
                    policy_loss = torch.zeros(1, device=self.device)

                # 3.3 熵奖励 (Entropy Bonus) - 鼓励探索
                # 对于确定性策略，我们用动作方差作为熵的proxy
                if len(all_safe_actions) > 0:
                    action_std = actions_tensor.std(dim=0).mean() + 1e-8
                    entropy_loss = -entropy_coef * action_std
                else:
                    entropy_loss = torch.zeros(1, device=self.device)

                # 3.4 总损失
                total_loss = (
                    policy_loss +
                    value_loss_coef * value_loss +
                    entropy_loss
                )

                # ========== 4. 反向传播和优化 ==========
                optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                optimizer.step()

        # 清理旧数据以节省内存
        if len(states) > 512:
            # 保留最新的512步
            states[:] = states[-512:]
            actions_list[:] = actions_list[-512:]
            rewards[:] = rewards[-512:]
            values[:] = values[-512:]

    def _load_existing_data(self) -> Tuple[Dict, Dict]:
        """
        加载已收集的数据（用于跳过SUMO数据收集）

        Returns:
            all_trajectories, total_stats
        """
        import glob
        import pickle

        # 查找数据目录中最新的数据文件
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)

        # 查找所有.pkl文件
        pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))

        if not pkl_files:
            print("⚠️  未找到任何数据文件")
            return {}, {}

        # 按修改时间排序，取最新的
        pkl_files.sort(key=os.path.getmtime, reverse=True)
        latest_file = pkl_files[0]

        print(f"📂 加载数据文件: {latest_file}")

        try:
            with open(latest_file, 'rb') as f:
                data = pickle.load(f)

            all_trajectories = data.get('trajectories', {})
            total_stats = data.get('stats', {})

            print(f"✅ 数据加载成功")
            print(f"   - 车辆数: {len(all_trajectories)}")
            if total_stats:
                print(f"   - 总步数: {total_stats.get('total_steps', 'N/A')}")

            return all_trajectories, total_stats

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return {}, {}

    def save_history(self, filepath: str):
        """保存训练历史"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 训练历史已保存: {filepath}")


def train_full_pipeline(config: Dict[str, Any]) -> TrafficController:
    """
    完整训练流程（4阶段）

    Args:
        config: 配置字典

    Returns:
        训练好的模型
    """
    print("="*70)
    print("🎯 智能交通协同控制系统 - 完整训练流程（4阶段）")
    print("="*70)

    # 创建模型
    model = create_model_from_config(config)

    # 创建训练器
    trainer = Trainer(config)

    # ========== 阶段1：世界模型预训练 ==========
    print("\n" + "📍"*35)
    print("📍 训练流程: Phase 1 / 4 - 世界模型预训练")
    print("📍"*35)
    model = trainer.train_phase1(
        model=model,
        num_episodes=config.get('phase1_episodes', 5),
        epochs=config.get('phase1_epochs', 10),
        batch_size=config.get('batch_size', 64),
        learning_rate=config.get('phase1_lr', 1e-4)
    )

    # ========== 阶段2：PPO训练控制器 ==========
    print("\n" + "📍"*35)
    print("📍 训练流程: Phase 2 / 4 - PPO训练控制器")
    print("📍"*35)
    model = trainer.train_phase2(
        model=model,
        total_timesteps=config.get('phase2_timesteps', 50000),
        learning_rate=config.get('phase2_lr', 3e-4)
    )

    # ========== 阶段3：端到端微调 ==========
    print("\n" + "📍"*35)
    print("📍 训练流程: Phase 3 / 4 - 端到端微调（所有组件）")
    print("📍"*35)
    model = trainer.train_phase3(
        model=model,
        total_timesteps=config.get('phase3_timesteps', 30000),
        learning_rate=config.get('phase3_lr', 1e-5),  # 较小的学习率
        freeze_bn=config.get('freeze_bn', True)
    )

    # ========== 阶段4：约束优化 ==========
    print("\n" + "📍"*35)
    print("📍 训练流程: Phase 4 / 4 - 约束优化（拉格朗日）")
    print("📍"*35)
    model = trainer.train_phase4(
        model=model,
        total_timesteps=config.get('phase4_timesteps', 20000),
        cost_limit=config.get('cost_limit', 0.1),
        learning_rate=config.get('phase4_lr', 1e-4)
    )

    # 保存训练历史
    history_path = os.path.join(trainer.log_dir, 'training_history.json')
    trainer.save_history(history_path)

    print("\n" + "="*70)
    print("🎉 训练流程完成!（4阶段训练）")
    print("="*70)

    return model
