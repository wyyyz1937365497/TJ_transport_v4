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
from ..utils.multi_gpu import setup_multi_gpu_training, adjust_hyperparameters_for_multi_gpu, print_gpu_info, monitor_gpu_usage


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

        # 打印GPU信息并设置多GPU训练
        print_gpu_info()

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if num_gpus > 1:
            print(f"\n🚀 检测到 {num_gpus} 张GPU，启用多GPU训练")
            model = setup_multi_gpu_training(model)

            # 调整超参数
            phase1_config = {
                'batch_size': batch_size,
                'lr': learning_rate
            }
            adjusted_config = adjust_hyperparameters_for_multi_gpu(phase1_config, num_gpus)
            batch_size = adjusted_config['batch_size']
            learning_rate = adjusted_config['lr']
        else:
            model = model.to(self.device)

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
            num_workers=4 if num_gpus > 1 else 0,  # 多GPU时增加数据加载进程
            pin_memory=True if num_gpus > 1 else False,  # 加速GPU数据传输
            persistent_workers=True if num_gpus > 1 else False
        )

        # 3. 设置模型（已经设置过多GPU了，不需要再.to(device)）
        # 注意：DataParallel包装后需要通过.module访问原始模型
        model_to_use = model.module if hasattr(model, 'module') else model
        model_to_use.set_world_model_phase(1)

        # 冻结其他组件，只训练世界模型
        model_to_use.freeze_component('controller')
        model_to_use.freeze_component('safety')

        # 4. 优化器和损失
        # 注意：需要通过.module访问模型的参数
        actual_model = model.module if hasattr(model, 'module') else model

        optimizer = torch.optim.AdamW(
            actual_model.world_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        criterion = nn.MSELoss()

        # 🔥 关键修复：为多GPU训练创建包装模块
        if hasattr(model, 'module'):
            # 如果模型已被DataParallel包装，创建一个辅助模块来包装world_model
            # 这样DataParallel可以正确分配工作到多个GPU
            class WorldModelWrapper(nn.Module):
                """包装world_model以支持DataParallel"""
                def __init__(self, world_model):
                    super().__init__()
                    self.world_model = world_model

                def forward(self, gnn_embedding):
                    return self.world_model(gnn_embedding)

            wrapped_world_model = WorldModelWrapper(actual_model.world_model)
            # 对这个包装器应用DataParallel
            if num_gpus > 1:
                wrapped_world_model = nn.DataParallel(
                    wrapped_world_model,
                    device_ids=list(range(num_gpus)),
                    output_device=None
                )
                wrapped_world_model.to(self.device)
                print(f"\n🔥 WorldModel已启用多GPU训练 ({num_gpus} 张GPU)")
            else:
                wrapped_world_model = wrapped_world_model.to(self.device)
        else:
            # 单GPU情况
            wrapped_world_model = actual_model.world_model

        # 5. 训练循环
        best_loss = float('inf')
        phase1_history = []

        print(f"\n🏋️  开始训练 ({epochs} epochs)...")
        print(f"   - Batch size: {batch_size} (已为{num_gpus}GPU优化)")
        print(f"   - 学习率: {learning_rate:.6f} (已为{num_gpus}GPU优化)")
        print(f"   - 总样本数: {len(dataset)}")

        for epoch in range(epochs):
            epoch_start = time.time()
            actual_model.world_model.train()

            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(dataloader):
                # ========== 完整实现：使用真实特征 ==========
                current_states = batch['current']  # [B, T, 3] where T=sequence_length
                future_states = batch['future']    # [B, T_future, 3] where T_future=future_steps

                # 提取车道和时间信息
                current_lane_ids = batch['current_lane_ids']      # [B, T]
                current_lane_indices = batch['current_lane_indices']  # [B, T]
                current_timestamps = batch['current_timestamps']  # [B, T]

                B, T, _ = current_states.shape

                # 1. 从当前状态序列中提取最后一步的状态
                # current_states: [B, T, 3] -> [B, 3]
                # 每个样本包含：[位置, 速度, 加速度]
                current_position = current_states[:, -1, 0:1].float().to(self.device)  # [B, 1]
                current_speed = current_states[:, -1, 1:2].float().to(self.device)    # [B, 1]
                current_accel = current_states[:, -1, 2:3].float().to(self.device)   # [B, 1]

                # 2. 提取车道信息（真实数据！）
                # current_lane_ids: [B, T] -> 取最后一步 -> [B]
                # current_lane_indices: [B, T] -> 取最后一步 -> [B]
                if isinstance(current_lane_ids, np.ndarray):
                    current_lane_ids = current_lane_ids[:, -1]  # [B]
                    current_lane_indices = current_lane_indices[:, -1]  # [B]
                else:
                    current_lane_ids = torch.tensor(current_lane_ids[:, -1]) if hasattr(current_lane_ids, '__len__') else torch.zeros(B, dtype=torch.long)
                    current_lane_indices = torch.tensor(current_lane_indices[:, -1]) if hasattr(current_lane_indices, '__len__') else torch.zeros(B, dtype=torch.long)

                # 3. 提取时间信息（真实数据！）
                if isinstance(current_timestamps, np.ndarray):
                    current_time = current_timestamps[:, -1]  # [B]
                else:
                    current_time = torch.tensor(current_timestamps[:, -1]) if hasattr(current_timestamps, '__len__') else torch.zeros(B)

                # 归一化时间（除以最大时间360秒）
                current_time_normalized = torch.tensor(current_time, dtype=torch.float32).to(self.device) / 360.0  # [B]

                # 4. 计算序列统计特征（从历史序列计算）
                # 位置、速度、加速度的标准差
                position_std = current_states[:, :, 0].std(dim=1, keepdim=True).float().to(self.device)  # [B, 1]
                speed_std = current_states[:, :, 1].std(dim=1, keepdim=True).float().to(self.device)      # [B, 1]
                accel_std = current_states[:, :, 2].std(dim=1, keepdim=True).float().to(self.device)     # [B, 1]

                # 5. 构建完整节点特征 [B, node_dim=9] - 使用真实数据！
                # Node features: [pos_x, pos_y, speed, accel, lane_id, pos_std, speed_std, accel_std, time]
                node_features_list = [
                    current_position,                           # [B, 1] - x位置（真实）
                    torch.zeros_like(current_position),         # [B, 1] - y位置（假设直线道路）
                    current_speed,                              # [B, 1] - 速度（真实）
                    current_accel,                              # [B, 1] - 加速度（真实）
                    current_lane_indices.unsqueeze(1).float().to(self.device),  # [B, 1] - 车道索引（真实！）
                    position_std,                               # [B, 1] - 位置std（真实计算）
                    speed_std,                                  # [B, 1] - 速度std（真实计算）
                    accel_std,                                  # [B, 1] - 加速度std（真实计算）
                    current_time_normalized.unsqueeze(1)        # [B, 1] - 时间（真实！）
                ]

                node_features = torch.cat(node_features_list, dim=1)  # [B, 9]

                # 6. 构建图结构（使用实际的车道信息）
                actual_model = model.module if hasattr(model, 'module') else model

                # 创建车辆状态字典（使用真实数据）
                vehicle_states_dict = {}
                for i in range(B):
                    veh_id = f"veh_{i}"
                    vehicle_states_dict[veh_id] = {
                        'position': float(current_position[i].cpu().numpy()),
                        'speed': float(current_speed[i].cpu().numpy()),
                        'acceleration': float(current_accel[i].cpu().numpy()),
                        'lane_id': str(int(current_lane_ids[i])),  # 真实车道ID
                        'lane_index': int(current_lane_indices[i]),  # 真实车道索引
                        'road_id': 'E0'
                    }

                # 使用GraphBuilder构建图
                icv_ids = set(vehicle_states_dict.keys())  # 所有车辆都是ICV（训练时）
                graph_data = actual_model.graph_builder.build_graph(vehicle_states_dict, icv_ids)

                # 7. 通过GNN提取特征（多GPU支持）
                if num_gpus > 1 and hasattr(model, 'module'):
                    gnn_output = actual_model.risk_gnn(
                        node_features=graph_data.x.to(self.device),
                        edge_index=graph_data.edge_index.to(self.device),
                        edge_features=graph_data.edge_attr.to(self.device),
                        batch=graph_data.batch.to(self.device) if hasattr(graph_data, 'batch') else None
                    )
                else:
                    gnn_output = actual_model.risk_gnn(
                        node_features=graph_data.x.to(self.device),
                        edge_index=graph_data.edge_index.to(self.device),
                        edge_features=graph_data.edge_attr.to(self.device),
                        batch=graph_data.batch.to(self.device) if hasattr(graph_data, 'batch') else None
                    )

                # GNN输出: [num_nodes, gnn_output_dim=256]
                gnn_embedding = gnn_output  # [B, 256]

                # 8. 通过WorldModel预测未来状态（多GPU）
                predictions = wrapped_world_model(gnn_embedding)
                next_state_pred = predictions['next_state']

                # 9. 计算真实的损失
                # 未来状态: [B, T_future, 3]
                future_position = future_states[:, 0, 0:1].float().to(self.device)  # [B, 1]
                future_speed = future_states[:, 0, 1:2].float().to(self.device)    # [B, 1]
                future_accel = future_states[:, 0, 2:3].float().to(self.device)   # [B, 1]

                # 将未来状态拼接为 [B, 3]
                future_state = torch.cat([future_position, future_speed, future_accel], dim=1)  # [B, 3]

                # 预测状态需要与未来状态对齐
                if next_state_pred.dim() == 2 and next_state_pred.size(1) == 256:
                    if not hasattr(self, 'state_projection'):
                        self.state_projection = nn.Linear(256, 3).to(self.device)

                    pred_state = self.state_projection(next_state_pred)  # [B, 3]
                else:
                    pred_state = next_state_pred  # [B, 3]

                # 计算MSE损失
                loss = criterion(pred_state, future_state)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actual_model.world_model.parameters(), max_norm=1.0)
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

            # 打印进度
            print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s")

            # 定期显示GPU使用情况
            if num_gpus > 1 and (epoch + 1) % 5 == 0:
                monitor_gpu_usage()

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                checkpoint_path = os.path.join(self.checkpoint_dir, 'world_model_phase1.pth')

                # 保存时使用 actual_model（已经是解包后的）
                actual_model.save_checkpoint(checkpoint_path, epoch, optimizer.state_dict())

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
